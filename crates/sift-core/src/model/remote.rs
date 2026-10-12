//! Reading a GGUF tensor directory over HTTP, without downloading the model.
//!
//! A GGUF file puts its entire tensor directory at the front: magic, metadata, then one
//! record per tensor. Everything needed to reason about a model — architecture, expert
//! count, per-expert size, dtype mix — is in the first few megabytes of what may be a
//! 90 GB file.
//!
//! So this module implements [`Read`] + [`Seek`] over HTTP range requests, which means
//! [`super::gguf::Gguf::parse`] works on a remote file **unchanged**. There is no second
//! parser to keep in sync, and a bug fixed for local files is fixed for remote ones.
//!
//! Prior art: [`gpustack/gguf-parser-go`](https://github.com/gpustack/gguf-parser-go) does
//! this in Go and does it well. This is the same idea in Rust, with machine measurement
//! and engine routing built on top.
//!
//! # Why `curl` rather than an HTTP crate
//!
//! It is already on every machine this targets, it handles TLS, redirects, proxies and
//! HuggingFace's auth headers, and it keeps the dependency tree small enough that the
//! whole crate builds in seconds. The cost is a subprocess per fetch, which is negligible
//! against network latency. It sits behind [`Fetch`], so tests drive [`RemoteFile`] with
//! scripted responses instead of a network.

use std::io::{self, Read, Seek, SeekFrom};
use std::process::{Command, Stdio};

use crate::cache::{Cache, Entry};

/// How much to fetch on the first request, unless the caller knows better.
///
/// Most GGUF directories fit comfortably here. Large-vocabulary models are the exception:
/// the tokenizer lives in metadata, and a 150k-token vocabulary can push the directory
/// past a megabyte on its own.
const INITIAL_FETCH: u64 = 1 << 20;

/// Ceiling on how much of a remote file we will pull while reading the directory.
///
/// This is a guard, not a budget. If parsing a directory needs more than this, something
/// is wrong with our understanding of the format and we should fail loudly rather than
/// quietly stream gigabytes over someone's connection.
const MAX_FETCH: u64 = 64 << 20;

/// Header bytes allowed on top of the body before a response is cut off.
///
/// Redirect chains to a signed CDN URL carry a few kilobytes of headers per hop.
const HEADER_ALLOWANCE: u64 = 64 << 10;

/// One HTTP response to a range request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    /// Status of the final response, after redirects. 304 means the validator matched.
    pub status: u16,
    pub body: Vec<u8>,
    /// Where `body` starts in the file, from `Content-Range`, when the server sent one.
    pub range_start: Option<u64>,
    /// Total file size from `Content-Range`, when the server reports one.
    pub total: Option<u64>,
    /// The `ETag` to send back next time.
    pub etag: Option<String>,
}

/// Something that can fetch a byte range of a URL.
///
/// The seam between [`RemoteFile`]'s bookkeeping and the network. [`CurlFetch`] is the
/// real one; tests supply scripted responses, which is how a 304, a server that ignores
/// `Range`, a short body or an ETag that changes mid-read get exercised without a server.
pub trait Fetch: Send + Sync {
    /// Fetch `[start, end]` inclusive, optionally conditional on an ETag.
    ///
    /// A non-2xx answer other than 304 and 416 is an error; those two come back as
    /// responses, since the caller decides what they mean.
    fn range(
        &self,
        url: &str,
        start: u64,
        end: u64,
        validator: Option<&str>,
    ) -> io::Result<Fetched>;
}

/// [`Fetch`] through the system `curl`.
#[derive(Debug, Clone, Default)]
pub struct CurlFetch;

impl Fetch for CurlFetch {
    fn range(
        &self,
        url: &str,
        start: u64,
        end: u64,
        validator: Option<&str>,
    ) -> io::Result<Fetched> {
        let mut cmd = Command::new("curl");
        cmd.args(["-sSL", "--max-time", "60", "-D", "-", "-r"])
            .arg(format!("{start}-{end}"));
        // No `--fail`: 304 is the answer a validator hopes for and 416 means "past the
        // end", so the status is read here instead of being turned into an exit code.
        if let Some(etag) = validator {
            cmd.arg("-H").arg(format!("If-None-Match: {etag}"));
        }
        cmd.arg(url).stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| io::Error::other(format!("running curl: {e}")))?;
        let mut stdout = child.stdout.take().expect("stdout is piped");

        // Read with a ceiling. A server that ignores `Range` answers 200 with the whole
        // file, and without a ceiling that is a 90 GB download into memory.
        let cap = (end - start + 1).saturating_add(HEADER_ALLOWANCE);
        let mut out = Vec::new();
        let mut chunk = vec![0u8; 64 << 10];
        loop {
            let n = stdout.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..n]);
            if out.len() as u64 > cap {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::other(format!(
                    "fetching {url}: the server sent more than the {} bytes asked for, so it \
                     is ignoring the Range header; refusing to download the whole file",
                    end - start + 1
                )));
            }
        }
        let mut stderr = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut stderr);
        }
        let status = child.wait()?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "fetching {url}: {}",
                if stderr.trim().is_empty() {
                    "curl failed".to_string()
                } else {
                    stderr.trim().to_string()
                }
            )));
        }

        let response = parse_response(&out);
        if response.status >= 400 && response.status != 416 {
            return Err(io::Error::other(format!(
                "fetching {url}: HTTP {}",
                response.status
            )));
        }
        Ok(response)
    }
}

/// A remote file readable through HTTP range requests.
///
/// Bytes are fetched lazily into one growing prefix held in memory. Seeking backwards is
/// free; reading past what has been fetched fetches the missing tail.
pub struct RemoteFile {
    url: String,
    /// Prefix of the file fetched so far.
    buf: Vec<u8>,
    /// Current cursor, which may sit beyond `buf.len()`.
    pos: u64,
    /// Total size from the server, if it reported one.
    total: Option<u64>,
    /// The validator the bytes in `buf` were served under.
    etag: Option<String>,
    /// Bytes actually transferred, so callers can prove we did not download the model.
    fetched: u64,
    /// Size of the first request.
    initial: u64,
    /// Cached prefix from a previous run, awaiting validation.
    ///
    /// Held rather than used until the server confirms it: the tool's claim is that it
    /// reads the real header of the real file, and serving an unvalidated cache would
    /// undercut the reason to use it.
    cached: Option<Entry>,
    /// Where [`Self::persist`] writes, if anywhere.
    cache: Option<Cache>,
    /// Whether `buf` holds bytes the cache does not have yet.
    dirty: bool,
    fetcher: Box<dyn Fetch>,
}

impl RemoteFile {
    /// Open a remote file through `curl` and the user's header cache. No bytes are fetched
    /// until the first read.
    pub fn new(url: impl Into<String>) -> Self {
        Self::with_fetcher(url, Box::new(CurlFetch), Cache::user())
    }

    /// Open a remote file through any [`Fetch`], with an optional cache.
    pub fn with_fetcher(
        url: impl Into<String>,
        fetcher: Box<dyn Fetch>,
        cache: Option<Cache>,
    ) -> Self {
        let url = url.into();
        let cached = cache.as_ref().and_then(|c| c.load(&url));
        Self {
            url,
            buf: Vec::new(),
            pos: 0,
            total: None,
            etag: None,
            fetched: 0,
            initial: INITIAL_FETCH,
            cached,
            cache,
            dirty: false,
            fetcher,
        }
    }

    /// Size the first request.
    ///
    /// A safetensors reader wants exactly 8 bytes and then exactly its header; the second
    /// and later shards of a split GGUF carry no tokenizer and need far less than the
    /// first. Either way, a megabyte up front is a megabyte wasted per file.
    pub fn with_initial_fetch(mut self, bytes: u64) -> Self {
        self.initial = bytes.max(1);
        self
    }

    /// Total bytes transferred so far.
    ///
    /// The point of this module is that this stays tiny relative to the file. Tests assert
    /// on it, so a regression that starts downloading payload fails rather than merely
    /// being slow.
    pub fn bytes_fetched(&self) -> u64 {
        self.fetched
    }

    /// Total file size as reported by the server, if known.
    pub fn total_size(&self) -> Option<u64> {
        self.total
    }

    /// The URL being read.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Record what was read in the header cache.
    ///
    /// Called by the reader once it has parsed the bytes successfully, never before. A
    /// truncated or garbled body stored under a valid ETag would be served — and
    /// revalidated with a 304 — on every later run.
    pub fn persist(&mut self) {
        if !self.dirty {
            return;
        }
        if let Some(cache) = &self.cache {
            cache.store(&self.url, &self.buf, self.total, self.etag.as_deref());
        }
        self.dirty = false;
    }

    /// Whether the whole file is already held.
    fn complete(&self) -> bool {
        self.total.is_some_and(|t| self.buf.len() as u64 >= t)
    }

    /// Ensure at least `want` bytes of prefix are buffered, or the whole file if shorter.
    fn ensure(&mut self, want: u64) -> io::Result<()> {
        if self.buf.len() as u64 >= want || self.complete() {
            return Ok(());
        }
        if want > MAX_FETCH {
            return Err(io::Error::other(format!(
                "refusing to fetch {want} bytes of {}: a model header should not need more \
                 than {MAX_FETCH}. The file may not be what its name says, or may be \
                 truncated.",
                self.url
            )));
        }
        if self.buf.is_empty() {
            self.fetch_from_start(want)
        } else {
            self.fetch_tail(want)
        }
    }

    /// First fetch: `[0, want)`, conditional on the cache when the cache could satisfy it.
    fn fetch_from_start(&mut self, want: u64) -> io::Result<()> {
        // Offer the cached ETag only when the cached prefix would answer this read: a 304
        // on a shorter prefix proves the file is unchanged and still leaves us without the
        // bytes. A cached file that is complete satisfies any read.
        let validator = self
            .cached
            .as_ref()
            .filter(|c| {
                c.body.len() as u64 >= want || c.total.is_some_and(|t| c.body.len() as u64 >= t)
            })
            .map(|c| c.etag.clone());
        let res = self
            .fetcher
            .range(&self.url, 0, want - 1, validator.as_deref())?;

        if res.status == 304 {
            // The server confirmed the bytes on disk are the bytes it has. Not counted as
            // fetched, because nothing was.
            let entry = match (validator, self.cached.take()) {
                (Some(_), Some(entry)) => entry,
                _ => {
                    return Err(io::Error::other(format!(
                        "fetching {}: the server answered 304 Not Modified to a request that \
                         carried no validator",
                        self.url
                    )))
                }
            };
            self.total = entry.total.or(self.total);
            self.etag = Some(entry.etag);
            self.buf = entry.body;
            return Ok(());
        }

        // A fresh body supersedes whatever was cached, so a later read does not offer a
        // validator for content we have already replaced.
        self.cached = None;
        self.accept(res, 0, want - 1, true)
    }

    /// Later fetches: only the missing tail, `[buf.len(), want)`.
    ///
    /// Re-fetching from zero each time paid for the prefix twice — or, on a directory that
    /// needs four doublings, four times.
    fn fetch_tail(&mut self, want: u64) -> io::Result<()> {
        let start = self.buf.len() as u64;
        let res = self.fetcher.range(&self.url, start, want - 1, None)?;

        if res.status == 416 {
            // Asked for bytes past the end: what we hold is the whole file.
            self.total = Some(res.total.unwrap_or(start).min(start));
            return Ok(());
        }
        // The file changed between requests. The prefix and the tail would describe two
        // different files, so start over rather than splice them.
        if let (Some(old), Some(new)) = (&self.etag, &res.etag) {
            if old != new {
                self.buf.clear();
                self.etag = None;
                self.total = None;
                return self.fetch_from_start(want);
            }
        }
        self.accept(res, start, want - 1, false)
    }

    /// Fold a response to `[start, end]` into the buffer.
    fn accept(&mut self, res: Fetched, start: u64, end: u64, first: bool) -> io::Result<()> {
        let asked = end - start + 1;
        match res.status {
            206 => {
                if res.range_start.is_some_and(|s| s != start) {
                    return Err(io::Error::other(format!(
                        "fetching {}: asked for bytes from {start}, got a range starting at {}",
                        self.url,
                        res.range_start.unwrap_or(0)
                    )));
                }
                if res.body.len() as u64 > asked {
                    return Err(io::Error::other(format!(
                        "fetching {}: the server sent {} bytes for a {asked}-byte range",
                        self.url,
                        res.body.len()
                    )));
                }
                // A short body is fine — the server may cap ranges — but an empty one that
                // does not reach the end would loop forever.
                if res.body.is_empty() && res.total.is_none_or(|t| start < t) {
                    return Err(io::Error::other(format!(
                        "fetching {}: an empty range response before the end of the file",
                        self.url
                    )));
                }
                self.fetched += res.body.len() as u64;
                self.total = res.total.or(self.total);
                if first || self.etag.is_none() {
                    self.etag = res.etag;
                }
                self.buf.extend_from_slice(&res.body);
            }
            200 => {
                // The server ignored `Range` and sent the file from its first byte. That is
                // only usable when the whole file is shorter than what we asked to hold —
                // anything longer was already cut off by the fetcher's ceiling, and this
                // check makes the rule hold for any `Fetch`.
                if res.body.len() as u64 > end + 1 {
                    return Err(io::Error::other(format!(
                        "fetching {}: the server ignored the Range header and sent {} bytes",
                        self.url,
                        res.body.len()
                    )));
                }
                self.fetched += res.body.len() as u64;
                self.total = Some(res.body.len() as u64);
                self.etag = res.etag;
                self.buf = res.body;
            }
            416 if first => {
                // An empty file.
                self.total = Some(0);
                self.buf.clear();
            }
            other => {
                return Err(io::Error::other(format!(
                    "fetching {}: unexpected HTTP {other} to a range request",
                    self.url
                )))
            }
        }
        self.dirty = true;
        Ok(())
    }
}

impl Read for RemoteFile {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let need = self.pos.saturating_add(out.len() as u64);

        // Grow geometrically so a directory slightly larger than the initial fetch does
        // not cause a request per read — but never below what this read needs, which is
        // what lets a safetensors header be fetched in exactly two requests.
        if (self.buf.len() as u64) < need {
            let target = need
                .max(self.initial)
                .max((self.buf.len() as u64).saturating_mul(2))
                .min(MAX_FETCH.max(need));
            self.ensure(target)?;
        }

        let start = self.pos as usize;
        if start >= self.buf.len() {
            return Ok(0); // genuine end of file
        }
        let n = out.len().min(self.buf.len() - start);
        out[..n].copy_from_slice(&self.buf[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for RemoteFile {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let new = match from {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => {
                // Seeking from the end needs the size; one small request gets it.
                let total = match self.total {
                    Some(t) => t,
                    None => {
                        self.ensure(1)?;
                        self.total.ok_or_else(|| {
                            io::Error::other(format!(
                                "{} did not report a size, so SeekFrom::End is unavailable",
                                self.url
                            ))
                        })?
                    }
                };
                total as i128 + d as i128
            }
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot seek before the start of the file",
            ));
        }
        self.pos = u64::try_from(new)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek past the end of u64"))?;
        Ok(self.pos)
    }
}

/// Split a curl `-D -` dump into header blocks and the body, and read what matters.
///
/// Redirects and `100 Continue` put several header blocks ahead of the body. Each block
/// ends at a blank line, and another block follows only if the next bytes start a status
/// line — so the split walks forward block by block from the start and never looks for a
/// blank line inside the body, which a binary body can contain.
///
/// Status and `Content-Range` describe the final response and come from the last block
/// only. The validator prefers HuggingFace's `x-linked-etag` from any block — it rides on
/// the 302, not on the CDN response that follows — and otherwise takes the final block's
/// own `etag`: a redirect hop's `etag` describes the redirect, not the file.
fn parse_response(raw: &[u8]) -> Fetched {
    let mut blocks: Vec<String> = Vec::new();
    let mut at = 0usize;
    while raw[at..].starts_with(b"HTTP/") {
        let Some(end) = find(&raw[at..], b"\r\n\r\n") else {
            // Headers that never end: treat what is there as the last block.
            blocks.push(String::from_utf8_lossy(&raw[at..]).into_owned());
            at = raw.len();
            break;
        };
        blocks.push(String::from_utf8_lossy(&raw[at..at + end + 4]).into_owned());
        at += end + 4;
    }
    let body = raw[at..].to_vec();
    let last = blocks.last().map(String::as_str).unwrap_or("");

    let header = |block: &str, name: &str| -> Option<String> {
        let prefix = format!("{name}:");
        block
            .lines()
            .rfind(|l| l.to_ascii_lowercase().starts_with(&prefix))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let content_range = header(last, "content-range");
    Fetched {
        status: parse_status(last).unwrap_or(200),
        body,
        range_start: content_range.as_deref().and_then(parse_range_start),
        total: content_range.as_deref().and_then(parse_range_total),
        etag: blocks
            .iter()
            .rev()
            .find_map(|b| header(b, "x-linked-etag"))
            .or_else(|| header(last, "etag")),
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Status code of a header block's status line.
fn parse_status(block: &str) -> Option<u16> {
    block
        .lines()
        .find(|l| l.starts_with("HTTP/"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
}

/// The total from a `Content-Range: bytes 0-1023/5242880` value.
fn parse_range_total(value: &str) -> Option<u64> {
    value.rsplit('/').next().and_then(|t| t.trim().parse().ok())
}

/// The first byte from a `Content-Range: bytes 1024-2047/5242880` value.
fn parse_range_start(value: &str) -> Option<u64> {
    let range = value.trim().strip_prefix("bytes")?.trim();
    range.split(['-', '/']).next()?.trim().parse().ok()
}

/// Build a HuggingFace resolve URL for a file in a repo.
pub fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn content_range_is_parsed_for_its_start_and_total() {
        assert_eq!(parse_range_total("bytes 0-1023/5242880"), Some(5_242_880));
        assert_eq!(parse_range_start("bytes 1024-2047/5242880"), Some(1024));
        // `bytes */1234` means unsatisfiable; the total is still after the slash.
        assert_eq!(parse_range_total("bytes */1234"), Some(1234));
        assert_eq!(parse_range_start("bytes */1234"), None);
    }

    #[test]
    fn content_range_is_matched_case_insensitively() {
        let r = parse_response(b"HTTP/1.1 206\r\nContent-Range: bytes 0-99/12345\r\n\r\n");
        assert_eq!(r.total, Some(12_345));
    }

    #[test]
    fn a_missing_content_range_yields_no_size() {
        assert_eq!(parse_response(b"HTTP/2 200\r\n\r\n").total, None);
    }

    #[test]
    fn redirect_blocks_are_skipped_and_the_body_follows_the_last_one() {
        let raw = b"HTTP/2 302\r\nlocation: /x\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 0-3/99\r\n\r\nBODY";
        let r = parse_response(raw);
        assert_eq!(r.body, b"BODY");
        assert_eq!(r.status, 206);
        assert_eq!(r.total, Some(99));
    }

    #[test]
    fn a_body_containing_a_blank_line_is_not_split_again() {
        // Regression: scanning for the *last* `\r\n\r\n` cut a binary body at whatever
        // happened to look like a blank line inside it.
        let raw = b"HTTP/2 206\r\ncontent-range: bytes 0-11/99\r\n\r\nAB\r\n\r\nCDEFGH";
        let r = parse_response(raw);
        assert_eq!(r.body, b"AB\r\n\r\nCDEFGH");
        assert_eq!(r.total, Some(99));
    }

    #[test]
    fn a_continue_block_is_not_mistaken_for_the_response() {
        let raw = b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 206 Partial\r\ncontent-range: bytes 0-1/2\r\n\r\nok";
        let r = parse_response(raw);
        assert_eq!((r.status, r.body.as_slice()), (206, &b"ok"[..]));
    }

    #[test]
    fn a_response_without_headers_is_all_body() {
        let r = parse_response(b"just body");
        assert_eq!(r.body, b"just body");
    }

    #[test]
    fn hf_urls_are_built_in_the_resolve_form() {
        assert_eq!(
            hf_url("allenai/OLMoE-1B-7B-0924-Instruct-GGUF", "olmoe-q4.gguf"),
            "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-q4.gguf"
        );
    }

    /// Every request a [`Fake`] was asked: `(start, end, validator)`.
    type RequestLog = Arc<Mutex<Vec<(u64, u64, Option<String>)>>>;

    /// A file served by a scripted server, recording every request it was asked.
    struct Fake {
        file: Vec<u8>,
        etag: Mutex<String>,
        /// Ignore `Range` and send the whole file with a 200.
        ignore_range: bool,
        /// Cap each range response at this many bytes.
        cap: Option<usize>,
        /// Answer 304 to any validator, whether or not it matches.
        always_304: bool,
        log: RequestLog,
    }

    impl Fake {
        fn new(file: Vec<u8>) -> Self {
            Self {
                file,
                etag: Mutex::new("\"v1\"".into()),
                ignore_range: false,
                cap: None,
                always_304: false,
                log: Arc::default(),
            }
        }
    }

    impl Fetch for Fake {
        fn range(
            &self,
            _: &str,
            start: u64,
            end: u64,
            validator: Option<&str>,
        ) -> io::Result<Fetched> {
            self.log
                .lock()
                .unwrap()
                .push((start, end, validator.map(str::to_owned)));
            let etag = self.etag.lock().unwrap().clone();
            if self.always_304 || validator == Some(etag.as_str()) {
                return Ok(Fetched {
                    status: 304,
                    body: vec![],
                    range_start: None,
                    total: None,
                    etag: Some(etag),
                });
            }
            let len = self.file.len() as u64;
            if self.ignore_range {
                return Ok(Fetched {
                    status: 200,
                    body: self.file.clone(),
                    range_start: None,
                    total: None,
                    etag: Some(etag),
                });
            }
            if start >= len {
                return Ok(Fetched {
                    status: 416,
                    body: vec![],
                    range_start: None,
                    total: Some(len),
                    etag: Some(etag),
                });
            }
            let mut stop = end.min(len - 1);
            if let Some(cap) = self.cap {
                stop = stop.min(start + cap as u64 - 1);
            }
            Ok(Fetched {
                status: 206,
                body: self.file[start as usize..=stop as usize].to_vec(),
                range_start: Some(start),
                total: Some(len),
                etag: Some(etag),
            })
        }
    }

    fn bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    fn open(fake: Fake, cache: Option<Cache>) -> RemoteFile {
        RemoteFile::with_fetcher("https://example.invalid/m.gguf", Box::new(fake), cache)
    }

    #[test]
    fn remote_growth_fetches_only_the_missing_tail() {
        let file = bytes(10_000);
        let fake = Fake::new(file.clone());
        let log = Arc::clone(&fake.log);
        let mut f = open(fake, None).with_initial_fetch(1000);

        let mut head = vec![0u8; 500];
        f.read_exact(&mut head).expect("first read");
        let mut rest = vec![0u8; 2000];
        f.read_exact(&mut rest).expect("second read");
        assert_eq!([head, rest].concat(), file[..2500]);
        let log = log.lock().unwrap();
        assert_eq!(log[0].0, 0, "first request from zero");
        assert_eq!(
            log[1].0, 1000,
            "second request starts where the buffer ends"
        );
        assert_eq!(f.bytes_fetched(), 2500, "no byte paid for twice");
    }

    #[test]
    fn remote_a_capped_server_is_read_through_in_short_bodies() {
        let file = bytes(5000);
        let fake = Fake {
            cap: Some(700),
            ..Fake::new(file.clone())
        };
        let mut f = open(fake, None).with_initial_fetch(4096);
        let mut out = Vec::new();
        f.read_to_end(&mut out).expect("read");
        assert_eq!(out, file);
    }

    #[test]
    fn remote_a_small_file_from_a_server_without_range_support_is_accepted() {
        let file = bytes(300);
        let fake = Fake {
            ignore_range: true,
            ..Fake::new(file.clone())
        };
        let mut f = open(fake, None);
        let mut out = Vec::new();
        f.read_to_end(&mut out).expect("read");
        assert_eq!(out, file);
        assert_eq!(f.total_size(), Some(300));
    }

    #[test]
    fn remote_a_server_ignoring_range_on_a_large_file_is_an_error() {
        let fake = Fake {
            ignore_range: true,
            ..Fake::new(bytes(5000))
        };
        let mut f = open(fake, None).with_initial_fetch(100);
        let mut out = [0u8; 10];
        let err = f.read_exact(&mut out).expect_err("must refuse");
        assert!(err.to_string().contains("ignored the Range"), "{err}");
    }

    #[test]
    fn remote_a_304_serves_the_cached_body_without_counting_it_as_fetched() {
        let d = tempfile::tempdir().expect("temp dir");
        let cache = Cache::at(d.path());
        let file = bytes(3000);
        cache.store(
            "https://example.invalid/m.gguf",
            &file[..2000],
            Some(3000),
            Some("\"v1\""),
        );

        let fake = Fake::new(file.clone());
        let log = Arc::clone(&fake.log);
        let mut f = open(fake, Some(cache)).with_initial_fetch(1000);
        let mut out = vec![0u8; 1500];
        f.read_exact(&mut out).expect("read");
        assert_eq!(out, file[..1500]);
        assert_eq!(f.bytes_fetched(), 0);
        assert_eq!(log.lock().unwrap()[0].2.as_deref(), Some("\"v1\""));
    }

    #[test]
    fn remote_a_complete_cached_small_file_is_revalidated_not_refetched() {
        // Regression: a file shorter than the first request was cached whole, but the
        // validator was offered only when the cached body reached the requested length —
        // which a short file never does — so it was downloaded again on every run.
        let d = tempfile::tempdir().expect("temp dir");
        let cache = Cache::at(d.path());
        let file = bytes(600);
        cache.store(
            "https://example.invalid/m.gguf",
            &file,
            Some(600),
            Some("\"v1\""),
        );

        let mut f = open(Fake::new(file.clone()), Some(cache));
        let mut out = Vec::new();
        f.read_to_end(&mut out).expect("read");
        assert_eq!(out, file);
        assert_eq!(f.bytes_fetched(), 0, "a 304, not a download");
    }

    #[test]
    fn remote_a_304_to_a_request_without_a_validator_is_an_error_not_an_abort() {
        // Regression: this was an `expect`, and `panic = "abort"` in release turned a
        // misbehaving proxy into a dead process.
        let fake = Fake {
            always_304: true,
            ..Fake::new(bytes(100))
        };
        let mut f = open(fake, None);
        let err = f.read(&mut [0u8; 4]).expect_err("must be an error");
        assert!(err.to_string().contains("304"), "{err}");
    }

    #[test]
    fn remote_a_file_that_changes_mid_read_is_refetched_from_the_start() {
        let file = bytes(4000);
        let fake = Arc::new(Fake::new(file.clone()));
        struct Shared(Arc<Fake>);
        impl Fetch for Shared {
            fn range(&self, u: &str, s: u64, e: u64, v: Option<&str>) -> io::Result<Fetched> {
                self.0.range(u, s, e, v)
            }
        }
        let mut f = RemoteFile::with_fetcher("u", Box::new(Shared(Arc::clone(&fake))), None)
            .with_initial_fetch(1000);
        let mut first = vec![0u8; 1000];
        f.read_exact(&mut first).expect("first read");
        *fake.etag.lock().unwrap() = "\"v2\"".into();
        let mut rest = vec![0u8; 500];
        f.read_exact(&mut rest).expect("second read");

        let log = fake.log.lock().unwrap();
        assert_eq!(log[1].0, 1000, "tail request");
        assert_eq!(log[2].0, 0, "etag changed, so the prefix was fetched again");
    }

    #[test]
    fn remote_nothing_is_cached_until_the_caller_persists() {
        let d = tempfile::tempdir().expect("temp dir");
        let url = "https://example.invalid/m.gguf";
        let mut f = open(Fake::new(bytes(500)), Some(Cache::at(d.path())));
        let mut out = Vec::new();
        f.read_to_end(&mut out).expect("read");
        assert!(
            Cache::at(d.path()).load(url).is_none(),
            "not before a successful parse"
        );
        f.persist();
        let e = Cache::at(d.path()).load(url).expect("stored");
        assert_eq!(e.body.len(), 500);
    }

    #[test]
    fn seeking_before_the_start_is_rejected() {
        let mut f = open(Fake::new(vec![]), None);
        assert!(f.seek(SeekFrom::Current(-1)).is_err());
    }

    #[test]
    fn seeking_forward_does_not_fetch_on_its_own() {
        // Only reads should cost bytes; a parser that seeks around a directory it has
        // already buffered must not re-fetch.
        let fake = Fake::new(bytes(10));
        let log = Arc::clone(&fake.log);
        let mut f = open(fake, None);
        f.seek(SeekFrom::Start(1_000_000))
            .expect("seek is arithmetic");
        assert_eq!(f.bytes_fetched(), 0);
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn a_zero_length_read_is_not_an_error_and_fetches_nothing() {
        let mut f = open(Fake::new(bytes(10)), None);
        assert_eq!(f.read(&mut []).expect("empty read"), 0);
        assert_eq!(f.bytes_fetched(), 0);
    }
}

#[cfg(test)]
mod etag_tests {
    use super::*;

    /// A real HuggingFace exchange: a 302 carrying the stable hash, then the CDN's 206.
    const HF_HEADERS: &[u8] = b"HTTP/2 302 \r\n\
x-linked-etag: \"8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f\"\r\n\
location: https://us.aws.cdn.hf.co/xet-bridge-us/signed?Expires=1785686539\r\n\
\r\n\
HTTP/2 206 \r\n\
etag: \"7ff148f83a88d5645abb8352cec4fa6fee204846ae3866297f06c983dad8a997\"\r\n\
content-range: bytes 0-1023/4213512672\r\n\
\r\n";

    #[test]
    fn the_stable_linked_etag_wins_over_the_cdn_one() {
        // The CDN etag describes an object behind a signed URL that expires and whose
        // signature encodes the byte range, so sending it back gets a fresh 302 and the
        // whole body again. The linked etag gets a 304 before the redirect is followed.
        let etag = parse_response(HF_HEADERS).etag.expect("an etag");
        assert!(etag.contains("8c310f14"), "took the CDN etag: {etag}");
        assert!(etag.starts_with('"') && etag.ends_with('"'), "quotes kept");
    }

    #[test]
    fn an_etag_carried_only_by_the_redirect_is_found() {
        let raw = b"HTTP/2 302\r\nx-linked-etag: \"abc\"\r\nlocation: /x\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 0-0/1\r\n\r\nZ";
        assert_eq!(parse_response(raw).etag.as_deref(), Some("\"abc\""));
    }

    #[test]
    fn a_redirect_hops_own_etag_is_not_taken_for_the_files() {
        let raw = b"HTTP/2 302\r\netag: \"redirect\"\r\nlocation: /x\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 0-0/1\r\n\r\nZ";
        assert_eq!(parse_response(raw).etag, None);
    }

    #[test]
    fn a_plain_etag_is_used_when_there_is_no_linked_one() {
        let r = parse_response(b"HTTP/1.1 200 OK\r\netag: \"abc123\"\r\n\r\n");
        assert_eq!(r.etag.as_deref(), Some("\"abc123\""));
    }

    #[test]
    fn absent_or_empty_etags_are_none_rather_than_an_empty_string() {
        // An empty validator would be sent as `If-None-Match: ` and match nothing, turning
        // every request into a cache miss that still paid for the header.
        assert_eq!(parse_response(b"HTTP/1.1 200 OK\r\n\r\n").etag, None);
        assert_eq!(
            parse_response(b"HTTP/1.1 200 OK\r\netag:   \r\n\r\n").etag,
            None
        );
    }

    #[test]
    fn the_final_status_is_taken_not_the_redirect() {
        // Reading the first status line would report 302 for every ranged fetch and the
        // 304 path would never trigger.
        assert_eq!(parse_response(HF_HEADERS).status, 206);
        assert_eq!(parse_response(b"HTTP/2 304 \r\n\r\n").status, 304);
    }

    #[test]
    fn the_total_size_still_comes_from_the_content_range() {
        assert_eq!(parse_response(HF_HEADERS).total, Some(4_213_512_672));
    }
}
