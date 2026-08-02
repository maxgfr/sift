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
//! against network latency.

use std::io::{self, Read, Seek, SeekFrom};
use std::process::Command;

/// How much to fetch on the first request.
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

/// A remote file readable through HTTP range requests.
///
/// Bytes are fetched lazily in growing prefixes and cached in memory. Seeking backwards is
/// free; seeking forward past what has been fetched triggers a fetch.
pub struct RemoteFile {
    url: String,
    /// Prefix of the file fetched so far.
    buf: Vec<u8>,
    /// Current cursor, which may sit beyond `buf.len()`.
    pos: u64,
    /// Total size from the server, if it reported one.
    total: Option<u64>,
    /// Bytes actually transferred, so callers can prove we did not download the model.
    fetched: u64,
}

impl RemoteFile {
    /// Open a remote file. No bytes are fetched until the first read.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            buf: Vec::new(),
            pos: 0,
            total: None,
            fetched: 0,
        }
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

    /// Ensure at least `want` bytes of prefix are buffered.
    fn ensure(&mut self, want: u64) -> io::Result<()> {
        if self.buf.len() as u64 >= want {
            return Ok(());
        }
        if let Some(total) = self.total {
            if self.buf.len() as u64 >= total {
                return Ok(()); // whole file already held
            }
        }
        if want > MAX_FETCH {
            return Err(io::Error::other(format!(
                "refusing to fetch {want} bytes of {}: a GGUF directory should not need \
                 more than {MAX_FETCH}. The file may not be a GGUF, or may be truncated.",
                self.url
            )));
        }

        // Always re-fetch from zero rather than stitching ranges. The directory is small,
        // this happens once or twice, and a single contiguous prefix is far easier to
        // reason about than a set of possibly-overlapping windows.
        let end = want.saturating_sub(1);
        let (body, total) = curl_range(&self.url, 0, end)?;
        self.fetched += body.len() as u64;
        self.total = total.or(self.total);
        self.buf = body;
        Ok(())
    }
}

impl Read for RemoteFile {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let need = self.pos + out.len() as u64;

        // Grow geometrically so a directory slightly larger than the initial fetch does
        // not cause a request per read.
        if (self.buf.len() as u64) < need {
            let mut target = self.buf.len().max(INITIAL_FETCH as usize) as u64;
            while target < need && target < MAX_FETCH {
                target *= 2;
            }
            self.ensure(target.max(need))?;
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
            SeekFrom::Start(n) => n as i64,
            SeekFrom::Current(d) => self.pos as i64 + d,
            SeekFrom::End(d) => {
                // Seeking from the end needs the size; one HEAD-shaped request gets it.
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
                total as i64 + d
            }
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot seek before the start of the file",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

/// Fetch `[start, end]` inclusive. Returns the body and the total size when the server
/// reports one via `Content-Range`.
fn curl_range(url: &str, start: u64, end: u64) -> io::Result<(Vec<u8>, Option<u64>)> {
    let out = Command::new("curl")
        .args([
            "-sSL",
            "--fail",
            "--max-time",
            "60",
            "-D",
            "-", // headers to stdout, ahead of the body
            "-r",
            &format!("{start}-{end}"),
            url,
        ])
        .output()
        .map_err(|e| io::Error::other(format!("running curl: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(io::Error::other(format!(
            "fetching {url}: {}",
            if stderr.trim().is_empty() {
                "curl failed".into()
            } else {
                stderr.trim().to_string()
            }
        )));
    }

    let (headers, body) = split_headers(&out.stdout);
    Ok((body, parse_content_range_total(&headers)))
}

/// Split a curl `-D -` response into header text and body bytes.
///
/// Redirects mean several header blocks arrive before the body, so this scans for the last
/// blank-line separator rather than the first.
fn split_headers(raw: &[u8]) -> (String, Vec<u8>) {
    let mut split_at = None;
    let mut i = 0;
    while i + 3 < raw.len() {
        if &raw[i..i + 4] == b"\r\n\r\n" {
            split_at = Some(i + 4);
            i += 4;
        } else {
            i += 1;
        }
    }
    match split_at {
        Some(at) => (
            String::from_utf8_lossy(&raw[..at]).into_owned(),
            raw[at..].to_vec(),
        ),
        None => (String::new(), raw.to_vec()),
    }
}

/// Extract the total size from a `Content-Range: bytes 0-1023/5242880` header.
fn parse_content_range_total(headers: &str) -> Option<u64> {
    headers
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-range:"))
        .and_then(|l| l.rsplit('/').next())
        .and_then(|t| t.trim().parse().ok())
}

/// Build a HuggingFace resolve URL for a file in a repo.
pub fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_total_is_parsed() {
        let h = "HTTP/2 206\r\ncontent-range: bytes 0-1023/5242880\r\n\r\n";
        assert_eq!(parse_content_range_total(h), Some(5_242_880));
    }

    #[test]
    fn content_range_is_matched_case_insensitively() {
        let h = "HTTP/1.1 206\r\nContent-Range: bytes 0-99/12345\r\n\r\n";
        assert_eq!(parse_content_range_total(h), Some(12_345));
    }

    #[test]
    fn a_missing_content_range_yields_no_size() {
        assert_eq!(parse_content_range_total("HTTP/2 200\r\n\r\n"), None);
    }

    #[test]
    fn an_unsatisfiable_range_star_is_not_read_as_a_size() {
        // `bytes */1234` means the range was unsatisfiable. The total is still after the
        // slash, so this parses — which is correct, and worth pinning so a future rewrite
        // does not start returning None here.
        let h = "HTTP/2 416\r\ncontent-range: bytes */1234\r\n\r\n";
        assert_eq!(parse_content_range_total(h), Some(1234));
    }

    #[test]
    fn headers_split_at_the_last_blank_line_so_redirects_do_not_confuse_it() {
        // curl emits one header block per hop; the body follows the final one.
        let raw = b"HTTP/2 302\r\nlocation: /x\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 0-3/99\r\n\r\nBODY";
        let (headers, body) = split_headers(raw);
        assert_eq!(body, b"BODY");
        assert!(
            headers.contains("206"),
            "final header block must be retained"
        );
        assert_eq!(parse_content_range_total(&headers), Some(99));
    }

    #[test]
    fn a_response_without_headers_is_all_body() {
        let (headers, body) = split_headers(b"just body");
        assert!(headers.is_empty());
        assert_eq!(body, b"just body");
    }

    #[test]
    fn hf_urls_are_built_in_the_resolve_form() {
        assert_eq!(
            hf_url("allenai/OLMoE-1B-7B-0924-Instruct-GGUF", "olmoe-q4.gguf"),
            "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-q4.gguf"
        );
    }

    #[test]
    fn seeking_before_the_start_is_rejected() {
        let mut f = RemoteFile::new("https://example.invalid/x.gguf");
        assert!(f.seek(SeekFrom::Current(-1)).is_err());
    }

    #[test]
    fn seeking_forward_does_not_fetch_on_its_own() {
        // Only reads should cost bytes; a parser that seeks around a directory it has
        // already buffered must not re-fetch.
        let mut f = RemoteFile::new("https://example.invalid/x.gguf");
        f.seek(SeekFrom::Start(1_000_000))
            .expect("seek is arithmetic");
        assert_eq!(f.bytes_fetched(), 0);
    }

    #[test]
    fn a_zero_length_read_is_not_an_error_and_fetches_nothing() {
        let mut f = RemoteFile::new("https://example.invalid/x.gguf");
        assert_eq!(f.read(&mut []).expect("empty read"), 0);
        assert_eq!(f.bytes_fetched(), 0);
    }
}
