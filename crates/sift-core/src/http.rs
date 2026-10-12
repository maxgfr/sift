//! Building `curl` calls: HuggingFace authentication and URL hygiene in one place.
//!
//! Every request `sift` makes goes through `curl` — the hub listing, the `config.json`,
//! the header ranges — and every one of them needs the same two things done the same way:
//! a token for gated repos, sent only to HuggingFace and never on a command line, and URLs
//! that curl does not reinterpret.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};

/// The user's HuggingFace token, from wherever the HuggingFace tools keep it.
///
/// `HF_TOKEN`, then the older `HUGGING_FACE_HUB_TOKEN`, then the file `hf auth login`
/// writes: `$HF_HOME/token`, `~/.cache/huggingface/token` by default.
pub fn token() -> Option<String> {
    token_from(
        std::env::var_os("HF_TOKEN"),
        std::env::var_os("HUGGING_FACE_HUB_TOKEN"),
        std::env::var_os("HF_HOME"),
        std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")),
    )
}

/// [`token`] from explicit inputs, so the lookup order is testable without the process
/// environment.
fn token_from(
    hf_token: Option<OsString>,
    hub_token: Option<OsString>,
    hf_home: Option<OsString>,
    home: Option<OsString>,
) -> Option<String> {
    let clean = |s: String| {
        let t = s.trim().to_string();
        (!t.is_empty()).then_some(t)
    };
    if let Some(t) = hf_token.and_then(|v| v.into_string().ok()).and_then(clean) {
        return Some(t);
    }
    if let Some(t) = hub_token.and_then(|v| v.into_string().ok()).and_then(clean) {
        return Some(t);
    }
    let dir = match hf_home {
        Some(h) => PathBuf::from(h),
        None => PathBuf::from(home?).join(".cache/huggingface"),
    };
    std::fs::read_to_string(dir.join("token"))
        .ok()
        .and_then(clean)
}

/// Whether `url` points at HuggingFace itself, the only host a token is sent to.
///
/// Matched on the host exactly, so `huggingface.co.example.com` and a path that merely
/// mentions the name do not qualify.
pub fn is_huggingface(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit('@').next().unwrap_or_default();
    let host = host.split(':').next().unwrap_or_default();
    host == "huggingface.co" || host.ends_with(".huggingface.co")
}

/// Percent-encode a path, keeping `/` as the separator.
///
/// HuggingFace file names can hold spaces, `#`, `?` and brackets. Unencoded, the first
/// two end the path early and curl reads the last as a glob range.
pub fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// A `curl` invocation for one URL, with authentication when the URL is HuggingFace's.
///
/// The token travels on stdin as a header file (`-H @-`), never in argv, where any user
/// on the machine could read it from the process table.
pub struct Curl {
    cmd: Command,
    auth: Option<String>,
}

impl Curl {
    /// `curl -sS --globoff` for `url`, with `args` before it.
    ///
    /// `--globoff` because a URL is data here: `[` and `{` in a file name are characters,
    /// not curl's range syntax.
    pub fn new<I, S>(url: &str, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let mut cmd = Command::new("curl");
        cmd.args(["-sS", "--globoff"]).args(args);
        let auth = is_huggingface(url)
            .then(token)
            .flatten()
            .map(|t| format!("Authorization: Bearer {t}\n"));
        if auth.is_some() {
            cmd.args(["-H", "@-"]);
        }
        cmd.arg(url);
        Self { cmd, auth }
    }

    /// Whether a token is being sent.
    pub fn authenticated(&self) -> bool {
        self.auth.is_some()
    }

    /// Start curl with stdout and stderr piped.
    pub fn spawn(mut self) -> io::Result<Child> {
        self.cmd
            .stdin(if self.auth.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = self
            .cmd
            .spawn()
            .map_err(|e| io::Error::other(format!("running curl: {e}")))?;
        if let (Some(auth), Some(mut stdin)) = (self.auth, child.stdin.take()) {
            stdin.write_all(auth.as_bytes())?;
        }
        Ok(child)
    }

    /// Run curl to completion.
    pub fn output(self) -> io::Result<Output> {
        self.spawn()?.wait_with_output()
    }
}

/// What to tell someone whose request was refused.
///
/// The hub answers 401 for a repo that does not exist as well as for one that is gated, so
/// the message names both — and says whether a token was sent, since "set HF_TOKEN" is
/// useless advice to someone whose token was just rejected.
pub fn refusal_hint(status: u16, authenticated: bool) -> Option<String> {
    match (status, authenticated) {
        (401 | 403, false) => Some(
            "no such repo on HuggingFace, or it is private or gated: accept its terms on \
             huggingface.co and set HF_TOKEN (or run `hf auth login`)"
                .to_string(),
        ),
        (401 | 403, true) => Some(
            "no such repo on HuggingFace, or your token does not grant access to it: \
             accept its terms on huggingface.co with the account the token belongs to"
                .to_string(),
        ),
        (404, _) => Some("no such repo or file on HuggingFace".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_token_is_looked_up_in_the_order_the_hf_tools_use() {
        let d = tempfile::tempdir().expect("temp dir");
        std::fs::write(d.path().join("token"), "hf_file\n").expect("write");
        let home = || Some(OsString::from("/nonexistent"));
        let hf_home = || Some(d.path().as_os_str().to_owned());

        assert_eq!(
            token_from(
                Some("hf_env".into()),
                Some("hf_old".into()),
                hf_home(),
                home()
            ),
            Some("hf_env".into())
        );
        assert_eq!(
            token_from(None, Some("hf_old".into()), hf_home(), home()),
            Some("hf_old".into())
        );
        assert_eq!(
            token_from(None, None, hf_home(), home()),
            Some("hf_file".into())
        );
        assert_eq!(token_from(Some("  ".into()), None, None, home()), None);
    }

    #[test]
    fn the_default_token_file_lives_under_the_cache_directory() {
        let d = tempfile::tempdir().expect("temp dir");
        let dir = d.path().join(".cache/huggingface");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("token"), "hf_default").expect("write");
        assert_eq!(
            token_from(None, None, None, Some(d.path().as_os_str().to_owned())),
            Some("hf_default".into())
        );
    }

    #[test]
    fn a_token_goes_to_huggingface_and_nowhere_else() {
        assert!(is_huggingface("https://huggingface.co/api/models/x/y"));
        assert!(is_huggingface("https://cdn-lfs.huggingface.co/repos/x"));
        assert!(!is_huggingface("https://huggingface.co.evil.example/x"));
        assert!(!is_huggingface("https://example.com/huggingface.co/x"));
        assert!(
            !is_huggingface("http://huggingface.co/x"),
            "never over plain HTTP"
        );
        assert!(!is_huggingface("https://huggingface.co@evil.example/x"));
    }

    #[test]
    fn paths_are_percent_encoded_but_keep_their_slashes() {
        assert_eq!(
            encode_path("Q4_K_M/model-Q4_K_M.gguf"),
            "Q4_K_M/model-Q4_K_M.gguf"
        );
        assert_eq!(encode_path("a b#c?[1].gguf"), "a%20b%23c%3F%5B1%5D.gguf");
    }

    #[test]
    fn a_401_says_what_to_do_depending_on_whether_a_token_was_sent() {
        assert!(refusal_hint(401, false).unwrap().contains("HF_TOKEN"));
        assert!(!refusal_hint(401, true).unwrap().contains("set HF_TOKEN"));
        assert_eq!(refusal_hint(500, false), None);
    }
}
