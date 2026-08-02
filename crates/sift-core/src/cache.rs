//! Caching fetched GGUF headers in `~/.sift/cache/`, validated by ETag.
//!
//! A `fit` sweep reads one header per quantization — 25 for Qwen3-30B-A3B, 72 for a fully
//! split repo — and re-read every one of them on every invocation. Running the same command
//! twice paid the same megabytes twice.
//!
//! # Why ETags rather than an expiry
//!
//! A time-based cache has to choose between being wrong and being useless: short enough to
//! notice a re-upload means it rarely hits, long enough to be worth having means it can
//! serve a stale header. HuggingFace returns a strong ETag on every file, so the cache asks
//! instead of guessing. A conditional request costs one small round-trip and a `304 Not
//! Modified` proves the bytes on disk are still the bytes on the server.
//!
//! That matters more here than for most caches: the tool's whole claim is that it reads the
//! **real header of the real file**, so a cache that could serve a stale one would undercut
//! the reason to use it at all.

use std::io;
use std::path::{Path, PathBuf};

/// What was cached for one URL.
pub struct Entry {
    /// The prefix of the file that was fetched.
    pub body: Vec<u8>,
    /// Total file size the server reported.
    pub total: Option<u64>,
    /// The validator to send back as `If-None-Match`.
    pub etag: String,
}

/// The cache directory, or `None` when caching is unavailable or switched off.
///
/// `SIFT_NO_CACHE` disables it outright. Useful when debugging a parse against a file you
/// are actively re-uploading, where even a validated cache is one more thing to rule out.
pub fn dir() -> Option<PathBuf> {
    if std::env::var_os("SIFT_NO_CACHE").is_some() {
        return None;
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".sift/cache/headers"))
}

/// A filesystem-safe key for a URL.
///
/// FNV-1a rather than `DefaultHasher`, whose output is explicitly not stable across Rust
/// releases — a cache keyed on it would silently miss everything after a toolchain upgrade.
/// Collision risk is irrelevant here because the ETag, not the key, is what decides whether
/// the bytes are current.
fn key(url: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in url.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn paths(url: &str) -> Option<(PathBuf, PathBuf)> {
    let d = dir()?;
    let k = key(url);
    Some((d.join(format!("{k}.bin")), d.join(format!("{k}.json"))))
}

/// Load what was cached for `url`, if anything.
///
/// Any inconsistency — missing sidecar, unparseable metadata, a body whose length disagrees
/// with what was recorded — is treated as a miss. A cache is an optimisation, and a corrupt
/// one must degrade to a fetch rather than to a wrong answer.
pub fn load(url: &str) -> Option<Entry> {
    let (body_path, meta_path) = paths(url)?;
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_path).ok()?).ok()?;
    let etag = meta["etag"].as_str()?.to_string();
    let recorded_len = meta["len"].as_u64()?;
    let body = std::fs::read(&body_path).ok()?;

    if body.len() as u64 != recorded_len {
        return None;
    }
    Some(Entry {
        body,
        total: meta["total"].as_u64(),
        etag,
    })
}

/// Record a fetched prefix against its ETag.
///
/// Best effort throughout: a full disk or a read-only home should slow the tool down, never
/// break it. An entry without an ETag is not stored at all, since there would be no way to
/// prove later that it is still current.
pub fn store(url: &str, body: &[u8], total: Option<u64>, etag: Option<&str>) {
    let Some(etag) = etag else { return };
    let Some((body_path, meta_path)) = paths(url) else {
        return;
    };
    let Some(parent) = body_path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }

    // Body first, sidecar second. A crash between the two leaves a body with no metadata,
    // which `load` reads as a miss. The reverse order could leave metadata pointing at a
    // body that was never written.
    if write_atomically(&body_path, body).is_err() {
        return;
    }
    let meta = serde_json::json!({
        "url": url,
        "etag": etag,
        "len": body.len() as u64,
        "total": total,
    });
    let _ = write_atomically(&meta_path, meta.to_string().as_bytes());
}

/// Write via a temporary file and rename, so a reader never sees a half-written entry.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_is_stable_and_filesystem_safe() {
        let k = key("https://huggingface.co/org/repo/resolve/main/model.gguf");
        assert_eq!(
            k,
            key("https://huggingface.co/org/repo/resolve/main/model.gguf")
        );
        assert_eq!(k.len(), 16);
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn different_urls_get_different_keys() {
        assert_ne!(key("https://a/x.gguf"), key("https://a/y.gguf"));
    }

    #[test]
    fn caching_is_switched_off_by_the_environment_variable() {
        // Set and cleared inside one test rather than across two, because tests share a
        // process and an env var set in one would leak into the others.
        let before = std::env::var_os("SIFT_NO_CACHE");
        // SAFETY: single-threaded within this test; restored before returning.
        unsafe { std::env::set_var("SIFT_NO_CACHE", "1") };
        assert!(dir().is_none());
        unsafe {
            match before {
                Some(v) => std::env::set_var("SIFT_NO_CACHE", v),
                None => std::env::remove_var("SIFT_NO_CACHE"),
            }
        }
    }

    #[test]
    fn an_entry_without_an_etag_is_not_stored() {
        // There would be no way to prove later that it is still current, and a cache that
        // cannot be validated is exactly what this design refuses.
        let url = "https://example.invalid/no-etag.gguf";
        store(url, b"body", Some(4), None);
        assert!(load(url).is_none());
    }
}
