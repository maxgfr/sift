//! Working out what the user meant by a model reference.
//!
//! One argument covers three things a user might reasonably type:
//!
//! ```text
//! ./model.gguf                       a local file
//! unsloth/Qwen3-30B-A3B-GGUF         a repo — pick a file from it
//! unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M  a repo and a quantization
//! https://…/model.gguf               a direct URL
//! ```
//!
//! Resolution is deliberately ordered so that a path that exists always wins. Someone with
//! a directory named `org/repo` should get their file, not a network request.

use anyhow::{bail, Context, Result};
use sift_core::hub::{self, RepoFile};
use sift_core::model::{hf_url, Gguf, RemoteFile};
use std::path::{Path, PathBuf};

/// A resolved model reference.
#[derive(Debug)]
pub enum Source {
    /// A file on this machine.
    Local(PathBuf),
    /// A URL, plus the repo and file it came from when known.
    Remote {
        url: String,
        label: String,
        /// The `org/repo` this was resolved from, when it was. A safetensors file keeps
        /// its architecture numbers in the repo's `config.json`, so knowing the repo is
        /// what makes KV sizing possible for that format.
        repo: Option<String>,
    },
}

/// Split a `repo:QUANT` reference into its two halves.
///
/// Splits on the last colon so a URL-ish repo name is not mangled, and refuses a suffix
/// containing `/`, which would mean the colon was part of something else. Shared by every
/// command that takes a model reference, so `fit` and `route` cannot disagree about what
/// `org/repo:Q4_K_M` means.
pub fn split_quant(reference: &str) -> (&str, Option<&str>) {
    match reference.rsplit_once(':') {
        Some((repo, quant)) if !quant.is_empty() && !quant.contains('/') => (repo, Some(quant)),
        _ => (reference, None),
    }
}

/// Whether a reference that does not exist on disk was nonetheless meant as a path.
///
/// `org/repo` and `/models/x.gguf` both contain a slash; only the second should be answered
/// with "no such file" rather than a network request that fails with an HTTP status. A
/// weight-file extension, a leading `/`, `./`, `../` or `~`, or a Windows drive or UNC
/// prefix all settle it.
fn looks_like_path(reference: &str) -> bool {
    let lower = reference.to_ascii_lowercase();
    lower.ends_with(".gguf")
        || lower.ends_with(".safetensors")
        || reference.starts_with('/')
        || reference.starts_with("./")
        || reference.starts_with("../")
        || reference.starts_with('~')
        || reference.starts_with('\\')
        || reference
            .as_bytes()
            .get(1)
            .is_some_and(|&b| b == b':' && reference.as_bytes()[0].is_ascii_alphabetic())
            && reference.len() > 2
}

impl Source {
    /// Parse a user-supplied reference.
    ///
    /// `quant` optionally narrows a repo reference; it is also accepted inline as
    /// `repo:QUANT`.
    pub fn resolve(reference: &str, quant: Option<&str>) -> Result<Self> {
        // A real path always wins over a network guess.
        let as_path = Path::new(reference);
        if as_path.exists() {
            return Ok(Source::Local(as_path.to_path_buf()));
        }

        if reference.starts_with("http://") || reference.starts_with("https://") {
            return Ok(Source::Remote {
                url: reference.to_string(),
                label: reference
                    .rsplit('/')
                    .next()
                    .unwrap_or(reference)
                    .to_string(),
                repo: None,
            });
        }

        // A path that does not exist is a typo, not a hub repo. Asking HuggingFace about
        // `/models/x.gguf` produces a 404 that reads as a network problem.
        if looks_like_path(reference) {
            bail!("no such file: {reference}");
        }

        let (repo, inline_quant) = split_quant(reference);
        let wanted = quant.or(inline_quant);

        if !repo.contains('/') {
            bail!(
                "`{reference}` is neither an existing path nor an `org/repo` reference.\n\
                 Try:  sift inspect ./model.gguf\n   \
                 or:  sift inspect unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"
            );
        }

        let files = hub::list_weights(repo)?;
        let chosen = pick(&files, wanted).with_context(|| {
            let available: Vec<String> = files.iter().take(12).map(RepoFile::quant_label).collect();
            format!(
                "choosing a file from `{repo}`. Available: {}",
                available.join(", ")
            )
        })?;

        Ok(Source::Remote {
            url: hf_url(repo, &chosen.path),
            label: format!("{repo}:{}", chosen.quant_label()),
            repo: Some(repo.to_string()),
        })
    }

    /// The weight format this source is in.
    pub fn format(&self) -> sift_core::engine::Format {
        let name = match self {
            Source::Local(p) => p.to_string_lossy().to_string(),
            Source::Remote { url, .. } => url.clone(),
        };
        // Query strings on a signed CDN URL come after the filename, so match anywhere
        // rather than only at the end.
        if name.contains(".safetensors") {
            sift_core::engine::Format::Safetensors
        } else {
            sift_core::engine::Format::Gguf
        }
    }

    /// Read the model's shape, whatever format it is in. Payload is never fetched.
    ///
    /// Returns the shape and, for remote sources, how many bytes crossed the network —
    /// which callers print, because "we did not download the model" is a claim that should
    /// come with evidence.
    ///
    /// GGUF carries its architecture facts in its own header. A safetensors file keeps them
    /// in the repo's `config.json`, so they are fetched when the reference named a repo and
    /// reported as unavailable — not guessed — for a bare file or URL, which has no way to
    /// locate that document.
    pub fn read_shape(&self) -> Result<(sift_core::model::ModelShape, Option<u64>)> {
        if self.format() == sift_core::engine::Format::Safetensors {
            let (st, facts, fetched) = match self {
                Source::Local(p) => {
                    let mut f = std::fs::File::open(p)
                        .with_context(|| format!("opening {}", p.display()))?;
                    (
                        sift_core::model::Safetensors::parse(&mut f)
                            .with_context(|| format!("reading {}", p.display()))?,
                        Default::default(),
                        None,
                    )
                }
                Source::Remote { url, repo, .. } => {
                    let mut f = RemoteFile::new(url.clone());
                    let st = sift_core::model::Safetensors::parse(&mut f)
                        .with_context(|| format!("reading {url}"))?;
                    let mut bytes = f.bytes_fetched();
                    // Counted in the bytes-read figure: "nothing was downloaded" is a
                    // claim that should include every request made to back it.
                    let facts = match repo.as_deref().and_then(hub::fetch_config) {
                        Some((cfg, n)) => {
                            bytes += n;
                            sift_core::model::safetensors::facts_from_config(&cfg)
                        }
                        None => Default::default(),
                    };
                    (st, facts, Some(bytes))
                }
            };
            return Ok((
                sift_core::model::ModelShape::new(facts, st.tensors, 1),
                fetched,
            ));
        }

        let (g, fetched) = self.read_gguf()?;
        Ok((g.shape(), fetched))
    }

    /// Read the model's GGUF tensor directory. Payload is never fetched.
    pub fn read_gguf(&self) -> Result<(Gguf, Option<u64>)> {
        match self {
            Source::Local(p) => {
                let g = Gguf::open(p).with_context(|| format!("reading {}", p.display()))?;
                Ok((g, None))
            }
            Source::Remote { url, .. } => {
                let mut f = RemoteFile::new(url.clone());
                let g = Gguf::parse(&mut f).with_context(|| format!("reading {url}"))?;
                Ok((g, Some(f.bytes_fetched())))
            }
        }
    }

    /// How to describe this source to a human.
    pub fn label(&self) -> String {
        match self {
            Source::Local(p) => p.display().to_string(),
            Source::Remote { label, .. } => label.clone(),
        }
    }
}

/// Choose one file from a repo's weight listing.
///
/// With no preference, prefers the largest non-shard file, on the reasoning that a repo's
/// flagship quantization is usually its biggest single file. Split shards are excluded:
/// shard 2 of 3 is not a model, and its header describes only part of one.
fn pick<'a>(files: &'a [RepoFile], quant: Option<&str>) -> Result<&'a RepoFile> {
    let whole: Vec<&RepoFile> = files.iter().filter(|f| !f.is_shard()).collect();
    if whole.is_empty() {
        bail!("this repo only contains split shards, which cannot be inspected individually");
    }

    match quant {
        Some(want) => {
            let want_up = want.to_ascii_uppercase();
            whole
                .iter()
                .find(|f| f.quant_label() == want_up)
                // Fall back to a substring match so `Q4` finds `Q4_K_M`.
                .or_else(|| whole.iter().find(|f| f.quant_label().contains(&want_up)))
                .copied()
                .with_context(|| format!("no file matching quantization `{want}`"))
        }
        None => whole
            .iter()
            .max_by_key(|f| f.size.unwrap_or(0))
            .copied()
            .context("no candidate files"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile {
            path: path.into(),
            size: Some(size),
        }
    }

    #[test]
    fn an_explicit_quant_is_honoured() {
        let files = vec![f("M-Q2_K.gguf", 10), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(pick(&files, Some("Q4_K_M")).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn a_partial_quant_matches_by_substring() {
        let files = vec![f("M-Q2_K.gguf", 10), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(pick(&files, Some("Q4")).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn quant_matching_is_case_insensitive() {
        let files = vec![f("M-Q4_K_M.gguf", 20)];
        assert_eq!(pick(&files, Some("q4_k_m")).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn shards_are_never_chosen() {
        // A shard's header describes part of a model; reporting it as the model would be
        // wrong in a way the user cannot see.
        let files = vec![f("M-BF16-00001-of-00002.gguf", 100), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(pick(&files, None).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn a_repo_of_only_shards_is_an_error_not_a_silent_wrong_answer() {
        let files = vec![f("M-BF16-00001-of-00002.gguf", 100)];
        assert!(pick(&files, None).is_err());
    }

    #[test]
    fn an_unknown_quant_lists_what_was_available() {
        let files = vec![f("M-Q2_K.gguf", 10)];
        let err = pick(&files, Some("Q8_0")).unwrap_err().to_string();
        assert!(
            err.contains("Q8_0"),
            "the error should name what was asked for: {err}"
        );
    }

    #[test]
    fn a_bare_word_is_rejected_with_guidance_rather_than_a_network_call() {
        let err = Source::resolve("qwen3", None).unwrap_err().to_string();
        assert!(
            err.contains("org/repo"),
            "should explain the expected form: {err}"
        );
    }

    #[test]
    fn urls_are_passed_through() {
        let s = Source::resolve("https://example.com/a/model.gguf", None).unwrap();
        assert_eq!(s.label(), "model.gguf");
    }

    #[test]
    fn a_missing_path_is_reported_as_missing_not_asked_of_the_hub() {
        // `/nonexistent/x.gguf` contains a slash, which is all an `org/repo` needs. It must
        // still be answered as a file that is not there, without a network request.
        for reference in [
            "/nonexistent/model.gguf",
            "./missing.gguf",
            "../missing/model.safetensors",
            "~/models/missing.gguf",
            "models/missing.gguf",
            "org/repo.safetensors",
            "C:\\models\\missing.gguf",
        ] {
            let err = Source::resolve(reference, None).unwrap_err().to_string();
            assert!(
                err.contains("no such file"),
                "{reference} should be treated as a path: {err}"
            );
        }
    }

    #[test]
    fn a_repo_reference_is_not_mistaken_for_a_path() {
        assert!(!looks_like_path("unsloth/Qwen3-30B-A3B-GGUF"));
        assert!(!looks_like_path("unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"));
        assert!(!looks_like_path("Qwen/Qwen2.5-0.5B-Instruct"));
    }

    #[test]
    fn an_inline_quant_is_split_off_the_repo() {
        assert_eq!(
            split_quant("unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"),
            ("unsloth/Qwen3-30B-A3B-GGUF", Some("Q4_K_M"))
        );
        assert_eq!(
            split_quant("unsloth/Qwen3-30B-A3B-GGUF"),
            ("unsloth/Qwen3-30B-A3B-GGUF", None)
        );
    }

    #[test]
    fn a_colon_followed_by_a_slash_is_not_a_quant() {
        // A URL-shaped string reaching here must survive intact.
        assert_eq!(
            split_quant("https://example.com/a/b"),
            ("https://example.com/a/b", None)
        );
        assert_eq!(split_quant("org/repo:"), ("org/repo:", None));
    }
}
