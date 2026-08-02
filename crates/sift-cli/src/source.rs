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
    Remote { url: String, label: String },
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
            });
        }

        // `repo:QUANT` — split on the last colon so a URL-ish repo name is not mangled.
        let (repo, inline_quant) = match reference.rsplit_once(':') {
            Some((r, q)) if !q.contains('/') => (r, Some(q)),
            _ => (reference, None),
        };
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
    /// Architecture facts are only available for GGUF here. A safetensors file keeps them
    /// in the repo's `config.json`, which a single-file reference has no way to locate —
    /// so KV sizing is reported as unavailable rather than guessed.
    pub fn read_shape(&self) -> Result<(sift_core::model::ModelShape, Option<u64>)> {
        if self.format() == sift_core::engine::Format::Safetensors {
            let (st, fetched) = match self {
                Source::Local(p) => {
                    let mut f = std::fs::File::open(p)
                        .with_context(|| format!("opening {}", p.display()))?;
                    (
                        sift_core::model::Safetensors::parse(&mut f)
                            .with_context(|| format!("reading {}", p.display()))?,
                        None,
                    )
                }
                Source::Remote { url, .. } => {
                    let mut f = RemoteFile::new(url.clone());
                    let st = sift_core::model::Safetensors::parse(&mut f)
                        .with_context(|| format!("reading {url}"))?;
                    let bytes = f.bytes_fetched();
                    (st, Some(bytes))
                }
            };
            return Ok((
                sift_core::model::ModelShape::new(Default::default(), st.tensors, 1),
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
}
