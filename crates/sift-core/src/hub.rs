//! Listing what a HuggingFace repo contains, without downloading any of it.
//!
//! The HF API reports every file's exact byte size. Combined with
//! [`crate::model::remote`], that is enough to evaluate every quantization a repo offers —
//! sizes from the API, architecture and dtype mix from a few kilobytes of range requests —
//! before a single weight is transferred.

use std::process::Command;

/// One GGUF file offered by a repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    /// Path within the repo, e.g. `Qwen3-30B-A3B-Q4_K_M.gguf`.
    pub path: String,
    /// Exact size in bytes, when the API reports one.
    pub size: Option<u64>,
}

impl RepoFile {
    /// The quantization label inferred from the filename, e.g. `Q4_K_M`.
    ///
    /// Publishers are inconsistent: `Qwen3-30B-A3B-UD-Q2_K_XL.gguf`,
    /// `BF16/Model-BF16.gguf`, `olmoe-...-q4_k_m.gguf`. Matching on the *shape* of a part
    /// does not work — `30B` and `A3B` look exactly like tags — so this matches an explicit
    /// set of known quant prefixes instead. Predictable beats clever here: a wrong label on
    /// a recommendation is a user downloading the wrong 18 GB file.
    ///
    /// Falls back to the filename stem when nothing matches, which is always something the
    /// user can recognise.
    pub fn quant_label(&self) -> String {
        let stem = self
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&self.path)
            .trim_end_matches(".gguf");

        let parts: Vec<&str> = stem.split('-').collect();
        let mut take_from = parts.len();
        for (i, p) in parts.iter().enumerate().rev() {
            if is_quant_tag(p) {
                take_from = i;
            } else {
                break;
            }
        }
        if take_from < parts.len() {
            parts[take_from..].join("-").to_ascii_uppercase()
        } else {
            stem.to_string()
        }
    }

    /// Whether this file is one shard of a split model.
    ///
    /// Split shards must not be evaluated individually: shard 2 of 3 is not a model, and
    /// reporting its size as the model's size would be badly misleading.
    pub fn is_shard(&self) -> bool {
        self.path.contains("-of-")
    }
}

/// Whether a hyphen-separated filename part names a quantization.
///
/// Deliberately an allow-list of prefixes rather than a shape test. `30B` and `A3B` are
/// upper-case, contain digits and letters, and are *not* quant tags — any heuristic based
/// on character classes swallows them and reports `30B-A3B-Q4_K_M`.
fn is_quant_tag(part: &str) -> bool {
    let p = part.to_ascii_uppercase();
    if p.is_empty() {
        return false;
    }
    // Unsloth's "dynamic" marker, which prefixes a real tag: `UD-Q2_K_XL`.
    if p == "UD" {
        return true;
    }
    // Q4_K_M, Q8_0, IQ4_XS, F16, F32, BF16, TQ1_0, MXFP4 — all are a known letter prefix
    // immediately followed by a digit.
    for prefix in ["IQ", "BF", "TQ", "MXFP", "Q", "F"] {
        if let Some(rest) = p.strip_prefix(prefix) {
            if rest.starts_with(|c: char| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

/// Errors from talking to the hub.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("running curl: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("hub request failed for `{repo}`: {detail}")]
    Request { repo: String, detail: String },
    #[error("could not parse the hub response for `{repo}`: {detail}")]
    Parse { repo: String, detail: String },
    #[error("`{0}` has no .gguf files")]
    NoGguf(String),
}

/// List the GGUF files a repo offers, with sizes, without downloading anything.
pub fn list_gguf(repo: &str) -> Result<Vec<RepoFile>, HubError> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let out = Command::new("curl")
        .args(["-sSL", "--fail", "--max-time", "30", &url])
        .output()?;

    if !out.status.success() {
        return Err(HubError::Request {
            repo: repo.to_string(),
            detail: {
                let e = String::from_utf8_lossy(&out.stderr);
                if e.trim().is_empty() {
                    "no such repo, or it is gated and needs authentication".into()
                } else {
                    e.trim().to_string()
                }
            },
        });
    }

    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| HubError::Parse {
            repo: repo.to_string(),
            detail: e.to_string(),
        })?;

    let mut files: Vec<RepoFile> = json["siblings"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let path = s["rfilename"].as_str()?;
                    if !path.ends_with(".gguf") {
                        return None;
                    }
                    Some(RepoFile {
                        path: path.to_string(),
                        // The API spells it `size` with `?blobs=true`, `lfs.size` otherwise.
                        size: s["size"].as_u64().or_else(|| s["lfs"]["size"].as_u64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    if files.is_empty() {
        return Err(HubError::NoGguf(repo.to_string()));
    }

    files.sort_by_key(|f| f.size);
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str) -> RepoFile {
        RepoFile {
            path: path.into(),
            size: Some(1),
        }
    }

    #[test]
    fn common_quant_labels_are_recovered() {
        assert_eq!(f("Qwen3-30B-A3B-Q4_K_M.gguf").quant_label(), "Q4_K_M");
        assert_eq!(f("Qwen3-30B-A3B-IQ4_XS.gguf").quant_label(), "IQ4_XS");
        assert_eq!(f("Qwen3-30B-A3B-Q2_K.gguf").quant_label(), "Q2_K");
    }

    #[test]
    fn multi_part_unsloth_labels_survive() {
        // `UD-Q2_K_XL` is two hyphen-separated tag parts and must not be truncated to XL.
        assert_eq!(
            f("Qwen3-30B-A3B-UD-Q2_K_XL.gguf").quant_label(),
            "UD-Q2_K_XL"
        );
    }

    #[test]
    fn a_directory_prefix_is_ignored() {
        assert_eq!(f("BF16/Qwen3-30B-A3B-BF16.gguf").quant_label(), "BF16");
    }

    #[test]
    fn lowercase_names_are_recognised_and_normalised() {
        // allenai ships lowercase filenames. The tag is still a tag.
        assert_eq!(
            f("olmoe-1b-7b-0924-instruct-q4_k_m.gguf").quant_label(),
            "Q4_K_M"
        );
    }

    #[test]
    fn model_size_parts_are_not_mistaken_for_quant_tags() {
        // The bug this guards: `30B` and `A3B` pass every character-class test a quant tag
        // passes, so a shape heuristic reports `30B-A3B-Q4_K_M`.
        assert!(!is_quant_tag("30B"));
        assert!(!is_quant_tag("A3B"));
        assert!(!is_quant_tag("Instruct"));
        assert!(is_quant_tag("Q4_K_M"));
        assert!(is_quant_tag("IQ2_XXS"));
        assert!(is_quant_tag("BF16"));
        assert!(is_quant_tag("UD"));
        assert!(is_quant_tag("MXFP4"));
    }

    #[test]
    fn a_name_with_no_recognisable_tag_falls_back_to_the_stem() {
        // Inventing a label would be worse than showing what the file is actually called.
        assert_eq!(
            f("some-random-model.gguf").quant_label(),
            "some-random-model"
        );
    }

    #[test]
    fn split_shards_are_identifiable() {
        assert!(f("Qwen3-30B-A3B-BF16-00001-of-00002.gguf").is_shard());
        assert!(!f("Qwen3-30B-A3B-Q4_K_M.gguf").is_shard());
    }
}
