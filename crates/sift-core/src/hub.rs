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
        let file = self.path.rsplit('/').next().unwrap_or(&self.path);
        let stem = file
            .strip_suffix(".gguf")
            .or_else(|| file.strip_suffix(".safetensors"))
            .unwrap_or(file);

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
        shard_position(&self.path).is_some()
    }

    /// Whether this file is safetensors rather than GGUF.
    pub fn is_safetensors(&self) -> bool {
        self.path.ends_with(".safetensors")
    }
}

/// Where a file sits in a split set: `(name without the suffix, index, total)`.
///
/// GGUF splits are named `Model-Q4_K_M-00002-of-00003.gguf`. Matching the whole
/// `-<digits>-of-<digits>` shape rather than the bare `-of-` substring matters: a model
/// legitimately called `Mixture-of-Experts-Q4_K_M.gguf` is one file, and treating it as a
/// shard would drop it from the table entirely.
pub fn shard_position(path: &str) -> Option<(String, u32, u32)> {
    let (stem, ext) = if let Some(s) = path.strip_suffix(".gguf") {
        (s, ".gguf")
    } else {
        (path.strip_suffix(".safetensors")?, ".safetensors")
    };
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (base, index) = rest.rsplit_once('-')?;

    let total: u32 = total.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    if total == 0 || index == 0 || index > total {
        return None;
    }
    Some((format!("{base}{ext}"), index, total))
}

/// One model that ships as several files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSet {
    /// The path the set would have had as a single file, used for its quant label.
    pub base: RepoFile,
    /// Every part, ordered by shard index.
    pub parts: Vec<RepoFile>,
    /// How many parts the filenames claim exist.
    pub expected: u32,
}

impl ShardSet {
    /// Total bytes across every part.
    pub fn size(&self) -> Option<u64> {
        self.parts.iter().map(|p| p.size).sum()
    }

    /// Whether every part the filenames promise is actually present.
    ///
    /// An incomplete set is reported rather than silently summed: two of three shards add
    /// up to a plausible number that is wrong by a third, and a user acting on it
    /// downloads a model that cannot load.
    pub fn is_complete(&self) -> bool {
        self.parts.len() as u32 == self.expected
    }
}

/// Split whole files from shard sets.
///
/// Both are returned because both belong in the table. Shards used to be dropped outright,
/// which left `sift` silent on exactly the largest models — the cases where "will this
/// fit" is hardest and the download most expensive to get wrong.
pub fn group_shards(files: &[RepoFile]) -> (Vec<RepoFile>, Vec<ShardSet>) {
    use std::collections::BTreeMap;

    let mut whole = Vec::new();
    let mut sets: BTreeMap<String, (Vec<(u32, RepoFile)>, u32)> = BTreeMap::new();

    for f in files {
        match shard_position(&f.path) {
            Some((base, index, total)) => {
                let entry = sets.entry(base).or_insert_with(|| (Vec::new(), total));
                entry.0.push((index, f.clone()));
                // Trust the largest claim: a set whose parts disagree is malformed, and
                // over-counting makes `is_complete` false, which is the safe direction.
                entry.1 = entry.1.max(total);
            }
            None => whole.push(f.clone()),
        }
    }

    let sets = sets
        .into_iter()
        .map(|(base, (mut parts, expected))| {
            parts.sort_by_key(|(i, _)| *i);
            let size = parts.iter().map(|(_, p)| p.size).sum();
            ShardSet {
                base: RepoFile { path: base, size },
                parts: parts.into_iter().map(|(_, p)| p).collect(),
                expected,
            }
        })
        .collect();

    (whole, sets)
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
    #[error("`{0}` has no .gguf or .safetensors files")]
    NoWeights(String),
}

/// List the weight files a repo offers, with sizes, without downloading anything.
///
/// Both formats, because a repo that ships only safetensors is a repo `sift` should have
/// something to say about rather than a dead end.
pub fn list_weights(repo: &str) -> Result<Vec<RepoFile>, HubError> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let out = Command::new("curl")
        .args(["-sSL", "--fail", "--max-time", "30", &url])
        .output()?;

    if !out.status.success() {
        return Err(HubError::Request {
            repo: repo.to_string(),
            detail: describe_curl_failure(&String::from_utf8_lossy(&out.stderr)),
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
                    if !path.ends_with(".gguf") && !path.ends_with(".safetensors") {
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
        return Err(HubError::NoWeights(repo.to_string()));
    }

    files.sort_by_key(|f| f.size);
    Ok(files)
}

/// Turn curl's failure text into something a user can act on.
///
/// The hub answers `401` for a repo that does not exist as well as for one that is gated,
/// so a raw `curl: (22) The requested URL returned error: 401` reads as an authentication
/// problem to someone who merely mistyped a name. Name both possibilities.
fn describe_curl_failure(stderr: &str) -> String {
    let stderr = stderr.trim();
    let status = stderr
        .rsplit_once("returned error: ")
        .and_then(|(_, tail)| tail.split_whitespace().next())
        .and_then(|code| code.parse::<u16>().ok());
    match status {
        Some(code @ (401 | 404)) => format!(
            "no such repo on HuggingFace, or it is private or gated and needs \
             authentication (HTTP {code})"
        ),
        Some(code) => format!("HuggingFace answered HTTP {code}"),
        None if stderr.is_empty() => {
            "no such repo, or it is gated and needs authentication".to_string()
        }
        None => stderr.to_string(),
    }
}

/// Fetch a repo's `config.json`, which safetensors needs and GGUF does not.
///
/// Absence is not an error: plenty of repos omit it, and the honest consequence is that
/// the KV cache goes uncounted and the caller says so. Returns the parsed document and how
/// many bytes crossed the network to get it, so callers that report their traffic can
/// count it.
pub fn fetch_config(repo: &str) -> Option<(serde_json::Value, u64)> {
    let out = Command::new("curl")
        .args([
            "-sSL",
            "--fail",
            "--max-time",
            "20",
            &crate::model::safetensors::hf_config_url(repo),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = serde_json::from_slice(&out.stdout).ok()?;
    Some((value, out.stdout.len() as u64))
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

#[cfg(test)]
mod shard_tests {
    use super::*;

    fn sized(path: &str, size: u64) -> RepoFile {
        RepoFile {
            path: path.into(),
            size: Some(size),
        }
    }

    #[test]
    fn a_split_set_is_recognised_and_ordered() {
        let (base, i, n) = shard_position("Qwen3-235B-Q4_K_M-00002-of-00003.gguf").unwrap();
        assert_eq!(base, "Qwen3-235B-Q4_K_M.gguf");
        assert_eq!((i, n), (2, 3));
    }

    #[test]
    fn a_model_whose_name_contains_of_is_not_a_shard() {
        // The bug the old `contains("-of-")` test had: `Mixture-of-Experts` is a name, not
        // a split, and treating it as one dropped the file from the table entirely.
        assert!(shard_position("Mixture-of-Experts-Q4_K_M.gguf").is_none());
        assert!(!sized("Mixture-of-Experts-Q4_K_M.gguf", 1).is_shard());
        assert!(sized("M-Q4_K_M-00001-of-00002.gguf", 1).is_shard());
    }

    #[test]
    fn shards_are_grouped_and_their_sizes_summed() {
        let files = vec![
            sized("M-Q4_K_M-00002-of-00003.gguf", 20),
            sized("M-Q8_0.gguf", 99),
            sized("M-Q4_K_M-00001-of-00003.gguf", 10),
            sized("M-Q4_K_M-00003-of-00003.gguf", 30),
        ];
        let (whole, sets) = group_shards(&files);

        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].path, "M-Q8_0.gguf");

        assert_eq!(sets.len(), 1);
        let s = &sets[0];
        assert_eq!(s.size(), Some(60), "the model is the sum of its parts");
        assert!(s.is_complete());
        assert_eq!(s.base.quant_label(), "Q4_K_M");
        // Order matters: part 1 carries the header that the rest is read against.
        let order: Vec<&str> = s.parts.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(order[0], "M-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(order[2], "M-Q4_K_M-00003-of-00003.gguf");
    }

    #[test]
    fn an_incomplete_set_is_flagged_rather_than_silently_summed() {
        // Two of three shards add up to a plausible number that is wrong by a third. A
        // user acting on it downloads a model that cannot load.
        let files = vec![
            sized("M-Q4_K_M-00001-of-00003.gguf", 10),
            sized("M-Q4_K_M-00002-of-00003.gguf", 20),
        ];
        let (_, sets) = group_shards(&files);
        assert!(!sets[0].is_complete());
        assert_eq!(sets[0].expected, 3);
        assert_eq!(sets[0].parts.len(), 2);
    }

    #[test]
    fn two_quantizations_split_separately_stay_separate() {
        let files = vec![
            sized("M-Q4_K_M-00001-of-00002.gguf", 10),
            sized("M-Q4_K_M-00002-of-00002.gguf", 10),
            sized("M-Q8_0-00001-of-00002.gguf", 40),
            sized("M-Q8_0-00002-of-00002.gguf", 40),
        ];
        let (whole, sets) = group_shards(&files);
        assert!(whole.is_empty());
        assert_eq!(sets.len(), 2);
        let labels: Vec<String> = sets.iter().map(|s| s.base.quant_label()).collect();
        assert!(labels.contains(&"Q4_K_M".to_string()));
        assert!(labels.contains(&"Q8_0".to_string()));
    }

    #[test]
    fn a_size_the_api_did_not_report_makes_the_total_unknown_not_wrong() {
        let files = vec![
            sized("M-Q4_K_M-00001-of-00002.gguf", 10),
            RepoFile {
                path: "M-Q4_K_M-00002-of-00002.gguf".into(),
                size: None,
            },
        ];
        let (_, sets) = group_shards(&files);
        assert_eq!(sets[0].size(), None, "a partial sum would understate it");
    }

    #[test]
    fn nonsense_shard_numbering_is_not_treated_as_a_split() {
        assert!(shard_position("M-Q4_K_M-00000-of-00003.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-00004-of-00003.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-000x-of-00003.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-00001-of-.gguf").is_none());
    }

    #[test]
    fn a_401_from_the_hub_is_explained_as_a_missing_or_gated_repo() {
        // The hub answers 401 for a typo exactly as it does for a gated repo, so the
        // message must name both rather than send the user looking for a token.
        let msg = describe_curl_failure("curl: (22) The requested URL returned error: 401");
        assert!(msg.contains("no such repo"), "{msg}");
        assert!(msg.contains("gated"), "{msg}");
        assert!(msg.contains("401"), "{msg}");
    }

    #[test]
    fn a_404_gets_the_same_explanation() {
        let msg = describe_curl_failure("curl: (56) The requested URL returned error: 404\n");
        assert!(msg.contains("no such repo"), "{msg}");
        assert!(msg.contains("404"), "{msg}");
    }

    #[test]
    fn other_http_statuses_are_reported_as_such() {
        let msg = describe_curl_failure("curl: (22) The requested URL returned error: 503");
        assert_eq!(msg, "HuggingFace answered HTTP 503");
    }

    #[test]
    fn a_non_http_failure_keeps_curls_own_words() {
        let msg = describe_curl_failure("curl: (6) Could not resolve host: huggingface.co\n");
        assert_eq!(msg, "curl: (6) Could not resolve host: huggingface.co");
    }

    #[test]
    fn silence_from_curl_still_yields_a_sentence() {
        assert!(describe_curl_failure("   ").contains("no such repo"));
    }
}
