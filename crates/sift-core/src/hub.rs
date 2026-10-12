//! Listing what a HuggingFace repo contains, without downloading any of it.
//!
//! The HF API reports every file's exact byte size. Combined with
//! [`crate::model::remote`], that is enough to evaluate every quantization a repo offers —
//! sizes from the API, architecture and dtype mix from a few kilobytes of range requests —
//! before a single weight is transferred.

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

        // Parts are separated by `-` or `.`: Unsloth writes `Model-Q4_K_M`, TheBloke and
        // mradermacher `model.Q4_K_M`. Each part keeps its byte offset so the label can be
        // cut from the stem with its own separators intact (`UD-Q2_K_XL`).
        let mut parts: Vec<(usize, &str)> = Vec::new();
        let mut start = 0;
        for (i, c) in stem.char_indices() {
            if c == '-' || c == '.' {
                parts.push((start, &stem[start..i]));
                start = i + 1;
            }
        }
        parts.push((start, &stem[start..]));

        let mut take_from = None;
        for &(offset, p) in parts.iter().rev() {
            if is_quant_tag(p) {
                take_from = Some(offset);
            } else {
                break;
            }
        }
        match take_from {
            Some(offset) => stem[offset..].to_ascii_uppercase(),
            None => stem.to_string(),
        }
    }

    /// Whether this file is an adjunct shipped beside a model rather than a model.
    ///
    /// Vision projectors (`mmproj-F16.gguf`) and importance matrices sit in the same repos
    /// as the weights and parse as GGUF. Listed as quantizations, a 0.8 GB projector was
    /// the smallest "model" in the table and got recommended.
    pub fn is_adjunct(&self) -> bool {
        let file = self.path.rsplit('/').next().unwrap_or(&self.path);
        let lower = file.to_ascii_lowercase();
        lower.starts_with("mmproj") || lower.contains("imatrix")
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
    // Index 0 is allowed: most sets count `1..=N`, but some (openai/gpt-oss) count
    // `0..=N`, and whether a set is whole is [`ShardSet::is_complete`]'s question.
    if total == 0 || index > total {
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
    /// Each part's index, in the same order as `parts`.
    pub indices: Vec<u32>,
    /// How many parts the filenames claim exist.
    pub expected: u32,
    /// Whether every part names the same total.
    pub totals_agree: bool,
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
    ///
    /// Counting parts is not enough: `1 of 2` beside `2 of 3` and `3 of 3` is three parts
    /// of no model at all. Complete means every part agrees on `expected` and the indices
    /// run without a gap in one of the three numberings publishers actually use:
    /// `1..=N` (the convention), `0..N` (zero-based, N the count: MiniCPM5) and `0..=N`
    /// (zero-based, N the last index: gpt-oss).
    pub fn is_complete(&self) -> bool {
        let indices = || self.indices.iter().copied();
        let n = self.expected;
        self.totals_agree && (indices().eq(1..=n) || indices().eq(0..n) || indices().eq(0..=n))
    }
}

/// Split whole files from shard sets.
///
/// Both are returned because both belong in the table. Shards used to be dropped outright,
/// which left `sift` silent on exactly the largest models — the cases where "will this
/// fit" is hardest and the download most expensive to get wrong.
pub fn group_shards(files: &[RepoFile]) -> (Vec<RepoFile>, Vec<ShardSet>) {
    use std::collections::BTreeMap;

    /// Parts gathered under one base name, before they become a [`ShardSet`].
    struct Gathered {
        parts: Vec<(u32, RepoFile)>,
        expected: u32,
        totals_agree: bool,
    }

    let mut whole = Vec::new();
    let mut sets: BTreeMap<String, Gathered> = BTreeMap::new();

    for f in files {
        match shard_position(&f.path) {
            Some((base, index, total)) => {
                let entry = sets.entry(base).or_insert_with(|| Gathered {
                    parts: Vec::new(),
                    expected: total,
                    totals_agree: true,
                });
                entry.parts.push((index, f.clone()));
                // Trust the largest claim, and remember the disagreement: a set whose
                // parts disagree is malformed, and `is_complete` says so.
                entry.totals_agree &= entry.expected == total;
                entry.expected = entry.expected.max(total);
            }
            None => whole.push(f.clone()),
        }
    }

    let sets = sets
        .into_iter()
        .map(|(base, g)| {
            let Gathered {
                mut parts,
                expected,
                totals_agree,
            } = g;
            parts.sort_by_key(|(i, _)| *i);
            let size = parts.iter().map(|(_, p)| p.size).sum();
            ShardSet {
                base: RepoFile { path: base, size },
                indices: parts.iter().map(|(i, _)| *i).collect(),
                parts: parts.into_iter().map(|(_, p)| p).collect(),
                expected,
                totals_agree,
            }
        })
        .collect();

    (whole, sets)
}

/// Split whole files from multi-file models, using the repo's safetensors indexes first.
///
/// `indexed` pairs each `*.safetensors.index.json` with the files its `weight_map` names
/// (see [`index_parts`]). An index is the authoritative list of a model's parts, whatever
/// they are called: MiMo-V2.6's `model_pp0_ep0_shard0.safetensors` has no `-of-` for
/// [`shard_position`] to find, and without the index each of its 132 files was a "model".
/// Files no index claims are grouped by name as before.
pub fn group_models(
    files: &[RepoFile],
    indexed: &[(String, Vec<String>)],
) -> (Vec<RepoFile>, Vec<ShardSet>) {
    use std::collections::HashSet;
    let mut claimed: HashSet<&str> = HashSet::new();
    let mut sets = Vec::new();
    for (index, parts) in indexed {
        if parts.is_empty() {
            continue;
        }
        let found: Vec<RepoFile> = parts
            .iter()
            .filter_map(|p| files.iter().find(|f| &f.path == p).cloned())
            .collect();
        for p in parts {
            claimed.insert(p.as_str());
        }
        let base = index
            .strip_suffix(".index.json")
            .unwrap_or(index)
            .to_string();
        let size = if found.len() == parts.len() {
            found.iter().map(|f| f.size).sum()
        } else {
            None
        };
        sets.push(ShardSet {
            base: RepoFile { path: base, size },
            // Numbered in the index's order; a missing part leaves a gap at the end, which
            // is what makes `is_complete` false.
            indices: (1..=found.len() as u32).collect(),
            parts: found,
            expected: parts.len() as u32,
            totals_agree: true,
        });
    }
    let rest: Vec<RepoFile> = files
        .iter()
        .filter(|f| !claimed.contains(f.path.as_str()))
        .cloned()
        .collect();
    let (whole, by_name) = group_shards(&rest);
    sets.extend(by_name);
    (whole, sets)
}

/// The files a safetensors index names, resolved against the index's own directory, in
/// order and once each.
pub fn index_parts(index_path: &str, index: &serde_json::Value) -> Vec<String> {
    let dir = index_path
        .rsplit_once('/')
        .map(|(d, _)| format!("{d}/"))
        .unwrap_or_default();
    let mut parts: Vec<String> = index["weight_map"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str())
                .map(|f| format!("{dir}{f}"))
                .collect()
        })
        .unwrap_or_default();
    parts.sort();
    parts.dedup();
    parts
}

/// Fetch every safetensors index a listing holds, as [`group_models`] takes them.
///
/// One small document per multi-file model, fetched only for repos that hold safetensors.
/// An index that cannot be read is left out, and its files fall back to name grouping.
pub fn fetch_indexes(
    repo: &str,
    revision: &str,
    indexes: &[String],
    files: &[RepoFile],
) -> Vec<(String, Vec<String>)> {
    if !files.iter().any(RepoFile::is_safetensors) {
        return Vec::new();
    }
    indexes
        .iter()
        .filter_map(|path| {
            let url = crate::model::hf_url_at(repo, revision, path);
            let out = crate::http::Curl::new(&url, ["-L", "--fail", "--max-time", "20"])
                .output()
                .ok()?;
            if !out.status.success() {
                return None;
            }
            let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
            Some((path.clone(), index_parts(path, &json)))
        })
        .collect()
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

/// What a repo offers at one revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing {
    /// Weight files, smallest first.
    pub files: Vec<RepoFile>,
    /// The commit the listing describes.
    ///
    /// Every later fetch is pinned to it, so a push landing mid-sweep cannot make one
    /// quantization's header come from one commit and the next from another.
    pub sha: Option<String>,
    /// `*.safetensors.index.json` files, each naming one multi-file model's parts.
    pub indexes: Vec<String>,
}

impl Listing {
    /// The revision to fetch files at: the listed commit, else what was asked for.
    pub fn revision<'a>(&'a self, asked: Option<&'a str>) -> &'a str {
        self.sha.as_deref().or(asked).unwrap_or("main")
    }
}

/// List the weight files a repo offers, with sizes, without downloading anything.
///
/// Both formats, because a repo that ships only safetensors is a repo `sift` should have
/// something to say about rather than a dead end. Vision projectors and importance
/// matrices are left out: they are GGUF files and not models. `revision` is a branch, tag
/// or commit; `None` is the default branch.
pub fn list_weights(repo: &str, revision: Option<&str>) -> Result<Listing, HubError> {
    let url = match revision {
        Some(rev) => format!(
            "https://huggingface.co/api/models/{repo}/revision/{}?blobs=true",
            crate::http::encode_path(rev)
        ),
        None => format!("https://huggingface.co/api/models/{repo}?blobs=true"),
    };
    let curl = crate::http::Curl::new(&url, ["-L", "--fail", "--max-time", "30"]);
    let authenticated = curl.authenticated();
    let out = curl.output()?;

    if !out.status.success() {
        return Err(HubError::Request {
            repo: repo.to_string(),
            detail: describe_curl_failure(&String::from_utf8_lossy(&out.stderr), authenticated),
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
                .filter(|f| !f.is_adjunct())
                .collect()
        })
        .unwrap_or_default();

    if files.is_empty() {
        return Err(HubError::NoWeights(repo.to_string()));
    }

    files.sort_by_key(|f| f.size);
    let indexes = json["siblings"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s["rfilename"].as_str())
                .filter(|p| p.ends_with(".safetensors.index.json"))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    Ok(Listing {
        files,
        sha: json["sha"].as_str().map(str::to_owned),
        indexes,
    })
}

/// Turn curl's failure text into something a user can act on.
///
/// The hub answers `401` for a repo that does not exist as well as for one that is gated,
/// so a raw `curl: (22) The requested URL returned error: 401` reads as an authentication
/// problem to someone who merely mistyped a name. Name both possibilities, and say what to
/// do about the second.
fn describe_curl_failure(stderr: &str, authenticated: bool) -> String {
    let stderr = stderr.trim();
    let status = stderr
        .rsplit_once("returned error: ")
        .and_then(|(_, tail)| tail.split_whitespace().next())
        .and_then(|code| code.parse::<u16>().ok());
    match status {
        Some(code) => match crate::http::refusal_hint(code, authenticated) {
            Some(hint) => format!("{hint} (HTTP {code})"),
            None => format!("HuggingFace answered HTTP {code}"),
        },
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
pub fn fetch_config(repo: &str, revision: &str) -> Option<(serde_json::Value, u64)> {
    let url = crate::model::hf_url_at(repo, revision, "config.json");
    let out = crate::http::Curl::new(&url, ["-L", "--fail", "--max-time", "20"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = serde_json::from_slice(&out.stdout).ok()?;
    Some((value, out.stdout.len() as u64))
}

/// A file on HuggingFace named by its web URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubFile {
    pub repo: String,
    pub revision: String,
    pub path: String,
}

/// Read `https://huggingface.co/{org}/{repo}/(blob|resolve)/{rev}/{path}`.
///
/// The `blob` form is what a browser's address bar shows, and pasting it used to fetch an
/// HTML page and fail with a 404 from the GGUF parser's point of view. Both forms name the
/// same file, so both become a repo, a revision and a path.
pub fn parse_hub_url(url: &str) -> Option<HubFile> {
    let rest = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("https://hf.co/"))?;
    let rest = rest.split(['?', '#']).next()?;
    let mut it = rest.splitn(5, '/');
    let (org, name, kind, rev, path) = (it.next()?, it.next()?, it.next()?, it.next()?, it.next()?);
    if !matches!(kind, "blob" | "resolve") || org.is_empty() || name.is_empty() || path.is_empty() {
        return None;
    }
    Some(HubFile {
        repo: format!("{org}/{name}"),
        revision: percent_decode(rev),
        path: percent_decode(path),
    })
}

/// Undo `%XX` escapes. Malformed escapes are kept as written.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = |c: u8| (c as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
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
    fn dot_separated_names_are_split_like_hyphenated_ones() {
        // TheBloke and mradermacher write `model.Q4_K_M.gguf`; the whole stem used to come
        // back as the label and overflow the table.
        assert_eq!(
            f("mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf").quant_label(),
            "Q4_K_M"
        );
        assert_eq!(f("DeepSeek-V2-Lite.IQ3_XS.gguf").quant_label(), "IQ3_XS");
        assert_eq!(f("Model.i1-Q4_K_M.gguf").quant_label(), "Q4_K_M");
        // A version number with a dot is not mistaken for a separator before a tag.
        assert_eq!(f("Qwen2.5-7B-Instruct-Q8_0.gguf").quant_label(), "Q8_0");
    }

    #[test]
    fn projectors_and_importance_matrices_are_adjuncts_not_models() {
        assert!(f("mmproj-F32.gguf").is_adjunct());
        assert!(f("mmproj-model-f16.gguf").is_adjunct());
        assert!(f("sub/MMPROJ-BF16.gguf").is_adjunct());
        assert!(f("imatrix_unsloth.gguf").is_adjunct());
        assert!(!f("gemma-3-4b-it-Q4_K_M.gguf").is_adjunct());
    }

    #[test]
    fn hub_page_urls_are_read_as_repo_revision_and_file() {
        let blob = parse_hub_url(
            "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/blob/main/Qwen3-30B-A3B-Q4_K_M.gguf",
        )
        .expect("a hub file");
        assert_eq!(blob.repo, "unsloth/Qwen3-30B-A3B-GGUF");
        assert_eq!(blob.revision, "main");
        assert_eq!(blob.path, "Qwen3-30B-A3B-Q4_K_M.gguf");

        let nested = parse_hub_url(
            "https://huggingface.co/o/r/resolve/abc123/BF16/m%20x-BF16-00001-of-00002.gguf?download=true",
        )
        .expect("a hub file");
        assert_eq!(nested.revision, "abc123");
        assert_eq!(nested.path, "BF16/m x-BF16-00001-of-00002.gguf");

        assert!(parse_hub_url("https://huggingface.co/o/r").is_none());
        assert!(parse_hub_url("https://huggingface.co/o/r/tree/main/x.gguf").is_none());
        assert!(parse_hub_url("https://example.com/o/r/blob/main/x.gguf").is_none());
    }

    #[test]
    fn a_malformed_escape_is_kept_rather_than_panicking() {
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%zz%41"), "%zzA");
        assert_eq!(percent_decode("é%20x"), "é x");
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
    fn a_safetensors_index_groups_files_whatever_they_are_called() {
        // XiaomiMiMo/MiMo-V2.6-Pro-RL names its shards `model_pp0_ep0_shard0.safetensors`
        // and so on: no `-of-` anywhere, so each of 132 files became its own "model".
        let files = vec![
            sized("model_pp0_ep0_shard0.safetensors", 5),
            sized("model_pp0_ep1_shard0.safetensors", 7),
            sized("dflash/dflash_draft_model.safetensors", 1),
        ];
        let indexed = vec![(
            "model.safetensors.index.json".to_string(),
            vec![
                "model_pp0_ep0_shard0.safetensors".to_string(),
                "model_pp0_ep1_shard0.safetensors".to_string(),
            ],
        )];
        let (whole, sets) = group_models(&files, &indexed);
        assert_eq!(sets.len(), 1);
        assert!(sets[0].is_complete());
        assert_eq!(sets[0].base.path, "model.safetensors");
        assert_eq!(sets[0].size(), Some(12));
        assert_eq!(
            whole.len(),
            1,
            "the draft model is not part of the indexed one"
        );

        // An index naming a file the listing lacks is an incomplete model, not a smaller one.
        let missing = vec![(
            "model.safetensors.index.json".to_string(),
            vec!["a.safetensors".to_string(), "b.safetensors".to_string()],
        )];
        let (_, sets) = group_models(&[sized("a.safetensors", 1)], &missing);
        assert!(!sets[0].is_complete());
    }

    #[test]
    fn index_part_names_are_resolved_against_the_index_directory() {
        let index = serde_json::json!({"weight_map": {
            "a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-1.safetensors"
        }});
        assert_eq!(
            index_parts("sub/model.safetensors.index.json", &index),
            ["sub/model-1.safetensors", "sub/model-2.safetensors"]
        );
    }

    #[test]
    fn a_zero_based_set_is_a_complete_model() {
        // openai/gpt-oss-20b ships model-00000-of-00002 … model-00002-of-00002: three
        // files, numbered from zero. Rejecting index 0 split one model into a "whole file"
        // holding a third of its tensors and a set missing that third.
        let files = vec![
            sized("model-00000-of-00002.safetensors", 4),
            sized("model-00001-of-00002.safetensors", 4),
            sized("model-00002-of-00002.safetensors", 4),
        ];
        let (whole, sets) = group_shards(&files);
        assert!(whole.is_empty());
        assert_eq!(sets.len(), 1);
        assert!(sets[0].is_complete());
        assert_eq!(sets[0].parts.len(), 3);
        assert_eq!(sets[0].size(), Some(12));
    }

    #[test]
    fn a_zero_based_set_counted_to_n_is_complete_too() {
        // openbmb/MiniCPM5-2B is one file, `model-00000-of-00001`: numbered from zero with
        // the count, not the last index, after `of`.
        let one = vec![sized("model-00000-of-00001.safetensors", 4)];
        let (whole, sets) = group_shards(&one);
        assert!(whole.is_empty());
        assert!(sets[0].is_complete());

        let files: Vec<RepoFile> = (0..4)
            .map(|i| sized(&format!("model-{i:05}-of-00004.safetensors"), 1))
            .collect();
        let (_, sets) = group_shards(&files);
        assert!(sets[0].is_complete(), "0..4 of 4");
    }

    #[test]
    fn where_names_are_ambiguous_the_index_decides() {
        // `00000` and `00001` of `00002` is a whole model under the count convention and
        // two thirds of one under gpt-oss's last-index convention. Names cannot tell; the
        // index gpt-oss ships can.
        let files = vec![
            sized("model-00000-of-00002.safetensors", 4),
            sized("model-00001-of-00002.safetensors", 4),
        ];
        let indexed = vec![(
            "model.safetensors.index.json".to_string(),
            vec![
                "model-00000-of-00002.safetensors".to_string(),
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ],
        )];
        let (_, sets) = group_models(&files, &indexed);
        assert_eq!(sets.len(), 1);
        assert!(
            !sets[0].is_complete(),
            "the index names a third file the repo lacks"
        );
    }

    #[test]
    fn parts_that_disagree_on_the_total_are_not_a_complete_set() {
        let files = vec![
            sized("M-Q4_K_M-00001-of-00002.gguf", 10),
            sized("M-Q4_K_M-00002-of-00003.gguf", 20),
            sized("M-Q4_K_M-00003-of-00003.gguf", 30),
        ];
        let (_, sets) = group_shards(&files);
        assert_eq!(sets.len(), 1);
        assert!(!sets[0].totals_agree);
        assert!(
            !sets[0].is_complete(),
            "1 of 2, 2 of 3, 3 of 3 is not one model"
        );
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
        // Index 0 is not nonsense — some publishers number from zero, see
        // `a_zero_based_set_is_a_complete_model` — but an index past the total is, and so
        // is a total of zero.
        assert!(shard_position("M-Q4_K_M-00004-of-00003.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-00000-of-00000.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-000x-of-00003.gguf").is_none());
        assert!(shard_position("M-Q4_K_M-00001-of-.gguf").is_none());
    }

    #[test]
    fn a_401_from_the_hub_is_explained_as_a_missing_or_gated_repo() {
        // The hub answers 401 for a typo exactly as it does for a gated repo, so the
        // message must name both rather than send the user looking for a token.
        let msg = describe_curl_failure("curl: (22) The requested URL returned error: 401", false);
        assert!(msg.contains("no such repo"), "{msg}");
        assert!(msg.contains("gated"), "{msg}");
        assert!(msg.contains("401"), "{msg}");
        assert!(msg.contains("HF_TOKEN"), "{msg}");
    }

    #[test]
    fn a_404_gets_the_same_explanation() {
        let msg =
            describe_curl_failure("curl: (56) The requested URL returned error: 404\n", false);
        assert!(msg.contains("no such repo"), "{msg}");
        assert!(msg.contains("404"), "{msg}");
    }

    #[test]
    fn other_http_statuses_are_reported_as_such() {
        let msg = describe_curl_failure("curl: (22) The requested URL returned error: 503", false);
        assert_eq!(msg, "HuggingFace answered HTTP 503");
    }

    #[test]
    fn a_non_http_failure_keeps_curls_own_words() {
        let msg =
            describe_curl_failure("curl: (6) Could not resolve host: huggingface.co\n", false);
        assert_eq!(msg, "curl: (6) Could not resolve host: huggingface.co");
    }

    #[test]
    fn silence_from_curl_still_yields_a_sentence() {
        assert!(describe_curl_failure("   ", false).contains("no such repo"));
    }
}
