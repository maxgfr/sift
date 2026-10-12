//! Working out what the user meant by a model reference.
//!
//! One argument covers everything a user might reasonably type:
//!
//! ```text
//! ./model.gguf  ~/models/x.gguf  C:\m\x.gguf   a local file
//! unsloth/Qwen3-30B-A3B-GGUF                   a repo — pick a model from it
//! unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M            a repo and a quantization
//! https://huggingface.co/org/repo/blob/main/x  a file on the hub, as a browser shows it
//! https://…/model.gguf                         any other URL
//! ```
//!
//! [`RepoRef::parse`] reads all of these, for every command, so `fit` and `inspect` cannot
//! disagree about what `org/repo:Q4_K_M` means. Resolution is ordered so that a path that
//! exists always wins: someone with a directory named `org/repo` should get their file,
//! not a network request.

use anyhow::{bail, Context, Result};
use sift_core::hub::{self, HubFile, RepoFile};
use sift_core::model::{hf_url_at, Gguf, RemoteFile};
use std::path::{Path, PathBuf};

/// What a model reference names, before anything is fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepoRef {
    /// A path on this machine. It exists, or was unmistakably meant as a path.
    Path(PathBuf),
    /// A file on HuggingFace, at a revision.
    HubFile(HubFile),
    /// Any other URL.
    Url(String),
    /// An `org/repo`, optionally narrowed to one quantization.
    Repo { repo: String, quant: Option<String> },
}

impl RepoRef {
    /// Read a user-supplied reference. `quant_flag` is `--quant`, which may also be given
    /// inline as `repo:QUANT` — but not both differently.
    pub fn parse(reference: &str, quant_flag: Option<&str>) -> Result<Self> {
        let expanded = expand_home(reference);
        let as_path = Path::new(&expanded);
        if as_path.exists() {
            return Ok(RepoRef::Path(as_path.to_path_buf()));
        }

        if let Some(f) = hub::parse_hub_url(reference) {
            return Ok(RepoRef::HubFile(f));
        }
        if let Some(repo) = hub_repo_url(reference) {
            return Ok(RepoRef::Repo {
                repo,
                quant: quant_flag.map(str::to_owned),
            });
        }
        if reference.starts_with("http://") || reference.starts_with("https://") {
            return Ok(RepoRef::Url(reference.to_string()));
        }

        // A path that does not exist is a typo, not a hub repo. Asking HuggingFace about
        // `/models/x.gguf` produces a 404 that reads as a network problem.
        if looks_like_path(reference) {
            bail!("no such file: {}", as_path.display());
        }

        let (repo, inline) = split_quant(reference);
        let quant = match (inline, quant_flag) {
            (Some(a), Some(b)) if !a.eq_ignore_ascii_case(b) => {
                bail!("`{reference}` asks for {a} and --quant asks for {b}; name one quantization")
            }
            (a, b) => a.or(b).map(str::to_owned),
        };
        let parts: Vec<&str> = repo.split('/').collect();
        if parts.len() != 2
            || parts
                .iter()
                .any(|p| p.is_empty() || p.contains(char::is_whitespace))
        {
            bail!(
                "`{reference}` is neither an existing path nor an `org/repo` reference.\n\
                 Try:  sift inspect ./model.gguf\n   \
                 or:  sift inspect unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"
            );
        }
        Ok(RepoRef::Repo {
            repo: repo.to_string(),
            quant,
        })
    }
}

/// `~` and `~/…` expanded to the home directory; anything else unchanged.
fn expand_home(reference: &str) -> String {
    let rest = match reference.strip_prefix('~') {
        Some(r) if r.is_empty() || r.starts_with('/') || r.starts_with('\\') => r,
        _ => return reference.to_string(),
    };
    match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        Some(home) => format!("{}{rest}", PathBuf::from(home).display()),
        None => reference.to_string(),
    }
}

/// `https://huggingface.co/org/repo` — a repo's page, with nothing after it.
fn hub_repo_url(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("https://hf.co/"))?;
    let rest = rest.split(['?', '#']).next()?.trim_end_matches('/');
    let mut it = rest.split('/');
    let (org, name) = (it.next()?, it.next()?);
    (it.next().is_none() && !org.is_empty() && !name.is_empty()).then(|| format!("{org}/{name}"))
}

/// A resolved model reference.
#[derive(Debug)]
pub enum Source {
    /// A file on this machine.
    Local(PathBuf),
    /// A model on the network: one URL, or every part of a split model in order.
    Remote {
        urls: Vec<String>,
        label: String,
        /// The `org/repo` this was resolved from, when it was. A safetensors file keeps
        /// its architecture numbers in the repo's `config.json`, so knowing the repo is
        /// what makes KV sizing possible for that format.
        repo: Option<String>,
        /// The revision the URLs are pinned to, for fetching the matching `config.json`.
        revision: String,
    },
}

/// A model's shape, read from wherever it lives.
pub struct Loaded {
    pub shape: sift_core::model::ModelShape,
    /// Bytes over the network, for a remote source.
    pub fetched: Option<Fetched>,
    /// The weight format, which routing depends on.
    pub format: sift_core::engine::Format,
    /// The parsed GGUF directory of the first part, kept so a caller that needs its byte
    /// offsets does not fetch the header a second time.
    pub gguf: Option<Gguf>,
}

/// The `config.json` beside a local safetensors file, if there is one.
///
/// Every MLX and HuggingFace download keeps it there. Without it a 1-bit MLX model reads
/// as 23.8 bits per weight and its KV cache goes unsized.
pub fn local_config(file: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read(file.parent()?.join("config.json")).ok()?;
    serde_json::from_slice(&text).ok()
}

/// Bytes that crossed the network to read a remote model's shape.
///
/// Kept in two figures because they answer two different questions. `header` is the part
/// of the *model file* that was read, and zero means the cached header was revalidated
/// with a `304`. `config` is the repo's `config.json`, a separate small document that is
/// not part of the model at all, so folding it into the first figure would misreport a
/// cached header as a partial read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fetched {
    pub header: u64,
    pub config: u64,
}

impl Fetched {
    /// Everything that crossed the network, for the headline figure.
    pub fn total(self) -> u64 {
        self.header + self.config
    }
}

/// Split a `repo:QUANT` reference into its two halves.
///
/// Splits on the last colon so a URL-ish repo name is not mangled, and refuses a suffix
/// containing `/`, which would mean the colon was part of something else.
pub fn split_quant(reference: &str) -> (&str, Option<&str>) {
    match reference.rsplit_once(':') {
        // A trailing colon with nothing after it is a slip of the finger, not a
        // quantization, and not part of the repo name either.
        Some((repo, "")) => (repo, None),
        Some((repo, quant)) if !quant.contains('/') => (repo, Some(quant)),
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
    let b = reference.as_bytes();
    lower.ends_with(".gguf")
        || lower.ends_with(".safetensors")
        || reference.starts_with('/')
        || reference.starts_with("./")
        || reference.starts_with("../")
        || reference.starts_with(".\\")
        || reference.starts_with("..\\")
        || reference.starts_with('~')
        || reference.starts_with('\\')
        || (b.len() > 2
            && b[0].is_ascii_alphabetic()
            && b[1] == b':'
            && matches!(b[2], b'\\' | b'/'))
}

/// One model a repo offers: a single file, or every part of a split set.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Model {
    label: String,
    /// Where the label came from, to tell two models of the same label apart.
    path: String,
    parts: Vec<String>,
    size: u64,
    split: bool,
}

/// Every model a repo listing holds. Incomplete sets are not models: two of three shards
/// describe a model that cannot be loaded.
fn models(files: &[RepoFile], indexed: &[(String, Vec<String>)]) -> Vec<Model> {
    let (whole, sets) = hub::group_models(files, indexed);
    whole
        .iter()
        .map(|f| Model {
            label: f.quant_label(),
            path: f.path.clone(),
            parts: vec![f.path.clone()],
            size: f.size.unwrap_or(0),
            split: false,
        })
        .chain(sets.iter().filter(|s| s.is_complete()).map(|s| Model {
            label: s.base.quant_label(),
            path: s.base.path.clone(),
            parts: s.parts.iter().map(|p| p.path.clone()).collect(),
            size: s.size().unwrap_or(0),
            split: true,
        }))
        .collect()
}

impl Source {
    /// Resolve a user-supplied reference to something readable.
    ///
    /// `quant` optionally narrows a repo reference; it is also accepted inline as
    /// `repo:QUANT`.
    pub fn resolve(reference: &str, quant: Option<&str>) -> Result<Self> {
        match RepoRef::parse(reference, quant)? {
            RepoRef::Path(p) => {
                if p.is_dir() {
                    bail!("{}", directory_hint(&p));
                }
                Ok(Source::Local(p))
            }
            // A HuggingFace page URL — the `blob` form a browser shows, or `resolve` —
            // names a repo, a revision and a file. Read as a raw URL, the `blob` form
            // fetched an HTML page; read this way it is the file, with the repo's
            // config.json in reach.
            RepoRef::HubFile(f) => {
                let label = RepoFile {
                    path: f.path.clone(),
                    size: None,
                }
                .quant_label();
                Ok(Source::Remote {
                    urls: vec![hf_url_at(&f.repo, &f.revision, &f.path)],
                    label: format!("{}:{label}", f.repo),
                    repo: Some(f.repo),
                    revision: f.revision,
                })
            }
            RepoRef::Url(url) => Ok(Source::Remote {
                label: url.rsplit('/').next().unwrap_or(&url).to_string(),
                urls: vec![url],
                repo: None,
                revision: "main".to_string(),
            }),
            RepoRef::Repo { repo, quant } => {
                let listing = hub::list_weights(&repo, None)?;
                let revision = listing.revision(None).to_string();
                let indexed =
                    hub::fetch_indexes(&repo, &revision, &listing.indexes, &listing.files);
                let offered = models(&listing.files, &indexed);
                let chosen = pick(&offered, quant.as_deref())
                    .with_context(|| format!("choosing a model from `{repo}`"))?;
                Ok(Source::Remote {
                    urls: chosen
                        .parts
                        .iter()
                        .map(|p| hf_url_at(&repo, &revision, p))
                        .collect(),
                    label: format!("{repo}:{}", chosen.label),
                    repo: Some(repo),
                    revision,
                })
            }
        }
    }

    /// Whether this source names safetensors rather than GGUF.
    ///
    /// Matched anywhere in the name, since a signed CDN URL carries its query string after
    /// the filename.
    fn is_safetensors(&self) -> bool {
        match self {
            Source::Local(p) => p.to_string_lossy().contains(".safetensors"),
            Source::Remote { urls, .. } => urls.iter().any(|u| u.contains(".safetensors")),
        }
    }

    /// Read the model's shape, whatever format it is in. Payload is never fetched.
    ///
    /// Returns the shape, its format and, for remote sources, how many bytes crossed the
    /// network — which callers print, because "we did not download the model" is a claim
    /// that should come with evidence. A split model is read part by part and merged,
    /// since each part carries only its own slice of the tensor directory.
    ///
    /// GGUF carries its architecture facts in its own header. A safetensors file keeps them
    /// in `config.json`: the repo's, when the reference named a repo, or the one beside a
    /// local file — the layout every MLX and HF download has. A bare file or URL with
    /// neither is reported as unsized, not guessed at.
    pub fn read_shape(&self) -> Result<Loaded> {
        let safetensors = self.is_safetensors();
        match self {
            Source::Local(p) if safetensors => {
                let mut f =
                    std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
                let st = sift_core::model::Safetensors::parse(&mut f)
                    .with_context(|| format!("reading {}", p.display()))?;
                let config = local_config(p);
                Ok(Loaded {
                    shape: sift_core::model::safetensors::model_shape(vec![st], config.as_ref()),
                    fetched: None,
                    format: sift_core::engine::format_of(&p.to_string_lossy(), config.as_ref()),
                    gguf: None,
                })
            }
            Source::Local(p) => {
                let g = Gguf::open(p).with_context(|| format!("reading {}", p.display()))?;
                Ok(Loaded {
                    shape: g.shape(),
                    fetched: None,
                    format: sift_core::engine::Format::Gguf,
                    gguf: Some(g),
                })
            }
            Source::Remote {
                urls,
                repo,
                revision,
                ..
            } if safetensors => {
                let mut parts = Vec::with_capacity(urls.len());
                let mut header = 0;
                for url in urls {
                    let mut f = RemoteFile::new(url.clone()).with_initial_fetch(8);
                    parts.push(
                        sift_core::model::Safetensors::parse(&mut f)
                            .with_context(|| format!("reading {url}"))?,
                    );
                    f.persist();
                    header += f.bytes_fetched();
                }
                // Counted, but separately from the header: "nothing was downloaded" is a
                // claim that should include every request made to back it, and a 660-byte
                // config must not turn a revalidated header into "0.00 MiB of the model
                // was read".
                let (config, n) = match repo.as_deref().and_then(|r| hub::fetch_config(r, revision))
                {
                    Some((cfg, n)) => (Some(cfg), n),
                    None => (None, 0),
                };
                Ok(Loaded {
                    shape: sift_core::model::safetensors::model_shape(parts, config.as_ref()),
                    fetched: Some(Fetched { header, config: n }),
                    format: sift_core::engine::format_of(&urls[0], config.as_ref()),
                    gguf: None,
                })
            }
            Source::Remote { urls, .. } => {
                let mut parts = Vec::with_capacity(urls.len());
                let mut header = 0;
                for (i, url) in urls.iter().enumerate() {
                    let mut f = RemoteFile::new(url.clone());
                    // Only the first part of a split model carries the tokenizer.
                    if i > 0 {
                        f = f.with_initial_fetch(64 << 10);
                    }
                    parts.push(Gguf::parse(&mut f).with_context(|| format!("reading {url}"))?);
                    f.persist();
                    header += f.bytes_fetched();
                }
                let shape = sift_core::model::ModelShape::sharded(&parts)
                    .context("a model with no parts")?;
                Ok(Loaded {
                    shape,
                    fetched: Some(Fetched { header, config: 0 }),
                    format: sift_core::engine::Format::Gguf,
                    gguf: parts.into_iter().next(),
                })
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

/// What to say when a directory is passed where a model file was expected.
fn directory_hint(dir: &Path) -> String {
    let mut weights: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .is_some_and(|e| e == "gguf" || e == "safetensors")
                })
                .map(|p| p.display().to_string())
                .collect()
        })
        .unwrap_or_default();
    weights.sort();
    match weights.first() {
        Some(first) => format!(
            "{} is a directory; pass one of its weight files, e.g.\n  {first}",
            dir.display()
        ),
        None => format!(
            "{} is a directory with no .gguf or .safetensors file in it",
            dir.display()
        ),
    }
}

/// Choose one model from a repo.
///
/// With no preference, the largest single file at the repo root, on the reasoning that a
/// repo's flagship quantization is usually its biggest one; a repo of split sets only
/// (every safetensors repo past a few GB) gets its largest complete set.
///
/// With a preference, an exact label wins, case-insensitively. Failing that a substring
/// match is accepted only when it is the only one: `Q4` against `IQ4_XS`, `Q4_0` and
/// `Q4_K_M` is a question, and answering it with the smallest silently inspected a
/// different file from the one meant.
fn pick<'a>(models: &'a [Model], quant: Option<&str>) -> Result<&'a Model> {
    if models.is_empty() {
        bail!(
            "no complete model in this repo: only some parts of a split model are listed, \
             which cannot be read as a model"
        );
    }
    let labels = || {
        models
            .iter()
            .map(|m| m.label.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let Some(want) = quant else {
        // At the repo root if anything is: a subdirectory holds an alternative — gpt-oss's
        // `original/` checkpoint, Unsloth's `BF16/` split — not the model a repo is named for.
        let root: Vec<&Model> = models.iter().filter(|m| !m.path.contains('/')).collect();
        let pool: Vec<&Model> = if root.is_empty() {
            models.iter().collect()
        } else {
            root
        };
        return pool
            .iter()
            .filter(|m| !m.split)
            .max_by_key(|m| m.size)
            .or_else(|| pool.iter().max_by_key(|m| m.size))
            .copied()
            .context("no candidate models");
    };

    let exact: Vec<&Model> = models
        .iter()
        .filter(|m| m.label.eq_ignore_ascii_case(want))
        .collect();
    let matches = if exact.is_empty() {
        let up = want.to_ascii_uppercase();
        models
            .iter()
            .filter(|m| m.label.to_ascii_uppercase().contains(&up))
            .collect()
    } else {
        exact
    };
    match matches.as_slice() {
        [one] => Ok(one),
        [] => bail!(
            "no file matching quantization `{want}`. Available: {}",
            labels()
        ),
        several => bail!(
            "`{want}` matches {}; name one",
            several
                .iter()
                .map(
                    |m| if several.iter().filter(|o| o.label == m.label).count() > 1 {
                        m.path.as_str()
                    } else {
                        m.label.as_str()
                    }
                )
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`pick`] over a listing, owning its answer.
    fn choose(files: &[RepoFile], quant: Option<&str>) -> Result<Model> {
        pick(&models(files, &[]), quant).cloned()
    }

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile {
            path: path.into(),
            size: Some(size),
        }
    }

    #[test]
    fn an_explicit_quant_is_honoured() {
        let files = vec![f("M-Q2_K.gguf", 10), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(
            choose(&files, Some("Q4_K_M")).unwrap().path,
            "M-Q4_K_M.gguf"
        );
    }

    #[test]
    fn a_partial_quant_matches_by_substring() {
        let files = vec![f("M-Q2_K.gguf", 10), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(choose(&files, Some("Q4")).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn quant_matching_is_case_insensitive() {
        let files = vec![f("M-Q4_K_M.gguf", 20)];
        assert_eq!(
            choose(&files, Some("q4_k_m")).unwrap().path,
            "M-Q4_K_M.gguf"
        );
    }

    #[test]
    fn shards_are_never_chosen() {
        // A shard's header describes part of a model; reporting it as the model would be
        // wrong in a way the user cannot see.
        let files = vec![f("M-BF16-00001-of-00002.gguf", 100), f("M-Q4_K_M.gguf", 20)];
        assert_eq!(choose(&files, None).unwrap().path, "M-Q4_K_M.gguf");
    }

    #[test]
    fn a_repo_of_only_shards_is_an_error_not_a_silent_wrong_answer() {
        let files = vec![f("M-BF16-00001-of-00002.gguf", 100)];
        assert!(choose(&files, None).is_err());
    }

    #[test]
    fn an_unknown_quant_lists_what_was_available() {
        let files = vec![f("M-Q2_K.gguf", 10)];
        let err = choose(&files, Some("Q8_0")).unwrap_err().to_string();
        assert!(
            err.contains("Q8_0"),
            "the error should name what was asked for: {err}"
        );
    }

    #[test]
    fn an_ambiguous_partial_quant_asks_rather_than_picking_the_smallest() {
        // Regression: `Q4` silently became IQ4_XS, the smallest of five matches.
        let files = vec![
            f("M-IQ4_XS.gguf", 16),
            f("M-Q4_0.gguf", 17),
            f("M-Q4_K_M.gguf", 18),
        ];
        let err = choose(&files, Some("Q4")).unwrap_err().to_string();
        assert!(err.contains("name one"), "{err}");
        assert!(err.contains("IQ4_XS") && err.contains("Q4_K_M"), "{err}");
    }

    #[test]
    fn the_model_label_of_a_safetensors_repo_matches_without_case() {
        let files = vec![
            f("model-00001-of-00002.safetensors", 5),
            f("model-00002-of-00002.safetensors", 5),
        ];
        let m = choose(&files, Some("MODEL")).unwrap();
        assert_eq!(m.parts.len(), 2, "every part of the set, in order");
        assert_eq!(m.parts[0], "model-00001-of-00002.safetensors");
    }

    #[test]
    fn a_split_safetensors_repo_is_one_readable_model() {
        // Regression: a repo holding only `model-0000x-of-0000N.safetensors` could not be
        // inspected at all — "this repo only contains split shards".
        let files = vec![
            f("model-00002-of-00002.safetensors", 4),
            f("model-00001-of-00002.safetensors", 4),
        ];
        let m = choose(&files, None).unwrap();
        assert!(m.split);
        assert_eq!(m.size, 8);
    }

    #[test]
    fn a_root_level_model_is_preferred_over_one_in_a_subdirectory() {
        // gpt-oss-20b keeps its original-format checkpoint in `original/` beside the
        // transformers one at the root; the root one is what every engine loads.
        let files = vec![
            f("original/model.safetensors", 13),
            f("model-00000-of-00002.safetensors", 4),
            f("model-00001-of-00002.safetensors", 4),
            f("model-00002-of-00002.safetensors", 4),
        ];
        let m = choose(&files, None).unwrap();
        assert_eq!(m.parts.len(), 3, "picked {:?}", m.path);
    }

    #[test]
    fn an_inline_quant_and_a_conflicting_flag_are_refused() {
        let err = RepoRef::parse("org/repo:Q4_K_M", Some("Q8_0"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("name one"), "{err}");
        // The same quantization twice is not a conflict.
        assert_eq!(
            RepoRef::parse("org/repo:q8_0", Some("Q8_0")).unwrap(),
            RepoRef::Repo {
                repo: "org/repo".into(),
                quant: Some("q8_0".into())
            }
        );
    }

    #[test]
    fn references_are_classified_without_touching_the_network() {
        assert!(matches!(
            RepoRef::parse("https://huggingface.co/o/r/blob/main/x.gguf", None).unwrap(),
            RepoRef::HubFile(_)
        ));
        assert_eq!(
            RepoRef::parse("https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF", None).unwrap(),
            RepoRef::Repo {
                repo: "unsloth/Qwen3-30B-A3B-GGUF".into(),
                quant: None
            }
        );
        assert!(matches!(
            RepoRef::parse("https://example.com/m.gguf", None).unwrap(),
            RepoRef::Url(_)
        ));
        assert!(RepoRef::parse("org/repo/extra", None).is_err());
    }

    #[test]
    fn a_tilde_is_expanded_to_the_home_directory() {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
        if let Some(home) = home {
            let want = format!("{}/models/x.gguf", PathBuf::from(home).display());
            assert_eq!(expand_home("~/models/x.gguf"), want);
        }
        assert_eq!(
            expand_home("~user/x"),
            "~user/x",
            "only the caller's own home"
        );
    }

    #[test]
    fn a_directory_is_answered_with_the_weight_files_inside_it() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("model.safetensors"), b"x").unwrap();
        let err = Source::resolve(d.path().to_str().unwrap(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a directory"), "{err}");
        assert!(err.contains("model.safetensors"), "{err}");
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
    fn a_hub_blob_url_resolves_to_the_file_at_its_revision() {
        // Regression: the `blob` URL a browser shows was fetched as-is, which returned an
        // HTML page and failed as a 404 from the parser's side.
        let s = Source::resolve(
            "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/blob/main/Qwen3-30B-A3B-Q4_K_M.gguf",
            None,
        )
        .unwrap();
        match s {
            Source::Remote {
                urls, label, repo, ..
            } => {
                assert_eq!(
                    urls[0],
                    "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
                );
                assert_eq!(label, "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M");
                assert_eq!(repo.as_deref(), Some("unsloth/Qwen3-30B-A3B-GGUF"));
            }
            other => panic!("expected a remote source, got {other:?}"),
        }
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
            "C:\\x.gguf",
            ".\\missing.gguf",
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
        assert!(
            looks_like_path("C:\\models"),
            "a drive path without an extension"
        );
        assert!(!looks_like_path("C:x"), "not a drive path");
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
        // A trailing colon is dropped rather than sent to the hub as part of the name.
        assert_eq!(split_quant("org/repo:"), ("org/repo", None));
    }
}
