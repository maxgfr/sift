//! `sift ls` — every local model, across tools that do not know about each other.
//!
//! LM Studio, Ollama and colibri each keep their own library and none of them can see the
//! others. A user with three engines installed has no way to answer "what do I already
//! have, and which of it actually runs well here" without opening three applications and
//! doing the arithmetic by hand.
//!
//! The two layouts are genuinely different and both are handled on their own terms:
//!
//! - **LM Studio, colibri and `~/.sift`** store weight files in a directory tree, so the
//!   path is the name. A model split across files is one model, and a safetensors model
//!   reads the `config.json` beside it.
//! - **Ollama** stores content-addressed blobs — `blobs/sha256-<digest>`, no extension, no
//!   hint of what they are — and keeps the names in separate manifests. Listing the blob
//!   directory would report a pile of hashes, so the manifests are read instead and each
//!   model's weights layer is followed to its blob.

use anyhow::Result;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// A model found on this machine.
pub struct LocalModel {
    /// Which tool's library it came from.
    pub source: &'static str,
    /// Name as that tool would show it.
    pub name: String,
    /// Every file the model is stored in, in order. One for the ordinary case.
    pub paths: Vec<PathBuf>,
    pub size_bytes: u64,
    /// Other names the same weights go by (Ollama tags that share a blob).
    pub aliases: Vec<String>,
}

/// The user's home directory: `HOME`, or `USERPROFILE` on Windows.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Find every local model across the engines this knows about.
///
/// Missing directories are not an error: most machines have one or two of these engines,
/// and an absent library is the normal case rather than a failure.
pub fn discover() -> Vec<LocalModel> {
    let Some(home) = home_dir() else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for (source, dir) in [
        ("LM Studio", lmstudio_models_dir(&home)),
        ("colibri", home.join(".colibri/models")),
        ("sift", home.join(".sift/models")),
    ] {
        found.extend(collect_weights(&dir, source));
    }
    if let Some(dir) = ollama_models_dir(&home) {
        found.extend(ollama_models(&dir));
    }

    found.sort_by(|a, b| a.source.cmp(b.source).then(a.name.cmp(&b.name)));
    found
}

/// LM Studio's model folder: the one its settings name, else the default.
///
/// The folder can be moved in LM Studio's settings, which record it as `downloadsFolder`.
fn lmstudio_models_dir(home: &Path) -> PathBuf {
    std::fs::read(home.join(".lmstudio/settings.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["downloadsFolder"].as_str().map(PathBuf::from))
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| home.join(".lmstudio/models"))
}

/// Ollama's model store: `OLLAMA_MODELS`, the user's `~/.ollama/models`, or the system
/// service's `/usr/share/ollama/.ollama/models` that Linux packages install.
fn ollama_models_dir(home: &Path) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("OLLAMA_MODELS").map(PathBuf::from) {
        return Some(dir);
    }
    [
        home.join(".ollama/models"),
        PathBuf::from("/usr/share/ollama/.ollama/models"),
    ]
    .into_iter()
    .find(|d| d.join("manifests").is_dir())
}

/// Every weight file under `root`, grouped into models.
///
/// Symlinks are followed — LM Studio users link models in from other disks — with a depth
/// bound and a set of canonical paths visited, so a link loop cannot hang the tool and a
/// file linked twice is listed once.
fn collect_weights(root: &Path, source: &'static str) -> Vec<LocalModel> {
    let mut files: Vec<(PathBuf, u64)> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    walk(root, 0, &mut seen, &mut files);

    // Split models — `Model-00001-of-00003.gguf`, `model-00001-of-00002.safetensors` — are
    // one model each. Grouped on their directory and base name, so two quantizations in
    // one folder stay two models.
    let mut groups: BTreeMap<PathBuf, Vec<(u32, PathBuf, u64)>> = BTreeMap::new();
    let mut whole = Vec::new();
    for (path, size) in files {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        match sift_core::hub::shard_position(&name) {
            Some((base, index, _)) => groups.entry(path.with_file_name(base)).or_default().push((
                index,
                path.clone(),
                size,
            )),
            None => whole.push((path, size)),
        }
    }

    let display = |p: &Path| {
        p.strip_prefix(root)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string()
    };
    let mut out: Vec<LocalModel> = whole
        .into_iter()
        .map(|(path, size)| LocalModel {
            source,
            name: display(&path),
            paths: vec![path],
            size_bytes: size,
            aliases: Vec::new(),
        })
        .collect();
    for (base, mut parts) in groups {
        parts.sort_by_key(|p| p.0);
        out.push(LocalModel {
            source,
            name: display(&base),
            size_bytes: parts.iter().map(|p| p.2).sum(),
            paths: parts.into_iter().map(|p| p.1).collect(),
            aliases: Vec::new(),
        });
    }
    out
}

fn walk(dir: &Path, depth: u32, seen: &mut HashSet<PathBuf>, out: &mut Vec<(PathBuf, u64)>) {
    if depth > 8 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // `fs::metadata` follows symlinks; `DirEntry::metadata` would describe the link.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        let Ok(canonical) = std::fs::canonicalize(&path) else {
            continue;
        };
        if !seen.insert(canonical) {
            continue;
        }
        if meta.is_dir() {
            walk(&path, depth + 1, seen, out);
            continue;
        }
        // MLX models ship as safetensors and live in the same libraries, so a scan for
        // `.gguf` alone reports a partial answer on any machine that has one.
        let is_weights = path
            .extension()
            .is_some_and(|e| e == "gguf" || e == "safetensors");
        // Vision projectors and speculative-decoding heads ship alongside a model and are
        // not one. Listing them as models would have the user trying to chat with an
        // encoder. The same rule the hub listing applies.
        let adjunct = sift_core::hub::RepoFile {
            path: path.to_string_lossy().to_string(),
            size: None,
        }
        .is_adjunct();
        if is_weights && !adjunct {
            out.push((path, meta.len()));
        }
    }
}

/// The name Ollama shows for a manifest at `parts` (host, namespace…, model, tag).
///
/// `registry.ollama.ai/library/qwen3/4b` is `qwen3:4b`; a user namespace on the same
/// registry keeps its namespace (`user/model:tag`); any other host keeps everything
/// (`hf.co/unsloth/Qwen3-GGUF:Q4_K_M`). Showing only the last two parts made an `hf.co`
/// pull and a library model of the same name indistinguishable.
fn ollama_name(parts: &[String]) -> Option<String> {
    let (tag, rest) = parts.split_last()?;
    let (model, prefix) = rest.split_last()?;
    let prefix: Vec<&str> = prefix.iter().map(String::as_str).collect();
    let path = match prefix.as_slice() {
        ["registry.ollama.ai", "library"] | [] => model.clone(),
        ["registry.ollama.ai", ns @ ..] => format!("{}/{model}", ns.join("/")),
        all => format!("{}/{model}", all.join("/")),
    };
    Some(format!("{path}:{tag}"))
}

/// Read Ollama's manifests and resolve each model to its weights blob.
///
/// The blob directory alone is unusable — `sha256-1a2b…` files with no extension and no
/// indication of which is a model, which a projector and which a template. The manifest
/// names the media type, so the model layer can be picked out and given its real name.
/// Several tags can point at one blob (`qwen3:latest` and `qwen3:8b`); that is one model
/// on disk, listed once with its other names alongside.
fn ollama_models(root: &Path) -> Vec<LocalModel> {
    let manifests = root.join("manifests");
    let blobs = root.join("blobs");

    let mut files = Vec::new();
    collect_files(&manifests, 0, &mut files);
    files.sort();

    let mut by_blob: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
    for manifest in files {
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(digest) = json["layers"].as_array().and_then(|ls| {
            ls.iter()
                .find(|l| l["mediaType"].as_str() == Some("application/vnd.ollama.image.model"))
                .and_then(|l| l["digest"].as_str())
        }) else {
            continue;
        };
        let rel = manifest.strip_prefix(&manifests).unwrap_or(&manifest);
        let parts: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        let Some(name) = ollama_name(&parts) else {
            continue;
        };
        // Digests are written `sha256:abc…` in the manifest and `sha256-abc…` on disk.
        by_blob
            .entry(blobs.join(digest.replace(':', "-")))
            .or_default()
            .push(name);
    }

    by_blob
        .into_iter()
        .filter_map(|(blob, mut names)| {
            let size = std::fs::metadata(&blob).ok()?.len();
            names.sort();
            let name = names.remove(0);
            Some(LocalModel {
                source: "Ollama",
                name,
                paths: vec![blob],
                size_bytes: size,
                aliases: names,
            })
        })
        .collect()
}

/// Collect every regular file under a directory, depth-bounded.
fn collect_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > 8 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match std::fs::metadata(&path) {
            Ok(m) if m.is_dir() => collect_files(&path, depth + 1, out),
            Ok(_) => out.push(path),
            Err(_) => continue,
        }
    }
}

/// One local model, evaluated against this machine.
pub struct Row {
    pub model: LocalModel,
    /// `None` when the header could not be read.
    pub regime: Option<sift_core::engine::Regime>,
    pub bits_per_weight: Option<f64>,
    pub is_moe: bool,
    /// An embedding model: it encodes text and generates nothing, so it has no decode
    /// speed to estimate.
    pub is_embedding: bool,
    pub format: Option<sift_core::engine::Format>,
    pub bytes_per_token: u64,
    /// Tensors of a dtype this build cannot size, left out of `bits_per_weight`.
    pub dropped_tensors: u64,
    /// `None` when the model does not fit, where memory bandwidth no longer sets the pace,
    /// and for embedding models.
    pub est_tok_s: Option<f64>,
    /// Set when the header could not be read, so the row still appears rather than
    /// vanishing without explanation.
    pub error: Option<String>,
}

/// Evaluate every discovered model against this machine.
///
/// Headers are read in parallel on scoped threads, as `fit` reads remote ones: a library
/// of a dozen models is a dozen independent file reads.
pub fn evaluate(
    models: Vec<LocalModel>,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Vec<Row> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
        .min(models.len().max(1));
    let queue = std::sync::Mutex::new(models.into_iter().enumerate());
    let out = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some((i, m)) = queue.lock().expect("queue lock").next() else {
                    return;
                };
                let row = evaluate_one(m, mem_bytes_per_sec, usable_ram, context_tokens);
                out.lock().expect("results lock").push((i, row));
            });
        }
    });
    let mut rows = out.into_inner().expect("results lock");
    rows.sort_by_key(|(i, _)| *i);
    rows.into_iter().map(|(_, r)| r).collect()
}

/// What reading a local model's files yields.
struct Read {
    shape: sift_core::model::ModelShape,
    format: sift_core::engine::Format,
    /// The GGUF declares a pooling type, which only embedding models carry.
    pooling: bool,
}

/// A local model's shape and format, read from its files.
fn read_local(m: &LocalModel) -> Result<Read> {
    use sift_core::model;
    let first = m.paths.first().ok_or_else(|| anyhow::anyhow!("no files"))?;
    if first.extension().is_some_and(|e| e == "safetensors") {
        let mut parts = Vec::with_capacity(m.paths.len());
        for p in &m.paths {
            let mut f = std::fs::File::open(p)?;
            parts.push(model::Safetensors::parse(&mut f)?);
        }
        // The architecture lives in the config.json beside the weights; without one the
        // KV cache goes uncounted rather than guessed.
        let config = crate::source::local_config(first);
        let format = sift_core::engine::format_of(&first.to_string_lossy(), config.as_ref());
        return Ok(Read {
            shape: model::safetensors::model_shape(parts, config.as_ref()),
            format,
            pooling: false,
        });
    }
    // Ollama blobs are not all GGUF, and a corrupt file is a real possibility; the error
    // becomes the row rather than a silent omission. `Gguf::open` buffers, so the header's
    // thousands of small reads are not thousands of syscalls.
    let parts = m
        .paths
        .iter()
        .map(model::Gguf::open)
        .collect::<Result<Vec<_>, _>>()?;
    let shape = model::ModelShape::sharded(&parts).ok_or_else(|| anyhow::anyhow!("no parts"))?;
    Ok(Read {
        shape,
        format: sift_core::engine::Format::Gguf,
        pooling: parts
            .first()
            .is_some_and(|g| g.arch_value("pooling_type").is_some()),
    })
}

/// Whether a model encodes rather than generates.
///
/// An embedding GGUF declares a pooling type, BERT-family architectures are encoders by
/// construction, and an embedding model built on a decoder (Qwen3-Embedding) says so in
/// its name.
fn is_embedding(shape: &sift_core::model::ModelShape, name: &str, pooling: bool) -> bool {
    let arch = shape
        .architecture()
        .unwrap_or_default()
        .to_ascii_lowercase();
    pooling || arch.contains("bert") || name.to_ascii_lowercase().contains("embed")
}

fn evaluate_one(
    m: LocalModel,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Row {
    use sift_core::engine::Regime;
    use sift_core::model::{self, TokenTraffic};

    let Read {
        shape,
        format,
        pooling,
    } = match read_local(&m) {
        Ok(r) => r,
        Err(e) => return errored(m, format!("{e:#}")),
    };
    let embedding = is_embedding(&shape, &m.name, pooling);

    let moe = model::infer_moe_shape(&shape);
    let traffic = match &moe {
        Some(moe) => TokenTraffic {
            expert_bytes: moe.expert_bytes_per_token(),
            trunk_bytes: shape
                .total_tensor_bytes()
                .saturating_sub(moe.total_expert_bytes()),
        },
        None => TokenTraffic {
            expert_bytes: 0,
            trunk_bytes: shape.total_tensor_bytes(),
        },
    };

    let kv = model::infer_kv_shape(&shape)
        .map(|k| k.bytes_at(context_tokens, model::KV_F16_BYTES))
        .unwrap_or(0);
    let regime = Regime::classify(m.size_bytes.saturating_add(kv), usable_ram);
    let efficiency = if moe.is_some() {
        crate::fit::MOE_EFFICIENCY
    } else {
        crate::fit::DENSE_EFFICIENCY
    };

    Row {
        regime: Some(regime),
        bits_per_weight: shape.bits_per_weight(),
        is_moe: moe.is_some(),
        is_embedding: embedding,
        format: Some(format),
        bytes_per_token: traffic.total(),
        dropped_tensors: shape.dropped_tensors,
        est_tok_s: match regime {
            _ if embedding => None,
            Regime::Fits | Regime::Tight => {
                Some(traffic.tokens_per_sec(mem_bytes_per_sec * efficiency, 0.0))
            }
            Regime::Oversized => None,
        },
        error: None,
        model: m,
    }
}

fn errored(model: LocalModel, why: String) -> Row {
    Row {
        model,
        regime: None,
        bits_per_weight: None,
        is_moe: false,
        is_embedding: false,
        format: None,
        bytes_per_token: 0,
        dropped_tensors: 0,
        est_tok_s: None,
        error: Some(why),
    }
}

/// Print the table.
pub fn report(rows: &[Row], context_tokens: u64, json: bool) -> Result<()> {
    if json {
        let items: Vec<_> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "source": r.model.source,
                    "name": r.model.name,
                    "aliases": r.model.aliases,
                    "path": r.model.paths.first().map(|p| p.display().to_string()),
                    "paths": r.model.paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                    "size_bytes": r.model.size_bytes,
                    "format": r.format,
                    // Null for a row whose header could not be read: "unknown" is not
                    // "does not fit".
                    "regime": r.regime,
                    "bits_per_weight": r.bits_per_weight,
                    "is_moe": r.is_moe,
                    "is_embedding": r.is_embedding,
                    "bytes_per_token": r.bytes_per_token,
                    "dropped_tensors": r.dropped_tensors,
                    "estimated_tokens_per_sec": r.est_tok_s,
                    "error": r.error,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "context_tokens": context_tokens,
                "models": items,
            }))?
        );
        return Ok(());
    }

    if rows.is_empty() {
        println!("no local models found.");
        println!("  Looked in LM Studio's model folder, Ollama's store, ~/.colibri/models and");
        println!("  ~/.sift/models. If your library lives elsewhere, sift does not know");
        println!("  about it yet — that is a gap, not a claim that you have nothing.");
        return Ok(());
    }

    println!(
        "  {:<12} {:<44} {:>9} {:>7} {:>6} {:>10}",
        "engine", "model", "size", "fits", "bpw", "est tok/s"
    );
    for r in rows {
        if let Some(e) = &r.error {
            println!(
                "  {:<12} {:<44} {:>6.2} GB   unreadable: {e}",
                r.model.source,
                truncate(&r.model.name, 44),
                r.model.size_bytes as f64 / 1e9,
            );
            continue;
        }
        let mut notes = vec![if r.is_moe { "moe" } else { "dense" }.to_string()];
        if r.is_embedding {
            notes.push("embedding".to_string());
        }
        if r.format == Some(sift_core::engine::Format::Mlx) {
            notes.push("mlx".to_string());
        }
        if r.model.paths.len() > 1 {
            notes.push(format!("{} parts", r.model.paths.len()));
        }
        if r.dropped_tensors > 0 {
            notes.push(format!("{} unsized tensors", r.dropped_tensors));
        }
        if !r.model.aliases.is_empty() {
            notes.push(format!("also {}", r.model.aliases.join(", ")));
        }
        println!(
            "  {:<12} {:<44} {:>6.2} GB {:>7} {:>6} {:>10}  {}",
            r.model.source,
            truncate(&r.model.name, 44),
            r.model.size_bytes as f64 / 1e9,
            match r.regime {
                Some(sift_core::engine::Regime::Fits) => "yes",
                Some(sift_core::engine::Regime::Tight) => "tight",
                Some(sift_core::engine::Regime::Oversized) | None => "no",
            },
            match r.bits_per_weight {
                Some(b) => format!("{b:.2}"),
                None => "?".into(),
            },
            match r.est_tok_s {
                Some(t) => format!("{t:.0}"),
                None if r.is_embedding => "—".into(),
                None => "disk-bound".into(),
            },
            notes.join(", "),
        );
    }

    println!(
        "\n  fits assumes {context_tokens} tokens of context. tok/s are estimates from\n  \
         measured memory bandwidth, not measurements; embedding models generate nothing\n  \
         and get none."
    );
    Ok(())
}

/// Shorten a name to fit its column, keeping the end.
///
/// The end is what distinguishes `…-Q4_K_M` from `…-Q8_0`, so trimming from the front
/// keeps the part a reader needs.
fn truncate(s: &str, width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= width {
        return s.to_string();
    }
    let tail: String = chars[chars.len() - (width - 1)..].iter().collect();
    format!("…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_name_is_left_alone() {
        assert_eq!(truncate("Qwen3-9B-Q4_K_M.gguf", 44), "Qwen3-9B-Q4_K_M.gguf");
    }

    #[test]
    fn a_long_name_keeps_its_end_because_that_is_the_quant() {
        // Trimming from the end would leave every row of a library reading the same, since
        // the vendor prefix is shared and the quantization is the last thing on the line.
        let long = "lmstudio-community/Some-Very-Long-Vendor-Name/Model-Name-Q4_K_M.gguf";
        let t = truncate(long, 30);
        assert_eq!(t.chars().count(), 30);
        assert!(t.starts_with('…'));
        assert!(t.ends_with("Q4_K_M.gguf"));
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        // A name with non-ASCII in it must not panic on a byte boundary.
        let t = truncate("modèle-très-long-avec-des-accents-Q4_K_M.gguf", 20);
        assert_eq!(t.chars().count(), 20);
    }

    fn parts(p: &str) -> Vec<String> {
        p.split('/').map(str::to_owned).collect()
    }

    #[test]
    fn ollama_names_keep_their_namespace_and_host() {
        // Regression: only the last two path parts were shown, so `hf.co/…:Q4_K_M` read
        // as a bare `Qwen3-GGUF:Q4_K_M`.
        assert_eq!(
            ollama_name(&parts("registry.ollama.ai/library/qwen3/4b")).unwrap(),
            "qwen3:4b"
        );
        assert_eq!(
            ollama_name(&parts("registry.ollama.ai/someone/model/latest")).unwrap(),
            "someone/model:latest"
        );
        assert_eq!(
            ollama_name(&parts("hf.co/unsloth/Qwen3-GGUF/Q4_K_M")).unwrap(),
            "hf.co/unsloth/Qwen3-GGUF:Q4_K_M"
        );
    }

    fn write_manifest(root: &Path, rel: &str, digest: &str) {
        let p = root.join("manifests").join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let m = serde_json::json!({"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest}
        ]});
        std::fs::write(p, m.to_string()).unwrap();
    }

    #[test]
    fn tags_sharing_a_blob_are_one_model_with_aliases() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("blobs")).unwrap();
        std::fs::write(d.path().join("blobs/sha256-aa"), vec![0u8; 10]).unwrap();
        write_manifest(
            d.path(),
            "registry.ollama.ai/library/qwen3/latest",
            "sha256:aa",
        );
        write_manifest(d.path(), "registry.ollama.ai/library/qwen3/8b", "sha256:aa");
        let models = ollama_models(d.path());
        assert_eq!(models.len(), 1, "one blob on disk is one model");
        assert_eq!(models[0].name, "qwen3:8b");
        assert_eq!(models[0].aliases, ["qwen3:latest"]);
    }

    #[test]
    fn split_files_are_one_model_and_two_quants_in_a_folder_stay_two() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("org/Model-GGUF");
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "Model-Q8_0-00001-of-00002.gguf",
            "Model-Q8_0-00002-of-00002.gguf",
            "Model-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "eagle3-Model-BF16.gguf",
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let mut models = collect_weights(d.path(), "LM Studio");
        models.sort_by(|a, b| a.name.cmp(&b.name));
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        // Names are paths relative to the library, in the platform's own separator.
        let rel = |f: &str| {
            Path::new("org")
                .join("Model-GGUF")
                .join(f)
                .to_string_lossy()
                .to_string()
        };
        assert_eq!(names, [rel("Model-Q4_K_M.gguf"), rel("Model-Q8_0.gguf")]);
        assert_eq!(models[1].paths.len(), 2);
        assert_eq!(models[1].size_bytes, 2);
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_followed_and_a_file_linked_twice_is_listed_once() {
        let d = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::write(elsewhere.path().join("m.gguf"), b"x").unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), d.path().join("linked")).unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), d.path().join("again")).unwrap();
        // And a loop, which must not hang the walk.
        std::os::unix::fs::symlink(d.path(), d.path().join("loop")).unwrap();
        let models = collect_weights(d.path(), "LM Studio");
        assert_eq!(
            models.len(),
            1,
            "{:?}",
            models.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn lm_studio_s_own_setting_names_its_model_folder() {
        let home = tempfile::tempdir().unwrap();
        let custom = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join(".lmstudio")).unwrap();
        std::fs::write(
            home.path().join(".lmstudio/settings.json"),
            serde_json::json!({"downloadsFolder": custom.path()}).to_string(),
        )
        .unwrap();
        assert_eq!(lmstudio_models_dir(home.path()), custom.path());
        let bare = tempfile::tempdir().unwrap();
        assert_eq!(
            lmstudio_models_dir(bare.path()),
            bare.path().join(".lmstudio/models")
        );
    }

    #[test]
    fn an_unreadable_row_has_no_regime_rather_than_oversized() {
        let r = errored(
            LocalModel {
                source: "Ollama",
                name: "x".into(),
                paths: vec![],
                size_bytes: 1,
                aliases: vec![],
            },
            "bad".into(),
        );
        assert_eq!(r.regime, None);
    }
}
