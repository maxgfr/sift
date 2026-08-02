//! `sift ls` — every local model, across tools that do not know about each other.
//!
//! LM Studio, Ollama and colibri each keep their own library and none of them can see the
//! others. A user with three engines installed has no way to answer "what do I already
//! have, and which of it actually runs well here" without opening three applications and
//! doing the arithmetic by hand.
//!
//! The two layouts are genuinely different and both are handled on their own terms:
//!
//! - **LM Studio, colibri and `~/.sift`** store `.gguf` files in a directory tree, so the
//!   filename is the name.
//! - **Ollama** stores content-addressed blobs — `blobs/sha256-<digest>`, no extension, no
//!   hint of what they are — and keeps the names in separate manifests. Listing the blob
//!   directory would report a pile of hashes, so the manifests are read instead and each
//!   model's weights layer is followed to its blob.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// A model found on this machine.
pub struct LocalModel {
    /// Which tool's library it came from.
    pub source: &'static str,
    /// Name as that tool would show it.
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// Find every local model across the engines this knows about.
///
/// Missing directories are not an error: most machines have one or two of these engines,
/// and an absent library is the normal case rather than a failure.
pub fn discover() -> Vec<LocalModel> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for (source, dir) in [
        ("LM Studio", home.join(".lmstudio/models")),
        ("colibri", home.join(".colibri/models")),
        ("sift", home.join(".sift/models")),
    ] {
        collect_gguf(&dir, source, &dir, &mut found);
    }
    found.extend(ollama_models(&home.join(".ollama/models")));

    found.sort_by(|a, b| a.source.cmp(b.source).then(a.name.cmp(&b.name)));
    found
}

/// Walk a directory tree collecting `.gguf` files.
///
/// Depth is bounded because a library is a handful of levels deep and a symlink loop in
/// someone's home directory should not hang the tool.
fn collect_gguf(dir: &Path, source: &'static str, root: &Path, out: &mut Vec<LocalModel>) {
    fn walk(dir: &Path, source: &'static str, root: &Path, depth: u32, out: &mut Vec<LocalModel>) {
        if depth > 8 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };

            if meta.is_dir() {
                walk(&path, source, root, depth + 1, out);
                continue;
            }
            if path.extension().is_none_or(|e| e != "gguf") {
                continue;
            }
            // Vision projectors ship alongside a model and are not one. Listing them as
            // models would have the user trying to chat with an encoder.
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            if file_name.starts_with("mmproj") {
                continue;
            }

            out.push(LocalModel {
                source,
                name: path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string(),
                path: path.clone(),
                size_bytes: meta.len(),
            });
        }
    }
    walk(dir, source, root, 0, out);
}

/// Read Ollama's manifests and resolve each model to its weights blob.
///
/// The blob directory alone is unusable — `sha256-1a2b…` files with no extension and no
/// indication of which is a model, which a projector and which a template. The manifest
/// names the media type, so the model layer can be picked out and given its real name.
fn ollama_models(root: &Path) -> Vec<LocalModel> {
    let manifests = root.join("manifests");
    let blobs = root.join("blobs");
    let mut out = Vec::new();

    let mut files = Vec::new();
    collect_files(&manifests, 0, &mut files);

    for manifest in files {
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };

        let Some(layer) = json["layers"].as_array().and_then(|ls| {
            ls.iter()
                .find(|l| l["mediaType"].as_str() == Some("application/vnd.ollama.image.model"))
        }) else {
            continue;
        };
        let Some(digest) = layer["digest"].as_str() else {
            continue;
        };

        // Digests are written `sha256:abc…` in the manifest and `sha256-abc…` on disk.
        let blob = blobs.join(digest.replace(':', "-"));
        let Ok(meta) = std::fs::metadata(&blob) else {
            continue;
        };

        // `manifests/registry.ollama.ai/library/llama3/latest` is the model `llama3:latest`.
        let rel = manifest.strip_prefix(&manifests).unwrap_or(&manifest);
        let parts: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        let name = match parts.len() {
            0 => continue,
            1 => parts[0].clone(),
            n => format!("{}:{}", parts[n - 2], parts[n - 1]),
        };

        out.push(LocalModel {
            source: "Ollama",
            name,
            path: blob,
            size_bytes: meta.len(),
        });
    }
    out
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
        match entry.metadata() {
            Ok(m) if m.is_dir() => collect_files(&path, depth + 1, out),
            Ok(_) => out.push(path),
            Err(_) => continue,
        }
    }
}

/// One local model, evaluated against this machine.
pub struct Row {
    pub model: LocalModel,
    pub regime: sift_core::engine::Regime,
    pub bits_per_weight: Option<f64>,
    pub is_moe: bool,
    pub bytes_per_token: u64,
    /// `None` when the model does not fit, where memory bandwidth no longer sets the pace.
    pub est_tok_s: Option<f64>,
    /// Set when the header could not be read, so the row still appears rather than
    /// vanishing without explanation.
    pub error: Option<String>,
}

/// Evaluate every discovered model against this machine.
pub fn evaluate(
    models: Vec<LocalModel>,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Vec<Row> {
    use sift_core::engine::Regime;
    use sift_core::model::{self, TokenTraffic};

    models
        .into_iter()
        .map(|m| {
            let mut file = match std::fs::File::open(&m.path) {
                Ok(f) => f,
                Err(e) => return errored(m, e.to_string()),
            };
            let g = match model::Gguf::parse(&mut file) {
                Ok(g) => g,
                // Ollama blobs are not all GGUF, and a corrupt file is a real possibility.
                // Saying so beats dropping the row silently.
                Err(e) => return errored(m, e.to_string()),
            };

            let shape = g.shape();
            let (traffic, is_moe) = match model::infer_moe_shape(&shape) {
                Some(moe) => (
                    TokenTraffic {
                        expert_bytes: moe.expert_bytes_per_token(),
                        trunk_bytes: shape
                            .total_tensor_bytes()
                            .saturating_sub(moe.total_expert_bytes()),
                    },
                    true,
                ),
                None => (
                    TokenTraffic {
                        expert_bytes: 0,
                        trunk_bytes: shape.total_tensor_bytes(),
                    },
                    false,
                ),
            };

            let kv = model::infer_kv_shape(&shape)
                .map(|k| k.bytes_at(context_tokens, model::KV_F16_BYTES))
                .unwrap_or(0);
            let regime = Regime::classify(m.size_bytes.saturating_add(kv), usable_ram);
            let efficiency = if is_moe {
                crate::fit::MOE_EFFICIENCY
            } else {
                crate::fit::DENSE_EFFICIENCY
            };

            Row {
                model: m,
                regime,
                bits_per_weight: shape.bits_per_weight(),
                is_moe,
                bytes_per_token: traffic.total(),
                est_tok_s: match regime {
                    Regime::Fits | Regime::Tight => {
                        Some(traffic.tokens_per_sec(mem_bytes_per_sec * efficiency, 0.0))
                    }
                    Regime::Oversized => None,
                },
                error: None,
            }
        })
        .collect()
}

fn errored(model: LocalModel, why: String) -> Row {
    Row {
        model,
        regime: sift_core::engine::Regime::Oversized,
        bits_per_weight: None,
        is_moe: false,
        bytes_per_token: 0,
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
                    "path": r.model.path.display().to_string(),
                    "size_bytes": r.model.size_bytes,
                    "regime": r.regime,
                    "bits_per_weight": r.bits_per_weight,
                    "is_moe": r.is_moe,
                    "bytes_per_token": r.bytes_per_token,
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
        println!("  Looked in ~/.lmstudio/models, ~/.ollama/models, ~/.colibri/models and");
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
        println!(
            "  {:<12} {:<44} {:>6.2} GB {:>7} {:>6} {:>10}  {}",
            r.model.source,
            truncate(&r.model.name, 44),
            r.model.size_bytes as f64 / 1e9,
            match r.regime {
                sift_core::engine::Regime::Fits => "yes",
                sift_core::engine::Regime::Tight => "tight",
                sift_core::engine::Regime::Oversized => "no",
            },
            match r.bits_per_weight {
                Some(b) => format!("{b:.2}"),
                None => "?".into(),
            },
            match r.est_tok_s {
                Some(t) => format!("{t:.0}"),
                None => "disk-bound".into(),
            },
            if r.is_moe { "moe" } else { "dense" },
        );
    }

    println!(
        "\n  fits assumes {context_tokens} tokens of context. tok/s are estimates from\n  \
         measured memory bandwidth, not measurements."
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
}
