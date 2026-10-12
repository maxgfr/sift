//! Reading a safetensors header, locally or over HTTP.
//!
//! The format is simpler than GGUF and reads the same way: eight little-endian bytes give
//! the length of a JSON header, then that many bytes of it. Every tensor appears as
//!
//! ```json
//! "model.layers.0.mlp.experts.0.gate_proj.weight": {
//!   "dtype": "BF16", "shape": [1024, 2048], "data_offsets": [210243584, 214437888]
//! }
//! ```
//!
//! so a tensor's size is the difference of its two offsets. **No dtype table is needed**,
//! which matters: an FP8 or MXFP4 variant that a hardcoded table would not know about is
//! sized correctly here by arithmetic that cannot go stale.
//!
//! # What the header does not carry
//!
//! Architecture. GGUF puts `block_count` and `attention.head_count_kv` in its own header;
//! a safetensors repo keeps them in a separate `config.json`. So sizing a KV cache takes a
//! second small fetch — see [`facts_from_config`] — and without it `sift` reports weights
//! only and says so, rather than assuming a shape.

use std::io::{self, Read, Seek, SeekFrom};

use super::{kv, ArchFacts, LayerKind, ModelShape, TensorEntry};

/// Ceiling on the JSON header we will read.
///
/// Headers are a few hundred kilobytes; a 141 KB one is typical for a 7B model. A length
/// prefix far above this means the file is not safetensors, or the first eight bytes were
/// read from the wrong offset — either way, failing loudly beats allocating gigabytes on
/// the strength of a number we do not trust.
const MAX_HEADER_BYTES: u64 = 256 << 20;

/// Errors from reading a safetensors header.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("header claims {0} bytes, which is not a safetensors file")]
    ImplausibleHeader(u64),
    #[error("header is not valid JSON: {0}")]
    Json(String),
    #[error("header has no tensors")]
    Empty,
    #[error("tensor `{0}` has a shape or offsets that overflow a 64-bit size")]
    Overflow(String),
}

/// A parsed safetensors header. No payload is read.
#[derive(Debug, Clone)]
pub struct Safetensors {
    pub tensors: Vec<TensorEntry>,
    /// Each tensor's dtype string, in the same order as `tensors`.
    pub dtypes: Vec<String>,
    /// Total bytes the tensors occupy.
    pub total_bytes: u64,
}

impl Safetensors {
    /// Parse the header from anything readable and seekable.
    ///
    /// Takes `Read + Seek` for the same reason [`super::gguf::Gguf::parse`] does: a local
    /// file and a [`super::RemoteFile`] then go through identical code, so there is no
    /// second parser to keep in sync.
    pub fn parse<R: Read + Seek>(src: &mut R) -> Result<Self, SafetensorsError> {
        src.seek(SeekFrom::Start(0))?;

        let mut len_bytes = [0u8; 8];
        src.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);

        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(SafetensorsError::ImplausibleHeader(header_len));
        }

        let mut raw = vec![0u8; header_len as usize];
        src.read_exact(&mut raw)?;

        let json: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| SafetensorsError::Json(e.to_string()))?;
        let obj = json
            .as_object()
            .ok_or_else(|| SafetensorsError::Json("header is not an object".into()))?;

        let mut tensors = Vec::with_capacity(obj.len());
        let mut dtypes = Vec::with_capacity(obj.len());
        let mut total_bytes = 0u64;

        for (name, spec) in obj {
            // Not a tensor: the format reserves this key for free-form strings.
            if name == "__metadata__" {
                continue;
            }
            let Some(offsets) = spec["data_offsets"].as_array() else {
                continue;
            };
            let (Some(start), Some(end)) = (
                offsets.first().and_then(serde_json::Value::as_u64),
                offsets.get(1).and_then(serde_json::Value::as_u64),
            ) else {
                continue;
            };

            let dims: Vec<u64> = spec["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default();

            // Size comes from the offsets, never from dtype x element count. The offsets
            // are what the file actually reserves, and they are right for a quantization
            // this code has never heard of.
            let size_bytes = end.saturating_sub(start);
            let overflow = || SafetensorsError::Overflow(name.clone());
            total_bytes = total_bytes.checked_add(size_bytes).ok_or_else(overflow)?;
            let n_elems = dims
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or_else(overflow)?;

            tensors.push(TensorEntry {
                name: name.clone(),
                n_elems,
                dims,
                size_bytes,
            });
            dtypes.push(spec["dtype"].as_str().unwrap_or_default().to_string());
        }

        if tensors.is_empty() {
            return Err(SafetensorsError::Empty);
        }
        Ok(Self {
            tensors,
            dtypes,
            total_bytes,
        })
    }
}

/// Nested blocks a multimodal `config.json` keeps its language model's numbers in.
///
/// Gemma 3, Llama 4 and Qwen3.5 put `num_hidden_layers` and friends under `text_config`;
/// InternVL uses `llm_config`, others `language_config`. Reading only the top level found
/// none of them, so every multimodal repo came out with an unsized KV cache.
const TEXT_BLOCKS: [&str; 3] = ["text_config", "llm_config", "language_config"];

/// Read the architecture numbers out of a HuggingFace `config.json`.
///
/// Names differ from GGUF's throughout — `num_key_value_heads` rather than
/// `attention.head_count_kv` — which is exactly why [`ArchFacts`] exists: the difference
/// is absorbed here once instead of branching inside the KV arithmetic.
///
/// Each key is looked up at the top level first, then in the text block of a multimodal
/// config, key by key: a wrapper config can state `architectures` itself and leave the
/// layer count to its text model.
pub fn facts_from_config(config: &serde_json::Value) -> ArchFacts {
    let get = |k: &str| -> Option<&serde_json::Value> {
        std::iter::once(config)
            .chain(TEXT_BLOCKS.iter().map(|b| &config[*b]))
            .map(|c| &c[k])
            .find(|v| !v.is_null())
    };
    let u = |k: &str| get(k).and_then(serde_json::Value::as_u64);
    let s = |k: &str| get(k).and_then(serde_json::Value::as_str);

    let block_count = u("num_hidden_layers");
    let model_type = TEXT_BLOCKS
        .iter()
        .find_map(|b| config[*b]["model_type"].as_str())
        .or_else(|| config["model_type"].as_str())
        .unwrap_or_default();

    // `use_sliding_window: false` (Qwen2) means the stated window is not used at all.
    let sliding_window = match get("use_sliding_window").and_then(serde_json::Value::as_bool) {
        Some(false) => None,
        _ => u("sliding_window").or_else(|| u("attention_chunk_size")),
    };

    let layer_kinds = block_count.and_then(|n| {
        let explicit = get("layer_types").and_then(|v| v.as_array()).map(|types| {
            types
                .iter()
                .map(|t| match t.as_str().unwrap_or_default() {
                    "sliding_attention" | "chunked_attention" => LayerKind::Sliding,
                    "linear_attention" | "mamba" | "conv" | "recurrent" => LayerKind::Linear,
                    // Unknown kinds are sized as full: overstating is the safe error.
                    _ => LayerKind::Full,
                })
                .collect()
        });
        let pattern =
            sliding_window
                .filter(|&w| w > 0)
                .and_then(|_| match u("sliding_window_pattern") {
                    Some(every) => Some((every, false)),
                    None => kv::default_sliding_pattern(model_type),
                });
        kv::resolve_layer_kinds(n, explicit, u("full_attention_interval"), pattern)
    });

    // Multi-head latent attention (DeepSeek V2/V3, Kimi K2) caches one latent of
    // `kv_lora_rank + qk_rope_head_dim` per token per layer, whatever the head count.
    let mla_latent = match (u("kv_lora_rank"), u("qk_rope_head_dim")) {
        (Some(rank), Some(rope)) => Some(rank + rope),
        _ => None,
    };

    ArchFacts {
        architecture: get("architectures")
            .and_then(|a| a[0].as_str())
            .or_else(|| s("model_type"))
            .map(str::to_owned),
        block_count,
        head_count: u("num_attention_heads"),
        // Absent means multi-head attention, where query and KV head counts are equal by
        // definition. Left as `None` rather than copied, so `infer_kv_shape` applies that
        // fallback in one place for both formats.
        head_count_kv: u("num_key_value_heads"),
        embedding_length: u("hidden_size"),
        // `head_dim: null` is common and means "derive it", which `as_u64` already gives.
        key_length: u("head_dim"),
        context_length: u("max_position_embeddings"),
        expert_used_count: u("num_experts_per_tok")
            .or_else(|| u("num_experts_per_token"))
            .or_else(|| u("moe_topk")),
        // Each family names it differently: Mixtral `num_local_experts`, Qwen and OLMoE
        // `num_experts`, DeepSeek `n_routed_experts`.
        expert_count: u("num_local_experts")
            .or_else(|| u("num_experts"))
            .or_else(|| u("n_routed_experts"))
            .or_else(|| u("moe_num_experts")),
        value_length: None,
        key_length_swa: None,
        value_length_swa: None,
        head_count_kv_layers: None,
        sliding_window,
        layer_kinds,
        mla_latent,
    }
}

/// How a quantized checkpoint packs several weights into one stored integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    /// `bits` per weight in a U32/I32: MLX (`quantization.bits`) and GPTQ / AWQ
    /// (`quantization_config.bits`).
    Bits(u64),
    /// OCP MXFP4 (`quant_method: "mxfp4"`): two 4-bit weights per byte of `*_blocks`.
    Mxfp4,
}

/// What `config.json` says about packed weights, if anything.
pub fn packing(config: &serde_json::Value) -> Option<Packing> {
    let bits = config["quantization"]["bits"]
        .as_u64()
        .or_else(|| config["quantization_config"]["bits"].as_u64())
        .filter(|&b| (1..=16).contains(&b));
    match bits {
        Some(b) => Some(Packing::Bits(b)),
        None => (config["quantization_config"]["quant_method"].as_str() == Some("mxfp4"))
            .then_some(Packing::Mxfp4),
    }
}

/// Whether `config.json` describes an MLX checkpoint.
///
/// `mlx_lm.convert` writes a top-level `quantization` block; transformers never does.
pub fn is_mlx(config: &serde_json::Value) -> bool {
    config["quantization"].is_object()
}

/// Width in bits of an integer dtype, which is what packed weights are stored as.
fn int_width(dtype: &str) -> Option<u64> {
    match dtype {
        "U8" | "I8" => Some(8),
        "U16" | "I16" => Some(16),
        "U32" | "I32" => Some(32),
        "U64" | "I64" => Some(64),
        _ => None,
    }
}

/// Turn packed integer tensors into logical weight counts.
///
/// A safetensors shape counts *stored* elements. An MLX 4-bit `weight` of U32 holds eight
/// weights per element, so taking the shape at face value under-counts parameters by 8×
/// and reports 30-odd bits per weight for a 4-bit model. The packed tensor's count is
/// scaled by `width / bits`, and its side tensors — scales, zero points, MLX biases, GPTQ
/// `g_idx` — keep their bytes but count no weights, since they are not weights.
///
/// Returns `false` when an integer tensor is left that no recognised scheme explains
/// (bitsandbytes, or MXFP4 with no config to say so). Bits per weight is then unknown,
/// and saying so beats inventing one.
fn unpack(tensors: &mut [TensorEntry], dtypes: &[String], packing: Option<Packing>) -> bool {
    use std::collections::HashMap;
    let index: HashMap<String, usize> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();
    let mut side = vec![false; tensors.len()];
    let mut packed = vec![false; tensors.len()];

    for (i, t) in tensors.iter().enumerate() {
        if int_width(&dtypes[i]).is_none() {
            continue;
        }
        let (base, sides): (&str, &[&str]) = match packing {
            Some(Packing::Bits(_)) => {
                if let Some(b) = t.name.strip_suffix(".qweight") {
                    (b, &[".qzeros", ".scales", ".g_idx"])
                } else if let Some(b) = t.name.strip_suffix(".weight") {
                    (b, &[".scales", ".biases"])
                } else {
                    continue;
                }
            }
            Some(Packing::Mxfp4) => {
                if let Some(b) = t.name.strip_suffix("_blocks") {
                    (b, &["_scales"])
                } else if let Some(b) = t.name.strip_suffix(".blocks") {
                    // gpt-oss's original checkpoint spells it with a dot.
                    (b, &[".scales"])
                } else {
                    continue;
                }
            }
            None => continue,
        };
        let found: Vec<usize> = sides
            .iter()
            .filter_map(|s| index.get(&format!("{base}{s}")).copied())
            .collect();
        // A packed weight always travels with its scales. Without them this is some
        // other integer tensor, and is left for the check below.
        if !found.iter().any(|&j| tensors[j].name.ends_with("scales")) {
            continue;
        }
        packed[i] = true;
        for j in found {
            side[j] = true;
        }
    }

    // FP4 recognised from shapes alone. MiMo-V2.6 and DeepSeek-V4 store experts as bytes
    // holding two 4-bit weights, each group of 16 or 32 sharing one byte-sized scale, while
    // their configs describe only the FP8 rest of the model. A byte tensor whose sibling
    // scale has one entry per 16 or 32 of its weights is that layout and nothing else.
    let mut fp4 = vec![false; tensors.len()];
    for (i, t) in tensors.iter().enumerate() {
        if packed[i] || side[i] || !matches!(dtypes[i].as_str(), "U8" | "I8") {
            continue;
        }
        let base = t.name.strip_suffix(".weight").unwrap_or(&t.name);
        let scale = [".weight_scale", ".scale", ".scales"]
            .iter()
            .find_map(|s| index.get(&format!("{base}{s}")).copied())
            .filter(|&j| matches!(dtypes[j].as_str(), "U8" | "I8" | "F8_E8M0"));
        let Some(j) = scale else { continue };
        let (Some(&w), Some(&g)) = (t.dims.last(), tensors[j].dims.last()) else {
            continue;
        };
        if g > 0 && (w * 2) % g == 0 && matches!(w * 2 / g, 16 | 32) {
            fp4[i] = true;
            side[j] = true;
        }
    }

    let mut understood = true;
    for (i, t) in tensors.iter_mut().enumerate() {
        if fp4[i] {
            t.n_elems = t.n_elems.saturating_mul(2);
        } else if packed[i] {
            let width = int_width(&dtypes[i]).unwrap_or(32);
            t.n_elems = match packing {
                Some(Packing::Bits(bits)) => t.n_elems.saturating_mul(width) / bits,
                Some(Packing::Mxfp4) => t.n_elems.saturating_mul(width / 4),
                None => t.n_elems,
            };
        } else if side[i] {
            t.n_elems = 0;
        } else if int_width(&dtypes[i]).is_some() {
            understood = false;
        }
    }
    understood
}

/// One model's shape from its safetensors headers and its `config.json`.
///
/// The one place a safetensors model is assembled, so `fit`, `inspect`, `route` and `ls`
/// cannot disagree about what a checkpoint holds.
pub fn model_shape(parts: Vec<Safetensors>, config: Option<&serde_json::Value>) -> ModelShape {
    let shard_count = parts.len() as u32;
    let mut tensors = Vec::new();
    let mut dtypes = Vec::new();
    for p in parts {
        tensors.extend(p.tensors);
        dtypes.extend(p.dtypes);
    }
    let understood = unpack(&mut tensors, &dtypes, config.and_then(packing));
    let facts = config.map(facts_from_config).unwrap_or_default();
    ModelShape {
        precision_unknown: !understood,
        ..ModelShape::new(facts, tensors, shard_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn file_with(header: &str) -> Cursor<Vec<u8>> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        // A byte of "payload", which must never be read.
        v.push(0);
        Cursor::new(v)
    }

    #[test]
    fn tensor_size_comes_from_the_offsets_not_a_dtype_table() {
        // The reason it is done this way: `SOMETHING_NEW` is a dtype this code has never
        // heard of, and its size is still exactly right.
        let mut f = file_with(
            r#"{"__metadata__":{"format":"pt"},
                "a.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},
                "b.weight":{"dtype":"SOMETHING_NEW","shape":[4],"data_offsets":[16,20]}}"#,
        );
        let st = Safetensors::parse(&mut f).expect("parse");
        assert_eq!(st.tensors.len(), 2, "__metadata__ is not a tensor");
        assert_eq!(st.total_bytes, 20);

        let b = st.tensors.iter().find(|t| t.name == "b.weight").unwrap();
        assert_eq!(b.size_bytes, 4);
        assert_eq!(b.n_elems, 4);
    }

    #[test]
    fn a_header_length_that_cannot_be_right_is_rejected_not_allocated() {
        // Eight bytes read at the wrong offset produce an enormous length. Allocating on
        // the strength of it would be a denial of service on ourselves.
        let mut v = u64::MAX.to_le_bytes().to_vec();
        v.extend_from_slice(b"{}");
        let err = Safetensors::parse(&mut Cursor::new(v)).expect_err("must reject");
        assert!(matches!(err, SafetensorsError::ImplausibleHeader(_)));
    }

    #[test]
    fn a_header_with_no_tensors_is_an_error_rather_than_an_empty_model() {
        let mut f = file_with(r#"{"__metadata__":{"format":"pt"}}"#);
        assert!(matches!(
            Safetensors::parse(&mut f).expect_err("must reject"),
            SafetensorsError::Empty
        ));
    }

    #[test]
    fn a_shape_that_overflows_is_refused_rather_than_wrapped() {
        let mut f = file_with(
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296,2],"data_offsets":[0,4]}}"#,
        );
        assert!(matches!(
            Safetensors::parse(&mut f).expect_err("must reject"),
            SafetensorsError::Overflow(_)
        ));
    }

    #[test]
    fn config_keys_are_translated_to_the_shared_names() {
        // Real OLMoE config values.
        let cfg = serde_json::json!({
            "architectures": ["OlmoeForCausalLM"],
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "hidden_size": 2048,
            "max_position_embeddings": 4096,
            "num_experts": 64,
            "num_experts_per_tok": 8,
        });
        let f = facts_from_config(&cfg);
        assert_eq!(f.architecture.as_deref(), Some("OlmoeForCausalLM"));
        assert_eq!(f.block_count, Some(16));
        assert_eq!(f.head_count_kv, Some(16));
        assert_eq!(f.embedding_length, Some(2048));
        assert_eq!(f.context_length, Some(4096));
        assert_eq!(f.expert_used_count, Some(8));
        // Not stated by this model, so left for `infer_kv_shape` to derive.
        assert_eq!(f.key_length, None);
    }

    fn kv_of(cfg: serde_json::Value) -> crate::model::KvShape {
        crate::model::kv::kv_shape_from_facts(&facts_from_config(&cfg)).expect("sizeable")
    }

    #[test]
    fn kv_gemma3_multimodal_config_is_read_through_text_config() {
        // Real unsloth/gemma-3-4b-it values: everything that sizes the cache is nested.
        let kv = kv_of(serde_json::json!({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text", "num_hidden_layers": 34,
                "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
                "hidden_size": 2560, "max_position_embeddings": 131072,
                "sliding_window": 1024, "sliding_window_pattern": 6,
            },
            "vision_config": { "num_hidden_layers": 27, "hidden_size": 1152 },
        }));
        assert_eq!((kv.full_layers, kv.sliding_layers), (5, 29));
        let per_layer = 4 * (256 + 256) * 2;
        assert_eq!(
            kv.bytes_at(131_072, 2),
            5 * per_layer * 131_072 + 29 * per_layer * 1024
        );
        assert_eq!(kv.train_context, Some(131_072));
    }

    #[test]
    fn kv_deepseek_v3_config_is_sized_as_mla() {
        // Real deepseek-ai/DeepSeek-V3 values. ~2.3 GB at 32k, where 128 heads of
        // 128-wide K and V gave ~65 GB.
        let kv = kv_of(serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 61,
            "num_attention_heads": 128, "num_key_value_heads": 128, "hidden_size": 7168,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
            "v_head_dim": 128, "max_position_embeddings": 163840,
        }));
        let bytes = kv.bytes_at(32_768, 2);
        assert_eq!(bytes, 61 * (512 + 64) * 32_768 * 2);
        assert!((bytes as f64 / 1e9 - 2.30).abs() < 0.01, "{bytes}");
    }

    #[test]
    fn kv_llama4_text_config_with_chunked_layers() {
        let kv = kv_of(serde_json::json!({
            "architectures": ["Llama4ForConditionalGeneration"],
            "text_config": {
                "model_type": "llama4_text", "num_hidden_layers": 4,
                "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
                "hidden_size": 5120, "attention_chunk_size": 8192,
                "layer_types": ["chunked_attention", "chunked_attention",
                                "chunked_attention", "full_attention"],
            },
        }));
        assert_eq!((kv.full_layers, kv.sliding_layers), (1, 3));
        assert_eq!(kv.window, Some(8192));
        let per_layer = 8 * 256 * 2;
        assert_eq!(
            kv.bytes_at(65_536, 2),
            per_layer * 65_536 + 3 * per_layer * 8192
        );
    }

    #[test]
    fn kv_qwen35_layer_types_mark_linear_attention_layers() {
        let types: Vec<&str> = (0..8)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let kv = kv_of(serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 8, "num_attention_heads": 16, "num_key_value_heads": 4,
                "head_dim": 256, "hidden_size": 4096, "layer_types": types,
                "full_attention_interval": 4,
            },
        }));
        assert_eq!((kv.full_layers, kv.recurrent_layers), (2, 6));
    }

    #[test]
    fn kv_a_null_head_dim_is_derived_and_an_unused_window_is_ignored() {
        // Qwen2 states a 32k window and `use_sliding_window: false`.
        let f = facts_from_config(&serde_json::json!({
            "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
            "hidden_size": 896, "head_dim": null, "sliding_window": 32768,
            "use_sliding_window": false, "model_type": "qwen2",
        }));
        assert_eq!(f.sliding_window, None);
        let kv = crate::model::kv::kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!(kv.head_dim, 64);
        assert!(!kv.is_mixed());
    }

    #[test]
    fn a_config_missing_everything_yields_no_facts_rather_than_zeros() {
        // Zeros would make `infer_kv_shape` compute a cache of size zero and report that
        // everything fits.
        let f = facts_from_config(&serde_json::json!({}));
        assert_eq!(f, ArchFacts::default());
    }

    /// A header with the given `(name, dtype, shape, bytes)` tensors, laid end to end.
    fn header(tensors: &[(&str, &str, &[u64], u64)]) -> Safetensors {
        let mut obj = serde_json::Map::new();
        let mut off = 0u64;
        for (name, dtype, shape, bytes) in tensors {
            obj.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes]}),
            );
            off += bytes;
        }
        Safetensors::parse(&mut file_with(&serde_json::Value::Object(obj).to_string()))
            .expect("parse")
    }

    #[test]
    fn mlx_4bit_weights_count_eight_per_u32_and_land_near_four_and_a_half_bits() {
        // Real mlx-community/Qwen3-4B-4bit geometry: group size 64, BF16 scales and biases.
        let st = header(&[
            (
                "model.layers.0.mlp.up_proj.weight",
                "U32",
                &[9728, 320],
                9728 * 320 * 4,
            ),
            (
                "model.layers.0.mlp.up_proj.scales",
                "BF16",
                &[9728, 40],
                9728 * 40 * 2,
            ),
            (
                "model.layers.0.mlp.up_proj.biases",
                "BF16",
                &[9728, 40],
                9728 * 40 * 2,
            ),
        ]);
        let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
        let shape = model_shape(vec![st], Some(&cfg));
        assert_eq!(shape.total_parameters(), 9728 * 2560, "logical weights");
        let bpw = shape.bits_per_weight().expect("known");
        assert!((bpw - 4.5).abs() < 1e-9, "{bpw}");
    }

    #[test]
    fn awq_and_gptq_qweights_are_unpacked_with_their_side_tensors_weightless() {
        let st = header(&[
            (
                "model.layers.0.mlp.up_proj.qweight",
                "I32",
                &[2560, 1216],
                2560 * 1216 * 4,
            ),
            (
                "model.layers.0.mlp.up_proj.qzeros",
                "I32",
                &[20, 1216],
                20 * 1216 * 4,
            ),
            (
                "model.layers.0.mlp.up_proj.scales",
                "F16",
                &[20, 9728],
                20 * 9728 * 2,
            ),
            ("model.layers.0.mlp.up_proj.g_idx", "I32", &[2560], 2560 * 4),
        ]);
        let cfg = serde_json::json!({"quantization_config": {"bits": 4, "quant_method": "awq"}});
        let shape = model_shape(vec![st], Some(&cfg));
        assert_eq!(shape.total_parameters(), 2560 * 9728);
        let bpw = shape.bits_per_weight().expect("known");
        assert!((4.0..4.5).contains(&bpw), "{bpw}");
    }

    #[test]
    fn mxfp4_blocks_hold_two_weights_per_byte() {
        let st = header(&[
            (
                "model.layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[32, 2880, 90, 16],
                32 * 2880 * 90 * 16,
            ),
            (
                "model.layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[32, 2880, 90],
                32 * 2880 * 90,
            ),
        ]);
        let cfg = serde_json::json!({"quantization_config": {"quant_method": "mxfp4"}});
        let shape = model_shape(vec![st], Some(&cfg));
        assert!((shape.bits_per_weight().unwrap() - 4.25).abs() < 1e-9);
    }

    #[test]
    fn fp4_experts_are_recognised_from_their_shapes_without_a_config_saying_so() {
        // MiMo-V2.6 and DeepSeek-V4.1 store experts as U8 holding two FP4 weights per byte,
        // with one U8 scale per 32 weights — and their configs describe the FP8 rest of the
        // model, not the experts. The shapes say it: 1024 bytes, 64 scales, 32 per scale.
        let st = header(&[
            (
                "model.layers.1.mlp.experts.0.down_proj.weight",
                "U8",
                &[6144, 1024],
                6144 * 1024,
            ),
            (
                "model.layers.1.mlp.experts.0.down_proj.weight_scale",
                "U8",
                &[6144, 64],
                6144 * 64,
            ),
            (
                "layers.1.ffn.experts.0.w1.weight",
                "I8",
                &[2304, 2560],
                2304 * 2560,
            ),
            // DeepSeek-V4.1 types its scales as what they are, E8M0, not as bytes.
            (
                "layers.1.ffn.experts.0.w1.scale",
                "F8_E8M0",
                &[2304, 160],
                2304 * 160,
            ),
        ]);
        let cfg = serde_json::json!({"quantization_config": {"quant_method": "fp8"}});
        let shape = model_shape(vec![st], Some(&cfg));
        assert_eq!(shape.total_parameters(), 6144 * 2048 + 2304 * 5120);
        assert!((shape.bits_per_weight().expect("known") - 4.25).abs() < 1e-9);
    }

    #[test]
    fn gpt_oss_original_blocks_are_unpacked_like_the_hf_ones() {
        let st = header(&[
            (
                "block.0.mlp.mlp1_weight.blocks",
                "U8",
                &[32, 5760, 90, 16],
                32 * 5760 * 90 * 16,
            ),
            (
                "block.0.mlp.mlp1_weight.scales",
                "U8",
                &[32, 5760, 90],
                32 * 5760 * 90,
            ),
        ]);
        let cfg = serde_json::json!({"quantization_config": {"quant_method": "mxfp4"}});
        let shape = model_shape(vec![st], Some(&cfg));
        assert!((shape.bits_per_weight().expect("known") - 4.25).abs() < 1e-9);
    }

    #[test]
    fn an_unrecognised_packed_tensor_makes_precision_unknown_rather_than_invented() {
        // Regression: a 1-bit MLX model read 23.8 bits per weight when its config was not
        // consulted. A packed tensor nothing explains is an unknown, not a number.
        let st = header(&[
            ("model.layers.0.mlp.up_proj.weight", "U32", &[16, 4], 256),
            ("model.layers.0.mlp.up_proj.scales", "F16", &[16, 1], 32),
        ]);
        let shape = model_shape(vec![st], None);
        assert_eq!(shape.bits_per_weight(), None);
    }

    #[test]
    fn an_ordinary_bf16_checkpoint_is_untouched() {
        let st = header(&[("a.weight", "BF16", &[4, 4], 32)]);
        let shape = model_shape(vec![st], Some(&serde_json::json!({})));
        assert_eq!(shape.bits_per_weight(), Some(16.0));
    }

    #[test]
    fn an_mlx_config_is_recognised_by_its_quantization_block() {
        assert!(is_mlx(
            &serde_json::json!({"quantization": {"bits": 1, "group_size": 128}})
        ));
        assert!(!is_mlx(
            &serde_json::json!({"quantization_config": {"bits": 4}})
        ));
    }
}
