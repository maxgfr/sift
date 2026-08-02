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

use super::{ArchFacts, TensorEntry};

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
    #[error("reading safetensors header: {0}")]
    Io(#[from] io::Error),
    #[error("header claims {0} bytes, which is not a safetensors file")]
    ImplausibleHeader(u64),
    #[error("header is not valid JSON: {0}")]
    Json(String),
    #[error("header has no tensors")]
    Empty,
}

/// A parsed safetensors header. No payload is read.
#[derive(Debug, Clone)]
pub struct Safetensors {
    pub tensors: Vec<TensorEntry>,
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
            total_bytes += size_bytes;

            tensors.push(TensorEntry {
                name: name.clone(),
                n_elems: dims.iter().product::<u64>(),
                dims,
                size_bytes,
            });
        }

        if tensors.is_empty() {
            return Err(SafetensorsError::Empty);
        }
        Ok(Self {
            tensors,
            total_bytes,
        })
    }
}

/// Read the architecture numbers out of a HuggingFace `config.json`.
///
/// Names differ from GGUF's throughout — `num_key_value_heads` rather than
/// `attention.head_count_kv` — which is exactly why [`ArchFacts`] exists: the difference
/// is absorbed here once instead of branching inside the KV arithmetic.
pub fn facts_from_config(config: &serde_json::Value) -> ArchFacts {
    let u = |k: &str| config.get(k).and_then(serde_json::Value::as_u64);

    ArchFacts {
        architecture: config["architectures"][0]
            .as_str()
            .or_else(|| config["model_type"].as_str())
            .map(str::to_owned),
        block_count: u("num_hidden_layers"),
        head_count: u("num_attention_heads"),
        // Absent means multi-head attention, where query and KV head counts are equal by
        // definition. Left as `None` rather than copied, so `infer_kv_shape` applies that
        // fallback in one place for both formats.
        head_count_kv: u("num_key_value_heads"),
        embedding_length: u("hidden_size"),
        key_length: u("head_dim"),
        context_length: u("max_position_embeddings"),
        expert_used_count: u("num_experts_per_tok").or_else(|| u("num_experts_per_token")),
    }
}

/// Group a repo's safetensors files, keyed on the whole model.
///
/// `model-00001-of-00003.safetensors` is one part of one model, exactly like a GGUF split.
/// Returned as `(base, parts)` so the caller can read every part's header — each carries
/// only its own slice of the tensor map.
pub fn shard_position(path: &str) -> Option<(String, u32, u32)> {
    let stem = path.strip_suffix(".safetensors")?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (base, index) = rest.rsplit_once('-')?;

    let total: u32 = total.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    if total == 0 || index == 0 || index > total {
        return None;
    }
    Some((format!("{base}.safetensors"), index, total))
}

/// URL of a repo's `config.json`, which carries the architecture numbers.
pub fn hf_config_url(repo: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/config.json")
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

    #[test]
    fn a_config_missing_everything_yields_no_facts_rather_than_zeros() {
        // Zeros would make `infer_kv_shape` compute a cache of size zero and report that
        // everything fits.
        let f = facts_from_config(&serde_json::json!({}));
        assert_eq!(f, ArchFacts::default());
    }

    #[test]
    fn split_safetensors_are_recognised() {
        let (base, i, n) = shard_position("model-00002-of-00003.safetensors").unwrap();
        assert_eq!(base, "model.safetensors");
        assert_eq!((i, n), (2, 3));
        assert!(shard_position("model.safetensors").is_none());
    }
}
