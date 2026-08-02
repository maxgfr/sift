//! Reading model files without loading them.
//!
//! The planner has to reason about a model it may not be able to hold: how many experts,
//! how big each one is, where its bytes live. All of that comes from the tensor
//! directory, which is a few hundred kilobytes at the front of the file. Payload is never
//! touched here.

pub mod experts;
pub mod ggml;
pub mod gguf;
pub mod remote;

pub use experts::{
    expert_ranges, expert_slice, ExpertError, ExpertRanges, ExpertSlice, TokenTraffic,
};
pub use ggml::{BlockLayout, GgmlType};
// Re-exported at the module root so callers reach for one `model::` path.
pub use self::KV_F16_BYTES as KV_DEFAULT_BYTES;
pub use gguf::{Gguf, GgufError, TensorInfo, Value};
pub use remote::{hf_url, RemoteFile};

/// One model's shape, whether it ships as a single file or several.
///
/// The largest models ship split — `Model-Q4_K_M-00001-of-00003.gguf` and friends — and
/// each part carries only its own slice of the tensor directory. Asking a single shard for
/// the model's size, parameter count or expert layout gives an answer that is confidently
/// a third of the truth.
///
/// This is the analysis view, deliberately separate from [`Gguf`]: a `Gguf` can address
/// bytes because its offsets are relative to its own file, and a merged view cannot. That
/// distinction is kept in the type rather than in a comment nobody reads, so
/// [`expert_ranges`] still takes a `Gguf` and cannot accidentally be handed a shard set.
pub struct ModelShape<'a> {
    metadata: &'a std::collections::HashMap<String, Value>,
    tensors: Vec<&'a TensorInfo>,
    /// Files this model is stored in. 1 for the ordinary case.
    pub shard_count: u32,
}

impl<'a> ModelShape<'a> {
    /// View a single-file model.
    pub fn single(g: &'a Gguf) -> Self {
        Self {
            metadata: &g.metadata,
            tensors: g.tensors.iter().collect(),
            shard_count: 1,
        }
    }

    /// Merge the parts of a split model, in shard order.
    ///
    /// Metadata comes from the first part: every shard repeats the architecture keys, and
    /// the first is the one whose presence is guaranteed. Returns `None` for an empty
    /// slice, since a model with no parts has no shape.
    pub fn sharded(parts: &'a [Gguf]) -> Option<Self> {
        let first = parts.first()?;
        Some(Self {
            metadata: &first.metadata,
            tensors: parts.iter().flat_map(|g| g.tensors.iter()).collect(),
            shard_count: parts.len() as u32,
        })
    }

    /// The model architecture string, e.g. `qwen3moe`.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(Value::as_str)
    }

    /// Read an architecture-scoped metadata integer.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_u64)
    }

    /// Look a tensor up by exact name, across every shard.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name).copied()
    }

    /// Total payload bytes across every shard.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(|t| t.size_bytes()).sum()
    }

    /// Total weight count across every shard.
    pub fn total_parameters(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.size_bytes().is_some())
            .map(|t| t.n_elems())
            .sum()
    }

    /// Mean bits stored per weight. See [`Gguf::bits_per_weight`].
    pub fn bits_per_weight(&self) -> Option<f64> {
        let params = self.total_parameters();
        if params == 0 {
            return None;
        }
        Some(self.total_tensor_bytes() as f64 * 8.0 / params as f64)
    }
}

impl Gguf {
    /// This file viewed as a whole model.
    pub fn shape(&self) -> ModelShape<'_> {
        ModelShape::single(self)
    }
}

/// What a model's attention layers cost to keep a context in memory.
///
/// Weights are not the whole footprint, and on long contexts they are not even the larger
/// part: a 30B model at 4 bits is ~18 GB of weights, and 128k tokens of KV cache on a
/// 48-layer model is another 12 GB. Judging "does it fit" on the file size alone answers a
/// question nobody asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvShape {
    /// Transformer blocks, each of which holds its own K and V.
    pub layers: u64,
    /// Key/value heads. Fewer than query heads under grouped-query attention, which is
    /// most modern models and is exactly why assuming `head_count` here overstates the
    /// cache by the GQA ratio — 8x on some models.
    pub kv_heads: u64,
    /// Width of one head.
    pub head_dim: u64,
    /// Context the model was trained for, when it says so.
    pub train_context: Option<u64>,
}

/// Bytes per KV element at the precision engines default to.
///
/// f16. Engines can be told to use 8- or 4-bit KV, which is why
/// [`KvShape::bytes_at`] takes this as an argument rather than baking it in — but the
/// default is what a user who has not gone looking will get.
pub const KV_F16_BYTES: u64 = 2;

impl KvShape {
    /// Bytes of KV cache held at `context_tokens`.
    ///
    /// Two tensors per layer — K and V — each `kv_heads * head_dim` wide per token.
    pub fn bytes_at(&self, context_tokens: u64, bytes_per_element: u64) -> u64 {
        2u64.saturating_mul(self.layers)
            .saturating_mul(self.kv_heads)
            .saturating_mul(self.head_dim)
            .saturating_mul(context_tokens)
            .saturating_mul(bytes_per_element)
    }
}

/// Infer the KV cache shape from metadata.
///
/// Returns `None` when any of the three factors is missing, because a partial answer here
/// would be a silently low one — and a low KV estimate makes a model look like it fits
/// when it does not, which is the failure mode with the highest cost to the user.
pub fn infer_kv_shape(g: &ModelShape) -> Option<KvShape> {
    let layers = g.arch_u64("block_count")?;
    // Grouped-query attention means K/V heads are usually fewer than query heads. Fall
    // back to `head_count` only when `head_count_kv` is absent, which means multi-head
    // attention where the two are equal by definition.
    let kv_heads = g
        .arch_u64("attention.head_count_kv")
        .or_else(|| g.arch_u64("attention.head_count"))?;

    // Some architectures state the head width outright; the rest imply it. Deepseek-style
    // models with a decoupled head dimension state it, and deriving it from the embedding
    // width would be wrong for them.
    let head_dim = match g.arch_u64("attention.key_length") {
        Some(d) => d,
        None => {
            let embedding = g.arch_u64("embedding_length")?;
            let heads = g.arch_u64("attention.head_count")?;
            if heads == 0 {
                return None;
            }
            embedding / heads
        }
    };

    if layers == 0 || kv_heads == 0 || head_dim == 0 {
        return None;
    }

    Some(KvShape {
        layers,
        kv_heads,
        head_dim,
        train_context: g.arch_u64("context_length"),
    })
}

/// A model's MoE shape, as far as the tensor directory reveals it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeShape {
    /// Number of transformer blocks carrying stacked expert tensors.
    pub moe_layers: u32,
    /// Routed experts per MoE layer.
    pub experts_per_layer: u64,
    /// Experts activated per token per layer (top-k).
    pub experts_per_token: u64,
    /// Total bytes of routed-expert weights across every MoE layer.
    ///
    /// Summed per layer rather than extrapolated from one, because dynamic quantizations
    /// give different layers different precision.
    pub total_expert_bytes: u64,
}

impl MoeShape {
    /// Mean bytes for one expert's three projections.
    ///
    /// An average across layers, so it is a description rather than a basis for
    /// arithmetic. Use [`Self::expert_bytes_per_token`] for traffic.
    pub fn mean_bytes_per_expert(&self) -> u64 {
        let experts = self.experts_per_layer * self.moe_layers as u64;
        if experts == 0 {
            return 0;
        }
        self.total_expert_bytes / experts
    }

    /// Bytes of routed-expert weights read per token, assuming every access misses.
    ///
    /// Scales the whole model's expert bytes by the activation ratio. This is the number a
    /// naive `bandwidth / file_size` model gets wrong by the sparsity factor — roughly 16×
    /// on a 128-expert top-8 model.
    pub fn expert_bytes_per_token(&self) -> u64 {
        (self.total_expert_bytes as f64 * self.activation_ratio()) as u64
    }

    /// Total bytes of routed-expert weights in the whole model.
    pub fn total_expert_bytes(&self) -> u64 {
        self.total_expert_bytes
    }

    /// Fraction of expert weights touched by a single token.
    ///
    /// This is the sparsity that makes the whole approach work: on a 128-expert top-8
    /// model it is 6.25%, so RAM demand tracks *activated* parameters rather than total.
    pub fn activation_ratio(&self) -> f64 {
        if self.experts_per_layer == 0 {
            return 0.0;
        }
        self.experts_per_token as f64 / self.experts_per_layer as f64
    }
}

/// Infer a model's MoE shape from its tensor directory.
///
/// Returns `None` for dense models, which have no stacked expert tensors.
///
/// `experts_per_token` comes from metadata (`{arch}.expert_used_count`) because it is a
/// routing decision, not a property of the stored weights — nothing in the tensor shapes
/// reveals it. Absent that key we assume 8, the near-universal default, and callers that
/// care should check metadata themselves.
pub fn infer_moe_shape(g: &ModelShape) -> Option<MoeShape> {
    let mut moe_layers = 0u32;
    let mut experts_per_layer = 0u64;
    let mut total_expert_bytes = 0u64;

    for layer in 0..u32::MAX {
        let name = format!("blk.{layer}.ffn_gate_exps.weight");
        let Some(t) = g.tensor(&name) else { break };
        moe_layers += 1;

        if experts_per_layer == 0 && t.dims.len() == 3 {
            experts_per_layer = t.dims[2];
        }

        // Sum every layer, never extrapolate from layer 0. Dynamic quantizations —
        // Unsloth's UD- family, and llama.cpp's own Q4_K_M convention — deliberately give
        // different layers different precision. Measuring one layer and multiplying gives
        // a per-token figure that can be *larger* for a smaller file, which is visibly
        // absurd and destroys trust in every number beside it.
        for proj in ["gate", "up", "down"] {
            let n = format!("blk.{layer}.ffn_{proj}_exps.weight");
            if let Some(t) = g.tensor(&n) {
                if let Some(b) = t.size_bytes() {
                    total_expert_bytes += b;
                }
            }
        }
    }

    if moe_layers == 0 || experts_per_layer == 0 || total_expert_bytes == 0 {
        return None;
    }

    let experts_per_token = g.arch_u64("expert_used_count").unwrap_or(8);

    Some(MoeShape {
        moe_layers,
        experts_per_layer,
        experts_per_token,
        total_expert_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_30b_a3b_sparsity_and_traffic() {
        // Real Qwen3-30B-A3B geometry at Q4_K.
        let shape = MoeShape {
            moe_layers: 48,
            experts_per_layer: 128,
            experts_per_token: 8,
            total_expert_bytes: 884_736 * 3 * 128 * 48,
        };

        assert!(
            (shape.activation_ratio() - 0.0625).abs() < 1e-9,
            "8 of 128 is 6.25%"
        );

        let per_token_gb = shape.expert_bytes_per_token() as f64 / 1e9;
        assert!(
            (per_token_gb - 1.019).abs() < 0.01,
            "expected ~1.02 GB/token, got {per_token_gb:.3}"
        );

        let total_gb = shape.total_expert_bytes() as f64 / 1e9;
        assert!(
            (total_gb - 16.3).abs() < 0.5,
            "expected ~16 GB of expert weights, got {total_gb:.1}"
        );
    }

    #[test]
    fn a_token_reads_the_activation_ratio_of_all_expert_bytes() {
        let shape = MoeShape {
            moe_layers: 48,
            experts_per_layer: 128,
            experts_per_token: 8,
            total_expert_bytes: 884_736 * 3 * 128 * 48,
        };
        let ratio = shape.expert_bytes_per_token() as f64 / shape.total_expert_bytes() as f64;
        assert!((ratio - shape.activation_ratio()).abs() < 1e-9);
    }

    #[test]
    fn per_token_traffic_never_exceeds_total_expert_bytes() {
        // Regression. The first implementation measured one layer's expert size and
        // multiplied by the layer count. On a dynamically quantized model — where Unsloth
        // deliberately gives layers different precision — that produced a per-token figure
        // *larger for a smaller file*: UD-IQ1_S (9.04 GB) reported 2.596 GB/token while
        // Q3_K_M (14.71 GB) reported 0.934. Visibly absurd, and it discredits every number
        // printed beside it.
        //
        // Summing per layer makes the invariant structural: a token cannot read more
        // expert bytes than the model contains.
        let shape = MoeShape {
            moe_layers: 48,
            experts_per_layer: 128,
            experts_per_token: 8,
            total_expert_bytes: 17_000_000_000,
        };
        assert!(
            shape.expert_bytes_per_token() < shape.total_expert_bytes(),
            "a token cannot read more than the whole model"
        );
        assert!(shape.expert_bytes_per_token() > 0);
    }

    #[test]
    fn mean_expert_size_is_reported_but_not_used_for_traffic() {
        let shape = MoeShape {
            moe_layers: 2,
            experts_per_layer: 4,
            experts_per_token: 2,
            total_expert_bytes: 800,
        };
        assert_eq!(shape.mean_bytes_per_expert(), 100, "800 over 8 experts");
        // Traffic comes from the activation ratio, not from mean x count x layers.
        assert_eq!(
            shape.expert_bytes_per_token(),
            400,
            "half the experts, so half"
        );
    }

    #[test]
    fn kv_cache_grows_linearly_with_context() {
        // Real Qwen3-30B-A3B attention geometry: 48 layers, 4 KV heads, 128-wide.
        let kv = KvShape {
            layers: 48,
            kv_heads: 4,
            head_dim: 128,
            train_context: Some(262_144),
        };

        // 2 * 48 * 4 * 128 * 4096 * 2 = 402,653,184
        let at_4k = kv.bytes_at(4096, KV_F16_BYTES);
        assert_eq!(at_4k, 402_653_184);
        assert_eq!(kv.bytes_at(8192, KV_F16_BYTES), at_4k * 2);
        assert_eq!(kv.bytes_at(0, KV_F16_BYTES), 0);
    }

    #[test]
    fn kv_cache_at_full_context_is_large_enough_to_change_the_answer() {
        // The reason this exists at all. Qwen3-30B-A3B at Q4_K_M is ~18.6 GB of weights;
        // its full 262k context adds ~25 GB on top. A `fits` verdict that ignores the
        // cache is not optimistic by a rounding error, it is wrong by more than the model.
        let kv = KvShape {
            layers: 48,
            kv_heads: 4,
            head_dim: 128,
            train_context: Some(262_144),
        };
        let full = kv.bytes_at(262_144, KV_F16_BYTES);
        assert!(
            full > 25_000_000_000,
            "expected >25 GB at full context, got {full}"
        );
    }

    #[test]
    fn grouped_query_attention_is_not_mistaken_for_multi_head() {
        // The trap: using `head_count` where `head_count_kv` was meant. Qwen3 has 32 query
        // heads and 4 KV heads, so that mistake overstates the cache 8x — and 8x on a
        // multi-gigabyte figure turns a model that fits into one that does not.
        let gqa = KvShape {
            layers: 48,
            kv_heads: 4,
            head_dim: 128,
            train_context: None,
        };
        let as_if_mha = KvShape {
            kv_heads: 32,
            ..gqa.clone()
        };
        assert_eq!(
            as_if_mha.bytes_at(4096, KV_F16_BYTES),
            gqa.bytes_at(4096, KV_F16_BYTES) * 8
        );
    }

    #[test]
    fn quantized_kv_scales_the_cache_down() {
        let kv = KvShape {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            train_context: None,
        };
        assert_eq!(
            kv.bytes_at(4096, 1),
            kv.bytes_at(4096, KV_F16_BYTES) / 2,
            "8-bit KV halves it"
        );
    }

    #[test]
    fn a_dense_model_has_no_moe_shape() {
        let shape = MoeShape {
            moe_layers: 0,
            experts_per_layer: 0,
            experts_per_token: 0,
            total_expert_bytes: 0,
        };
        assert_eq!(shape.activation_ratio(), 0.0, "must not divide by zero");
    }
}
