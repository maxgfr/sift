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
pub mod safetensors;

pub use experts::{
    expert_ranges, expert_slice, ExpertError, ExpertRanges, ExpertSlice, TokenTraffic,
};
pub use ggml::{BlockLayout, GgmlType};
// Re-exported at the module root so callers reach for one `model::` path.
pub use self::KV_F16_BYTES as KV_DEFAULT_BYTES;
pub use gguf::{Gguf, GgufError, TensorInfo, Value};
pub use remote::{hf_url, RemoteFile};
pub use safetensors::{Safetensors, SafetensorsError};

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
pub struct ModelShape {
    /// Architecture facts, however the source format spelled them.
    pub facts: ArchFacts,
    tensors: Vec<TensorEntry>,
    /// Files this model is stored in. 1 for the ordinary case.
    pub shard_count: u32,
    /// Tensors left out of every total because their dtype could not be sized.
    ///
    /// Nonzero means the size, parameter count and bits per weight all describe less than
    /// the whole file, and callers say so rather than presenting them as complete.
    pub dropped_tensors: u64,
}

/// One tensor, reduced to what analysis needs.
///
/// Deliberately not [`TensorInfo`]: that carries a GGUF dtype and a file-relative offset,
/// neither of which a safetensors file has in the same terms — there, a tensor's size is
/// the difference of its two data offsets and its dtype is a string. Sizes and shapes are
/// the only things the fit arithmetic needs, so they are all this holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    /// Dimensions as the file stores them. GGUF puts the expert count last on a stacked
    /// expert tensor, which is how [`infer_moe_shape`] reads it off.
    pub dims: Vec<u64>,
    pub n_elems: u64,
    pub size_bytes: u64,
}

/// The architecture numbers, normalised across formats.
///
/// GGUF keeps these in its own header under architecture-scoped keys
/// (`qwen3moe.attention.head_count_kv`); a safetensors repo keeps them in a separate
/// `config.json` under HuggingFace's names (`num_key_value_heads`). Normalising here means
/// [`infer_kv_shape`] is written once and is right for both, rather than growing a branch
/// per format in the place where a mistake is most expensive.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchFacts {
    pub architecture: Option<String>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub embedding_length: Option<u64>,
    /// Explicit head width, where the architecture states one rather than implying it.
    pub key_length: Option<u64>,
    pub context_length: Option<u64>,
    /// Experts activated per token, the top-k of the router.
    pub expert_used_count: Option<u64>,
    /// Routed experts per MoE layer, where the metadata states it.
    pub expert_count: Option<u64>,
}

impl ModelShape {
    /// Build from parts, taking the facts from the first.
    pub fn new(facts: ArchFacts, tensors: Vec<TensorEntry>, shard_count: u32) -> Self {
        Self {
            facts,
            tensors,
            shard_count,
            dropped_tensors: 0,
        }
    }

    /// View a single-file GGUF model.
    pub fn single(g: &Gguf) -> Self {
        Self {
            dropped_tensors: unsized_tensors(g),
            ..Self::new(gguf_facts(g), gguf_entries(g), 1)
        }
    }

    /// Merge the parts of a split GGUF model, in shard order.
    ///
    /// Facts come from the first part, because only the first part carries them:
    /// `llama-gguf-split` writes the full metadata into split 0 and little more than the
    /// `split.*` keys into the rest. Returns `None` for an empty slice, since a model with
    /// no parts has no shape.
    pub fn sharded(parts: &[Gguf]) -> Option<Self> {
        let first = parts.first()?;
        Some(Self {
            dropped_tensors: parts.iter().map(unsized_tensors).sum(),
            ..Self::new(
                gguf_facts(first),
                parts.iter().flat_map(gguf_entries).collect(),
                parts.len() as u32,
            )
        })
    }

    /// The model architecture string, e.g. `qwen3moe`.
    pub fn architecture(&self) -> Option<&str> {
        self.facts.architecture.as_deref()
    }

    /// Every tensor, across every shard.
    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }

    /// Look a tensor up by exact name, across every shard.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total payload bytes across every shard.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.size_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Total weight count across every shard.
    pub fn total_parameters(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.n_elems)
            .fold(0, u64::saturating_add)
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

/// Tensors whose size we can compute, as analysis entries.
///
/// A tensor of unknown dtype is dropped rather than counted at zero, so the byte total and
/// the parameter total always cover the same set — otherwise [`ModelShape::bits_per_weight`]
/// would divide a partial numerator by a whole denominator and understate precision.
fn gguf_entries(g: &Gguf) -> Vec<TensorEntry> {
    g.tensors
        .iter()
        .filter_map(|t| {
            t.size_bytes().map(|size_bytes| TensorEntry {
                name: t.name.clone(),
                dims: t.dims.clone(),
                n_elems: t.n_elems(),
                size_bytes,
            })
        })
        .collect()
}

/// Tensors [`gguf_entries`] had to leave out, counted so the omission is visible.
fn unsized_tensors(g: &Gguf) -> u64 {
    g.tensors
        .iter()
        .filter(|t| t.size_bytes().is_none())
        .count() as u64
}

/// Read the architecture-scoped metadata GGUF stores.
fn gguf_facts(g: &Gguf) -> ArchFacts {
    ArchFacts {
        architecture: g.architecture().map(str::to_owned),
        block_count: g.arch_u64("block_count"),
        head_count: g.arch_u64("attention.head_count"),
        head_count_kv: g.arch_u64("attention.head_count_kv"),
        embedding_length: g.arch_u64("embedding_length"),
        key_length: g.arch_u64("attention.key_length"),
        context_length: g.arch_u64("context_length"),
        expert_used_count: g.arch_u64("expert_used_count"),
        expert_count: g.arch_u64("expert_count"),
    }
}

impl Gguf {
    /// This file viewed as a whole model.
    pub fn shape(&self) -> ModelShape {
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
    let f = &g.facts;
    let layers = f.block_count?;
    // Grouped-query attention means K/V heads are usually fewer than query heads. Fall
    // back to `head_count` only when `head_count_kv` is absent, which means multi-head
    // attention where the two are equal by definition.
    let kv_heads = f.head_count_kv.or(f.head_count)?;

    // Some architectures state the head width outright; the rest imply it. Deepseek-style
    // models with a decoupled head dimension state it, and deriving it from the embedding
    // width would be wrong for them.
    let head_dim = match f.key_length {
        Some(d) => d,
        None => {
            let embedding = f.embedding_length?;
            let heads = f.head_count?;
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
        train_context: f.context_length,
    })
}

/// A model's MoE shape, as far as the tensor directory reveals it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeShape {
    /// Number of transformer blocks carrying routed-expert tensors.
    pub moe_layers: u32,
    /// Routed experts per MoE layer.
    pub experts_per_layer: u64,
    /// Experts activated per token per layer (top-k).
    pub experts_per_token: u64,
    /// Whether `experts_per_token` is the assumed default rather than a stated figure.
    ///
    /// Callers print it: the traffic figure scales linearly with top-k, so a guessed 8 on a
    /// top-2 model is a 4× error in every number derived from it.
    pub experts_per_token_assumed: bool,
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
        match self.experts_per_layer.checked_mul(self.moe_layers as u64) {
            Some(experts) if experts > 0 => self.total_expert_bytes / experts,
            _ => 0,
        }
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

/// Top-k assumed when a model does not state it.
///
/// The commonest value among published MoE models, and always reported as an assumption.
pub const ASSUMED_EXPERTS_PER_TOKEN: u64 = 8;

/// Infer a model's MoE shape from its tensor directory.
///
/// Returns `None` for dense models, which have no routed-expert tensors.
///
/// Every tensor is classified on its own name rather than by probing layer numbers from
/// zero. Probing stopped at the first layer without experts, which is layer 0 on DeepSeek
/// and Kimi (dense leading layers) and layer 1 on Llama 4 (MoE every other layer) — and
/// reported each of them as dense.
///
/// `experts_per_token` comes from metadata because it is a routing decision, not a property
/// of the stored weights. Absent it, [`ASSUMED_EXPERTS_PER_TOKEN`] is used and
/// [`MoeShape::experts_per_token_assumed`] says so.
pub fn infer_moe_shape(g: &ModelShape) -> Option<MoeShape> {
    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Default)]
    struct Layer {
        /// Distinct expert indices seen in per-expert tensors.
        ids: BTreeSet<u64>,
        /// Largest expert count read off a stacked tensor's expert dimension.
        stacked: u64,
    }

    let mut layers: BTreeMap<u64, Layer> = BTreeMap::new();
    let mut total_expert_bytes = 0u64;
    for t in g.tensors() {
        let Some((layer, which)) = classify_expert_tensor(&t.name, &t.dims) else {
            continue;
        };
        let entry = layers.entry(layer).or_default();
        match which {
            ExpertIndex::One(j) => {
                entry.ids.insert(j);
            }
            ExpertIndex::Stacked(n) => entry.stacked = entry.stacked.max(n),
        }
        // Sum every layer, never extrapolate from one. Dynamic quantizations deliberately
        // give different layers different precision, and multiplying one layer out gave a
        // per-token figure that was *larger* for a smaller file.
        total_expert_bytes = total_expert_bytes.saturating_add(t.size_bytes);
    }

    // The widest layer, not the first: a shard boundary can leave the lowest-numbered layer
    // holding only some of its experts.
    let seen = layers
        .values()
        .map(|l| (l.ids.len() as u64).max(l.stacked))
        .max()
        .unwrap_or(0);
    let experts_per_layer = g.facts.expert_count.filter(|&n| n > 0).unwrap_or(seen);
    let moe_layers = layers.len() as u32;

    if moe_layers == 0 || experts_per_layer == 0 || total_expert_bytes == 0 {
        return None;
    }

    let (experts_per_token, experts_per_token_assumed) =
        match g.facts.expert_used_count.filter(|&k| k > 0) {
            Some(k) => (k, false),
            None => (ASSUMED_EXPERTS_PER_TOKEN, true),
        };

    Some(MoeShape {
        moe_layers,
        experts_per_layer,
        // A top-k above the expert count is a metadata error; a token cannot read more
        // experts than exist.
        experts_per_token: experts_per_token.min(experts_per_layer),
        experts_per_token_assumed,
        total_expert_bytes,
    })
}

/// How an expert tensor identifies its expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertIndex {
    /// One expert per tensor, numbered.
    One(u64),
    /// Every expert of the layer stacked into one tensor, this many of them.
    Stacked(u64),
}

/// Recognise a routed-expert weight tensor, in any of the layouts publishers ship.
///
/// ```text
/// blk.3.ffn_gate_exps.weight                         GGUF, stacked; experts are dims[2]
/// blk.3.ffn_gate_up_exps.weight                      GGUF, gate and up fused
/// blk.3.ffn_gate.5.weight                            GGUF, one tensor per expert (old Mixtral)
/// model.layers.3.mlp.experts.5.gate_proj.weight      HF, one tensor per expert
/// model.layers.3.block_sparse_moe.experts.5.w1.weight  Mixtral, Phi-MoE
/// model.layers.3.mlp.experts.gate_up_proj            HF fused 3-D; experts are dims[0]
/// model.layers.3.mlp.experts.down_proj_blocks        gpt-oss MXFP4 blocks and scales
/// model.layers.3.mlp.switch_mlp.gate_proj.weight     MLX stacked; experts are dims[0]
/// ```
///
/// Quantization side tensors (`qweight`, `scales`, `qzeros`, MLX `biases`, FP8
/// `weight_scale_inv`) are counted too: they are read with the weights on every access,
/// and leaving them out understates a quantized model's expert traffic. Routers, shared
/// experts and plain biases are not routed-expert weights and never match.
fn classify_expert_tensor(name: &str, dims: &[u64]) -> Option<(u64, ExpertIndex)> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let (layer, rest) = rest.split_once('.')?;
        let layer = layer.parse().ok()?;
        // Stacked: `ffn_{gate,up,down,gate_up}_exps.weight`, expert count innermost-last.
        if let Some(proj) = rest
            .strip_prefix("ffn_")
            .and_then(|r| r.strip_suffix("_exps.weight"))
        {
            if matches!(proj, "gate" | "up" | "down" | "gate_up") && dims.len() == 3 {
                return Some((layer, ExpertIndex::Stacked(dims[2])));
            }
            return None;
        }
        // Per expert: `ffn_{gate,up,down}.{j}.weight`.
        let rest = rest.strip_prefix("ffn_")?;
        let (proj, rest) = rest.split_once('.')?;
        let (j, tail) = rest.split_once('.')?;
        if matches!(proj, "gate" | "up" | "down") && tail == "weight" {
            return Some((layer, ExpertIndex::One(j.parse().ok()?)));
        }
        return None;
    }

    // HuggingFace names put the layer after `layers.`, under whatever prefix a multimodal
    // wrapper adds (`model.`, `model.language_model.`, `language_model.model.`).
    let at = name.find("layers.")?;
    let rest = &name[at + "layers.".len()..];
    let (layer, rest) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;

    let (block, rest) = rest.split_once('.')?;
    if !matches!(block, "mlp" | "block_sparse_moe" | "feed_forward") {
        return None;
    }
    let side = |s: &str| {
        matches!(
            s,
            "weight" | "qweight" | "scales" | "qzeros" | "biases" | "weight_scale_inv"
        )
    };
    let proj = |s: &str| {
        matches!(
            s,
            "gate_proj" | "up_proj" | "down_proj" | "gate_up_proj" | "w1" | "w2" | "w3"
        )
    };

    if let Some(rest) = rest.strip_prefix("switch_mlp.") {
        let (p, tail) = rest.split_once('.')?;
        return (proj(p) && side(tail) && dims.len() == 3)
            .then(|| (layer, ExpertIndex::Stacked(dims[0])));
    }
    let rest = rest.strip_prefix("experts.")?;
    match rest.split_once('.') {
        // Per expert: `experts.{j}.{proj}.{suffix}`.
        Some((j, tail)) if j.bytes().all(|b| b.is_ascii_digit()) => {
            let (p, suffix) = tail.split_once('.')?;
            let j = j.parse().ok()?;
            (proj(p) && side(suffix)).then_some((layer, ExpertIndex::One(j)))
        }
        // Fused 3-D: `experts.{proj}` or gpt-oss `experts.{proj}_{blocks,scales}`, with the
        // expert count outermost. A `_bias` tensor is a bias, not a weight.
        Some(_) => None,
        None => {
            let p = rest
                .strip_suffix("_blocks")
                .or_else(|| rest.strip_suffix("_scales"))
                .unwrap_or(rest);
            (proj(p) && dims.len() >= 2).then(|| (layer, ExpertIndex::Stacked(dims[0])))
        }
    }
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
            experts_per_token_assumed: false,
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
            experts_per_token_assumed: false,
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
            experts_per_token_assumed: false,
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
            experts_per_token_assumed: false,
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

    fn entry(name: &str, dims: &[u64], size_bytes: u64) -> TensorEntry {
        TensorEntry {
            name: name.into(),
            dims: dims.to_vec(),
            n_elems: dims.iter().product(),
            size_bytes,
        }
    }

    fn shape_of(tensors: Vec<TensorEntry>, facts: ArchFacts) -> ModelShape {
        ModelShape::new(facts, tensors, 1)
    }

    #[test]
    fn moe_deepseek_style_dense_leading_layer_is_not_mistaken_for_a_dense_model() {
        // Regression: probing from `blk.0` stopped at the first layer without experts,
        // which on DeepSeek and Kimi is layer 0 — and reported the whole model as dense.
        let mut t = vec![
            entry("blk.0.ffn_gate.weight", &[2048, 10944], 100),
            entry("blk.0.ffn_up.weight", &[2048, 10944], 100),
        ];
        for l in 1..4 {
            for p in ["gate", "up", "down"] {
                t.push(entry(
                    &format!("blk.{l}.ffn_{p}_exps.weight"),
                    &[2048, 1408, 64],
                    1000,
                ));
            }
            t.push(entry(
                &format!("blk.{l}.ffn_gate_shexp.weight"),
                &[2048, 2816],
                50,
            ));
        }
        let facts = ArchFacts {
            expert_count: Some(64),
            expert_used_count: Some(6),
            ..Default::default()
        };
        let moe = infer_moe_shape(&shape_of(t, facts)).expect("moe");
        assert_eq!(moe.moe_layers, 3);
        assert_eq!(moe.experts_per_layer, 64);
        assert_eq!(moe.experts_per_token, 6);
        assert!(!moe.experts_per_token_assumed);
        assert_eq!(
            moe.total_expert_bytes,
            9 * 1000,
            "shared experts are trunk, not routed"
        );
    }

    #[test]
    fn moe_interleaved_layers_are_all_counted() {
        // Llama 4 puts experts on every other layer.
        let mut t = Vec::new();
        for l in [0, 2, 4] {
            for p in ["gate", "up", "down"] {
                t.push(entry(
                    &format!("blk.{l}.ffn_{p}_exps.weight"),
                    &[8, 8, 16],
                    10,
                ));
            }
        }
        for l in [1, 3] {
            t.push(entry(&format!("blk.{l}.ffn_gate.weight"), &[8, 8], 10));
        }
        let moe = infer_moe_shape(&shape_of(t, ArchFacts::default())).expect("moe");
        assert_eq!(moe.moe_layers, 3);
        assert_eq!(
            moe.experts_per_layer, 16,
            "read off dims[2] when metadata is silent"
        );
        assert_eq!(moe.total_expert_bytes, 90);
    }

    #[test]
    fn moe_fused_gate_up_stacked_tensor_is_recognised() {
        let t = vec![
            entry("blk.0.ffn_gate_up_exps.weight", &[64, 256, 32], 200),
            entry("blk.0.ffn_down_exps.weight", &[128, 64, 32], 100),
        ];
        let moe = infer_moe_shape(&shape_of(t, ArchFacts::default())).expect("moe");
        assert_eq!(moe.experts_per_layer, 32);
        assert_eq!(moe.total_expert_bytes, 300);
    }

    #[test]
    fn moe_mixtral_hf_layout_is_recognised_and_its_router_is_not_counted() {
        let mut t = Vec::new();
        for l in 0..2 {
            t.push(entry(
                &format!("model.layers.{l}.block_sparse_moe.gate.weight"),
                &[8, 64],
                7,
            ));
            for j in 0..8 {
                for w in ["w1", "w2", "w3"] {
                    t.push(entry(
                        &format!("model.layers.{l}.block_sparse_moe.experts.{j}.{w}.weight"),
                        &[64, 224],
                        10,
                    ));
                }
            }
        }
        let facts = ArchFacts {
            expert_count: Some(8),
            expert_used_count: Some(2),
            ..Default::default()
        };
        let moe = infer_moe_shape(&shape_of(t, facts)).expect("moe");
        assert_eq!((moe.moe_layers, moe.experts_per_layer), (2, 8));
        assert_eq!(moe.experts_per_token, 2);
        assert_eq!(moe.total_expert_bytes, 2 * 8 * 3 * 10, "router excluded");
    }

    #[test]
    fn moe_old_per_expert_gguf_layout_is_recognised() {
        // TheBloke's Mixtral GGUFs predate stacked experts: one tensor per expert.
        let mut t = Vec::new();
        for j in 0..8 {
            for p in ["gate", "up", "down"] {
                t.push(entry(&format!("blk.0.ffn_{p}.{j}.weight"), &[64, 224], 10));
            }
        }
        t.push(entry("blk.0.ffn_gate_inp.weight", &[64, 8], 5));
        let moe = infer_moe_shape(&shape_of(t, ArchFacts::default())).expect("moe");
        assert_eq!(moe.experts_per_layer, 8);
        assert_eq!(moe.total_expert_bytes, 240);
    }

    #[test]
    fn moe_gpt_oss_fused_mxfp4_tensors_are_recognised_without_their_biases() {
        let mut t = Vec::new();
        for l in 0..2 {
            let pre = format!("model.layers.{l}.mlp.experts");
            t.push(entry(
                &format!("{pre}.gate_up_proj_blocks"),
                &[32, 5760, 90, 16],
                1000,
            ));
            t.push(entry(
                &format!("{pre}.gate_up_proj_scales"),
                &[32, 5760, 90],
                60,
            ));
            t.push(entry(&format!("{pre}.gate_up_proj_bias"), &[32, 5760], 3));
            t.push(entry(
                &format!("{pre}.down_proj_blocks"),
                &[32, 2880, 90, 16],
                500,
            ));
            t.push(entry(
                &format!("{pre}.down_proj_scales"),
                &[32, 2880, 90],
                30,
            ));
            t.push(entry(
                &format!("model.layers.{l}.mlp.router.weight"),
                &[32, 2880],
                9,
            ));
        }
        let facts = ArchFacts {
            expert_used_count: Some(4),
            ..Default::default()
        };
        let moe = infer_moe_shape(&shape_of(t, facts)).expect("moe");
        assert_eq!((moe.moe_layers, moe.experts_per_layer), (2, 32));
        assert_eq!(moe.total_expert_bytes, 2 * (1000 + 60 + 500 + 30));
    }

    #[test]
    fn moe_mlx_switch_mlp_with_quantization_side_tensors_is_recognised() {
        let mut t = Vec::new();
        for p in ["gate_proj", "up_proj", "down_proj"] {
            for s in ["weight", "scales", "biases"] {
                t.push(entry(
                    &format!("model.layers.0.mlp.switch_mlp.{p}.{s}"),
                    &[128, 768, 256],
                    10,
                ));
            }
        }
        let moe = infer_moe_shape(&shape_of(t, ArchFacts::default())).expect("moe");
        assert_eq!(moe.experts_per_layer, 128);
        assert_eq!(moe.total_expert_bytes, 90);
    }

    #[test]
    fn moe_an_unstated_top_k_is_assumed_flagged_and_clamped_to_the_expert_count() {
        let t = vec![entry("blk.0.ffn_gate_exps.weight", &[8, 8, 4], 10)];
        let moe = infer_moe_shape(&shape_of(t, ArchFacts::default())).expect("moe");
        assert!(moe.experts_per_token_assumed);
        assert_eq!(
            moe.experts_per_token, 4,
            "8 assumed, but only 4 experts exist"
        );
        assert!(moe.activation_ratio() <= 1.0);
    }

    #[test]
    fn moe_a_dense_model_has_none() {
        let t = vec![
            entry("blk.0.ffn_gate.weight", &[8, 8], 10),
            entry("model.layers.0.mlp.gate_proj.weight", &[8, 8], 10),
            entry(
                "model.layers.0.mlp.shared_experts.gate_proj.weight",
                &[8, 8],
                10,
            ),
        ];
        assert!(infer_moe_shape(&shape_of(t, ArchFacts::default())).is_none());
    }

    #[test]
    fn a_dense_model_has_no_moe_shape() {
        let shape = MoeShape {
            moe_layers: 0,
            experts_per_layer: 0,
            experts_per_token: 0,
            experts_per_token_assumed: false,
            total_expert_bytes: 0,
        };
        assert_eq!(shape.activation_ratio(), 0.0, "must not divide by zero");
    }
}
