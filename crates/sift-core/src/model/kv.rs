//! What a model's attention layers cost to keep a context in memory.
//!
//! Weights are not the whole footprint, and on long contexts they are not even the larger
//! part: a 30B model at 4 bits is ~18 GB of weights, and 128k tokens of KV cache on a
//! 48-layer model is another 12 GB. Judging "does it fit" on the file size alone answers a
//! question nobody asked.
//!
//! # Why per layer
//!
//! One formula — `2 × layers × kv_heads × head_dim × context` — was right for every model
//! until models stopped giving every layer the same attention. Three designs now break it,
//! all in the direction of overstating the cache, which makes a model that fits look like
//! one that does not:
//!
//! - **Sliding-window layers** keep only the last `window` tokens. Gemma 3 has one global
//!   layer in six; the other five hold 1024 tokens however long the context.
//! - **Linear-attention and recurrent layers** keep a fixed-size state and no KV at all.
//!   Qwen3.5 has one attention layer in four.
//! - **Multi-head latent attention** (DeepSeek V2/V3, Kimi) caches one compressed latent
//!   per token instead of a K and a V per head — 576 values where the naive formula
//!   counts 128 heads × 320.
//!
//! So the cache is described layer by layer, and the summary fields `inspect` has always
//! printed are kept alongside for readers that only want the headline.

use super::ArchFacts;

/// How one transformer block attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Attends over the whole context, so its cache grows with it.
    Full,
    /// Attends over the last `sliding_window` tokens only.
    Sliding,
    /// Linear attention or a recurrent block: fixed-size state, no KV cache.
    Linear,
}

/// The KV cache one layer holds per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayer {
    /// Key/value heads. Fewer than query heads under grouped-query attention.
    pub kv_heads: u64,
    /// Width of one key head.
    pub k_dim: u64,
    /// Width of one value head. Zero under MLA, where V is read out of the cached latent.
    pub v_dim: u64,
    /// Tokens this layer keeps, when it keeps fewer than the whole context.
    pub window: Option<u64>,
}

/// Tokens a sliding-window cache holds beyond its window: llama.cpp's default `n_ubatch`,
/// one batch of new tokens arriving while the window is still full.
pub const SWA_BATCH_TOKENS: u64 = 512;

impl KvLayer {
    /// Bytes this layer holds at `context_tokens`.
    ///
    /// A sliding layer is sized as llama.cpp sizes it, since every GGUF engine runs on it:
    /// the window plus one batch of [`SWA_BATCH_TOKENS`], padded to 256 cells, never more
    /// than the context (`llama-kv-cache-iswa.cpp`). Counting the window alone put
    /// gemma-3-1b's cache at 27 MiB where llama.cpp allocates 38.
    pub fn bytes_at(&self, context_tokens: u64, bytes_per_element: u64) -> u64 {
        let tokens = match self.window {
            Some(w) => {
                let cells = context_tokens.min(w.saturating_add(SWA_BATCH_TOKENS));
                context_tokens.min(cells.div_ceil(256).saturating_mul(256))
            }
            None => context_tokens,
        };
        self.kv_heads
            .saturating_mul(self.k_dim.saturating_add(self.v_dim))
            .saturating_mul(tokens)
            .saturating_mul(bytes_per_element)
    }
}

/// A model's KV cache, layer by layer, with the headline figures alongside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvShape {
    /// Transformer blocks that hold a KV cache: full plus sliding.
    pub layers: u64,
    /// Key/value heads of a full-attention layer. Fewer than query heads under
    /// grouped-query attention, which is most modern models and is exactly why assuming
    /// `head_count` here overstates the cache by the GQA ratio — 8x on some models.
    pub kv_heads: u64,
    /// Width of one head of a full-attention layer.
    pub head_dim: u64,
    /// Context the model was trained for, when it says so.
    pub train_context: Option<u64>,
    /// Layers attending over the whole context.
    pub full_layers: u64,
    /// Layers attending over a sliding window.
    pub sliding_layers: u64,
    /// Linear-attention or recurrent layers, which hold no KV cache.
    pub recurrent_layers: u64,
    /// The sliding window, when any layer has one.
    pub window: Option<u64>,
    /// Every KV-holding layer, in order.
    pub per_layer: Vec<KvLayer>,
    /// Bytes of state the recurrent layers keep whatever the context, when the metadata
    /// sizes it. Zero when it does not, which understates by tens of MiB at most.
    pub state_bytes: u64,
}

/// More layers than this is not a model but a corrupt or hostile header.
///
/// The largest published models have under 200. Sizing is per layer, so an unchecked
/// `block_count` of 2^40 would allocate a vector that large and abort the process.
pub const MAX_LAYERS: u64 = 1 << 16;

/// Bytes per KV element at the precision engines default to.
///
/// f16. Engines can be told to use 8- or 4-bit KV, which is why
/// [`KvShape::bytes_at`] takes this as an argument rather than baking it in — but the
/// default is what a user who has not gone looking will get.
pub const KV_F16_BYTES: u64 = 2;

impl KvShape {
    /// A model whose every layer is full attention with the same geometry.
    pub fn uniform(layers: u64, kv_heads: u64, head_dim: u64, train_context: Option<u64>) -> Self {
        let layer = KvLayer {
            kv_heads,
            k_dim: head_dim,
            v_dim: head_dim,
            window: None,
        };
        Self {
            layers,
            kv_heads,
            head_dim,
            train_context,
            full_layers: layers,
            sliding_layers: 0,
            recurrent_layers: 0,
            window: None,
            per_layer: vec![layer; layers as usize],
            state_bytes: 0,
        }
    }

    /// Bytes of context memory held at `context_tokens`: the KV cache plus the fixed
    /// recurrent state.
    ///
    /// Summed per layer: `kv_heads × (k_dim + v_dim) × min(context, window)` each. The
    /// recurrent state is f32 in llama.cpp whatever the KV precision, so it is added as is.
    pub fn bytes_at(&self, context_tokens: u64, bytes_per_element: u64) -> u64 {
        self.per_layer
            .iter()
            .map(|l| l.bytes_at(context_tokens, bytes_per_element))
            .fold(self.state_bytes, u64::saturating_add)
    }

    /// Whether the cache is anything other than every layer attending in full.
    pub fn is_mixed(&self) -> bool {
        self.sliding_layers > 0 || self.recurrent_layers > 0
    }
}

/// Infer the KV cache shape from architecture facts. See [`super::infer_kv_shape`].
///
/// Returns `None` when a factor is missing, because a partial answer here would be a
/// silently low one — and a low KV estimate makes a model look like it fits when it does
/// not, which is the failure mode with the highest cost to the user.
///
/// Where the metadata is silent about a layer's kind, the layer is taken to be full
/// attention. That overstates the cache of a model whose pattern went unrecorded, which is
/// the safe direction to be wrong in.
pub fn kv_shape_from_facts(facts: &ArchFacts) -> Option<KvShape> {
    let f = facts;
    let n = f.block_count.filter(|&n| n > 0 && n <= MAX_LAYERS)?;

    // Grouped-query attention means K/V heads are usually fewer than query heads. Fall
    // back to `head_count` only when `head_count_kv` is absent, which means multi-head
    // attention where the two are equal by definition.
    let default_heads = f.head_count_kv.or(f.head_count);

    // Some architectures state the head width outright; the rest imply it. DeepSeek-style
    // models with a decoupled head dimension state it, and deriving it from the embedding
    // width would be wrong for them.
    let k_full = match f.key_length {
        Some(d) => Some(d),
        None => match (f.embedding_length, f.head_count) {
            (Some(e), Some(h)) if h > 0 => Some(e / h),
            _ => None,
        },
    };
    let v_full = f.value_length.or(k_full);

    let mut per_layer = Vec::with_capacity(n as usize);
    let (mut full, mut sliding, mut recurrent) = (0u64, 0u64, 0u64);
    for i in 0..n as usize {
        let kind = f
            .layer_kinds
            .as_ref()
            .and_then(|k| k.get(i).copied())
            .unwrap_or(LayerKind::Full);
        let heads = f
            .head_count_kv_layers
            .as_ref()
            .and_then(|h| h.get(i).copied())
            .or(default_heads);
        // A per-layer head count of zero is how GGUF marks a layer with no attention.
        if kind == LayerKind::Linear || heads == Some(0) {
            recurrent += 1;
            continue;
        }
        // A sliding layer with no stated window is sized as full: the honest fallback for a
        // number we do not have is the larger one.
        let window = match kind {
            LayerKind::Sliding => f.sliding_window.filter(|&w| w > 0),
            _ => None,
        };
        let layer = match f.mla_latent {
            // MLA caches one latent per token, shared by every head, and reads V out of
            // it — so one "head" as wide as the latent, and no separate V.
            Some(latent) => KvLayer {
                kv_heads: 1,
                k_dim: latent,
                v_dim: 0,
                window,
            },
            None => {
                let k = match window {
                    Some(_) => f.key_length_swa.or(k_full),
                    None => k_full,
                }?;
                let v = match window {
                    Some(_) => f.value_length_swa.or(v_full),
                    None => v_full,
                }
                .unwrap_or(k);
                KvLayer {
                    kv_heads: heads?,
                    k_dim: k,
                    v_dim: v,
                    window,
                }
            }
        };
        if layer.kv_heads == 0 || layer.k_dim == 0 {
            return None;
        }
        match window {
            Some(_) => sliding += 1,
            None => full += 1,
        }
        per_layer.push(layer);
    }

    // The headline figures describe a full-attention layer when there is one, since that
    // is the layer whose cost grows with the context.
    let headline = per_layer
        .iter()
        .find(|l| l.window.is_none())
        .or(per_layer.first())
        .copied();
    Some(KvShape {
        layers: full + sliding,
        kv_heads: headline.map_or(0, |l| l.kv_heads),
        head_dim: headline.map_or(0, |l| l.k_dim),
        train_context: f.context_length,
        full_layers: full,
        sliding_layers: sliding,
        recurrent_layers: recurrent,
        window: per_layer.iter().find_map(|l| l.window),
        per_layer,
        state_bytes: f
            .recurrent_state_per_layer
            .map_or(0, |s| s.saturating_mul(recurrent)),
    })
}

/// Expand the attention pattern a model states into one kind per layer.
///
/// Shared by both format readers so GGUF and `config.json` agree on what a pattern means.
/// `explicit` wins outright; otherwise `full_attention_interval` marks every Nth layer
/// full and the rest linear (Qwen3-Next, Qwen3.5), and a sliding pattern marks layers
/// sliding by llama.cpp's rule — every Nth layer global, the last of each group unless
/// `dense_first`, in which case the first.
pub fn resolve_layer_kinds(
    n_layers: u64,
    explicit: Option<Vec<LayerKind>>,
    full_attention_interval: Option<u64>,
    sliding_pattern: Option<(u64, bool)>,
) -> Option<Vec<LayerKind>> {
    if let Some(kinds) = explicit {
        return Some(kinds);
    }
    if n_layers > MAX_LAYERS {
        return None;
    }
    let n = n_layers as usize;
    if let Some(every) = full_attention_interval.filter(|&k| k > 0) {
        return Some(
            (0..n as u64)
                .map(|i| {
                    if (i + 1) % every == 0 {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect(),
        );
    }
    let (every, dense_first) = sliding_pattern.filter(|&(k, _)| k > 0)?;
    Some(
        (0..n as u64)
            .map(|i| {
                let global = if dense_first {
                    i % every == 0
                } else {
                    i % every == every - 1
                };
                if global {
                    LayerKind::Full
                } else {
                    LayerKind::Sliding
                }
            })
            .collect(),
    )
}

/// The sliding-window pattern an architecture uses when its metadata does not say.
///
/// `(every, dense_first)`, copied from the `load_swa_pattern` calls in llama.cpp's
/// `src/models/*.cpp`, with the HuggingFace `model_type` spellings of the same families.
/// An architecture absent here gets no sliding window, because llama.cpp gives it none:
/// it caches every layer in full whatever `sliding_window` says.
pub fn default_sliding_pattern(arch: &str) -> Option<(u64, bool)> {
    Some(match arch {
        "gemma2" => (2, false),
        "gemma3" | "gemma3_text" => (6, false),
        "gemma3n" | "gemma3n_text" => (5, false),
        "gpt-oss" | "gpt_oss" => (2, false),
        "cohere2" | "olmo2" | "exaone4" | "afmoe" | "llama4" | "llama4_text" => (4, false),
        "smallthinker" | "laguna" => (4, true),
        _ => return None,
    })
}

/// The sliding window an architecture uses when its metadata does not state one.
///
/// Gemma 2 predates the key and llama.cpp hardcodes 4096; Llama 4's chunked attention is
/// 8192 unless the file says otherwise.
pub fn default_sliding_window(arch: &str) -> Option<u64> {
    match arch {
        "gemma2" => Some(4096),
        "llama4" => Some(8192),
        _ => None,
    }
}

/// Whether an architecture has no attention at all: every layer recurrent, no KV cache.
///
/// Mamba and RWKV state their layer count and nothing about heads, which read as "the
/// cache cannot be sized" when the cache is zero. Spelled as llama.cpp names them in GGUF
/// and as transformers names them in `model_type`. Hybrids (Jamba, Granite 4, Nemotron-H)
/// are not here: their attention layers do hold a cache, stated per layer.
pub fn attention_free(arch: &str) -> bool {
    matches!(
        arch,
        "mamba"
            | "mamba2"
            | "falcon_mamba"
            | "rwkv"
            | "rwkv5"
            | "rwkv6"
            | "rwkv7"
            | "arwkv7"
            | "rwkv6qwen2"
    )
}

/// Every Nth layer full attention, the rest linear, for families that default it.
///
/// llama.cpp's `qwen35`, `qwen35moe` and `qwen3next` loaders all fall back to 4.
pub fn default_full_attention_interval(arch: &str) -> Option<u64> {
    matches!(arch, "qwen35" | "qwen35moe" | "qwen3next").then_some(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_grows_linearly_with_context() {
        // Real Qwen3-30B-A3B attention geometry: 48 layers, 4 KV heads, 128-wide.
        let kv = KvShape::uniform(48, 4, 128, Some(262_144));

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
        let kv = KvShape::uniform(48, 4, 128, Some(262_144));
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
        let gqa = KvShape::uniform(48, 4, 128, None);
        let as_if_mha = KvShape::uniform(48, 32, 128, None);
        assert_eq!(
            as_if_mha.bytes_at(4096, KV_F16_BYTES),
            gqa.bytes_at(4096, KV_F16_BYTES) * 8
        );
    }

    #[test]
    fn quantized_kv_scales_the_cache_down() {
        let kv = KvShape::uniform(32, 8, 128, None);
        assert_eq!(
            kv.bytes_at(4096, 1),
            kv.bytes_at(4096, KV_F16_BYTES) / 2,
            "8-bit KV halves it"
        );
    }

    fn facts(n: u64, heads_kv: u64, head_dim: u64) -> ArchFacts {
        ArchFacts {
            block_count: Some(n),
            head_count_kv: Some(heads_kv),
            key_length: Some(head_dim),
            value_length: Some(head_dim),
            ..Default::default()
        }
    }

    #[test]
    fn kv_gemma3_27b_sliding_window_at_128k() {
        // Gemma-3-27B: 62 layers, 16 KV heads, head_dim 128, window 1024, one global layer
        // in six (layers 5, 11, … 59: ten of them).
        let mut f = facts(62, 16, 128);
        f.sliding_window = Some(1024);
        f.layer_kinds = resolve_layer_kinds(62, None, None, default_sliding_pattern("gemma3"));
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!((kv.full_layers, kv.sliding_layers), (10, 52));
        assert_eq!(kv.window, Some(1024));

        let per_token_layer = 16 * (128 + 128) * 2; // heads x (k + v) x f16
                                                    // Sliding layers hold window + 512 cells, as llama.cpp allocates them.
        let expected = 10 * per_token_layer * 131_072 + 52 * per_token_layer * (1024 + 512);
        assert_eq!(kv.bytes_at(131_072, KV_F16_BYTES), expected);
        // The uniform formula would have charged every layer the whole context: 5x more.
        let naive = KvShape::uniform(62, 16, 128, None).bytes_at(131_072, KV_F16_BYTES);
        assert!(naive > expected * 5);
    }

    #[test]
    fn kv_sliding_layers_hold_what_llama_cpp_allocates_for_them() {
        // Measured: llama.cpp allocates 38.00 MiB of KV for gemma-3-1b-it at 4096 tokens —
        // 16 MiB for its 4 global layers and 22 MiB for its 22 sliding ones, whose cache is
        // `window + n_ubatch` = 512 + 512 = 1024 cells, not the 512-token window.
        let mut f = facts(26, 1, 256);
        f.sliding_window = Some(512);
        f.layer_kinds = resolve_layer_kinds(26, None, None, default_sliding_pattern("gemma3"));
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!((kv.full_layers, kv.sliding_layers), (4, 22));
        assert_eq!(kv.bytes_at(4096, KV_F16_BYTES), 38 << 20);
        // A context shorter than window + batch is all the cache can hold.
        assert_eq!(kv.bytes_at(256, KV_F16_BYTES), 26 * 512 * 256 * 2);
    }

    #[test]
    fn kv_recurrent_layers_add_their_fixed_state_as_llama_cpp_allocates_it() {
        // Measured: granite-4.0-h-350m at 4096 tokens holds 16.00 MiB of KV in its 4
        // attention layers and 21.57 MiB of recurrent state in its 28 Mamba-2 layers
        // (S 21.00 + R 0.57). The state does not grow with the context, and it is there.
        let mut f = facts(32, 4, 64);
        let mut heads = vec![0u64; 32];
        for i in [10, 13, 17, 27] {
            heads[i] = 4;
        }
        f.head_count_kv_layers = Some(heads);
        let per_layer = 128 * 1536 * 4 + 3 * (1536 + 2 * 128) * 4;
        f.recurrent_state_per_layer = Some(per_layer);
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!(kv.state_bytes, 28 * per_layer);
        assert_eq!(kv.bytes_at(4096, KV_F16_BYTES), (16 << 20) + 28 * per_layer);
        assert_eq!(kv.bytes_at(8192, KV_F16_BYTES), (32 << 20) + 28 * per_layer);
        assert!((kv.state_bytes as f64 / 1_048_576.0 - 21.57).abs() < 0.01);
    }

    #[test]
    fn kv_qwen35_hybrid_counts_only_its_attention_layers() {
        // Qwen3.5-9B: 32 layers, full attention every 4th, 4 KV heads, k = v = 256.
        let mut f = facts(32, 4, 256);
        f.layer_kinds =
            resolve_layer_kinds(32, None, default_full_attention_interval("qwen35"), None);
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!((kv.full_layers, kv.recurrent_layers), (8, 24));
        assert_eq!(
            kv.bytes_at(4096, KV_F16_BYTES),
            8 * 4 * (256 + 256) * 4096 * 2
        );
    }

    #[test]
    fn kv_mla_caches_one_latent_per_token() {
        // DeepSeek-V3: 61 layers, kv_lora_rank 512 + qk_rope_head_dim 64. llama.cpp keeps
        // only the 576-wide latent and no V cache: 61 x 576 x 32768 x 2 ≈ 2.30 GB, where
        // the per-head formula (128 heads x (192 + 128)) claimed ~164 GB.
        let f = ArchFacts {
            block_count: Some(61),
            head_count: Some(128),
            head_count_kv: Some(128),
            mla_latent: Some(512 + 64),
            ..Default::default()
        };
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        let bytes = kv.bytes_at(32_768, KV_F16_BYTES);
        assert_eq!(bytes, 61 * 576 * 32_768 * 2);
        assert!((bytes as f64 / 1e9 - 2.30).abs() < 0.01);
    }

    #[test]
    fn kv_per_layer_head_counts_are_honoured_and_zero_means_no_attention() {
        // GGUF stores `head_count_kv` as an array for models whose layers differ.
        let mut f = facts(4, 8, 64);
        f.head_count_kv_layers = Some(vec![8, 0, 4, 0]);
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!((kv.full_layers, kv.recurrent_layers), (2, 2));
        assert_eq!(kv.bytes_at(100, 1), (8 + 4) * (64 + 64) * 100);
    }

    #[test]
    fn kv_sliding_layers_may_use_their_own_head_width() {
        let mut f = facts(2, 2, 512);
        f.sliding_window = Some(10);
        f.key_length_swa = Some(256);
        f.value_length_swa = Some(256);
        f.layer_kinds = Some(vec![LayerKind::Sliding, LayerKind::Full]);
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        // At 100 tokens the sliding layer's window + 512 cells exceed the context, so it
        // holds all 100 tokens, at its own narrower width.
        assert_eq!(kv.bytes_at(100, 1), 2 * 512 * 100 + 2 * 1024 * 100);
        // Past window + batch it stops growing.
        assert_eq!(
            kv.bytes_at(10_000, 1),
            2 * 512 * (10u64 + 512).div_ceil(256) * 256 + 2 * 1024 * 10_000
        );
    }

    #[test]
    fn kv_a_sliding_layer_without_a_window_is_sized_as_full() {
        let mut f = facts(2, 1, 8);
        f.layer_kinds = Some(vec![LayerKind::Sliding, LayerKind::Sliding]);
        let kv = kv_shape_from_facts(&f).expect("sizeable");
        assert_eq!(kv.full_layers, 2, "no window stated, so no window assumed");
    }

    #[test]
    fn kv_missing_geometry_is_none_rather_than_zero() {
        assert!(kv_shape_from_facts(&ArchFacts::default()).is_none());
        let f = ArchFacts {
            block_count: Some(4),
            ..Default::default()
        };
        assert!(kv_shape_from_facts(&f).is_none(), "no heads, no width");
    }

    #[test]
    fn kv_an_absurd_layer_count_is_refused_rather_than_allocated() {
        // A hostile header can claim 2^40 layers; sizing it per layer would allocate a
        // vector that large and abort the process.
        let f = facts(1 << 40, 8, 128);
        assert!(kv_shape_from_facts(&f).is_none());
        assert_eq!(
            resolve_layer_kinds(1 << 40, None, Some(4), None),
            None,
            "no per-layer vector for an implausible model"
        );
    }

    #[test]
    fn kv_attention_free_architectures_are_all_recurrent() {
        for arch in ["mamba", "mamba2", "rwkv6", "rwkv7", "falcon_mamba"] {
            assert!(attention_free(arch), "{arch}");
        }
        assert!(!attention_free("jamba"), "hybrid: some layers attend");
        assert!(!attention_free("llama"));
    }

    #[test]
    fn kv_sliding_patterns_follow_llama_cpp() {
        use LayerKind::*;
        // Gemma 3: the 6th of every six is global.
        let g = resolve_layer_kinds(7, None, None, Some((6, false))).unwrap();
        assert_eq!(
            g,
            [Sliding, Sliding, Sliding, Sliding, Sliding, Full, Sliding]
        );
        // gpt-oss: alternating, starting sliding.
        let o = resolve_layer_kinds(4, None, None, Some((2, false))).unwrap();
        assert_eq!(o, [Sliding, Full, Sliding, Full]);
        // dense_first: the first of each group is global.
        let d = resolve_layer_kinds(5, None, None, Some((4, true))).unwrap();
        assert_eq!(d, [Full, Sliding, Sliding, Sliding, Full]);
        // No pattern at all: the reader says nothing, and every layer stays full.
        assert_eq!(resolve_layer_kinds(5, None, None, None), None);
    }
}
