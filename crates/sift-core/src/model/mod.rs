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
pub use gguf::{Gguf, GgufError, TensorInfo, Value};
pub use remote::{hf_url, RemoteFile};

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
pub fn infer_moe_shape(g: &Gguf) -> Option<MoeShape> {
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
