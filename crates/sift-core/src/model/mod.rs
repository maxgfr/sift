//! Reading model files without loading them.
//!
//! The planner has to reason about a model it may not be able to hold: how many experts,
//! how big each one is, where its bytes live. All of that comes from the tensor
//! directory, which is a few hundred kilobytes at the front of the file. Payload is never
//! touched here.

pub mod experts;
pub mod ggml;
pub mod gguf;

pub use experts::{expert_ranges, expert_slice, ExpertError, ExpertRanges, ExpertSlice, TokenTraffic};
pub use ggml::{BlockLayout, GgmlType};
pub use gguf::{Gguf, GgufError, TensorInfo, Value};

/// A model's MoE shape, as far as the tensor directory reveals it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeShape {
    /// Number of transformer blocks carrying stacked expert tensors.
    pub moe_layers: u32,
    /// Routed experts per MoE layer.
    pub experts_per_layer: u64,
    /// Experts activated per token per layer (top-k).
    pub experts_per_token: u64,
    /// Bytes for one expert's three projections.
    pub bytes_per_expert: u64,
}

impl MoeShape {
    /// Bytes of routed-expert weights read per token, assuming every access misses.
    pub fn expert_bytes_per_token(&self) -> u64 {
        self.bytes_per_expert * self.experts_per_token * self.moe_layers as u64
    }

    /// Total bytes of routed-expert weights in the whole model.
    pub fn total_expert_bytes(&self) -> u64 {
        self.bytes_per_expert * self.experts_per_layer * self.moe_layers as u64
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
    let mut bytes_per_expert = 0u64;

    for layer in 0..u32::MAX {
        let name = format!("blk.{layer}.ffn_gate_exps.weight");
        let Some(t) = g.tensor(&name) else { break };
        moe_layers += 1;

        if experts_per_layer == 0 && t.dims.len() == 3 {
            experts_per_layer = t.dims[2];

            // Sum the three projections rather than tripling gate: down is stored
            // transposed and a model could in principle size it differently.
            let mut total = 0u64;
            for proj in ["gate", "up", "down"] {
                let n = format!("blk.{layer}.ffn_{proj}_exps.weight");
                if let Ok(s) = expert_slice(g, &n, 0) {
                    total += s.len;
                }
            }
            bytes_per_expert = total;
        }
    }

    if moe_layers == 0 || experts_per_layer == 0 || bytes_per_expert == 0 {
        return None;
    }

    let experts_per_token = g.arch_u64("expert_used_count").unwrap_or(8);

    Some(MoeShape {
        moe_layers,
        experts_per_layer,
        experts_per_token,
        bytes_per_expert,
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
            bytes_per_expert: 884_736 * 3,
        };

        assert!((shape.activation_ratio() - 0.0625).abs() < 1e-9, "8 of 128 is 6.25%");

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
            bytes_per_expert: 884_736 * 3,
        };
        let ratio = shape.expert_bytes_per_token() as f64 / shape.total_expert_bytes() as f64;
        assert!((ratio - shape.activation_ratio()).abs() < 1e-9);
    }

    #[test]
    fn a_dense_model_has_no_moe_shape() {
        let shape = MoeShape {
            moe_layers: 0,
            experts_per_layer: 0,
            experts_per_token: 0,
            bytes_per_expert: 0,
        };
        assert_eq!(shape.activation_ratio(), 0.0, "must not divide by zero");
    }
}
