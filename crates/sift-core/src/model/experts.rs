//! Locating a single expert inside a stacked expert tensor.
//!
//! GGUF does not store one tensor per expert. All experts of a layer share one stacked
//! tensor per projection:
//!
//! ```text
//! blk.0.ffn_gate_exps.weight   [2048, 768, 128]   <- 128 experts, gate
//! blk.0.ffn_up_exps.weight     [2048, 768, 128]   <- 128 experts, up
//! blk.0.ffn_down_exps.weight   [768, 2048, 128]   <- 128 experts, down
//! ```
//!
//! So there is no tensor name to look up for "expert 37". Its bytes must be computed:
//! each expert occupies a contiguous slice of the stacked tensor, and the slice for
//! expert *e* starts at `e * bytes_per_expert`.
//!
//! Two consequences shape the whole engine:
//!
//! - **One expert's gate, up and down live in three tensors far apart on disk.** Using
//!   one expert therefore costs three scattered reads. That is the access pattern that
//!   [`crate::model::ExpertSlice`] exists to make measurable, and the reason a repacked
//!   container is worth building.
//! - **The arithmetic must be exact.** A block-size mistake does not raise an error; it
//!   reads a misaligned window of a neighbouring expert and produces fluent nonsense.

use super::gguf::{Gguf, TensorInfo};

/// A byte range on disk holding exactly one expert's weights for one projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlice {
    /// Absolute file offset.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
}

impl ExpertSlice {
    /// End offset, exclusive.
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }
}

/// The three projections making up one MoE expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRanges {
    /// Gate projection slice.
    pub gate: ExpertSlice,
    /// Up projection slice.
    pub up: ExpertSlice,
    /// Down projection slice.
    pub down: ExpertSlice,
}

impl ExpertRanges {
    /// Total bytes that must be read to use this expert.
    pub fn total_bytes(&self) -> u64 {
        self.gate.len + self.up.len + self.down.len
    }

    /// Whether the three projections are contiguous on disk, in gate/up/down order.
    ///
    /// False for stock GGUF — which is precisely the cost a repacked container removes,
    /// turning three scattered reads into one. Used by the packer to verify its output
    /// and by the benchmark to report reads-per-token honestly.
    pub fn is_contiguous(&self) -> bool {
        self.gate.end() == self.up.offset && self.up.end() == self.down.offset
    }
}

/// Why an expert's byte range could not be computed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExpertError {
    #[error("tensor `{0}` is not present in this model")]
    MissingTensor(String),
    #[error("tensor `{name}` has rank {rank}, expected 3 (a stacked expert tensor)")]
    NotStacked { name: String, rank: usize },
    #[error("expert {index} is out of range; tensor `{name}` holds {count}")]
    OutOfRange { name: String, index: u64, count: u64 },
    #[error("tensor `{name}` uses dtype {dtype}, whose block layout this build does not know")]
    UnknownDtype { name: String, dtype: String },
    #[error(
        "tensor `{name}` holds {elems} weights across {count} experts, which is not a whole \
         number of {block}-weight blocks per expert; the layout assumption is wrong"
    )]
    NotBlockAligned { name: String, elems: u64, count: u64, block: u64 },
}

/// Compute the byte range of one expert inside a stacked expert tensor.
///
/// GGUF dimensions are innermost-first, so a stacked expert tensor's **last** dimension
/// is the expert count and the leading dimensions are one expert's matrix shape.
pub fn expert_slice(g: &Gguf, tensor_name: &str, expert: u64) -> Result<ExpertSlice, ExpertError> {
    let t: &TensorInfo = g
        .tensor(tensor_name)
        .ok_or_else(|| ExpertError::MissingTensor(tensor_name.to_string()))?;

    if t.dims.len() != 3 {
        return Err(ExpertError::NotStacked {
            name: t.name.clone(),
            rank: t.dims.len(),
        });
    }

    let count = t.dims[2];
    if expert >= count {
        return Err(ExpertError::OutOfRange {
            name: t.name.clone(),
            index: expert,
            count,
        });
    }

    let layout = t.dtype.block_layout().ok_or_else(|| ExpertError::UnknownDtype {
        name: t.name.clone(),
        dtype: t.dtype.name().to_string(),
    })?;

    // Weights in one expert's matrix.
    let per_expert_elems = t.dims[0] * t.dims[1];
    if per_expert_elems % layout.elems != 0 {
        return Err(ExpertError::NotBlockAligned {
            name: t.name.clone(),
            elems: per_expert_elems,
            count,
            block: layout.elems,
        });
    }

    let bytes_per_expert = per_expert_elems / layout.elems * layout.bytes;
    Ok(ExpertSlice {
        offset: g.absolute_offset(t) + expert * bytes_per_expert,
        len: bytes_per_expert,
    })
}

/// Compute all three projection ranges for one expert of one layer.
///
/// Tensor names follow the llama.cpp convention: `blk.{layer}.ffn_{gate,up,down}_exps.weight`.
pub fn expert_ranges(g: &Gguf, layer: u32, expert: u64) -> Result<ExpertRanges, ExpertError> {
    Ok(ExpertRanges {
        gate: expert_slice(g, &format!("blk.{layer}.ffn_gate_exps.weight"), expert)?,
        up: expert_slice(g, &format!("blk.{layer}.ffn_up_exps.weight"), expert)?,
        down: expert_slice(g, &format!("blk.{layer}.ffn_down_exps.weight"), expert)?,
    })
}

/// Per-token weight traffic for a MoE model, in bytes.
///
/// This is the number that sets the speed ceiling. Divide the machine's achievable
/// bandwidth by it and you have the maximum tokens per second, whatever the kernels do:
///
/// ```text
/// resident:  153 GB/s memory   / 1.1 GB per token  ->  ~139 tok/s ceiling
/// streamed:  6.15 GB/s NVMe    / 1.1 GB per token  ->  ~5.6 tok/s ceiling
/// ```
///
/// The 25× gap between those two lines is the entire reason residency planning matters.
#[derive(Debug, Clone, Copy)]
pub struct TokenTraffic {
    /// Bytes of routed-expert weights read per token, assuming every access misses.
    pub expert_bytes: u64,
    /// Bytes of always-active weights read per token (attention, embeddings, routers,
    /// shared experts, output projection).
    pub trunk_bytes: u64,
}

impl TokenTraffic {
    /// Total bytes read per token with a cold cache.
    pub fn total(&self) -> u64 {
        self.expert_bytes + self.trunk_bytes
    }

    /// Tokens per second achievable at a given bandwidth, given a cache hit rate.
    ///
    /// `hit_rate` applies only to expert traffic; trunk weights are read every token by
    /// definition and no cache policy changes that.
    pub fn tokens_per_sec(&self, bytes_per_sec: f64, hit_rate: f64) -> f64 {
        let hit = hit_rate.clamp(0.0, 1.0);
        let bytes = self.expert_bytes as f64 * (1.0 - hit) + self.trunk_bytes as f64;
        if bytes <= 0.0 {
            return f64::INFINITY;
        }
        bytes_per_sec / bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gguf::Gguf;
    use std::io::Cursor;

    const MAGIC: u32 = 0x4655_4747;

    /// Build a GGUF with the three stacked expert tensors of one layer.
    fn synth_moe_layer() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // no metadata

        // gate and up: [2048, 768, 128]; down: [768, 2048, 128]. All Q4_K.
        // Offsets are laid out consecutively, as a real file would.
        let per_expert = 2048u64 * 768 / 256 * 144; // 884,736
        let stacked = per_expert * 128;
        let specs: [(&str, [u64; 3], u64); 3] = [
            ("blk.0.ffn_gate_exps.weight", [2048, 768, 128], 0),
            ("blk.0.ffn_up_exps.weight", [2048, 768, 128], stacked),
            ("blk.0.ffn_down_exps.weight", [768, 2048, 128], stacked * 2),
        ];
        for (name, dims, off) in specs {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&3u32.to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
            b.extend_from_slice(&off.to_le_bytes());
        }
        b
    }

    fn model() -> Gguf {
        Gguf::parse(Cursor::new(synth_moe_layer())).expect("parse")
    }

    #[test]
    fn expert_zero_starts_at_the_tensor_origin() {
        let g = model();
        let t = g.tensor("blk.0.ffn_gate_exps.weight").expect("present");
        let s = expert_slice(&g, "blk.0.ffn_gate_exps.weight", 0).expect("slice");
        assert_eq!(s.offset, g.absolute_offset(t));
        assert_eq!(s.len, 884_736);
    }

    #[test]
    fn consecutive_experts_are_adjacent_and_equal_sized() {
        let g = model();
        let a = expert_slice(&g, "blk.0.ffn_gate_exps.weight", 5).expect("slice 5");
        let b = expert_slice(&g, "blk.0.ffn_gate_exps.weight", 6).expect("slice 6");
        assert_eq!(a.end(), b.offset, "experts must tile the stacked tensor without gaps");
        assert_eq!(a.len, b.len);
    }

    #[test]
    fn the_last_expert_ends_exactly_at_the_tensor_end() {
        let g = model();
        let t = g.tensor("blk.0.ffn_gate_exps.weight").expect("present");
        let last = expert_slice(&g, "blk.0.ffn_gate_exps.weight", 127).expect("slice 127");
        let tensor_end = g.absolute_offset(t) + t.size_bytes().expect("size");
        assert_eq!(last.end(), tensor_end, "slices must exactly cover the tensor");
    }

    #[test]
    fn an_out_of_range_expert_is_refused() {
        let g = model();
        match expert_slice(&g, "blk.0.ffn_gate_exps.weight", 128) {
            Err(ExpertError::OutOfRange { index: 128, count: 128, .. }) => {}
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn stock_gguf_experts_are_not_contiguous_across_projections() {
        // This is the finding that justifies a repacked container: using one expert costs
        // three reads scattered across the file, not one.
        let g = model();
        let r = expert_ranges(&g, 0, 3).expect("ranges");
        assert!(
            !r.is_contiguous(),
            "stock GGUF splits gate/up/down across three distant tensors"
        );
        assert_eq!(r.total_bytes(), 884_736 * 3);
    }

    #[test]
    fn down_projection_has_the_same_size_despite_transposed_dims() {
        let g = model();
        let r = expert_ranges(&g, 0, 0).expect("ranges");
        assert_eq!(r.gate.len, r.down.len, "768x2048 and 2048x768 hold equal weights");
    }

    #[test]
    fn a_missing_layer_names_the_tensor_it_looked_for() {
        let g = model();
        match expert_ranges(&g, 47, 0) {
            Err(ExpertError::MissingTensor(name)) => {
                assert_eq!(name, "blk.47.ffn_gate_exps.weight");
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn token_traffic_matches_the_hand_computed_qwen3_figure() {
        // Qwen3-30B-A3B at Q4_K: 8 experts x 48 layers x 3 projections x 884,736 bytes.
        let expert_bytes = 8 * 48 * 3 * 884_736u64;
        let t = TokenTraffic { expert_bytes, trunk_bytes: 0 };
        let gb = t.total() as f64 / 1e9;
        assert!((gb - 1.019).abs() < 0.01, "expected ~1.02 GB/token, got {gb:.3}");
    }

    #[test]
    fn hit_rate_scales_only_expert_traffic() {
        let t = TokenTraffic { expert_bytes: 1_000_000_000, trunk_bytes: 500_000_000 };

        // Cold: reads everything.
        let cold = t.tokens_per_sec(6.15e9, 0.0);
        assert!((cold - 6.15e9 / 1.5e9).abs() < 1e-6);

        // Perfect expert cache still pays the trunk every token.
        let hot = t.tokens_per_sec(6.15e9, 1.0);
        assert!((hot - 6.15e9 / 0.5e9).abs() < 1e-6);

        assert!(hot > cold);
    }

    #[test]
    fn hit_rate_is_clamped_rather_than_producing_nonsense() {
        let t = TokenTraffic { expert_bytes: 1_000, trunk_bytes: 1_000 };
        assert_eq!(t.tokens_per_sec(1e9, 2.0), t.tokens_per_sec(1e9, 1.0));
        assert_eq!(t.tokens_per_sec(1e9, -5.0), t.tokens_per_sec(1e9, 0.0));
    }
}
