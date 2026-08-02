//! Validation against a real MoE model on disk.
//!
//! The unit tests build synthetic GGUF headers, which proves the arithmetic is
//! self-consistent but not that it matches what a real quantizer writes. These tests read
//! actual bytes from an actual file at the offsets we compute.
//!
//! That distinction matters more than usual here. A wrong block size or a transposed
//! dimension does not raise an error — it reads a misaligned window of a neighbouring
//! expert and the model produces fluent nonsense. Reading real bytes and checking they
//! are distinct, in-bounds, and exactly tile the tensor is the only way to catch it.
//!
//! Set `SIFT_TEST_MOE` to a `.gguf` path, or drop one at `~/.sift/models/olmoe-q4km.gguf`.
//! Absent a model these tests skip rather than fail, so CI stays green without a 4 GB
//! download.

use sift_core::io::{CachePolicy, WeightFile};
use sift_core::model::{self, Gguf};
use std::path::PathBuf;

/// Locate a MoE model to test against, if one is available.
fn find_model() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SIFT_TEST_MOE") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join(".sift/models/olmoe-q4km.gguf");
    p.exists().then_some(p)
}

macro_rules! model_or_skip {
    () => {
        match find_model() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no MoE model (set SIFT_TEST_MOE to a .gguf path)");
                return;
            }
        }
    };
}

#[test]
fn opening_a_real_model_does_not_read_its_weights() {
    let path = model_or_skip!();
    let file_bytes = std::fs::metadata(&path).expect("stat").len();

    let before = sift_core::doctor::peak_rss_bytes();
    let g = Gguf::open(&path).expect("parse real gguf");
    let after = sift_core::doctor::peak_rss_bytes();
    let grew = after.saturating_sub(before);

    // The directory plus tokenizer metadata is tens of MiB at most; payload is gigabytes.
    // The point of a lazy reader is that this gap stays enormous.
    assert!(
        grew < file_bytes / 8,
        "opening grew RSS by {grew} bytes against a {file_bytes}-byte file; \
         payload is being read when it should not be"
    );
    assert!(!g.tensors.is_empty());
}

#[test]
fn expert_offsets_land_inside_the_file_and_tile_their_tensor() {
    let path = model_or_skip!();
    let g = Gguf::open(&path).expect("parse");
    let file_bytes = std::fs::metadata(&path).expect("stat").len();

    let Some(shape) = model::infer_moe_shape(&g) else {
        eprintln!("skipping: {} is dense, not MoE", path.display());
        return;
    };

    let name = "blk.0.ffn_gate_exps.weight";
    let tensor = g.tensor(name).expect("layer 0 gate experts");
    let tensor_start = g.absolute_offset(tensor);
    let tensor_end = tensor_start + tensor.size_bytes().expect("known size");

    let mut prev_end = tensor_start;
    for e in 0..shape.experts_per_layer {
        let s = model::expert_slice(&g, name, e).expect("slice");
        assert_eq!(
            s.offset,
            prev_end,
            "expert {e} must start where expert {} ended, leaving no gap",
            e.wrapping_sub(1)
        );
        assert!(
            s.end() <= file_bytes,
            "expert {e} ends at {} past the {file_bytes}-byte file",
            s.end()
        );
        prev_end = s.end();
    }

    assert_eq!(
        prev_end, tensor_end,
        "the experts must exactly cover the stacked tensor, with nothing left over"
    );
}

#[test]
fn different_experts_hold_different_bytes() {
    let path = model_or_skip!();
    let g = Gguf::open(&path).expect("parse");

    let Some(shape) = model::infer_moe_shape(&g) else {
        eprintln!("skipping: dense model");
        return;
    };
    assert!(
        shape.experts_per_layer >= 2,
        "need at least two experts to compare"
    );

    let name = "blk.0.ffn_gate_exps.weight";
    let a = model::expert_slice(&g, name, 0).expect("expert 0");
    let b = model::expert_slice(&g, name, 1).expect("expert 1");

    let wf = WeightFile::open(&path, CachePolicy::Cached).expect("open");
    // A prefix is enough: identical prefixes across two experts would mean the offsets
    // collapsed onto the same region.
    let n = 4096.min(a.len as usize);
    let mut buf_a = vec![0u8; n];
    let mut buf_b = vec![0u8; n];
    wf.read_at(&mut buf_a, a.offset).expect("read expert 0");
    wf.read_at(&mut buf_b, b.offset).expect("read expert 1");

    assert_ne!(
        buf_a, buf_b,
        "experts 0 and 1 read identical bytes; the per-expert stride is wrong"
    );
    assert!(
        buf_a.iter().any(|&x| x != 0),
        "expert 0 is entirely zero, which suggests the offset points outside the payload"
    );
}

#[test]
fn stock_layout_costs_three_scattered_reads_per_expert() {
    // This is the measured justification for a repacked container: the three projections
    // of one expert live in three tensors far apart, so using an expert is three seeks.
    let path = model_or_skip!();
    let g = Gguf::open(&path).expect("parse");

    if model::infer_moe_shape(&g).is_none() {
        eprintln!("skipping: dense model");
        return;
    }

    let r = model::expert_ranges(&g, 0, 0).expect("ranges");
    assert!(
        !r.is_contiguous(),
        "stock GGUF is expected to scatter gate/up/down; if this now passes, the \
         packer's premise has changed and the README claim needs revisiting"
    );

    // Quantify the scatter so a regression shows up as a number, not a boolean.
    //
    // Do not assume gate < up < down in the file. Real quantizers emit them in their own
    // order — OLMoE writes down, then gate, then up — so the span has to be computed from
    // the actual extremes rather than by subtracting one fixed end from one fixed start.
    let lo = r.gate.offset.min(r.up.offset).min(r.down.offset);
    let hi = r.gate.end().max(r.up.end()).max(r.down.end());
    let spread = hi - lo;
    assert!(
        spread > r.total_bytes() * 4,
        "expected the three projections to be far apart; they span {spread} bytes \
         while holding only {} bytes of payload",
        r.total_bytes()
    );
}

#[test]
fn projections_may_carry_different_dtypes() {
    // Q4_K_M gives `down` more bits than `gate`/`up` — a real, shipped instance of
    // spending bits where they matter. Any container we write must preserve per-projection
    // dtypes rather than assuming one type per expert.
    let path = model_or_skip!();
    let g = Gguf::open(&path).expect("parse");

    if model::infer_moe_shape(&g).is_none() {
        eprintln!("skipping: dense model");
        return;
    }

    let names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ];
    let dtypes: Vec<&str> = names
        .iter()
        .map(|n| g.tensor(n).expect("projection present").dtype.name())
        .collect();

    // Every projection must at least be a type we know how to size, or every downstream
    // offset is wrong.
    for (name, dtype) in names.iter().zip(&dtypes) {
        assert_ne!(
            *dtype, "unknown",
            "{name} has a dtype this build cannot size"
        );
    }

    // Sizes must be computed per projection, never by tripling one of them.
    let r = model::expert_ranges(&g, 0, 0).expect("ranges");
    assert_eq!(
        r.total_bytes(),
        r.gate.len + r.up.len + r.down.len,
        "per-expert size must sum the three projections individually"
    );
}

#[test]
fn per_token_traffic_matches_the_activation_ratio() {
    let path = model_or_skip!();
    let g = Gguf::open(&path).expect("parse");

    let Some(shape) = model::infer_moe_shape(&g) else {
        eprintln!("skipping: dense model");
        return;
    };

    // A token touches exactly `top_k / n_experts` of the expert weights. This is the
    // sparsity the whole design rests on, so it is checked against the real file rather
    // than assumed.
    let ratio = shape.expert_bytes_per_token() as f64 / shape.total_expert_bytes() as f64;
    assert!(
        (ratio - shape.activation_ratio()).abs() < 1e-9,
        "per-token traffic {ratio} disagrees with activation ratio {}",
        shape.activation_ratio()
    );
    assert!(
        ratio < 0.5,
        "a model this dense in activation would not benefit from residency planning"
    );
}
