//! GGML tensor element types and their block layouts.
//!
//! Quantized GGML types are **block** formats: `block_elems` weights share one packed
//! record of `block_bytes`. Everything `sift` does with byte offsets — in particular
//! locating one expert inside a stacked expert tensor — depends on getting these two
//! numbers right, so they are exhaustive and tested rather than approximated.

/// A GGML tensor element type, as stored in a GGUF tensor directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType(pub u32);

/// Block geometry for a quantized (or plain) GGML type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// Number of weights per block.
    pub elems: u64,
    /// Bytes per block.
    pub bytes: u64,
}

impl BlockLayout {
    /// Average bits per weight for this layout.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes as f64 * 8.0 / self.elems as f64
    }
}

impl GgmlType {
    /// Human-readable name, or `"unknown"` for types this build does not model.
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            39 => "MXFP4",
            _ => "unknown",
        }
    }

    /// Block geometry, or `None` if this build does not know the type.
    ///
    /// Returning `None` rather than guessing matters: a wrong block size silently
    /// produces wrong byte offsets, which reads the wrong expert and yields plausible
    /// garbage instead of an error.
    pub fn block_layout(self) -> Option<BlockLayout> {
        let (elems, bytes) = match self.0 {
            0 => (1, 4),    // F32
            1 => (1, 2),    // F16
            2 => (32, 18),  // Q4_0
            3 => (32, 20),  // Q4_1
            6 => (32, 22),  // Q5_0
            7 => (32, 24),  // Q5_1
            8 => (32, 34),  // Q8_0
            9 => (32, 36),  // Q8_1
            10 => (256, 82),   // Q2_K
            11 => (256, 110),  // Q3_K
            12 => (256, 144),  // Q4_K
            13 => (256, 176),  // Q5_K
            14 => (256, 210),  // Q6_K
            15 => (256, 292),  // Q8_K
            16 => (256, 66),   // IQ2_XXS
            17 => (256, 74),   // IQ2_XS
            18 => (256, 98),   // IQ3_XXS
            19 => (256, 50),   // IQ1_S
            20 => (32, 18),    // IQ4_NL
            21 => (256, 110),  // IQ3_S
            22 => (256, 82),   // IQ2_S
            23 => (256, 136),  // IQ4_XS
            24 => (1, 1),   // I8
            25 => (1, 2),   // I16
            26 => (1, 4),   // I32
            27 => (1, 8),   // I64
            28 => (1, 8),   // F64
            29 => (256, 56),   // IQ1_M
            30 => (1, 2),   // BF16
            39 => (32, 17),    // MXFP4: 32 4-bit weights + one E8M0 scale byte
            _ => return None,
        };
        Some(BlockLayout { elems, bytes })
    }

    /// Bytes needed to store `n_elems` weights of this type.
    ///
    /// Returns `None` if the type is unknown, or if `n_elems` is not a whole number of
    /// blocks — a partial block means our understanding of the tensor is wrong, and
    /// rounding it would corrupt every subsequent offset.
    pub fn bytes_for(self, n_elems: u64) -> Option<u64> {
        let layout = self.block_layout()?;
        if n_elems % layout.elems != 0 {
            return None;
        }
        Some(n_elems / layout.elems * layout.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_is_four_and_a_half_bits() {
        let l = GgmlType(12).block_layout().expect("Q4_K is known");
        assert_eq!(l.elems, 256);
        assert_eq!(l.bytes, 144);
        assert!((l.bits_per_weight() - 4.5).abs() < 1e-9);
    }

    #[test]
    fn mxfp4_is_four_and_a_quarter_bits() {
        // gpt-oss ships natively in MXFP4; getting this wrong would misplace every expert.
        let l = GgmlType(39).block_layout().expect("MXFP4 is known");
        assert!((l.bits_per_weight() - 4.25).abs() < 1e-9);
    }

    #[test]
    fn f32_is_thirty_two_bits() {
        let l = GgmlType(0).block_layout().expect("F32 is known");
        assert!((l.bits_per_weight() - 32.0).abs() < 1e-9);
    }

    #[test]
    fn byte_size_matches_a_hand_computed_expert() {
        // One Qwen3-30B-A3B expert matrix: 2048 x 768 weights at Q4_K.
        let elems = 2048u64 * 768;
        let bytes = GgmlType(12).bytes_for(elems).expect("whole blocks");
        assert_eq!(elems % 256, 0, "must divide into whole Q4_K blocks");
        assert_eq!(bytes, elems / 256 * 144);
        assert_eq!(bytes, 884_736);
    }

    #[test]
    fn a_partial_block_is_refused_rather_than_rounded() {
        // 100 weights is not a whole number of 256-weight Q4_K blocks. Rounding here
        // would silently shift every later tensor offset.
        assert_eq!(GgmlType(12).bytes_for(100), None);
    }

    #[test]
    fn unknown_types_report_none_instead_of_guessing() {
        assert_eq!(GgmlType(9999).block_layout(), None);
        assert_eq!(GgmlType(9999).bytes_for(256), None);
        assert_eq!(GgmlType(9999).name(), "unknown");
    }
}
