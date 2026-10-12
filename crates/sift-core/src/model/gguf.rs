//! Lazy GGUF reader.
//!
//! Reads the header, metadata and tensor directory — and stops. Tensor payload is never
//! touched, so opening a 100 GB model costs the same as opening a 1 GB one and the
//! process RSS does not move. That property is asserted in the tests, because it is the
//! whole point: a planner has to see the shape of a model it cannot hold.

use std::collections::HashMap;
use std::io::{self, Read, Seek};
use std::path::Path;

use super::ggml::GgmlType;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// A metadata value from the GGUF key/value block.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
    /// An array the reader stepped over without materialising, keeping only its length.
    ///
    /// The tokenizer vocabulary is the case that matters: 150k strings allocated per
    /// candidate, for a planner that never reads a token.
    Skipped {
        len: u64,
    },
}

impl Value {
    /// Interpret as an unsigned integer, whatever width it was stored at.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Interpret as an array of unsigned integers, booleans counting as 0 and 1.
    ///
    /// GGUF stores per-layer facts — KV heads, which layers are recurrent — as arrays on
    /// models whose layers differ. `None` for anything that is not such an array.
    pub fn as_u64_array(&self) -> Option<Vec<u64>> {
        match self {
            Value::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::Bool(b) => Some(*b as u64),
                    other => other.as_u64(),
                })
                .collect(),
            _ => None,
        }
    }

    /// Interpret as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Where one tensor lives and what shape it has.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.0.ffn_gate_exps.weight`.
    pub name: String,
    /// Dimensions, innermost first, exactly as GGUF stores them.
    pub dims: Vec<u64>,
    /// Element type.
    pub dtype: GgmlType,
    /// Byte offset relative to the start of the tensor-data region.
    pub rel_offset: u64,
}

impl TensorInfo {
    /// Total number of weights.
    ///
    /// Cannot overflow: [`Gguf::parse`] refuses a tensor whose dimensions multiply past
    /// `u64`, so every `TensorInfo` it hands out has a representable element count.
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Total size in bytes, or `None` if the dtype is unknown or the shape does not
    /// divide into whole quantization blocks.
    pub fn size_bytes(&self) -> Option<u64> {
        self.dtype.bytes_for(self.n_elems())
    }
}

/// A parsed GGUF file: header, metadata and tensor directory, with no payload read.
#[derive(Debug, Clone)]
pub struct Gguf {
    /// Format version from the header.
    pub version: u32,
    /// Metadata key/value pairs.
    pub metadata: HashMap<String, Value>,
    /// Tensor directory, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where tensor data begins.
    ///
    /// Add [`TensorInfo::rel_offset`] to this to get an absolute file offset.
    pub data_offset: u64,
}

/// Errors that can arise reading a GGUF directory.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("not a GGUF file: magic was {0:#010x}, expected {MAGIC:#010x}")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0}")]
    BadVersion(u32),
    #[error("unknown metadata value type {0}")]
    BadValueType(u32),
    #[error("{what} claims {n} entries, which exceeds the {limit} sanity limit")]
    Implausible {
        what: &'static str,
        n: u64,
        limit: u64,
    },
    #[error("string of {0} bytes is implausible")]
    ImplausibleString(u64),
    #[error(
        "this GGUF is big-endian (written for s390x or similar); sift reads little-endian \
         GGUF only, which is what every mainstream publisher ships"
    )]
    BigEndian,
    #[error("tensor `{0}` has a shape or offset that overflows a 64-bit size")]
    Overflow(String),
}

// A corrupt or hostile file can encode enormous counts. Refuse them rather than
// attempting a multi-gigabyte allocation from a four-byte field.
const MAX_TENSORS: u64 = 1 << 22;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u64 = 8;
const MAX_STRING: u64 = 1 << 26;
const MAX_ARRAY: u64 = 1 << 26;
/// Elements held across every array the reader keeps, tokenizer arrays excluded.
///
/// Each element costs a `Value`, so the per-array bound alone still allows a file with a
/// few dozen maximal arrays to ask for tens of gigabytes. Nothing a planner reads is an
/// array of more than a few hundred entries — per-layer head counts are the largest.
const MAX_HELD_ARRAY_ELEMS: u64 = 1 << 22;

/// Metadata arrays under this prefix are stepped over rather than read.
const SKIPPED_PREFIX: &str = "tokenizer.";

struct Reader<R> {
    inner: R,
    /// Array elements materialised so far, against [`MAX_HELD_ARRAY_ELEMS`].
    held: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn u8(&mut self) -> io::Result<u8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> io::Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(GgufError::ImplausibleString(len));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        // GGUF strings are UTF-8 but we do not want a malformed byte to abort a whole
        // model load; substituting is strictly more useful than failing here.
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value, GgufError> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let (elem_ty, n) = self.array_header()?;
                self.held = self.held.saturating_add(n);
                if self.held > MAX_HELD_ARRAY_ELEMS {
                    return Err(GgufError::Implausible {
                        what: "metadata array elements",
                        n: self.held,
                        limit: MAX_HELD_ARRAY_ELEMS,
                    });
                }
                let mut items = Vec::with_capacity(n.min(4096) as usize);
                for _ in 0..n {
                    items.push(self.value(elem_ty)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::BadValueType(other)),
        })
    }

    /// Read an array's element type and length, refusing what llama.cpp refuses.
    ///
    /// An array of arrays is rejected outright, as llama.cpp's own reader does: no
    /// publisher writes one, and accepting it would let a hostile file nest the per-array
    /// bound into a product of bounds.
    fn array_header(&mut self) -> Result<(u32, u64), GgufError> {
        let elem_ty = self.u32()?;
        if elem_ty == 9 {
            return Err(GgufError::BadValueType(9));
        }
        let n = self.u64()?;
        if n > MAX_ARRAY {
            return Err(GgufError::Implausible {
                what: "array",
                n,
                limit: MAX_ARRAY,
            });
        }
        Ok((elem_ty, n))
    }

    /// Step over an array without materialising it.
    ///
    /// Fixed-width elements are one seek. Strings are read through without being kept, so
    /// nothing is allocated either way.
    fn skip_array(&mut self) -> Result<Value, GgufError> {
        let (elem_ty, n) = self.array_header()?;
        let width = match elem_ty {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            8 => None,
            other => return Err(GgufError::BadValueType(other)),
        };
        match width {
            Some(w) => {
                let bytes = (n * w) as i64; // n <= 2^26 and w <= 8, so this cannot overflow
                self.inner.seek(io::SeekFrom::Current(bytes))?;
            }
            None => {
                // Read through rather than seek: a seek discards a `BufReader`'s buffer,
                // and 150k of them would refill it 150k times. Copying into a sink costs
                // a memcpy per string and allocates nothing.
                for _ in 0..n {
                    let len = self.u64()?;
                    if len > MAX_STRING {
                        return Err(GgufError::ImplausibleString(len));
                    }
                    let copied = io::copy(&mut (&mut self.inner).take(len), &mut io::sink())?;
                    if copied != len {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                }
            }
        }
        Ok(Value::Skipped { len: n })
    }
}

impl Gguf {
    /// Parse the directory of a GGUF file on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path.as_ref())?;
        // A buffered reader keeps the many small header reads from becoming many syscalls.
        Self::parse(io::BufReader::with_capacity(1 << 20, file))
    }

    /// Parse a GGUF directory from any seekable reader.
    pub fn parse<R: Read + Seek>(inner: R) -> Result<Self, GgufError> {
        let mut r = Reader { inner, held: 0 };

        let magic = r.u32()?;
        if magic == MAGIC.swap_bytes() {
            return Err(GgufError::BigEndian);
        }
        if magic != MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32()?;
        // A big-endian file keeps the magic bytes in order and swaps everything after
        // them, so its version 3 reads here as 50331648. Name that rather than report a
        // version number nobody has ever shipped.
        if (2..=3).contains(&version.swap_bytes()) {
            return Err(GgufError::BigEndian);
        }
        if !(2..=3).contains(&version) {
            return Err(GgufError::BadVersion(version));
        }

        let tensor_count = r.u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::Implausible {
                what: "tensor count",
                n: tensor_count,
                limit: MAX_TENSORS,
            });
        }
        let kv_count = r.u64()?;
        if kv_count > MAX_KV {
            return Err(GgufError::Implausible {
                what: "metadata count",
                n: kv_count,
                limit: MAX_KV,
            });
        }

        let mut metadata = HashMap::with_capacity(kv_count.min(4096) as usize);
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = if ty == 9 && key.starts_with(SKIPPED_PREFIX) {
                r.skip_array()?
            } else {
                r.value(ty)?
            };
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 16) as usize);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as u64;
            if n_dims > MAX_DIMS {
                return Err(GgufError::Implausible {
                    what: "tensor rank",
                    n: n_dims,
                    limit: MAX_DIMS,
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let dtype = GgmlType(r.u32()?);
            let rel_offset = r.u64()?;
            // Checked once here so every later product and sum on this tensor is safe.
            if dims
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .is_none()
            {
                return Err(GgufError::Overflow(name));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                rel_offset,
            });
        }

        // Tensor data starts at the next `general.alignment` boundary after the directory.
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| a.is_power_of_two())
            .unwrap_or(32);
        let pos = r.inner.stream_position()?;
        let data_offset = pos.div_ceil(alignment) * alignment;
        // Absolute offsets are `data_offset + rel_offset` everywhere downstream.
        if let Some(t) = tensors
            .iter()
            .find(|t| data_offset.checked_add(t.rel_offset).is_none())
        {
            return Err(GgufError::Overflow(t.name.clone()));
        }

        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Absolute file offset of a tensor's payload.
    pub fn absolute_offset(&self, t: &TensorInfo) -> u64 {
        self.data_offset + t.rel_offset
    }

    /// Look up a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The model architecture string, e.g. `qwen3moe`.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(Value::as_str)
    }

    /// Read an architecture-scoped metadata value of any type.
    pub fn arch_value(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}"))
    }

    /// Read an architecture-scoped metadata integer, e.g. `expert_count` resolves to
    /// `qwen3moe.expert_count`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_u64)
    }

    /// Where this file sits in a split model: `(index from 1, total)`.
    ///
    /// `None` for an ordinary single-file model. Read from `split.no` and `split.count`,
    /// which `llama-gguf-split` writes into every part — so a shard is recognisable from
    /// its own header, whatever it happens to be called on disk.
    pub fn split(&self) -> Option<(u64, u64)> {
        let count = self.metadata.get("split.count").and_then(Value::as_u64)?;
        let no = self.metadata.get("split.no").and_then(Value::as_u64)?;
        (count > 1 && no < count).then_some((no + 1, count))
    }

    /// Total payload bytes across all tensors whose size we can compute.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter_map(TensorInfo::size_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Total weight count across every tensor.
    ///
    /// Counts only tensors whose byte size is computable, so it pairs with
    /// [`Self::total_tensor_bytes`] over the same set — otherwise
    /// [`Self::bits_per_weight`] would divide a partial numerator by a whole denominator
    /// and understate precision.
    pub fn total_parameters(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.size_bytes().is_some())
            .map(TensorInfo::n_elems)
            .fold(0, u64::saturating_add)
    }

    /// Mean bits stored per weight, measured from this file's own tensor directory.
    ///
    /// The honest quality signal for a quantization, and deliberately **not** a lookup
    /// table keyed on the filename. A table has to be taught every new quant family —
    /// `IQ4_NL`, `MXFP4`, whatever Unsloth ships next month — and is wrong until someone
    /// updates it. This arithmetic works on a quantization invented this morning, which is
    /// the same reason the tool reads real headers instead of shipping a model catalog.
    ///
    /// It also captures what a name cannot: dynamic quantizations give different layers
    /// different precision, so two files both labelled `Q4_K_M` genuinely differ. This
    /// reports what each one actually stores.
    ///
    /// `None` when no tensor size is computable, which is the same condition under which
    /// every other size figure here is unavailable.
    pub fn bits_per_weight(&self) -> Option<f64> {
        let params = self.total_parameters();
        if params == 0 {
            return None;
        }
        Some(self.total_tensor_bytes() as f64 * 8.0 / params as f64)
    }

    /// Seek past the directory without reading payload.
    ///
    /// Exposed so callers can prove to themselves that opening a file did not fault in
    /// any weights.
    pub fn data_region_start(&self) -> u64 {
        self.data_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal but structurally valid GGUF in memory.
    fn synth() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        b.extend_from_slice(&2u64.to_le_bytes()); // 2 metadata entries

        let kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&8u32.to_le_bytes()); // string
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        kv_str(&mut b, "general.architecture", "qwen3moe");

        // qwen3moe.expert_count = 128, as u32
        let k = "qwen3moe.expert_count";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&4u32.to_le_bytes()); // u32
        b.extend_from_slice(&128u32.to_le_bytes());

        // One stacked expert tensor: [2048, 768, 128] at Q4_K.
        let name = "blk.0.ffn_gate_exps.weight";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // rank 3
        b.extend_from_slice(&2048u64.to_le_bytes());
        b.extend_from_slice(&768u64.to_le_bytes());
        b.extend_from_slice(&128u64.to_le_bytes());
        b.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
        b.extend_from_slice(&0u64.to_le_bytes()); // rel offset
        b
    }

    #[test]
    fn parses_header_metadata_and_directory() {
        let g = Gguf::parse(Cursor::new(synth())).expect("parse");
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("qwen3moe"));
        assert_eq!(g.arch_u64("expert_count"), Some(128));
        assert_eq!(g.tensors.len(), 1);

        let t = g
            .tensor("blk.0.ffn_gate_exps.weight")
            .expect("tensor present");
        assert_eq!(t.dims, vec![2048, 768, 128]);
        assert_eq!(t.dtype.name(), "Q4_K");
        assert_eq!(t.n_elems(), 2048 * 768 * 128);
        // 128 experts x 884,736 bytes each.
        assert_eq!(t.size_bytes(), Some(884_736 * 128));
    }

    #[test]
    fn data_offset_respects_alignment() {
        let g = Gguf::parse(Cursor::new(synth())).expect("parse");
        assert_eq!(g.data_offset % 32, 0, "default alignment is 32");
        assert!(g.data_offset > 0);
    }

    #[test]
    fn rejects_a_non_gguf_file() {
        let bytes = b"this is not a model file at all".to_vec();
        match Gguf::parse(Cursor::new(bytes)) {
            Err(GgufError::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn refuses_an_implausible_tensor_count_instead_of_allocating() {
        // A corrupt header claiming 2^40 tensors must not trigger a huge allocation.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(1u64 << 40).to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::Implausible {
                what: "tensor count",
                ..
            }) => {}
            other => panic!("expected Implausible, got {other:?}"),
        }
    }

    /// A header with the given metadata entries and no tensors.
    fn with_kv(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (k, typed_value) in entries {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(typed_value);
        }
        b
    }

    /// Type tag + array header + raw element bytes.
    fn array(elem_ty: u32, n: u64, payload: &[u8]) -> Vec<u8> {
        let mut v = 9u32.to_le_bytes().to_vec();
        v.extend_from_slice(&elem_ty.to_le_bytes());
        v.extend_from_slice(&n.to_le_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn u16_value(n: u16) -> Vec<u8> {
        let mut v = 2u32.to_le_bytes().to_vec();
        v.extend_from_slice(&n.to_le_bytes());
        v
    }

    #[test]
    fn an_array_of_arrays_is_refused_as_llama_cpp_refuses_it() {
        // Nesting would turn the per-array bound into a product of bounds.
        let inner = array(4, 0, &[]);
        let b = with_kv(&[("x", array(9, 1, &inner[4..]))]);
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::BadValueType(9)) => {}
            other => panic!("expected BadValueType(9), got {other:?}"),
        }
    }

    #[test]
    fn tokenizer_arrays_are_stepped_over_and_the_rest_still_parses() {
        let mut strings = Vec::new();
        for tok in ["<s>", "hello", "world"] {
            strings.extend_from_slice(&(tok.len() as u64).to_le_bytes());
            strings.extend_from_slice(tok.as_bytes());
        }
        let ids: Vec<u8> = [1i32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut arch = 8u32.to_le_bytes().to_vec();
        arch.extend_from_slice(&5u64.to_le_bytes());
        arch.extend_from_slice(b"llama");
        let b = with_kv(&[
            ("tokenizer.ggml.tokens", array(8, 3, &strings)),
            ("tokenizer.ggml.token_type", array(5, 3, &ids)),
            ("general.architecture", arch),
        ]);
        let g = Gguf::parse(Cursor::new(b)).expect("parse");
        assert_eq!(
            g.metadata.get("tokenizer.ggml.tokens"),
            Some(&Value::Skipped { len: 3 })
        );
        assert_eq!(
            g.metadata.get("tokenizer.ggml.token_type"),
            Some(&Value::Skipped { len: 3 })
        );
        // The reader landed exactly after both arrays, or this key would be garbage.
        assert_eq!(g.architecture(), Some("llama"));
    }

    #[test]
    fn a_truncated_tokenizer_array_is_an_error_not_a_silent_skip() {
        let mut strings = 10u64.to_le_bytes().to_vec();
        strings.extend_from_slice(b"abc"); // claims 10 bytes, has 3
        let b = with_kv(&[("tokenizer.ggml.tokens", array(8, 1, &strings))]);
        assert!(Gguf::parse(Cursor::new(b)).is_err());
    }

    #[test]
    fn arrays_held_in_memory_are_bounded_in_total_not_only_each() {
        // Two arrays each under the per-array limit, together over the global one. The
        // second count is checked before any of its elements is read, so it fails fast.
        let half = (MAX_HELD_ARRAY_ELEMS / 2) as usize;
        let b = with_kv(&[
            ("a", array(0, half as u64, &vec![0u8; half])),
            ("b", array(0, MAX_HELD_ARRAY_ELEMS, &[])),
        ]);
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::Implausible {
                what: "metadata array elements",
                ..
            }) => {}
            other => panic!("expected the global bound, got {other:?}"),
        }
    }

    #[test]
    fn a_big_endian_file_is_named_as_such() {
        // Regression: this used to read as "unsupported GGUF version 50331648".
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::BigEndian) => {}
            other => panic!("expected BigEndian, got {other:?}"),
        }
    }

    #[test]
    fn a_tensor_whose_shape_overflows_is_refused() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let name = "t";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&(1u64 << 40).to_le_bytes());
        b.extend_from_slice(&(1u64 << 40).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::Overflow(n)) => assert_eq!(n, "t"),
            other => panic!("expected Overflow, got {other:?}"),
        }
    }

    #[test]
    fn a_shard_says_which_part_of_the_model_it_is() {
        let b = with_kv(&[("split.no", u16_value(1)), ("split.count", u16_value(3))]);
        let g = Gguf::parse(Cursor::new(b)).expect("parse");
        assert_eq!(g.split(), Some((2, 3)), "split.no is zero-based");

        let whole = Gguf::parse(Cursor::new(synth())).expect("parse");
        assert_eq!(whole.split(), None);
    }

    #[test]
    fn rejects_an_unsupported_version() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&99u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        match Gguf::parse(Cursor::new(b)) {
            Err(GgufError::BadVersion(99)) => {}
            other => panic!("expected BadVersion, got {other:?}"),
        }
    }
}
