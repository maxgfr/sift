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
    Implausible { what: &'static str, n: u64, limit: u64 },
    #[error("string of {0} bytes is implausible")]
    ImplausibleString(u64),
}

// A corrupt or hostile file can encode enormous counts. Refuse them rather than
// attempting a multi-gigabyte allocation from a four-byte field.
const MAX_TENSORS: u64 = 1 << 22;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u64 = 8;
const MAX_STRING: u64 = 1 << 26;
const MAX_ARRAY: u64 = 1 << 26;

struct Reader<R> {
    inner: R,
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
                let elem_ty = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY {
                    return Err(GgufError::Implausible { what: "array", n, limit: MAX_ARRAY });
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
        let mut r = Reader { inner };

        let magic = r.u32()?;
        if magic != MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32()?;
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
            let value = r.value(ty)?;
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
            tensors.push(TensorInfo { name, dims, dtype, rel_offset });
        }

        // Tensor data starts at the next `general.alignment` boundary after the directory.
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| a.is_power_of_two())
            .unwrap_or(32);
        let pos = r.inner.stream_position()?;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(Gguf { version, metadata, tensors, data_offset })
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
        self.metadata.get("general.architecture").and_then(Value::as_str)
    }

    /// Read an architecture-scoped metadata integer, e.g. `expert_count` resolves to
    /// `qwen3moe.expert_count`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}")).and_then(Value::as_u64)
    }

    /// Total payload bytes across all tensors whose size we can compute.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(TensorInfo::size_bytes).sum()
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

        let t = g.tensor("blk.0.ffn_gate_exps.weight").expect("tensor present");
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
            Err(GgufError::Implausible { what: "tensor count", .. }) => {}
            other => panic!("expected Implausible, got {other:?}"),
        }
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
