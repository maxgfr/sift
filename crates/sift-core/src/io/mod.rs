//! The cold tier: explicit reads that never touch the page cache.
//!
//! # Why not mmap
//!
//! The obvious design is to mmap the weights file and let the kernel page it in. Two
//! independent engines measured that and rejected it: a 3.9 MB expert spans 240 × 16 KiB
//! pages on Apple Silicon, so one logical read becomes 240 faults instead of one NVMe
//! command. Apple's own *LLM in a Flash* team also used explicit reads with `F_NOCACHE`
//! across many threads rather than mmap.
//!
//! There is a dissenting measurement (llama.cpp discussion #18758 reports mmap winning
//! for MoE), which is why [`crate::doctor`] measures both on the actual machine instead
//! of trusting either result. This module implements the explicit-read side.
//!
//! # Why bypass the page cache
//!
//! A cache we don't control is a cache that competes with the weights we pinned on
//! purpose. `F_NOCACHE` keeps cold-tier reads from growing into the resident set, which
//! is what makes the RSS ceiling an actual guarantee rather than a hope.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Page size assumed when aligning buffers. Apple Silicon uses 16 KiB pages, not 4 KiB.
///
/// Prefer [`page_size`] over this constant; it is only the fallback.
pub const ASSUMED_PAGE_SIZE: usize = 16 * 1024;

/// Destination buffers are aligned to this boundary.
///
/// Measured on Apple Silicon: 2 MiB-aligned destinations read page-cache-resident data at
/// 16.8 GB/s versus 4.7 GB/s for 16 KiB alignment — a 3.6× difference for a `posix_memalign`
/// argument. It costs nothing, so we always do it.
pub const DEST_ALIGN: usize = 2 * 1024 * 1024;

/// Query the OS page size at runtime.
///
/// Never hardcode this: Apple Silicon is 16 KiB where most of the world is 4 KiB, and
/// alignment requirements downstream (Metal's `newBufferWithBytesNoCopy`) are expressed
/// in real pages.
pub fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name has no preconditions and cannot fail in a way
    // that matters here; a negative return is handled below.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        ASSUMED_PAGE_SIZE
    }
}

/// How the kernel should treat the page cache for a given file handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Let the kernel cache normally. Fast on repeat reads, but the cache grows into
    /// memory we may have promised to the resident set.
    Cached,
    /// `F_NOCACHE`: do not retain these pages.
    ///
    /// This is what makes a cold measurement honest and what keeps the cold tier from
    /// stealing RAM from pinned weights. Note it does **not** evict pages already
    /// resident — see [`WeightFile::open`] docs.
    Uncached,
}

/// A weights file opened for explicit positional reads.
///
/// Reads go through `pread`, which takes `&self` and ignores the file cursor, so a single
/// handle can be shared across threads without synchronisation.
pub struct WeightFile {
    file: File,
    len: u64,
}

impl WeightFile {
    /// Open a weights file with an explicit cache policy.
    ///
    /// # Cold-measurement caveat
    ///
    /// [`CachePolicy::Uncached`] stops *new* caching but cannot evict pages that are
    /// already resident. Opening a file the machine just read and calling this "cold" is
    /// the single most common way to publish a wrong number — you end up measuring RAM.
    /// Use a file the machine has not touched, or reboot.
    pub fn open(path: impl AsRef<Path>, policy: CachePolicy) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let len = file.metadata()?.len();

        if policy == CachePolicy::Uncached {
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is a live descriptor owned by `file` for the duration of this
            // call. Both commands take an int argument and only affect caching policy;
            // failure is advisory, so the result is deliberately not propagated.
            unsafe {
                // Do not retain these pages in the unified buffer cache.
                libc::fcntl(fd, libc::F_NOCACHE, 1);
                // Disable kernel readahead. Once we schedule our own reads, readahead
                // competes with us and spends bandwidth on a pattern it mispredicts.
                libc::fcntl(fd, libc::F_RDAHEAD, 0);
            }
        }

        Ok(Self { file, len })
    }

    /// Total file length in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read exactly `buf.len()` bytes at `offset`.
    ///
    /// Takes `&self`: `pread` does not touch the shared file cursor, so concurrent calls
    /// on one handle are safe and are the intended usage.
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, offset)
    }
}

/// A heap buffer aligned to [`DEST_ALIGN`].
///
/// Alignment of the *destination* measurably changes read throughput on Apple Silicon,
/// so this is not premature tidiness.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `AlignedBuf` uniquely owns its allocation and exposes it only through `&self` /
// `&mut self` slices, so it carries the same thread-safety as `Box<[u8]>`.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Allocate `len` bytes aligned to [`DEST_ALIGN`], zero-initialised.
    ///
    /// # Panics
    /// Panics if `len` is zero or the allocation fails.
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "AlignedBuf::new requires a non-zero length");
        let layout = std::alloc::Layout::from_size_align(len, DEST_ALIGN)
            .expect("length and alignment form a valid layout");
        // SAFETY: layout has non-zero size, checked above.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, len }
    }

    /// Borrow the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is a live allocation of exactly `len` bytes, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Borrow the buffer as a slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as above, shared borrow.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Buffer length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty. Always false; present to satisfy clippy.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, DEST_ALIGN)
            .expect("layout was valid at construction");
        // SAFETY: `ptr` came from `alloc_zeroed` with this exact layout and is freed once.
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn page_size_is_plausible() {
        let ps = page_size();
        assert!(
            ps.is_power_of_two(),
            "page size {ps} should be a power of two"
        );
        assert!(
            (4096..=65536).contains(&ps),
            "page size {ps} outside expected range"
        );
    }

    #[test]
    fn aligned_buf_is_actually_aligned() {
        let buf = AlignedBuf::new(4096);
        assert_eq!(buf.as_slice().as_ptr() as usize % DEST_ALIGN, 0);
        assert_eq!(buf.len(), 4096);
        assert!(
            buf.as_slice().iter().all(|&b| b == 0),
            "must be zero-initialised"
        );
    }

    #[test]
    fn reads_at_offset_without_moving_a_cursor() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        let data: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
        f.write_all(&data).expect("write");
        f.flush().expect("flush");

        let wf = WeightFile::open(f.path(), CachePolicy::Uncached).expect("open");
        assert_eq!(wf.len(), 8192);

        // Read the same range twice: pread must not advance any shared cursor.
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        wf.read_at(&mut a, 1024).expect("read a");
        wf.read_at(&mut b, 1024).expect("read b");
        assert_eq!(
            a, b,
            "repeated pread at one offset must return identical bytes"
        );
        assert_eq!(a[..], data[1024..1088]);
    }

    #[test]
    fn reading_past_the_end_is_an_error_not_a_short_read() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&[7u8; 100]).expect("write");
        f.flush().expect("flush");

        let wf = WeightFile::open(f.path(), CachePolicy::Cached).expect("open");
        let mut buf = [0u8; 200];
        assert!(
            wf.read_at(&mut buf, 0).is_err(),
            "short read must surface as an error"
        );
    }
}
