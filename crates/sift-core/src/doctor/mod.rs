//! Measure the machine.
//!
//! Every planning decision `sift` makes depends on four numbers that vary by an order of
//! magnitude across machines that carry the same marketing name:
//!
//! - how fast this SSD reads **cold**, at the block size we actually use;
//! - how fast memory streams, which sets the resident-mode ceiling;
//! - how much RAM we may hold before the OS starts paging;
//! - where the paging cliff actually is.
//!
//! Published figures are not a substitute. "6 GB/s" for an Apple SSD is a large
//! sequential number from a benchmark app; a base-chip 512 GB configuration with few NAND
//! dies behaves nothing like a 4 TB Max. And a benchmark that forgets to defeat the page
//! cache measures RAM and reports it as storage — a mistake serious enough that one
//! project discarded four of its own published results over it.

use std::path::Path;
use std::time::Instant;

use crate::io::{AlignedBuf, CachePolicy, WeightFile};

/// Result of a single disk read measurement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiskSample {
    /// Read size per operation, in bytes.
    pub block_bytes: usize,
    /// Number of concurrent reader threads.
    pub threads: usize,
    /// Total bytes read across all threads.
    pub total_bytes: u64,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Achieved throughput in GB/s (decimal GB, matching how drives are marketed).
    pub gb_per_sec: f64,
    /// Mean latency per read operation, in milliseconds.
    pub ms_per_read: f64,
}

/// Throughput above which a "cold" disk read is almost certainly a page-cache hit.
///
/// No consumer NVMe device reads at this rate. Apple Silicon internal SSDs top out
/// around 6–14 GB/s depending on capacity and generation; anything far above that is
/// memory pretending to be storage.
pub const IMPLAUSIBLE_DISK_GB_S: f64 = 20.0;

impl DiskSample {
    /// Whether this sample is fast enough that it was probably served from RAM.
    ///
    /// `F_NOCACHE` stops *new* caching but cannot evict pages already resident, so a file
    /// the machine touched recently will read at memory speed and look like a spectacular
    /// SSD. Publishing such a number is the most common way to be wrong in this field —
    /// one project discarded four of its own results over it — so the harness flags it
    /// rather than trusting the caller to remember.
    pub fn looks_cached(&self) -> bool {
        self.gb_per_sec > IMPLAUSIBLE_DISK_GB_S
    }
}

/// Memory bandwidth, measured STREAM-style.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemorySample {
    /// Bytes touched (read + written) during the measurement.
    pub total_bytes: u64,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Achieved bandwidth in GB/s.
    pub gb_per_sec: f64,
}

/// What the machine reports about itself, before we measure anything.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MachineFacts {
    /// Physical RAM in bytes.
    pub ram_bytes: u64,
    /// OS page size in bytes. 16 KiB on Apple Silicon.
    pub page_bytes: usize,
    /// Logical CPU count.
    pub cpus: usize,
    /// Hardware model identifier, e.g. `Mac17,2`.
    pub model: Option<String>,
    /// Maximum bytes the GPU may wire, from `iogpu.wired_limit_mb`.
    ///
    /// `None` means the sysctl is unset, in which case macOS applies an internal default
    /// well below physical RAM. This is why an 11 GB model can fail to load on a 16 GB
    /// machine: the limit, not the RAM, is what binds.
    pub gpu_wired_limit_bytes: Option<u64>,
}

impl MachineFacts {
    /// Collect what the OS will tell us without running any benchmark.
    pub fn collect() -> Self {
        Self {
            ram_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
            page_bytes: crate::io::page_size(),
            cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            model: sysctl_string("hw.model"),
            gpu_wired_limit_bytes: sysctl_u64("iogpu.wired_limit_mb")
                .filter(|&mb| mb > 0)
                .map(|mb| mb * 1024 * 1024),
        }
    }
}

/// Read an integer sysctl by name.
fn sysctl_u64(name: &str) -> Option<u64> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `cname` is a valid NUL-terminated string; `value`/`size` are live locals of
    // matching type and size. `sysctlbyname` writes at most `size` bytes into `value`.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    // Some sysctls are 32-bit; a 4-byte read leaves the high half as written zeros.
    Some(value)
}

/// Read a string sysctl by name.
fn sysctl_string(name: &str) -> Option<String> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut size: usize = 0;
    // SAFETY: querying with a null buffer asks for the required size; `size` is a live local.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || size == 0 {
        return None;
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` has exactly `size` bytes, which is what the previous call requested.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    buf.truncate(size);
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}

/// Measure random-read throughput at one block size and thread count.
///
/// `path` must point at a file the machine has **not** recently read, otherwise resident
/// pages turn this into a memory benchmark. [`CachePolicy::Uncached`] prevents new
/// caching but cannot evict what is already there.
///
/// Offsets are strided by a large odd multiple of the block size rather than drawn
/// randomly, which keeps the access pattern scattered without needing an RNG dependency
/// and stays reproducible across runs.
pub fn measure_random_read(
    path: &Path,
    block_bytes: usize,
    threads: usize,
    reads_per_thread: usize,
) -> std::io::Result<DiskSample> {
    let file = std::sync::Arc::new(WeightFile::open(path, CachePolicy::Uncached)?);
    let len = file.len();

    let span = len.saturating_sub(block_bytes as u64);
    if span == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "file is {len} bytes, too small for a {block_bytes}-byte block; \
                 use a larger sample file"
            ),
        ));
    }

    // Spawning threads costs tens of microseconds each. With a small read count that
    // overhead lands inside the timed region and reports throughput far above what the
    // hardware can do — 33 GB/s from an NVMe drive, in one observed run. So: spawn every
    // thread first, hold them at a barrier, and only start the clock once all of them are
    // ready to issue their first read.
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);

    for t in 0..threads {
        let file = std::sync::Arc::clone(&file);
        let barrier = std::sync::Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || -> std::io::Result<u64> {
            let mut buf = AlignedBuf::new(block_bytes);
            let mut read = 0u64;
            // A large odd stride walks the file without repeating quickly and without
            // degenerating into a sequential scan the readahead could predict. Each
            // thread starts a different fraction of the way in so they do not convoy.
            let stride = block_bytes as u64 * 7 + 4096;
            let mut off = (span / threads.max(1) as u64).wrapping_mul(t as u64) % span;

            // Allocation and first-touch of `buf` are done; wait for everyone.
            barrier.wait();

            for _ in 0..reads_per_thread {
                file.read_at(buf.as_mut_slice(), off)?;
                read += block_bytes as u64;
                off = (off + stride) % span;
            }
            Ok(read)
        }));
    }

    barrier.wait();
    let started = Instant::now();

    let mut total_bytes = 0u64;
    for h in handles {
        total_bytes += h
            .join()
            .map_err(|_| std::io::Error::other("reader thread panicked"))??;
    }

    let seconds = started.elapsed().as_secs_f64();
    let ops = (threads * reads_per_thread) as f64;
    Ok(DiskSample {
        block_bytes,
        threads,
        total_bytes,
        seconds,
        gb_per_sec: total_bytes as f64 / seconds / 1e9,
        ms_per_read: seconds * 1000.0 / ops,
    })
}

/// Measure sequential read throughput with a single reader.
pub fn measure_sequential_read(
    path: &Path,
    block_bytes: usize,
    blocks: usize,
) -> std::io::Result<DiskSample> {
    let file = WeightFile::open(path, CachePolicy::Uncached)?;
    let mut buf = AlignedBuf::new(block_bytes);

    let want = block_bytes as u64 * blocks as u64;
    if file.len() < want {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("need {want} bytes, file has {}", file.len()),
        ));
    }

    let started = Instant::now();
    let mut off = 0u64;
    for _ in 0..blocks {
        file.read_at(buf.as_mut_slice(), off)?;
        off += block_bytes as u64;
    }
    let seconds = started.elapsed().as_secs_f64();

    Ok(DiskSample {
        block_bytes,
        threads: 1,
        total_bytes: want,
        seconds,
        gb_per_sec: want as f64 / seconds / 1e9,
        ms_per_read: seconds * 1000.0 / blocks as f64,
    })
}

/// Measure memory bandwidth, STREAM-copy style.
///
/// This sets the ceiling for resident-mode decode. A model that fits in RAM cannot
/// generate faster than `bandwidth / bytes_read_per_token`, no matter how good the
/// kernels are — so this number, divided by the model's per-token weight traffic, is the
/// honest upper bound to quote.
///
/// `bytes` is the size of *each* of the two buffers; a copy touches twice that.
pub fn measure_memory_bandwidth(bytes: usize, iterations: usize) -> MemorySample {
    let n = bytes / std::mem::size_of::<u64>();
    let src = vec![1u64; n];
    let mut dst = vec![0u64; n];

    // Warm both buffers so we measure steady-state bandwidth rather than first-touch
    // page faults.
    dst.copy_from_slice(&src);

    let started = Instant::now();
    for _ in 0..iterations {
        dst.copy_from_slice(&src);
        // Defeat any optimiser that might notice the copy is redundant.
        std::hint::black_box(&dst);
    }
    let seconds = started.elapsed().as_secs_f64();

    // A copy reads one buffer and writes the other.
    let total_bytes = (bytes as u64) * 2 * iterations as u64;
    MemorySample {
        total_bytes,
        seconds,
        gb_per_sec: total_bytes as f64 / seconds / 1e9,
    }
}

/// Peak resident set size of this process so far, in bytes.
///
/// Used to assert that a configured RAM budget was actually honoured. `sift` treats the
/// budget as a hard ceiling, so this is a test assertion, not a diagnostic.
pub fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a live, correctly-typed local that `getrusage` fully initialises.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0;
    }
    // Darwin reports maxrss in bytes; Linux reports kilobytes.
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        usage.ru_maxrss as u64 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn machine_facts_are_sane() {
        let f = MachineFacts::collect();
        assert!(f.ram_bytes > 0, "physical RAM must be discoverable");
        assert!(f.cpus >= 1);
        assert!(f.page_bytes.is_power_of_two());
    }

    #[test]
    fn memory_bandwidth_is_positive_and_finite() {
        let s = measure_memory_bandwidth(4 << 20, 4);
        assert!(s.gb_per_sec > 0.0 && s.gb_per_sec.is_finite());
        assert_eq!(s.total_bytes, (4u64 << 20) * 2 * 4);
    }

    #[test]
    fn random_read_reports_consistent_totals() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![0xABu8; 1 << 20]).expect("write");
        f.flush().expect("flush");

        let s = measure_random_read(f.path(), 4096, 2, 8).expect("measure");
        assert_eq!(s.total_bytes, 4096 * 2 * 8);
        assert_eq!(s.threads, 2);
        assert!(s.gb_per_sec > 0.0);
        assert!(s.ms_per_read >= 0.0);
    }

    #[test]
    fn a_block_larger_than_the_file_is_rejected_not_silently_shrunk() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&[0u8; 1024]).expect("write");
        f.flush().expect("flush");

        let err = measure_random_read(f.path(), 1 << 20, 1, 1).expect_err("must reject");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
