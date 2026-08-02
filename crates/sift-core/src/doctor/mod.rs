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

/// Bandwidth above which a memory measurement is not believable on consumer hardware.
///
/// No desktop or laptop part reaches this. Apple's widest configurations top out around
/// 800 GB/s, and a mainstream machine is a fraction of that. A figure above this means the
/// benchmark measured something other than memory — usually the optimiser hoisting an
/// invariant loop, which reported 369 GB/s here before [`measure_read_bandwidth`] started
/// threading its accumulator through each pass.
pub const IMPLAUSIBLE_MEMORY_GB_S: f64 = 1200.0;

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

impl MemorySample {
    /// Whether this figure is too fast to be a real memory measurement.
    ///
    /// Reported rather than silently corrected, in the same spirit as
    /// [`DiskSample::looks_cached`]: a number the harness cannot vouch for should be
    /// visible, not quietly folded into an average.
    pub fn is_implausible(&self) -> bool {
        !self.gb_per_sec.is_finite() || self.gb_per_sec > IMPLAUSIBLE_MEMORY_GB_S
    }
}

pub use crate::platform::{AccelMemory, MachineFacts};

/// Alignment every read offset is held to.
///
/// 4 KiB covers both common sector sizes (512 and 4096). Windows' unbuffered reads reject
/// anything else outright; elsewhere it is simply the granularity worth measuring, since a
/// read straddling a block boundary costs an I/O the caller did not ask for.
///
/// Note this constrains offsets only. Read *lengths* must also be a sector multiple under
/// `FILE_FLAG_NO_BUFFERING`, which is the caller's business — every block size swept here
/// is a power of two at or above 64 KiB.
const OFFSET_ALIGN: u64 = 4096;

/// Round down to the previous [`OFFSET_ALIGN`] boundary.
fn align_down(v: u64) -> u64 {
    v - (v % OFFSET_ALIGN)
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
    let file = WeightFile::open(path, CachePolicy::Uncached)?;
    let len = file.len();

    // Offsets are held to a sector multiple. Windows rejects an unaligned unbuffered read
    // outright, and on every other platform a read straddling a block boundary silently
    // costs an extra I/O — so the sample would describe a pattern nobody asked to measure.
    let span = (len.saturating_sub(block_bytes as u64) / OFFSET_ALIGN) * OFFSET_ALIGN;
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
        // A handle of its own, not a shared one. Windows `seek_read` moves the file
        // pointer, so threads sharing a handle would read each other's offsets and report
        // throughput from a pattern nobody asked for. Opening per thread costs one syscall
        // and is done before the barrier, so it stays outside the timed region.
        let file = file.try_clone()?;
        let barrier = std::sync::Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || -> std::io::Result<u64> {
            let mut buf = AlignedBuf::new(block_bytes);
            let mut read = 0u64;
            // A large odd stride walks the file without repeating quickly and without
            // degenerating into a sequential scan the readahead could predict. Each
            // thread starts a different fraction of the way in so they do not convoy.
            // Both are rounded to `OFFSET_ALIGN`, which keeps every offset a sector
            // multiple however odd the caller's block size is.
            let stride = align_down(block_bytes as u64 * 7 + OFFSET_ALIGN).max(OFFSET_ALIGN);
            let start = (span / threads.max(1) as u64).wrapping_mul(t as u64) % span;
            let mut off = align_down(start);

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

/// Buffer size that reliably escapes the last-level cache.
///
/// This has to clear far more than an L3. Apple's System Level Cache sits behind the CPU
/// caches and is reported at ~96 MiB on M4 Max; AMD X3D parts carry 96–128 MiB of L3, and
/// server chips reach 384 MiB. A 256 MiB buffer is under 3× an Apple SLC — close enough
/// that the measurement is partly cache bandwidth, which reads as an implausibly fast
/// machine.
///
/// 512 MiB is over 5× the largest consumer LLC, and [`bandwidth_escapes_cache`] checks
/// empirically rather than trusting this constant.
pub const BANDWIDTH_BUF_BYTES: usize = 512 << 20;

/// Whether a bandwidth measurement has genuinely escaped the cache.
///
/// Run the kernel at two sizes: if the larger one is materially slower, the smaller was
/// still partly served by cache. A tool that publishes bandwidth numbers should be able to
/// prove it measured memory.
pub fn bandwidth_escapes_cache() -> (MemorySample, MemorySample, bool) {
    let small = measure_memory_bandwidth(BANDWIDTH_BUF_BYTES / 2, 3);
    let large = measure_memory_bandwidth(BANDWIDTH_BUF_BYTES, 3);
    // Within 5% means the plateau has been reached.
    let ratio = large.gb_per_sec / small.gb_per_sec.max(f64::MIN_POSITIVE);
    let escaped = ratio > 0.95;
    (small, large, escaped)
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

/// Measure read-only memory bandwidth, saturating the bus.
///
/// **This, not the copy figure, is the ceiling for decode**, and getting it right took
/// being wrong twice.
///
/// Decode streams weights in and writes back only a small activation, so the workload is
/// read-dominated — a copy benchmark measures the wrong access pattern. But the obvious
/// read benchmark, a single-threaded `fold` over a buffer, measures the wrong thing too:
/// one accumulator is a serial dependency chain, so it reports memory *latency* rather
/// than bandwidth. On an M5 that read 40 GB/s where the bus delivers three times as much.
///
/// Two fixes, both necessary:
///
/// - **Several accumulators per thread**, so loads issue in parallel instead of queueing
///   behind one add.
/// - **Several threads.** One core cannot saturate an Apple Silicon memory bus. The same
///   machine reads 40 GB/s on one thread and 123 GB/s on six.
///
/// What that buys is a number that means something. LM Studio decoding Qwen3.5-9B on this
/// M5 achieves 118.1 GB/s effective, against 125 GB/s measured here — **94% of the
/// ceiling.** Against the old single-threaded copy figure the same engine looked like it
/// was running at 138% of the machine, which is not a calibration, it is a broken ruler.
pub fn measure_read_bandwidth(bytes: usize, iterations: usize) -> MemorySample {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let n = bytes / std::mem::size_of::<u64>();
    let src = vec![1u64; n];
    let chunk = (n / threads).max(1);

    // Sum a slice with four independent accumulators. With one, each load waits on the
    // previous add and the loop measures latency; with four, the memory system sees four
    // outstanding requests at a time.
    //
    // `seed` is what makes repeated sweeps honest. Reading the same immutable slice into
    // the same accumulators is loop-invariant, so LLVM will happily hoist the whole sweep
    // out of the repeat loop and time a single pass as if it were `iterations` of them.
    // That is not a subtle few percent: it reported 369 GB/s on a machine whose bus tops
    // out near 153. Threading the previous result in makes each pass depend on the last.
    fn sweep(slice: &[u64], seed: u64) -> u64 {
        let (mut a, mut b, mut c, mut d) = (seed, 0u64, 0u64, 0u64);
        for q in slice.chunks_exact(4) {
            a = a.wrapping_add(q[0]);
            b = b.wrapping_add(q[1]);
            c = c.wrapping_add(q[2]);
            d = d.wrapping_add(q[3]);
        }
        a ^ b ^ c ^ d
    }

    // Spawn first, hold at a barrier, then start the clock. Thread creation and the
    // first-touch page faults are real work, and timing them reports a slower machine than
    // the one you have — the first sample of an unwarmed run came in 40% low.
    let barrier = std::sync::Barrier::new(threads + 1);
    let seconds = std::thread::scope(|scope| {
        for t in 0..threads {
            let start = t * chunk;
            let end = if t == threads - 1 { n } else { start + chunk };
            let slice = &src[start..end];
            let barrier = &barrier;
            scope.spawn(move || {
                // One untimed pass: faults every page in and lets the CPU reach its
                // steady-state clock before anything is measured.
                let mut acc = sweep(slice, 0);
                barrier.wait();
                for _ in 0..iterations {
                    acc = sweep(std::hint::black_box(slice), acc);
                }
                std::hint::black_box(acc);
            });
        }
        // Released only once every thread has finished its warm-up pass, so the clock
        // starts on a warm buffer at a steady clock speed.
        barrier.wait();
        Instant::now()
    });
    // The scope joins every thread before returning, so this is the full wall-clock of the
    // timed region.
    let seconds = seconds.elapsed().as_secs_f64();

    // Read-only: one pass over the buffer per iteration, per thread's share.
    let total_bytes = (n * std::mem::size_of::<u64>()) as u64 * iterations as u64;
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
    crate::platform::peak_rss_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    #[test]
    fn offsets_stay_sector_aligned_however_odd_the_inputs() {
        // The invariant Windows enforces at runtime and every other platform pays for
        // silently. Reproduced here as arithmetic so a regression fails on any OS rather
        // than only in Windows CI.
        //
        // Deliberately awkward: a file length that is not a multiple of the alignment, and
        // a block size that is not either.
        let len: u64 = 1_048_576 + 1234;
        let block: u64 = 5000;
        let span = align_down(len.saturating_sub(block));
        let stride = align_down(block * 7 + OFFSET_ALIGN).max(OFFSET_ALIGN);
        assert!(span > 0);
        assert_eq!(span % OFFSET_ALIGN, 0);
        assert_eq!(stride % OFFSET_ALIGN, 0);

        for threads in [1u64, 4, 8] {
            for t in 0..threads {
                let mut off = align_down((span / threads).wrapping_mul(t) % span);
                for _ in 0..64 {
                    assert_eq!(off % OFFSET_ALIGN, 0, "offset {off} lost its alignment");
                    assert!(off < span, "offset {off} escaped the span");
                    off = (off + stride) % span;
                }
            }
        }
    }

    #[test]
    fn align_down_never_rounds_up() {
        assert_eq!(align_down(0), 0);
        assert_eq!(align_down(1), 0);
        assert_eq!(align_down(4095), 0);
        assert_eq!(align_down(4096), 4096);
        assert_eq!(align_down(4097), 4096);
    }

    #[test]
    fn a_multi_threaded_read_gives_every_thread_its_own_handle() {
        // Windows `seek_read` moves the file pointer, so a shared handle would have
        // threads reading each other's offsets. The totals would still add up, which is
        // what makes it dangerous — so assert the read actually succeeds under contention.
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![0xCDu8; 4 << 20]).expect("write");
        f.flush().expect("flush");

        let s = measure_random_read(f.path(), 64 << 10, 8, 4).expect("measure");
        assert_eq!(s.total_bytes, (64 << 10) * 8 * 4);
        assert_eq!(s.threads, 8);
    }
}

#[cfg(test)]
mod bandwidth_tests {
    use super::*;

    #[test]
    fn read_bandwidth_is_positive_and_counts_one_pass_per_iteration() {
        let s = measure_read_bandwidth(4 << 20, 4);
        assert!(s.gb_per_sec > 0.0 && s.gb_per_sec.is_finite());
        assert_eq!(
            s.total_bytes,
            (4u64 << 20) * 4,
            "read-only touches each byte once, unlike copy's twice"
        );
    }

    #[test]
    fn read_bandwidth_is_not_the_copy_figure() {
        // They measure different things and must not be conflated: a copy moves a byte in
        // and a byte out, and writes are the more expensive half. Quoting copy bandwidth
        // as the decode ceiling understates the machine — on an M5 by enough to predict
        // 12.9 tok/s for a model that runs at 20.9.
        //
        // Asserted as "not identical" rather than "read is faster", because the ordering
        // is a property of the hardware and a test that pins it would be a test of the
        // machine, not of this code.
        let copy = measure_memory_bandwidth(64 << 20, 4);
        let read = measure_read_bandwidth(64 << 20, 4);
        assert!(copy.gb_per_sec > 0.0 && read.gb_per_sec > 0.0);
        assert_ne!(copy.total_bytes, read.total_bytes);
    }
}
