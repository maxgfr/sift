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
    /// Mean time one read took, in milliseconds, as a thread waiting on it saw it.
    pub ms_per_read: f64,
    /// Fraction of the sample file already in the page cache before this measurement,
    /// where the platform can tell.
    pub resident_before: Option<f64>,
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
            || self
                .resident_before
                .is_some_and(|f| f > RESIDENT_SAMPLE_LIMIT)
    }
}

/// Fraction of a sample file that may already be cached before a disk read stops being a
/// disk measurement.
///
/// The throughput test above catches a fully cached file. This catches the partly cached
/// one, whose reads are a blend of RAM and SSD and read as a fast, plausible drive.
pub const RESIDENT_SAMPLE_LIMIT: f64 = 0.10;

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
/// caching but cannot evict what is already there, so the file's residency is checked
/// first and recorded on the sample.
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
    let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, msg);
    if threads == 0 || reads_per_thread == 0 {
        return Err(invalid(format!(
            "{threads} threads x {reads_per_thread} reads measures nothing"
        )));
    }
    // Unbuffered reads (O_DIRECT, FILE_FLAG_NO_BUFFERING) refuse a length that is not a
    // sector multiple, so refuse it here, on every OS, with a message.
    if block_bytes == 0 || block_bytes as u64 % OFFSET_ALIGN != 0 {
        return Err(invalid(format!(
            "block size {block_bytes} is not a multiple of {OFFSET_ALIGN}"
        )));
    }

    let resident_before = crate::platform::resident_fraction(path);
    let file = WeightFile::open(path, CachePolicy::Uncached)?;
    let len = file.len();

    // Offsets are held to a sector multiple. Windows rejects an unaligned unbuffered read
    // outright, and on every other platform a read straddling a block boundary silently
    // costs an extra I/O — so the sample would describe a pattern nobody asked to measure.
    let span = (len.saturating_sub(block_bytes as u64) / OFFSET_ALIGN) * OFFSET_ALIGN;
    if span == 0 {
        return Err(invalid(format!(
            "file is {len} bytes, too small for a {block_bytes}-byte block; \
             use a larger sample file"
        )));
    }

    // Every handle is opened before any thread starts. Opening inside the spawn loop left
    // the threads already started waiting forever at the barrier when a later open failed.
    // A handle each, not a shared one: Windows `seek_read` moves the file pointer, so
    // threads sharing a handle would read each other's offsets.
    let handles: Vec<WeightFile> = (0..threads)
        .map(|_| file.try_clone())
        .collect::<std::io::Result<_>>()?;

    // Spawning threads costs tens of microseconds each, so every thread is spawned first,
    // held at a barrier, and only then starts. Each worker records when its own first
    // read was issued and its last returned; the sample spans the earliest start to the
    // latest end. The main thread holds no clock at all — it can be descheduled while the
    // workers run, and its view of the start is not when the reads began.
    let barrier = std::sync::Barrier::new(threads);
    let results: Vec<std::io::Result<(Instant, Instant, u64)>> = std::thread::scope(|scope| {
        let workers: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(t, file)| {
                let barrier = &barrier;
                scope.spawn(move || -> std::io::Result<(Instant, Instant, u64)> {
                    let mut buf = AlignedBuf::new(block_bytes);
                    let mut read = 0u64;
                    // A large odd stride walks the file without repeating quickly and
                    // without degenerating into a sequential scan readahead could predict.
                    // Each thread starts a different fraction of the way in so they do
                    // not convoy.
                    let stride =
                        align_down(block_bytes as u64 * 7 + OFFSET_ALIGN).max(OFFSET_ALIGN);
                    let start = (span / threads as u64).wrapping_mul(t as u64) % span;
                    let mut off = align_down(start);

                    // Allocation and first-touch of `buf` are done; wait for everyone.
                    barrier.wait();
                    let began = Instant::now();
                    for _ in 0..reads_per_thread {
                        file.read_at(buf.as_mut_slice(), off)?;
                        read += block_bytes as u64;
                        off = (off + stride) % span;
                    }
                    Ok((began, Instant::now(), read))
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                w.join()
                    .unwrap_or_else(|_| Err(std::io::Error::other("reader thread panicked")))
            })
            .collect()
    });

    let mut spans = Vec::with_capacity(threads);
    let mut total_bytes = 0u64;
    for r in results {
        let (began, ended, bytes) = r?;
        spans.push((began, ended));
        total_bytes += bytes;
    }
    let seconds = wall_seconds(&spans);
    Ok(DiskSample {
        block_bytes,
        threads,
        total_bytes,
        seconds,
        gb_per_sec: total_bytes as f64 / seconds / 1e9,
        // Each thread issues its reads one after another, so one read takes the whole
        // span divided by that thread's count — not by every thread's, which would be the
        // inverse of throughput and shrink as threads were added.
        ms_per_read: seconds * 1000.0 / reads_per_thread as f64,
        resident_before,
    })
}

/// Seconds from the earliest start to the latest end of a set of worker spans.
///
/// That is the wall clock of the parallel region seen from inside it. Any scheduling gap
/// before a worker starts or after one finishes only widens it, so noise can understate
/// the machine but never overstate it.
fn wall_seconds(spans: &[(Instant, Instant)]) -> f64 {
    let (Some(first), Some(last)) = (
        spans.iter().map(|s| s.0).min(),
        spans.iter().map(|s| s.1).max(),
    ) else {
        return 0.0;
    };
    // A zero-length region would divide to infinity; a nanosecond is below any real read.
    last.duration_since(first).as_secs_f64().max(1e-9)
}

/// Buffer size that reliably escapes the last-level cache.
///
/// This has to clear far more than an L3. Apple's System Level Cache sits behind the CPU
/// caches and is reported at ~96 MiB on M4 Max; AMD X3D parts carry 96–128 MiB of L3, and
/// server chips reach 384 MiB. A 256 MiB buffer is under 3× an Apple SLC — close enough
/// that the measurement is partly cache bandwidth, which reads as an implausibly fast
/// machine.
///
/// 512 MiB is over 5× the largest consumer LLC.
pub const BANDWIDTH_BUF_BYTES: usize = 512 << 20;

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
/// Three fixes, all necessary:
///
/// - **Several accumulators per thread**, so loads issue in parallel instead of queueing
///   behind one add.
/// - **Several threads.** One core cannot saturate an Apple Silicon memory bus. The same
///   machine reads 40 GB/s on one thread and 123 GB/s on six.
/// - **A clock that runs inside the work.** See below; this is the one that took longest to
///   find, because the number it produces is not merely wrong, it is impossible.
///
/// # The clock has to be held by a thread that is doing the work
///
/// This function used to release a barrier, call `Instant::now()` on the main thread, and
/// treat that as the start of the parallel region. On an idle machine that reads fine. It
/// also reported **372 GB/s on a bus that tops out near 153**, run to run, from the same
/// binary — because the main thread has nothing to do at the barrier and the OS is free to
/// deschedule it for several milliseconds while ten worker threads saturate memory. Those
/// milliseconds fall outside the timed window, and the whole window is only ~12 ms.
///
/// Each worker now records its own start and end, and the sample spans the earliest start
/// to the latest end: the wall clock of the parallel region as measured from inside it.
/// Taking the slowest worker's own duration instead — what this did next — still missed
/// any skew between workers' starts, and so could overstate the machine.
///
/// The lesson generalises to the rest of this module. A wrong number here is not caught by
/// any test that asks whether the code runs — only by asking whether the answer is possible.
pub fn measure_read_bandwidth(bytes: usize, iterations: usize) -> std::io::Result<MemorySample> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let n = bytes / std::mem::size_of::<u64>();
    if iterations == 0 || n < threads * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "{bytes} bytes x {iterations} passes over {threads} threads is too small to \
                 measure"
            ),
        ));
    }
    let chunk = n / threads;

    // Sum a slice with four independent accumulators. With one, each load waits on the
    // previous add and the loop measures latency; with four, the memory system sees four
    // outstanding requests at a time.
    //
    // `seed` is what makes repeated sweeps honest. Reading the same immutable slice into
    // the same accumulators is loop-invariant, so LLVM will happily hoist the sweep out of
    // the repeat loop and time one pass as if it were `iterations` of them. **Every**
    // accumulator is seeded from it, not just the first: seeding one and leaving three at
    // zero leaves three quarters of the work invariant and hoistable, which is the same
    // bug wearing a smaller number.
    fn sweep(slice: &[u64], seed: u64) -> u64 {
        let (mut a, mut b, mut c, mut d) = (
            seed,
            seed ^ 0x9e37_79b9_7f4a_7c15,
            seed.rotate_left(17),
            !seed,
        );
        for q in slice.chunks_exact(4) {
            a = a.wrapping_add(q[0]);
            b = b.wrapping_add(q[1]);
            c = c.wrapping_add(q[2]);
            d = d.wrapping_add(q[3]);
        }
        a ^ b ^ c ^ d
    }

    // Each worker allocates and first-touches its own segment, once, so on a NUMA machine
    // the pages land on the node of the core reading them — a buffer filled by the main
    // thread sits on one node and every other node reads it across the interconnect. The
    // segments outlive the three samples below, so the repeats measure memory rather than
    // the allocator.
    let segments: Vec<Vec<u64>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                let len = if t == threads - 1 {
                    n - chunk * (threads - 1)
                } else {
                    chunk
                };
                scope.spawn(move || vec![1u64; len])
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("allocation worker panicked"))
            .collect()
    });

    // One pass of the whole measurement. Every worker warms up before any starts its
    // clock, then records its own start and end; the sample spans the earliest start to
    // the latest end. See [`wall_seconds`].
    let sample = || -> f64 {
        let barrier = std::sync::Barrier::new(threads);
        let spans: Vec<(Instant, Instant)> = std::thread::scope(|scope| {
            let workers: Vec<_> = segments
                .iter()
                .map(|segment| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        // One untimed pass: faults every page in and lets the CPU reach its
                        // steady-state clock before anything is measured.
                        let mut acc = sweep(segment, 0);
                        barrier.wait();
                        let began = Instant::now();
                        for _ in 0..iterations {
                            acc = sweep(std::hint::black_box(segment), acc);
                        }
                        let ended = Instant::now();
                        std::hint::black_box(acc);
                        (began, ended)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|h| h.join().expect("bandwidth worker panicked"))
                .collect()
        });
        wall_seconds(&spans)
    };

    // Median of three, because one sample is one roll of the scheduler. The timed region is
    // ~12 ms; a background process waking inside it drags that sample down, and a single
    // low read halves every tok/s figure the tool prints from it.
    let mut samples = [sample(), sample(), sample()];
    samples.sort_by(|a, b| a.total_cmp(b));
    let seconds = samples[1];

    // Read-only: one pass over the buffer per iteration, per thread's share.
    let total_bytes = (n * std::mem::size_of::<u64>()) as u64 * iterations as u64;
    Ok(MemorySample {
        total_bytes,
        seconds,
        gb_per_sec: total_bytes as f64 / seconds / 1e9,
    })
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
    fn nonsense_parameters_are_refused_instead_of_panicking_or_dividing_by_zero() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![0u8; 1 << 20]).expect("write");
        f.flush().expect("flush");
        for (block, threads, reads) in [(4096, 0, 8), (4096, 2, 0), (5000, 2, 8), (0, 1, 1)] {
            let err =
                measure_random_read(f.path(), block, threads, reads).expect_err("must refuse");
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "{block} {threads} {reads}"
            );
        }
        assert_eq!(
            measure_read_bandwidth(64, 4).expect_err("too small").kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(
            measure_read_bandwidth(4 << 20, 0).is_err(),
            "zero passes is 0/0"
        );
    }

    #[test]
    fn a_missing_sample_file_is_an_error_and_starts_no_thread() {
        let err = measure_random_read(Path::new("/nonexistent/sift-sample"), 4096, 4, 4)
            .expect_err("no file");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn the_wall_clock_spans_the_earliest_start_to_the_latest_end() {
        // Two workers that each ran 10 ms, 5 ms apart: the region took 15 ms, and taking
        // either worker's own 10 ms would overstate the machine by half.
        let t0 = Instant::now();
        let ms = std::time::Duration::from_millis;
        let spans = [(t0, t0 + ms(10)), (t0 + ms(5), t0 + ms(15))];
        assert!((wall_seconds(&spans) - 0.015).abs() < 1e-9);
        assert_eq!(wall_seconds(&[]), 0.0);
    }

    #[test]
    fn latency_per_read_does_not_shrink_as_threads_are_added() {
        // Regression: ms_per_read divided by every thread's reads, so it was the inverse of
        // throughput and halved each time threads doubled.
        let s = DiskSample {
            block_bytes: 4096,
            threads: 8,
            total_bytes: 0,
            seconds: 1.0,
            gb_per_sec: 0.0,
            ms_per_read: 1000.0 / 100.0,
            resident_before: None,
        };
        assert_eq!(s.ms_per_read, 10.0);
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![0u8; 1 << 20]).expect("write");
        f.flush().expect("flush");
        let one = measure_random_read(f.path(), 4096, 1, 16).expect("one");
        assert!((one.ms_per_read - one.seconds * 1000.0 / 16.0).abs() < 1e-9);
    }

    #[test]
    fn a_sample_already_in_the_page_cache_is_flagged() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![7u8; 4 << 20]).expect("write");
        f.flush().expect("flush");
        let s = measure_random_read(f.path(), 64 << 10, 1, 4).expect("measure");
        if s.resident_before.is_some() {
            assert!(
                s.looks_cached(),
                "just written, so resident: {:?}",
                s.resident_before
            );
        }
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
        let s = measure_read_bandwidth(4 << 20, 4).expect("measure");
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
        let read = measure_read_bandwidth(64 << 20, 4).expect("measure");
        assert!(copy.gb_per_sec > 0.0 && read.gb_per_sec > 0.0);
        assert_ne!(copy.total_bytes, read.total_bytes);
    }
}
