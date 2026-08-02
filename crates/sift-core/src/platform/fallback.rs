//! Every other platform.
//!
//! Reached on the BSDs, illumos, and anything else with a Rust target and no module here.
//! The design rule is that an unported platform must **fail loudly at the point of use**
//! rather than quietly produce numbers. Reporting a plausible-looking zero would let
//! `sift fit` print a table that looks measured and is not — the exact failure mode this
//! crate is built to avoid.
//!
//! Reads still work, because `pread` is POSIX. Only the cache-bypass hint is missing, so
//! `open_uncached` degrades to a normal open; `doctor` independently flags any sample fast
//! enough to have come from RAM, so a cold measurement here reports as suspect rather than
//! as fast storage.

use super::{AccelMemory, MachineFacts};
use std::fs::File;
use std::io;
use std::path::Path;

pub fn machine_facts() -> MachineFacts {
    MachineFacts {
        // Zero, and honestly so: callers treat it as "unknown" and every regime becomes
        // Oversized, which reads as "sift cannot advise here" rather than a false answer.
        ram_bytes: 0,
        available_bytes: None,
        page_bytes: page_size(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        model: None,
        accel_memory: AccelMemory::Unknown,
    }
}

pub fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name has no preconditions; a negative return is
    // handled below.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        crate::io::ASSUMED_PAGE_SIZE
    }
}

pub fn peak_rss_bytes() -> u64 {
    0
}

/// No cache-bypass hint on an unported platform. A plain open, and `doctor`'s
/// page-cache detection is what keeps the resulting sample honest.
pub fn open_uncached(path: &Path) -> io::Result<File> {
    File::open(path)
}

pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}
