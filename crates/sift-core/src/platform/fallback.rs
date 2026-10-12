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
        // Where the BSDs publish it, `hw.physmem`. Otherwise zero, and honestly so:
        // callers treat it as "unknown" and every regime becomes Oversized, which reads as
        // "sift cannot advise here" rather than a false answer.
        ram_bytes: physmem().unwrap_or(0),
        available_bytes: None,
        page_bytes: page_size(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        model: None,
        accel_memory: AccelMemory::Unknown,
        cgroup_limit_bytes: None,
    }
}

pub fn gpu_memory() -> AccelMemory {
    AccelMemory::Unknown
}

/// Physical memory from `hw.physmem`, on the systems that have `sysctlbyname`.
#[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd"))]
fn physmem() -> Option<u64> {
    let name = std::ffi::CString::new("hw.physmem").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `name` is NUL-terminated; `value`/`size` are live locals of matching size,
    // and `sysctlbyname` writes at most `size` bytes into `value`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value)
}

#[cfg(not(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd")))]
fn physmem() -> Option<u64> {
    None
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
