//! macOS. The primary target, and the one whose quirks shaped the rest of the seam.

use super::{AccelMemory, MachineFacts};
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub fn machine_facts() -> MachineFacts {
    MachineFacts {
        ram_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
        // macOS exposes no single "available" figure. Reconstructing one from `vm_stat`
        // means deciding which of purgeable, compressed and inactive counts as available,
        // and that is a judgement call this crate would rather not make silently.
        available_bytes: None,
        page_bytes: page_size(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        model: sysctl_string("hw.model"),
        // `iogpu.wired_limit_mb` unset means macOS applies an internal default, itself
        // well below physical RAM. This is why an 11 GB model can fail to load on a 16 GB
        // machine: the limit, not the RAM, is what binds.
        accel_memory: match sysctl_u64("iogpu.wired_limit_mb").filter(|&mb| mb > 0) {
            Some(mb) => AccelMemory::Limited {
                bytes: mb * 1024 * 1024,
            },
            None => AccelMemory::PlatformDefault,
        },
    }
}

pub fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name has no preconditions; a negative return is
    // handled by the caller below.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        crate::io::ASSUMED_PAGE_SIZE
    }
}

pub fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a live, correctly-typed local that `getrusage` fully initialises.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0;
    }
    // Darwin reports maxrss in bytes, unlike Linux.
    usage.ru_maxrss as u64
}

pub fn open_uncached(path: &Path) -> io::Result<File> {
    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a live descriptor owned by `file` for the duration of these calls.
    // Both commands take an int argument and only affect caching policy; failure is
    // advisory, so the results are deliberately not propagated.
    unsafe {
        // Do not retain these pages in the unified buffer cache.
        libc::fcntl(fd, libc::F_NOCACHE, 1);
        // Disable kernel readahead. Once we schedule our own reads, readahead competes
        // with us and spends bandwidth on a pattern it mispredicts.
        libc::fcntl(fd, libc::F_RDAHEAD, 0);
    }
    Ok(file)
}

pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
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
