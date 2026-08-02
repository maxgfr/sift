//! Linux.
//!
//! Two things here are easy to get wrong, and both are recorded rather than assumed.
//!
//! **`MemFree` is not free memory.** It excludes reclaimable page cache, so on a machine
//! that has read any large file it understates what a new allocation can have — often by
//! most of RAM. `MemAvailable` is the kernel's own estimate of what a workload could take
//! without swapping. That is the question `sift` asks, so that is the field read.
//!
//! **`posix_fadvise(DONTNEED)` cannot evict dirty pages.** Hence the `fsync` first. It
//! also cannot evict pages another process has mapped, which is one more reason a "cold"
//! sample taken on a busy machine deserves the scepticism [`crate::doctor`] applies to it.

use super::{AccelMemory, MachineFacts};
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub fn machine_facts() -> MachineFacts {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    MachineFacts {
        ram_bytes: meminfo_kib(&meminfo, "MemTotal").unwrap_or(0),
        available_bytes: meminfo_kib(&meminfo, "MemAvailable"),
        page_bytes: page_size(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        model: dmi_product_name(),
        // No honest equivalent of the Apple wired limit. Discrete VRAM via NVML or amdgpu
        // sysfs is a different quantity — it is not a ceiling on host memory — and
        // pretending otherwise would be the mistake this crate exists to avoid.
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
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a live, correctly-typed local that `getrusage` fully initialises.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0;
    }
    // Linux reports maxrss in kilobytes, unlike Darwin's bytes.
    (usage.ru_maxrss as u64).saturating_mul(1024)
}

pub fn open_uncached(path: &Path) -> io::Result<File> {
    let file = File::open(path)?;
    let fd = file.as_raw_fd();

    // SAFETY: `fd` is a live descriptor owned by `file` for the duration of these calls.
    // All three are advisory and only affect caching; failures are deliberately ignored,
    // and `doctor` independently flags any sample fast enough to have been served from
    // RAM, so a silently-ineffective hint cannot turn into a published wrong number.
    unsafe {
        // Flush first: `DONTNEED` drops clean pages only, so anything dirty would survive.
        libc::fsync(fd);
        // Drop this file's pages from the page cache. Length 0 means "to end of file".
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        // Disable readahead — the analogue of Darwin's `F_RDAHEAD` off. Once we schedule
        // our own scattered reads, readahead spends bandwidth on a pattern it mispredicts.
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_RANDOM);
    }
    Ok(file)
}

pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

/// Read a `/proc/meminfo` field, converting its kibibytes to bytes.
///
/// Matches on `"{key}:"` rather than a prefix, so `MemTotal` cannot match `MemTotalFoo`
/// and — the case that actually exists — `MemFree` cannot be returned for a `MemAvailable`
/// query on a kernel that lists them adjacently.
fn meminfo_kib(meminfo: &str, key: &str) -> Option<u64> {
    let needle = format!("{key}:");
    meminfo
        .lines()
        .find_map(|l| l.strip_prefix(needle.as_str()))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

/// The machine's model name, as DMI reports it.
///
/// World-readable on every mainstream distribution, unlike `product_uuid` which is root
/// only. Absent in most containers and some VMs, where `None` is the honest answer.
fn dmi_product_name() -> Option<String> {
    let raw = std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name").ok()?;
    let name = raw.trim();
    // Firmware that was never configured leaves placeholder strings behind. Reporting
    // "To Be Filled By O.E.M." as a machine model is worse than reporting nothing.
    if name.is_empty()
        || name.eq_ignore_ascii_case("None")
        || name.to_ascii_lowercase().contains("to be filled")
        || name.to_ascii_lowercase().contains("system product name")
    {
        return None;
    }
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
MemTotal:       16333816 kB
MemFree:          204536 kB
MemAvailable:   12058112 kB
Buffers:            2048 kB
";

    #[test]
    fn meminfo_fields_are_converted_from_kib_to_bytes() {
        assert_eq!(meminfo_kib(SAMPLE, "MemTotal"), Some(16_333_816 * 1024));
        assert_eq!(meminfo_kib(SAMPLE, "MemAvailable"), Some(12_058_112 * 1024));
    }

    #[test]
    fn a_missing_field_is_none_rather_than_zero() {
        // Zero would read as "no memory" and make every model look oversized.
        assert_eq!(meminfo_kib(SAMPLE, "Hugepagesize"), None);
    }

    #[test]
    fn mem_available_is_not_confused_with_mem_free() {
        // The bug this guards: a prefix match on "Mem" or a `contains` would return
        // MemFree's 204 MB where MemAvailable's 12 GB was asked for — understating usable
        // memory by 60x and making everything look like it does not fit.
        let available = meminfo_kib(SAMPLE, "MemAvailable").unwrap();
        let free = meminfo_kib(SAMPLE, "MemFree").unwrap();
        assert!(available > free * 50, "must not have matched MemFree");
    }

    #[test]
    fn malformed_meminfo_does_not_panic() {
        assert_eq!(meminfo_kib("MemTotal: not-a-number kB", "MemTotal"), None);
        assert_eq!(meminfo_kib("", "MemTotal"), None);
        assert_eq!(meminfo_kib("MemTotal:", "MemTotal"), None);
    }
}
