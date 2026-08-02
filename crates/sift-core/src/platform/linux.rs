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
        // Linux has no equivalent of the Apple wired limit, so this is never `Limited`:
        // what the drivers report is a separate pool, and `AccelMemory` keeps the two
        // apart. See `discrete_vram`.
        accel_memory: discrete_vram(),
    }
}

/// Directory the kernel exposes one entry per DRM device under.
const DRM_CLASS: &str = "/sys/class/drm";

/// Proof the NVIDIA kernel module is loaded.
///
/// Checked before anything is spawned, so a machine with no NVIDIA driver — every AMD box,
/// every VM, every container — pays a `stat` and nothing more.
const NVIDIA_PROC: &str = "/proc/driver/nvidia/version";

/// Memory belonging to a discrete accelerator, as its driver reports it.
///
/// The largest device wins when there are several. Reporting the sum would describe one
/// enormous pool that no single model can be loaded into, and reporting the first would
/// depend on PCI enumeration order.
fn discrete_vram() -> AccelMemory {
    let amd = sysfs_vram_bytes(Path::new(DRM_CLASS)).map(|bytes| AccelMemory::Discrete {
        bytes,
        vendor: "amdgpu",
    });
    let nvidia = nvidia_vram_bytes().map(|bytes| AccelMemory::Discrete {
        bytes,
        vendor: "nvidia",
    });

    [amd, nvidia]
        .into_iter()
        .flatten()
        .max_by_key(|a| a.vram_bytes().unwrap_or(0))
        .unwrap_or(AccelMemory::Unknown)
}

/// Total VRAM from DRM sysfs, in bytes.
///
/// `mem_info_vram_total` is amdgpu's, and it is the honest source for that driver: a plain
/// integer the kernel publishes, world-readable, with no library to load and no process to
/// spawn. Intel's discrete parts expose the same file under `xe`.
///
/// Deliberately not the PCI BAR size from `/sys/bus/pci/devices/*/resource`, which looks
/// like it should work and does not: without resizable BAR it reports a 256 MiB aperture on
/// a 24 GB card.
///
/// Takes the root as an argument so the scan itself is testable against a fixture rather
/// than only on hardware nobody in CI has.
fn sysfs_vram_bytes(drm_root: &Path) -> Option<u64> {
    let entries = std::fs::read_dir(drm_root).ok()?;
    entries
        .flatten()
        // `/sys/class/drm` also holds one entry per connector — `card0-DP-1` and friends —
        // which carry no device memory file. Reading and discarding is simpler than
        // pattern-matching names that vary by driver.
        .filter_map(|e| std::fs::read_to_string(e.path().join("device/mem_info_vram_total")).ok())
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .filter(|&b| b > 0)
        .max()
}

/// Total VRAM of the largest NVIDIA device, in bytes.
///
/// # Why `nvidia-smi` and not NVML
///
/// NVML is the documented interface and would avoid a subprocess, but reaching it means
/// `dlopen`ing a proprietary shared library and calling through hand-written FFI — code no
/// machine here and no CI runner can execute, so it would ship unverified, and its failure
/// mode is a segfault in a tool whose entire pitch is that its numbers can be trusted.
/// `nvidia-smi` ships with the same driver, its failure mode is an empty string, and the
/// parsing is a pure function with tests.
///
/// Gated on the kernel module being loaded, so nothing is spawned on a machine without an
/// NVIDIA GPU.
fn nvidia_vram_bytes() -> Option<u64> {
    if !Path::new(NVIDIA_PROC).exists() {
        return None;
    }
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_smi_mib(&String::from_utf8_lossy(&out.stdout))
}

/// Largest device in `nvidia-smi --query-gpu=memory.total` output, converted to bytes.
///
/// One line per GPU, bare integers in MiB under `nounits`. A driver that cannot read a
/// device prints `[N/A]`, which parses as nothing rather than as zero.
fn parse_nvidia_smi_mib(out: &str) -> Option<u64> {
    out.lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .filter(|&mib| mib > 0)
        .max()
        .map(|mib| mib * 1024 * 1024)
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

    /// Build a fake `/sys/class/drm` holding the cards described.
    fn fake_drm(cards: &[(&str, Option<&str>)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp dir");
        for (name, vram) in cards {
            let dev = root.path().join(name).join("device");
            std::fs::create_dir_all(&dev).expect("mkdir");
            if let Some(v) = vram {
                std::fs::write(dev.join("mem_info_vram_total"), v).expect("write");
            }
        }
        root
    }

    #[test]
    fn amdgpu_vram_is_read_from_the_kernels_own_figure() {
        let root = fake_drm(&[("card0", Some("25753026560\n"))]);
        assert_eq!(sysfs_vram_bytes(root.path()), Some(25_753_026_560));
    }

    #[test]
    fn connectors_alongside_the_card_are_skipped_rather_than_failing_the_scan() {
        // `/sys/class/drm` holds `card0-DP-1`, `card0-HDMI-A-1` and so on next to the card
        // itself. An implementation that assumed every entry has the file would find
        // nothing on a machine that does have a GPU.
        let root = fake_drm(&[
            ("card0-DP-1", None),
            ("card0-HDMI-A-1", None),
            ("card0", Some("17163091968")),
        ]);
        assert_eq!(sysfs_vram_bytes(root.path()), Some(17_163_091_968));
    }

    #[test]
    fn the_largest_card_wins_rather_than_the_sum() {
        // Two cards are two pools. Summing them would report 32 GB that no single model can
        // be loaded into.
        let root = fake_drm(&[
            ("card0", Some("8589934592")),
            ("card1", Some("25769803776")),
        ]);
        assert_eq!(sysfs_vram_bytes(root.path()), Some(25_769_803_776));
    }

    #[test]
    fn a_machine_with_no_gpu_reports_nothing_rather_than_zero() {
        // Zero would read as "measured, and it has none", which is a different claim.
        let root = fake_drm(&[("card0", Some("0")), ("card1", None)]);
        assert_eq!(sysfs_vram_bytes(root.path()), None);
        assert_eq!(sysfs_vram_bytes(Path::new("/nonexistent/drm")), None);
    }

    #[test]
    fn nvidia_smi_output_is_read_in_mib_and_the_largest_device_wins() {
        assert_eq!(
            parse_nvidia_smi_mib("24564\n"),
            Some(24_564 * 1024 * 1024),
            "a single card"
        );
        assert_eq!(
            parse_nvidia_smi_mib("8192\n24564\n"),
            Some(24_564 * 1024 * 1024),
            "two cards are two pools, not one of 32 GB"
        );
    }

    #[test]
    fn an_unreadable_nvidia_device_is_not_counted_as_empty() {
        // `[N/A]` is what the driver prints for a device it cannot query — falling back to
        // zero would claim a measurement that failed.
        assert_eq!(parse_nvidia_smi_mib("[N/A]\n"), None);
        assert_eq!(parse_nvidia_smi_mib(""), None);
        assert_eq!(parse_nvidia_smi_mib("Failed to initialize NVML\n"), None);
        assert_eq!(
            parse_nvidia_smi_mib("[N/A]\n16384\n"),
            Some(16_384 * 1024 * 1024),
            "one bad device must not hide a good one"
        );
    }
}
