//! The per-OS seam.
//!
//! Everything that differs between macOS, Linux and Windows lives behind this module, so
//! the rest of the crate never sees a `cfg`. That boundary is load-bearing: `doctor` and
//! `io` are where the tool's honesty comes from, and honesty is easiest to audit when the
//! platform-specific parts are in one place with one shape.
//!
//! # What each OS has to answer
//!
//! - **How much memory may we actually use?** Never physical RAM. The OS, the compositor
//!   and the KV cache all take a share, and on Apple Silicon the GPU wired limit binds
//!   before physical RAM does.
//! - **How do we read a file without the page cache?** Required for a cold disk
//!   measurement to mean anything. Each OS spells this differently and none of them can
//!   evict pages that are already resident — see [`AccelMemory`] and
//!   [`crate::io::WeightFile::open`].
//! - **How much memory has this process actually used?** Used to check a measurement did
//!   not quietly balloon.
//!
//! # Two quantities that are never added together
//!
//! Accelerator memory is one word for two different things, and conflating them is the
//! mistake that makes competing tools untrustworthy — one of them maps every GPU below
//! 8 GB onto an 8 GB tier and then compares it against benchmark data recorded on 8 GB
//! hardware.
//!
//! - **A ceiling on host memory.** Apple's `iogpu.wired_limit_mb`: unified memory, so the
//!   limit binds what a model may occupy *in RAM*. [`AccelMemory::host_ceiling_bytes`]
//!   returns it, and `fits` is computed against it.
//! - **A discrete pool beside host memory.** NVIDIA and AMD VRAM. It is not a ceiling on
//!   anything the host allocates, and adding it to RAM would describe a machine nobody
//!   owns. [`AccelMemory::vram_bytes`] returns it, `doctor` prints it, and nothing else
//!   reads it — deliberately, until offload is modelled rather than guessed at.
//!
//! The type keeps them apart so the split survives contact with a future caller: there is
//! no accessor that returns "the accelerator's memory" without saying which kind.

use std::path::Path;

// Read only on Linux; compiled everywhere under test so its fixtures run on every OS.
#[cfg(any(target_os = "linux", test))]
mod cgroup;
#[cfg(target_os = "macos")]
mod darwin;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "macos")]
use darwin as imp;
#[cfg(target_os = "linux")]
use linux as imp;
#[cfg(windows)]
use windows as imp;

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
mod fallback;
#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
use fallback as imp;

/// What the OS will tell us about itself, before we measure anything.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MachineFacts {
    /// Physical RAM in bytes, or 0 if the OS would not say.
    pub ram_bytes: u64,
    /// Memory the OS says is available right now, where it will tell us.
    ///
    /// Distinct from free memory, and the distinction matters: Linux `MemFree` excludes
    /// reclaimable page cache and so understates what a new allocation can have, often by
    /// most of RAM. `MemAvailable` is the kernel's own estimate of what a workload could
    /// take without swapping, which is the question being asked here.
    ///
    /// `None` on macOS, which exposes no single equivalent figure.
    pub available_bytes: Option<u64>,
    /// OS page size in bytes. 16 KiB on Apple Silicon, 4 KiB most elsewhere.
    pub page_bytes: usize,
    /// Logical CPU count.
    pub cpus: usize,
    /// Hardware model identifier, e.g. `Mac17,2`.
    pub model: Option<String>,
    /// Ceiling on memory an accelerator may hold, where the platform defines one.
    pub accel_memory: AccelMemory,
    /// A memory limit imposed on this process's control group, when one binds below RAM.
    ///
    /// A container started with `--memory=2g` on a 64 GB host sees all 64 GB in
    /// `/proc/meminfo`; this is the 2 GB it may actually use. Already folded into
    /// `ram_bytes` and `available_bytes`, and reported so `doctor` can say why they are
    /// smaller than the machine.
    pub cgroup_limit_bytes: Option<u64>,
}

/// What we know about this machine's accelerator memory.
///
/// Five states, not two, because "we did not look", "the platform caps it and says by how
/// much", "the platform caps it and does not say", "a separate pool" and "a GPU that shares
/// system memory" lead to different advice. Collapsing them into `Option<u64>` would let
/// the tool imply a measurement it never made, or — worse — add a graphics card's VRAM to a
/// laptop's RAM and call the sum usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AccelMemory {
    /// A hard ceiling on host memory the platform reports, in bytes. On macOS this is
    /// `iogpu.wired_limit_mb`, which binds well before physical RAM does.
    Limited { bytes: u64 },
    /// The platform defines such a ceiling but it is unset, so an OS default applies —
    /// itself below physical RAM, and not published. `estimated_bytes` is sift's estimate
    /// of it, and is always reported as an estimate. macOS only.
    PlatformDefault { estimated_bytes: u64 },
    /// A discrete accelerator with its own memory, which is **not** a ceiling on host
    /// memory. Reported so the user knows `fits` is conservative on this machine, and
    /// named so they can check the detection.
    Discrete { bytes: u64, vendor: &'static str },
    /// An integrated GPU (an AMD APU, Intel graphics, a basic display adapter). What the
    /// driver reports is a carve-out of system memory, not memory of its own, so it is
    /// neither VRAM nor a ceiling.
    Integrated { bytes: u64, vendor: &'static str },
    /// Nothing found. Callers fall back to RAM minus a reserve, and should say so rather
    /// than implying a measurement.
    Unknown,
}

impl AccelMemory {
    /// The ceiling on **host** memory, when the platform imposes one.
    ///
    /// `None` for [`AccelMemory::Discrete`], and that is the whole point of the split: a
    /// 24 GB card on a 16 GB machine does not let a 20 GB model fit in RAM. Deciding what
    /// an engine can offload to a discrete GPU is a different calculation — one that
    /// depends on the engine, the layer split and the KV cache location — and until `sift`
    /// models it, saying nothing is the honest answer.
    ///
    /// For [`AccelMemory::PlatformDefault`] this is the estimate: macOS applies a default
    /// below physical RAM whether or not it says what it is, and planning against all of
    /// RAM would be planning against memory the GPU cannot wire.
    pub fn host_ceiling_bytes(self) -> Option<u64> {
        match self {
            AccelMemory::Limited { bytes } => Some(bytes),
            AccelMemory::PlatformDefault { estimated_bytes } => Some(estimated_bytes),
            _ => None,
        }
    }

    /// Memory belonging to a discrete accelerator, where one was found.
    ///
    /// Informational. Nothing in the fit arithmetic reads this. An integrated GPU has none.
    pub fn vram_bytes(self) -> Option<u64> {
        match self {
            AccelMemory::Discrete { bytes, .. } => Some(bytes),
            _ => None,
        }
    }

    /// Whether the ceiling is sift's estimate rather than a figure the OS reported.
    pub fn is_estimate(self) -> bool {
        matches!(self, AccelMemory::PlatformDefault { .. })
    }
}

/// The GPU wired-memory ceiling on Apple Silicon, from what the sysctls say.
///
/// `wired_limit_mb` is `iogpu.wired_limit_mb`; `legacy_limit_bytes` is the
/// `debug.iogpu.wired_limit` older releases used. Neither existing means no Apple GPU to
/// limit — an Intel Mac — and that is `Unknown`. A key that exists and reads 0 means the
/// OS default applies, which Apple does not publish: by observation it is about two thirds
/// of RAM up to 36 GiB and three quarters above, so that is the estimate. A set limit
/// above physical RAM binds nothing beyond RAM, so it is capped there.
pub fn apple_wired_limit(
    wired_limit_mb: Option<u64>,
    legacy_limit_bytes: Option<u64>,
    ram_bytes: u64,
) -> AccelMemory {
    let set = match (wired_limit_mb, legacy_limit_bytes) {
        (Some(mb), _) => Some(mb.saturating_mul(1024 * 1024)),
        (None, Some(bytes)) => Some(bytes),
        (None, None) => return AccelMemory::Unknown,
    };
    match set {
        Some(bytes) if bytes > 0 => AccelMemory::Limited {
            bytes: if ram_bytes > 0 {
                bytes.min(ram_bytes)
            } else {
                bytes
            },
        },
        _ => AccelMemory::PlatformDefault {
            estimated_bytes: apple_default_wired_limit(ram_bytes),
        },
    }
}

/// Whether a Windows display adapter's `DriverDesc` names an integrated GPU.
///
/// Intel graphics other than Arc, AMD's APU "Radeon(TM) Graphics" (no model number), and
/// the basic display adapter Windows falls back to. Their "dedicated memory" is a slice of
/// system RAM, and reporting it as VRAM would tell a laptop it has a graphics card.
pub fn is_integrated_adapter(desc: &str) -> bool {
    let d = desc.to_ascii_lowercase();
    (d.contains("intel") && !d.contains("arc"))
        || d.contains("basic display")
        || d.contains("basic render")
        || d.trim_end().ends_with("radeon(tm) graphics")
        || d.trim_end().ends_with("radeon graphics")
}

/// macOS's unpublished default GPU wired limit, estimated: ⅔ of RAM up to 36 GiB, ¾ above.
pub fn apple_default_wired_limit(ram_bytes: u64) -> u64 {
    if ram_bytes <= 36 * crate::GIB {
        ram_bytes / 3 * 2
    } else {
        ram_bytes / 4 * 3
    }
}

impl MachineFacts {
    /// Collect what the OS will tell us without running any benchmark.
    ///
    /// Discrete GPU memory is not looked for here: nothing that plans a fit reads it, and
    /// finding it can mean spawning `nvidia-smi`. See [`Self::collect_with_vram`].
    pub fn collect() -> Self {
        imp::machine_facts()
    }

    /// [`Self::collect`], plus a look for a discrete or integrated GPU's memory.
    ///
    /// For `doctor`, which reports it.
    pub fn collect_with_vram() -> Self {
        let mut facts = Self::collect();
        if facts.accel_memory == AccelMemory::Unknown {
            facts.accel_memory = imp::gpu_memory();
        }
        facts
    }
}

/// Query the OS page size at runtime.
///
/// Never hardcode this: Apple Silicon is 16 KiB where most of the world is 4 KiB, and
/// alignment requirements downstream are expressed in real pages.
pub fn page_size() -> usize {
    imp::page_size()
}

/// Peak resident set size of this process so far, in bytes.
///
/// Returns 0 when the platform will not say. Used to check that a measurement did not
/// quietly balloon, so a zero reads as "unknown", never as "nothing was used".
pub fn peak_rss_bytes() -> u64 {
    imp::peak_rss_bytes()
}

/// Open a file for positional reads, asking the OS not to cache the pages.
///
/// # The caveat that matters
///
/// Every platform's spelling of this — `F_NOCACHE`, `posix_fadvise(DONTNEED)`,
/// `FILE_FLAG_NO_BUFFERING` — stops *new* caching but **cannot evict pages already
/// resident**. Opening a file the machine just read and calling the result "cold" is the
/// single most common way to publish a wrong number: you measure RAM and report it as
/// storage. Use a file the machine has not touched, or reboot.
pub fn open_uncached(path: &Path) -> std::io::Result<std::fs::File> {
    imp::open_uncached(path)
}

/// Fraction of a file's pages that are resident in the page cache right now.
///
/// What every "uncached" open above cannot do — evict — this at least detects: a disk
/// sample taken from a file that is already in memory measures RAM, and `doctor` uses this
/// to say so instead of publishing it. `None` where the platform cannot tell (Windows) or
/// the file cannot be mapped.
pub fn resident_fraction(path: &Path) -> Option<f64> {
    #[cfg(unix)]
    {
        unix_resident_fraction(path)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// `mmap` + `mincore`, which Linux, macOS and the BSDs all have.
///
/// The mapping is never touched, so it faults nothing in: `mincore` only reports which
/// pages are already there.
#[cfg(unix)]
fn unix_resident_fraction(path: &Path) -> Option<f64> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(path).ok()?;
    let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if len == 0 {
        return None;
    }
    let page = page_size();
    let pages = len.div_ceil(page);
    let mut vec = vec![0u8; pages];
    // SAFETY: mapping a regular file read-only and shared; the pointer is checked against
    // MAP_FAILED, never dereferenced, and unmapped before returning. `vec` holds one byte
    // per page of the mapping, which is what `mincore` writes.
    unsafe {
        let addr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        );
        if addr == libc::MAP_FAILED {
            return None;
        }
        let rc = libc::mincore(addr, len, vec.as_mut_ptr() as *mut _);
        libc::munmap(addr, len);
        if rc != 0 {
            return None;
        }
    }
    let resident = vec.iter().filter(|&&b| b & 1 != 0).count();
    Some(resident as f64 / pages as f64)
}

/// Open a file for positional reads with normal caching.
pub fn open_cached(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(path)
}

/// Read exactly `buf.len()` bytes at `offset`, without disturbing any shared file cursor.
///
/// Unix gets this for free from `pread`. Windows does not: `seek_read` moves the handle's
/// file pointer, so two threads sharing one handle would corrupt each other's offsets.
/// Callers must therefore give **each thread its own handle** — see
/// [`crate::io::WeightFile::try_clone`] — rather than assume this call is safe to share.
pub fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    imp::read_exact_at(file, buf, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_reports_something_plausible_about_itself() {
        let f = MachineFacts::collect();
        assert!(f.ram_bytes > 0, "physical RAM must be discoverable");
        assert!(f.cpus >= 1);
        assert!(f.page_bytes.is_power_of_two());
        assert!(
            (4096..=65536).contains(&f.page_bytes),
            "page size {} outside the plausible range",
            f.page_bytes
        );
    }

    #[test]
    fn a_file_just_written_is_reported_resident() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&vec![1u8; 1 << 20]).expect("write");
        f.flush().expect("flush");
        if let Some(frac) = resident_fraction(f.path()) {
            assert!(
                frac > 0.5,
                "a freshly written file is in the page cache: {frac}"
            );
        }
    }

    #[test]
    fn an_unknown_accelerator_ceiling_yields_no_bytes() {
        // The distinction this guards: `Unknown` must never be mistaken for a measured 0,
        // which would make every model look oversized.
        assert_eq!(AccelMemory::Unknown.host_ceiling_bytes(), None);
        assert_eq!(
            AccelMemory::Limited { bytes: 42 }.host_ceiling_bytes(),
            Some(42)
        );
    }

    #[test]
    fn discrete_vram_is_never_offered_as_a_host_memory_ceiling() {
        // The failure this prevents: a 24 GB card on a 16 GB machine reporting that a 20 GB
        // model fits in RAM. Discrete memory is a separate pool, so it answers the VRAM
        // question and refuses the host one.
        let card = AccelMemory::Discrete {
            bytes: 24 * crate::GIB,
            vendor: "nvidia",
        };
        assert_eq!(card.host_ceiling_bytes(), None, "VRAM is not host memory");
        assert_eq!(card.vram_bytes(), Some(24 * crate::GIB));
    }

    #[test]
    fn an_unset_apple_wired_limit_is_an_estimate_below_ram() {
        // Regression: an unset limit used to fall back to RAM minus 4 GiB — 12 GiB on a
        // 16 GiB Mac whose GPU can wire about 10.7.
        let ram = 16 * crate::GIB;
        let a = apple_wired_limit(Some(0), None, ram);
        assert!(a.is_estimate());
        let est = a
            .host_ceiling_bytes()
            .expect("an estimate is still a ceiling");
        assert_eq!(est, ram / 3 * 2);
        assert!((crate::gib(est) - 10.67).abs() < 0.01);
        // Above 36 GiB the default is three quarters.
        assert_eq!(apple_default_wired_limit(64 * crate::GIB), 48 * crate::GIB);
    }

    #[test]
    fn an_absent_wired_limit_sysctl_is_unknown_not_a_default() {
        // An Intel Mac has no Apple GPU, so there is no wired limit to estimate.
        assert_eq!(
            apple_wired_limit(None, None, 16 * crate::GIB),
            AccelMemory::Unknown
        );
        // Older releases spell it `debug.iogpu.wired_limit`, in bytes.
        assert_eq!(
            apple_wired_limit(None, Some(8 * crate::GIB), 16 * crate::GIB),
            AccelMemory::Limited {
                bytes: 8 * crate::GIB
            }
        );
    }

    #[test]
    fn a_wired_limit_above_ram_is_capped_at_ram() {
        assert_eq!(
            apple_wired_limit(Some(64 * 1024), None, 16 * crate::GIB),
            AccelMemory::Limited {
                bytes: 16 * crate::GIB
            }
        );
    }

    #[test]
    fn windows_adapter_names_are_told_apart() {
        assert!(is_integrated_adapter("Intel(R) UHD Graphics 770"));
        assert!(is_integrated_adapter("Intel(R) Iris(R) Xe Graphics"));
        assert!(is_integrated_adapter("AMD Radeon(TM) Graphics"));
        assert!(is_integrated_adapter("Microsoft Basic Display Adapter"));
        assert!(!is_integrated_adapter("Intel(R) Arc(TM) A770 Graphics"));
        assert!(!is_integrated_adapter("NVIDIA GeForce RTX 4090"));
        assert!(!is_integrated_adapter("AMD Radeon RX 7900 XTX"));
    }

    #[test]
    fn integrated_gpu_memory_is_neither_vram_nor_a_ceiling() {
        let igpu = AccelMemory::Integrated {
            bytes: 512 << 20,
            vendor: "amdgpu",
        };
        assert_eq!(igpu.vram_bytes(), None);
        assert_eq!(igpu.host_ceiling_bytes(), None);
    }

    #[test]
    fn an_apple_wired_limit_is_not_reported_as_vram() {
        // The mirror image: unified memory the GPU may wire is host memory, so it must not
        // appear as a separate pool that could be added to it.
        let apple = AccelMemory::Limited {
            bytes: 12 * crate::GIB,
        };
        assert_eq!(apple.vram_bytes(), None);
        assert_eq!(AccelMemory::Unknown.vram_bytes(), None);
    }
}
