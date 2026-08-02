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
}

/// What we know about this machine's accelerator memory.
///
/// Four states, not two, because "we did not look", "we looked and there is no limit" and
/// "there is a separate pool that is not a limit at all" lead to different advice.
/// Collapsing them into `Option<u64>` would let the tool imply a measurement it never made,
/// or — worse — add a graphics card's VRAM to a laptop's RAM and call the sum usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AccelMemory {
    /// A hard ceiling on host memory the platform reports, in bytes. On macOS this is
    /// `iogpu.wired_limit_mb`, which binds well before physical RAM does.
    Limited { bytes: u64 },
    /// The platform defines such a ceiling but it is unset, so an OS default applies —
    /// itself below physical RAM. macOS only.
    PlatformDefault,
    /// A discrete accelerator with its own memory, which is **not** a ceiling on host
    /// memory. Reported so the user knows `fits` is conservative on this machine, and
    /// named so they can check the detection.
    Discrete { bytes: u64, vendor: &'static str },
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
    pub fn host_ceiling_bytes(self) -> Option<u64> {
        match self {
            AccelMemory::Limited { bytes } => Some(bytes),
            _ => None,
        }
    }

    /// Memory belonging to a discrete accelerator, where one was found.
    ///
    /// Informational. Nothing in the fit arithmetic reads this.
    pub fn vram_bytes(self) -> Option<u64> {
        match self {
            AccelMemory::Discrete { bytes, .. } => Some(bytes),
            _ => None,
        }
    }
}

impl MachineFacts {
    /// Collect what the OS will tell us without running any benchmark.
    pub fn collect() -> Self {
        imp::machine_facts()
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
    fn an_unknown_accelerator_ceiling_yields_no_bytes() {
        // The distinction this guards: `Unknown` must never be mistaken for a measured 0,
        // which would make every model look oversized.
        assert_eq!(AccelMemory::Unknown.host_ceiling_bytes(), None);
        assert_eq!(AccelMemory::PlatformDefault.host_ceiling_bytes(), None);
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
