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
//! # What is deliberately not abstracted
//!
//! Accelerator memory. `gpu_wired_limit_bytes` is an Apple concept — a ceiling on unified
//! memory the GPU may wire. There is no honest Linux or Windows equivalent to fill it
//! with, so those platforms report [`AccelMemory::Unknown`] and the caller falls back to
//! `RAM − reserve`. Reporting a guess here is exactly the mistake that makes competing
//! tools untrustworthy: one of them maps every GPU below 8 GB onto an 8 GB tier and then
//! compares it against benchmark data recorded on 8 GB hardware.

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

/// What we know about the memory ceiling an accelerator imposes.
///
/// Three states, not two, because "we did not look" and "we looked and there is no limit"
/// lead to different advice, and collapsing them into `None` would let the tool imply a
/// measurement it never made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AccelMemory {
    /// A hard ceiling the platform reports, in bytes. On macOS this is
    /// `iogpu.wired_limit_mb`, which binds well before physical RAM does.
    Limited { bytes: u64 },
    /// The platform defines such a ceiling but it is unset, so an OS default applies —
    /// itself below physical RAM. macOS only.
    PlatformDefault,
    /// This platform exposes no ceiling we can read. Callers must fall back to RAM minus a
    /// reserve, and should say so rather than implying a measurement.
    Unknown,
}

impl AccelMemory {
    /// The ceiling in bytes, when one is actually known.
    pub fn bytes(self) -> Option<u64> {
        match self {
            AccelMemory::Limited { bytes } => Some(bytes),
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
        assert_eq!(AccelMemory::Unknown.bytes(), None);
        assert_eq!(AccelMemory::PlatformDefault.bytes(), None);
        assert_eq!(AccelMemory::Limited { bytes: 42 }.bytes(), Some(42));
    }
}
