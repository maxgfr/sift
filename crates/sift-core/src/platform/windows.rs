//! Windows.
//!
//! Three traps here, all of which cost someone a wrong number before they were written
//! down:
//!
//! **`GetPhysicallyInstalledSystemMemory` is the wrong call.** It reads the SMBIOS table
//! and fails outright on virtual machines, where the table is absent or synthetic.
//! `GlobalMemoryStatusEx` asks the memory manager instead and works everywhere.
//!
//! **`seek_read` moves the handle's file pointer.** Unix `pread` does not, which is why
//! [`crate::io::WeightFile`] can be shared across threads there. Here two threads on one
//! handle would corrupt each other's offsets and produce a plausible-looking throughput
//! figure from a read pattern nobody intended. [`crate::io::WeightFile::try_clone`] exists
//! so each reader thread holds its own handle; that is a requirement, not an optimisation.
//!
//! **`FILE_FLAG_NO_BUFFERING` is strict.** Offsets, lengths and buffer addresses must all
//! be multiples of the volume sector size. `AlignedBuf` is 2 MiB-aligned and the block
//! sizes and strides `doctor` sweeps are all multiples of 4096, so this holds — but a new
//! caller passing an odd length will get `ERROR_INVALID_PARAMETER`, not a short read.

use super::{AccelMemory, MachineFacts};
use std::fs::File;
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;

use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_RANDOM_ACCESS};
use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::Registry::{
    RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_QWORD, RRF_RT_REG_SZ,
};
use windows_sys::Win32::System::SystemInformation::{
    GetSystemInfo, GlobalMemoryStatusEx, MEMORYSTATUSEX, SYSTEM_INFO,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub fn machine_facts() -> MachineFacts {
    let (total, available) = memory_status();
    MachineFacts {
        ram_bytes: total,
        available_bytes: available,
        page_bytes: page_size(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        model: system_product_name(),
        // Never `Limited`: Windows has no equivalent of the Apple wired limit, and what the
        // display driver reports is a separate pool. `AccelMemory` keeps the two apart.
        accel_memory: discrete_vram(),
    }
}

/// The display-adapter class key. This GUID is fixed by Windows and has not changed since
/// NT; every graphics driver registers a numbered subkey under it.
const DISPLAY_CLASS_KEY: &str =
    "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";

/// Adapters to look at. Numbered from `0000`, densely, and eight is well past any real
/// machine — the point of a bound is that a corrupt registry cannot spin here.
const MAX_ADAPTERS: u32 = 8;

/// Dedicated video memory, as the display driver recorded it.
///
/// # Why the registry and not DXGI
///
/// `IDXGIAdapter3::QueryVideoMemoryInfo` is the documented interface and reports live
/// budget as well as capacity. Reaching it from `windows-sys` means driving COM vtables by
/// hand — unsafe code that no CI runner here can exercise, since GitHub's Windows images
/// have no discrete GPU. `HardwareInformation.qwMemorySize` is a plain QWORD written by the
/// driver, read through the same `RegGetValueW` this file already uses for the machine
/// model, and its failure mode is a missing value rather than a bad pointer.
///
/// The older `HardwareInformation.MemorySize` is deliberately not consulted: it is a DWORD
/// and truncates on any card of 4 GB or more, which is every card worth reporting.
fn discrete_vram() -> AccelMemory {
    let best = (0..MAX_ADAPTERS)
        .filter_map(|i| {
            reg_qword(
                &format!("{DISPLAY_CLASS_KEY}\\{i:04}"),
                "HardwareInformation.qwMemorySize",
            )
        })
        .filter(|&b| b > 0)
        // The largest adapter, not the sum: two cards are two pools, and a laptop with
        // switchable graphics lists both its integrated and its discrete part.
        .max();

    match best {
        Some(bytes) => AccelMemory::Discrete {
            bytes,
            vendor: "display driver",
        },
        None => AccelMemory::Unknown,
    }
}

/// Read a `REG_QWORD` from `HKEY_LOCAL_MACHINE`.
fn reg_qword(subkey: &str, value: &str) -> Option<u64> {
    let subkey = wide(subkey);
    let value = wide(value);
    let mut data: u64 = 0;
    let mut size = std::mem::size_of::<u64>() as u32;

    // SAFETY: both strings are NUL-terminated wide buffers that outlive the call, and
    // `data`/`size` are live locals of exactly the size the flag promises the value is.
    // `RRF_RT_REG_QWORD` makes the call fail rather than write a differently-typed value.
    let rc = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_QWORD,
            std::ptr::null_mut(),
            &mut data as *mut u64 as *mut std::ffi::c_void,
            &mut size,
        )
    };
    (rc == 0 && size as usize == std::mem::size_of::<u64>()).then_some(data)
}

pub fn page_size() -> usize {
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a live, correctly-typed local that `GetSystemInfo` fully writes.
    unsafe { GetSystemInfo(&mut info) };
    let ps = info.dwPageSize as usize;
    if ps > 0 {
        ps
    } else {
        crate::io::ASSUMED_PAGE_SIZE
    }
}

pub fn peak_rss_bytes() -> u64 {
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: `counters` is a live local with `cb` set to its own size, which is the
    // contract `GetProcessMemoryInfo` documents. The pseudo-handle from
    // `GetCurrentProcess` needs no closing.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return 0;
    }
    counters.PeakWorkingSetSize as u64
}

pub fn open_uncached(path: &Path) -> io::Result<File> {
    File::options()
        .read(true)
        // NO_BUFFERING keeps these pages out of the cache manager, so a cold sample stays
        // cold. RANDOM_ACCESS suppresses prefetch, the analogue of Darwin's `F_RDAHEAD`
        // off: once we schedule our own scattered reads, prefetch spends bandwidth on a
        // pattern it mispredicts.
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS)
        .open(path)
}

pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;

    // `seek_read` may return a short read, and unlike `pread` it moves the file pointer.
    // Looping here covers the first; per-thread handles cover the second.
    let mut done = 0usize;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], offset + done as u64) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read past the end of the file",
                ))
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Total and available physical memory, from the memory manager.
fn memory_status() -> (u64, Option<u64>) {
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a live local with `dwLength` set to its own size, which is the
    // contract `GlobalMemoryStatusEx` documents.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return (0, None);
    }
    (status.ullTotalPhys, Some(status.ullAvailPhys))
}

/// The machine's model, as the firmware recorded it.
///
/// Readable without elevation. Motherboards that were never configured leave placeholder
/// strings here, which are filtered out — "System Product Name" is not a machine model.
fn system_product_name() -> Option<String> {
    let subkey = wide("HARDWARE\\DESCRIPTION\\System\\BIOS");
    let value = wide("SystemProductName");

    // Ask for the size first, then read into a buffer of exactly that size.
    let mut size: u32 = 0;
    // SAFETY: both strings are NUL-terminated wide buffers that outlive the call; a null
    // data pointer with a live `size` asks for the required byte count.
    let rc = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    };
    if rc != 0 || size == 0 {
        return None;
    }

    let mut buf = vec![0u16; size as usize / 2 + 1];
    let mut size_out = size;
    // SAFETY: `buf` holds at least `size` bytes, which is what the sizing call requested.
    let rc = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            &mut size_out,
        )
    };
    if rc != 0 {
        return None;
    }

    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let name = String::from_utf16_lossy(&buf[..len]).trim().to_string();
    if name.is_empty()
        || name.eq_ignore_ascii_case("None")
        || name.to_ascii_lowercase().contains("to be filled")
        || name.to_ascii_lowercase().contains("system product name")
    {
        return None;
    }
    Some(name)
}

/// NUL-terminated UTF-16, as every `W` entry point expects.
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_subkeys_are_zero_padded_to_four_digits() {
        // Windows writes `0000`, `0001`. Formatting `{i}` instead would look right in a
        // review and match nothing on a real machine.
        assert!(format!("{DISPLAY_CLASS_KEY}\\{:04}", 0).ends_with("\\0000"));
        assert!(format!("{DISPLAY_CLASS_KEY}\\{:04}", 7).ends_with("\\0007"));
    }

    #[test]
    fn reading_a_value_that_is_not_there_yields_nothing_rather_than_zero() {
        // Runs on every Windows machine, including CI runners with no discrete GPU: the
        // point is that an absent key is reported as absent, never as a card with no
        // memory.
        assert_eq!(
            reg_qword(
                "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}",
                "sift.NoSuchValue"
            ),
            None
        );
    }

    #[test]
    fn a_string_value_is_refused_rather_than_reinterpreted_as_a_number() {
        // `RRF_RT_REG_QWORD` is what stops eight bytes of UTF-16 being read as a memory
        // size. `SystemProductName` is a REG_SZ that exists on every machine.
        assert_eq!(
            reg_qword("HARDWARE\\DESCRIPTION\\System\\BIOS", "SystemProductName"),
            None
        );
    }

    #[test]
    fn detection_never_reports_a_host_memory_ceiling() {
        // Whatever this machine has, Windows must not produce `Limited`: that variant means
        // "the OS caps what a model may hold in RAM", which is an Apple concept.
        assert_eq!(discrete_vram().host_ceiling_bytes(), None);
    }
}
