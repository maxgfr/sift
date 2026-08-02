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
use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};
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
        // No equivalent of the Apple wired limit. Discrete VRAM via DXGI is a different
        // quantity — not a ceiling on host memory — so claiming it here would mislead.
        accel_memory: AccelMemory::Unknown,
    }
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
