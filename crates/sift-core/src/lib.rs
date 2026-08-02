//! `sift` — compile the model to the machine.
//!
//! Every local-inference tool takes a quantized file someone else chose and hopes it
//! fits. When it doesn't, the OS starts paging and throughput falls off a cliff.
//!
//! `sift` inverts that: it measures *this* machine, measures which weights *this* model
//! actually touches, and decides what stays resident. The resident-set size becomes a
//! number you set, not one that emerges from kernel paging behaviour.
//!
//! # Layout
//!
//! - [`doctor`] — measure the machine. Disk, memory bandwidth, and the paging cliff.
//! - [`model`] — read model files without loading them.
//! - [`io`] — the cold tier: explicit reads that bypass the page cache.

pub mod doctor;
pub mod engine;
pub mod hub;
pub mod io;
pub mod model;

/// Bytes in a binary gigabyte, used throughout for human-facing sizes.
pub const GIB: u64 = 1 << 30;
/// Bytes in a binary megabyte.
pub const MIB: u64 = 1 << 20;

/// Convert a byte count to GiB for display.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

/// Convert a byte count to MiB for display.
pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB as f64
}
