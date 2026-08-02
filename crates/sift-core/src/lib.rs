//! `sift` — will this model fit and run fast on this machine?
//!
//! Answering that today costs an hour: pull 18 GB, discover it is too slow, delete, guess
//! again. Every input needed to answer it up front is cheap. A model's shape lives in the
//! tensor directory at the front of the file, which HTTP range requests can read in a few
//! megabytes. What the machine can actually do is a measurement, and it takes seconds.
//!
//! `sift` never runs a model. It reads the header, measures the machine, and says which
//! quantization to download and which engine should run it. The alternatives either ship a
//! model catalog that rots or ask the user to type in their own memory bandwidth; this
//! crate is built so neither is necessary.
//!
//! # Layout
//!
//! - [`platform`] — the per-OS seam. Everything that differs between macOS, Linux and
//!   Windows lives here so the rest of the crate never sees a `cfg`.
//! - [`doctor`] — measure the machine: memory bandwidth, cold disk reads, what the OS
//!   reports about itself.
//! - [`model`] — read model files without loading them.
//! - [`hub`] — list what a HuggingFace repo offers, without downloading any of it.
//! - [`engine`] — which runtime should run a given model, and which to avoid.
//! - [`io`] — explicit reads that bypass the page cache, so a cold measurement is honest.

pub mod doctor;
pub mod engine;
pub mod hub;
pub mod io;
pub mod model;
pub mod platform;

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
