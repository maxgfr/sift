//! Head-to-head benchmark harness: `sift` versus LM Studio, on one machine.
//!
//! The thesis this repo defends is narrow and falsifiable, so the harness is built to
//! *disprove* it as readily as confirm it. Runs are labelled by regime:
//!
//! | regime | model vs RAM | expectation |
//! |--------|--------------|-------------|
//! | A      | fits easily  | LM Studio likely wins. Publish that. |
//! | B      | ~1.2x RAM    | LM Studio thrashes; this is the claim |
//! | C      | >2x RAM      | LM Studio cannot run it at all |
//!
//! A number without its regime is meaningless — the same engine is fast in A and slow in
//! C — so every record carries one, and the reporter refuses to average across them.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command as Proc;

/// Where the LM Studio CLI lives on a default install.
const LMS_DEFAULT: &str = ".lmstudio/bin/lms";

#[derive(Parser)]
#[command(
    name = "sift-bench",
    about = "Compare sift against LM Studio on this machine"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Report what is installed and runnable before benchmarking anything.
    Probe,
    /// List models LM Studio has locally, with the regime each falls into here.
    Models,
}

/// Which side of the RAM boundary a model sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
enum Regime {
    /// Comfortably resident.
    A,
    /// Near or just past the boundary — where paging starts to bite.
    B,
    /// Far past RAM; only a streaming engine can run it.
    C,
}

impl Regime {
    /// Classify by model size against usable RAM.
    ///
    /// Usable RAM is deliberately not physical RAM: macOS, the window server and the
    /// KV cache all want a share, and on Apple Silicon the GPU wired limit binds before
    /// physical RAM does.
    fn classify(model_bytes: u64, usable_ram_bytes: u64) -> Self {
        // No usable RAM means nothing can be held resident. Saturating to 1 byte would
        // instead make a 1-byte model look borderline, which is the wrong answer.
        if usable_ram_bytes == 0 {
            return Regime::C;
        }
        let ratio = model_bytes as f64 / usable_ram_bytes as f64;
        if ratio < 0.8 {
            Regime::A
        } else if ratio < 2.0 {
            Regime::B
        } else {
            Regime::C
        }
    }

    fn expectation(self) -> &'static str {
        match self {
            Regime::A => "LM Studio likely wins — publish it",
            Regime::B => "the claim: LM Studio thrashes here",
            Regime::C => "LM Studio cannot run this at all",
        }
    }
}

fn home() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

fn lms_path() -> Result<PathBuf> {
    let p = home()?.join(LMS_DEFAULT);
    if !p.exists() {
        bail!(
            "LM Studio CLI not found at {}. Install LM Studio, then run `lms bootstrap`.",
            p.display()
        );
    }
    Ok(p)
}

/// Physical RAM, minus a reserve for the OS.
///
/// The reserve is an estimate, not a measurement — which is exactly why `sift doctor`
/// locates the paging cliff empirically instead of trusting a constant like this one.
fn usable_ram_bytes() -> u64 {
    let facts = sift_core::doctor::MachineFacts::collect();
    // If the GPU wired limit is set, it binds before physical RAM does.
    if let Some(limit) = facts.gpu_wired_limit_bytes {
        return limit;
    }
    facts.ram_bytes.saturating_sub(4 * sift_core::GIB)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Probe => probe(),
        Cmd::Models => models(),
    }
}

fn probe() -> Result<()> {
    let facts = sift_core::doctor::MachineFacts::collect();
    println!("machine");
    println!(
        "  model        {}",
        facts.model.as_deref().unwrap_or("unknown")
    );
    println!("  ram          {:.1} GiB", sift_core::gib(facts.ram_bytes));
    println!(
        "  usable       {:.1} GiB (regime boundary)",
        sift_core::gib(usable_ram_bytes())
    );

    println!("\nlm studio");
    match lms_path() {
        Ok(p) => {
            println!("  cli          {}", p.display());
            match Proc::new(&p).arg("runtime").arg("ls").output() {
                Ok(out) if out.status.success() => {
                    let text = String::from_utf8_lossy(&out.stdout);
                    let selected: Vec<&str> = text
                        .lines()
                        .filter(|l| l.contains('✓'))
                        .map(str::trim)
                        .collect();
                    if selected.is_empty() {
                        println!("  runtimes     present, none marked selected");
                    } else {
                        println!("  selected runtimes:");
                        for line in selected {
                            println!("    {line}");
                        }
                    }
                }
                Ok(out) => println!(
                    "  runtimes     query failed: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                ),
                Err(e) => println!("  runtimes     could not run lms: {e}"),
            }
        }
        Err(e) => println!("  {e}"),
    }
    Ok(())
}

fn models() -> Result<()> {
    let root = home()?.join(".lmstudio/models");
    if !root.exists() {
        bail!("no LM Studio model directory at {}", root.display());
    }

    let usable = usable_ram_bytes();
    let mut found = Vec::new();
    walk_gguf(&root, &mut found)?;
    found.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    if found.is_empty() {
        println!("no .gguf files under {}", root.display());
        return Ok(());
    }

    println!(
        "usable RAM for regime classification: {:.1} GiB\n",
        sift_core::gib(usable)
    );
    for (path, size) in &found {
        let regime = Regime::classify(*size, usable);
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        println!(
            "  [{:?}] {:>7.2} GiB  {}\n         {}",
            regime,
            sift_core::gib(*size),
            name,
            regime.expectation()
        );
    }
    Ok(())
}

fn walk_gguf(dir: &Path, out: &mut Vec<(PathBuf, u64)>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_gguf(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            // Projector shards are not language models; counting them would misreport size.
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("mmproj") {
                continue;
            }
            let size = entry.metadata()?.len();
            out.push((path, size));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regimes_split_at_the_documented_boundaries() {
        let ram = 12 * sift_core::GIB;
        assert_eq!(Regime::classify(5 * sift_core::GIB, ram), Regime::A);
        assert_eq!(Regime::classify(11 * sift_core::GIB, ram), Regime::B);
        assert_eq!(Regime::classify(60 * sift_core::GIB, ram), Regime::C);
    }

    #[test]
    fn classification_never_divides_by_zero() {
        assert_eq!(Regime::classify(1, 0), Regime::C);
    }
}
