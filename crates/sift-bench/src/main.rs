//! Calibration harness: measure a real engine so `sift`'s estimates can be checked.
//!
//! `sift` does not run models, so this is not a competitor's benchmark — LM Studio is the
//! **instrument**, not the opponent. `sift fit` prints a tok/s estimate derived from
//! measured memory bandwidth times a hand-set efficiency factor; this harness produces the
//! measured number that estimate must be held against.
//!
//! Runs are labelled by regime, because a throughput figure without one is meaningless —
//! the same engine is fast when a model is resident and collapses when it is not:
//!
//! | regime | model vs RAM | what it measures |
//! |--------|--------------|------------------|
//! | A      | fits easily  | the resident ceiling: bandwidth-bound decode |
//! | B      | ~1.2x RAM    | the paging cliff, where the estimate stops applying |
//! | C      | >2x RAM      | disk-bound territory; most engines cannot run it at all |
//!
//! The reporter refuses to average across regimes for the same reason.
//!
//! Publishing where the estimator is *wrong* is the point. Every tool in this space
//! publishes only flattering numbers, which is precisely why none of them can be trusted
//! on the cases that matter.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command as Proc;

/// Where the LM Studio CLI lives on a default install.
const LMS_DEFAULT: &str = ".lmstudio/bin/lms";

#[derive(Parser)]
#[command(
    name = "sift-bench",
    about = "Measure a real engine on this machine, to check sift's estimates against"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

mod lmstudio;

#[derive(Subcommand)]
enum Cmd {
    /// Report what is installed and runnable before benchmarking anything.
    Probe,
    /// List models LM Studio has locally, with the regime each falls into here.
    Models,
    /// Measure LM Studio's decode throughput. This is what `sift fit` is checked against.
    Baseline {
        /// Model id as LM Studio's server reports it. Defaults to whatever is loaded.
        #[arg(long)]
        model: Option<String>,
        /// Server port.
        #[arg(long, default_value_t = lmstudio::DEFAULT_PORT)]
        port: u16,
        /// Tokens to generate per run.
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Measured runs, after a discarded warm-up.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Prompt to send.
        #[arg(long, default_value = "Explain what a mixture-of-experts layer does.")]
        prompt: String,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
    },
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
            Regime::A => "resident: this is where the estimate should hold",
            Regime::B => "at the paging cliff: the estimate stops applying here",
            Regime::C => "disk-bound: most engines cannot run this at all",
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
    // Where a platform reports an accelerator ceiling, it binds before physical RAM does.
    if let Some(limit) = facts.accel_memory.bytes() {
        return limit;
    }
    let estimate = facts.ram_bytes.saturating_sub(4 * sift_core::GIB);
    match facts.available_bytes {
        Some(available) => estimate.min(available),
        None => estimate,
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Probe => probe(),
        Cmd::Models => models(),
        Cmd::Baseline {
            model,
            port,
            max_tokens,
            repeats,
            prompt,
            json,
        } => baseline(model, port, max_tokens, repeats, &prompt, json),
    }
}

fn baseline(
    model: Option<String>,
    port: u16,
    max_tokens: u32,
    repeats: usize,
    prompt: &str,
    json: bool,
) -> Result<()> {
    if !lmstudio::server_is_up(port) {
        bail!(
            "no LM Studio server on port {port}.\n  \
             Start it with:  lms server start\n  \
             Then load a model:  lms load <model>"
        );
    }

    let loaded = lmstudio::loaded_models(port)?;
    let model = match model {
        Some(m) => m,
        None => loaded
            .first()
            .cloned()
            .context("no model is loaded; run `lms load <model>` first")?,
    };

    let runs = lmstudio::measure(port, &model, prompt, max_tokens, repeats)?;
    let median = lmstudio::median_tps(&runs).context("no runs completed")?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engine": "lm-studio",
                "model": model,
                "median_tokens_per_sec": median,
                "runs": runs,
            }))?
        );
        return Ok(());
    }

    println!("lm studio baseline");
    println!("  model            {model}");
    println!("  prompt tokens    {}", runs[0].prompt_tokens);
    println!("  tokens per run   {}", runs[0].completion_tokens);
    println!("\n  {:>5}  {:>9}  {:>10}", "run", "seconds", "tok/s");
    for (i, r) in runs.iter().enumerate() {
        println!(
            "  {:>5}  {:>9.2}  {:>10.2}",
            i + 1,
            r.seconds,
            r.tokens_per_sec
        );
    }
    println!("\n  median {median:.2} tok/s");
    println!(
        "\n  This is the number sift has to beat, on this machine, on this model.\n  \
         Regime matters: on a model that fits, LM Studio is expected to win."
    );

    Ok(())
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
