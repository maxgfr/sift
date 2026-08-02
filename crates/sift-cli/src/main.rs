//! `sift` — will this model run on your machine, and how fast?

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sift_core::doctor::{self, MachineFacts};
use sift_core::model;
use std::path::PathBuf;

mod source;
use source::Source;

#[derive(Parser)]
#[command(
    name = "sift",
    version,
    about = "Will this model fit and run fast on your machine? Answered before you download it.",
    long_about = "sift measures your machine, reads a model's real header straight from \
                  HuggingFace without downloading it, and tells you which quantization \
                  fits and roughly how many tokens per second to expect.\n\n\
                  There is no bundled model list, so a model uploaded ten minutes ago \
                  works exactly like one from last year."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure this machine: RAM, memory bandwidth, cold disk speed, GPU wired limit.
    Doctor {
        /// A file to read for the disk measurement. Must be one the machine has NOT
        /// recently read, or you will be measuring the page cache instead of the SSD.
        #[arg(long)]
        disk_sample: Option<PathBuf>,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// Read a model's shape. Works on a local file or straight off HuggingFace.
    Inspect {
        /// A .gguf path, an `org/repo`, an `org/repo:QUANT`, or a URL.
        model: String,
        /// Pick a quantization when `model` is a repo.
        #[arg(long)]
        quant: Option<String>,
        /// Also print every tensor.
        #[arg(long)]
        tensors: bool,
    },

    /// Show what a model would cost per token, and the resulting speed ceilings.
    Plan {
        /// A .gguf path, an `org/repo`, an `org/repo:QUANT`, or a URL.
        model: String,
        /// Pick a quantization when `model` is a repo.
        #[arg(long)]
        quant: Option<String>,
        /// Expert-cache hit rate to assume, 0.0 to 1.0.
        #[arg(long, default_value_t = 0.0)]
        hit_rate: f64,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Doctor { disk_sample, json } => cmd_doctor(disk_sample, json),
        Command::Inspect {
            model,
            quant,
            tensors,
        } => cmd_inspect(&Source::resolve(&model, quant.as_deref())?, tensors),
        Command::Plan {
            model,
            quant,
            hit_rate,
        } => cmd_plan(&Source::resolve(&model, quant.as_deref())?, hit_rate),
    }
}

fn cmd_doctor(disk_sample: Option<PathBuf>, json: bool) -> Result<()> {
    let facts = MachineFacts::collect();

    // 256 MiB per buffer, enough to overflow every cache level so we measure DRAM.
    let mem = doctor::measure_memory_bandwidth(256 << 20, 8);

    let mut disk = Vec::new();
    if let Some(path) = &disk_sample {
        // Every configuration must move the same volume, or the small-block rows spend
        // most of their timed region on thread startup and report absurd throughput.
        const BYTES_PER_THREAD: usize = 128 << 20;

        // Sweep block sizes: read granularity, not total volume, is what collapses
        // throughput on NVMe. Apple's own measurements put the usable floor at 32 KiB.
        for &block in &[64 << 10, 1 << 20, 4 << 20, 12 << 20] {
            for &threads in &[1usize, 4, 8] {
                let reads = (BYTES_PER_THREAD / block).max(8);
                match doctor::measure_random_read(path, block, threads, reads) {
                    Ok(s) => disk.push(s),
                    Err(e) => eprintln!("  skipped {block}B x{threads}: {e}"),
                }
            }
        }
    }

    if json {
        let out = serde_json::json!({
            "machine": {
                "model": facts.model,
                "ram_bytes": facts.ram_bytes,
                "page_bytes": facts.page_bytes,
                "cpus": facts.cpus,
                "gpu_wired_limit_bytes": facts.gpu_wired_limit_bytes,
            },
            "memory": mem,
            "disk": disk,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("machine");
    println!(
        "  model            {}",
        facts.model.as_deref().unwrap_or("unknown")
    );
    println!(
        "  ram              {:.1} GiB",
        sift_core::gib(facts.ram_bytes)
    );
    println!("  page size        {} KiB", facts.page_bytes / 1024);
    println!("  cpus             {}", facts.cpus);
    match facts.gpu_wired_limit_bytes {
        Some(b) => println!("  gpu wired limit  {:.1} GiB", sift_core::gib(b)),
        None => println!(
            "  gpu wired limit  unset -> macOS default, well below physical RAM.\n\
             {:>19}This, not your RAM, is what an 11 GB model hits first.\n\
             {:>19}Raise with: sudo sysctl iogpu.wired_limit_mb=<MB>",
            "", ""
        ),
    }

    println!("\nmemory bandwidth");
    println!("  STREAM copy      {:.1} GB/s", mem.gb_per_sec);
    println!(
        "  -> resident ceiling for a model reading 1.1 GB/token: {:.0} tok/s",
        mem.gb_per_sec * 1e9 / 1.1e9
    );

    if disk.is_empty() {
        println!("\ndisk: not measured (pass --disk-sample <file>)");
        println!("  Use a file this machine has NOT read recently. F_NOCACHE stops new");
        println!("  caching but cannot evict resident pages, so a warm file reports RAM.");
    } else {
        println!("\ndisk, cold random read");
        println!(
            "  {:>8}  {:>7}  {:>9}  {:>10}",
            "block", "threads", "GB/s", "ms/read"
        );
        for s in &disk {
            println!(
                "  {:>7}K  {:>7}  {:>9.2}  {:>10.3}{}",
                s.block_bytes / 1024,
                s.threads,
                s.gb_per_sec,
                s.ms_per_read,
                if s.looks_cached() {
                    "   <- page cache, not disk"
                } else {
                    ""
                }
            );
        }

        // The sweep warms its own sample: each configuration reads real bytes, so by the
        // last row much of a small file is resident and the numbers drift upward. Say so
        // rather than letting the reader assume all rows are equally cold.
        let swept_bytes: u64 = disk.iter().map(|s| s.total_bytes).sum();
        let sample_bytes = disk_sample
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0);
        if sample_bytes > 0 && swept_bytes > sample_bytes / 4 {
            println!(
                "\n  caveat: this sweep read {:.1} GiB of a {:.1} GiB sample, so later rows\n  \
                 are partly page-cache hits. Earlier rows are the trustworthy ones.\n  \
                 For a clean sweep use a sample at least 4x your RAM.",
                sift_core::gib(swept_bytes),
                sift_core::gib(sample_bytes)
            );
        }

        let cached = disk.iter().filter(|s| s.looks_cached()).count();
        // Take the best *trustworthy* sample. A cached row is not a disk measurement, and
        // quoting it would inflate every downstream ceiling.
        let best = disk
            .iter()
            .filter(|s| !s.looks_cached())
            .fold(0.0f64, |a, s| a.max(s.gb_per_sec));

        if cached > 0 {
            println!(
                "\n  {cached} of {} samples exceeded {:.0} GB/s, which no consumer NVMe can do.",
                disk.len(),
                sift_core::doctor::IMPLAUSIBLE_DISK_GB_S
            );
            println!("  Those pages were already resident: F_NOCACHE stops new caching but");
            println!("  cannot evict. For a real cold number, reboot or use an untouched file.");
        }

        if best > 0.0 {
            println!(
                "\n  best trustworthy {:.2} GB/s -> streamed ceiling at 1.1 GB/token: {:.1} tok/s",
                best,
                best * 1e9 / 1.1e9
            );
            println!(
                "  RAM is {:.0}x faster than this disk. That ratio is why residency matters.",
                mem.gb_per_sec / best
            );
        } else {
            println!("\n  no trustworthy cold sample; every read hit the page cache.");
        }
    }

    Ok(())
}

fn cmd_inspect(src: &Source, list_tensors: bool) -> Result<()> {
    let (g, fetched) = src.read_gguf()?;

    // Payload size is authoritative for remote files: the server's Content-Length covers
    // the whole file, but only the directory was read.
    let payload = g.total_tensor_bytes();

    println!("{}", src.label());
    println!("  gguf version     {}", g.version);
    println!(
        "  architecture     {}",
        g.architecture().unwrap_or("unknown")
    );
    println!("  tensors          {}", g.tensors.len());
    println!("  tensor payload   {:.2} GiB", sift_core::gib(payload));
    match fetched {
        // The headline claim of this tool, stated as a measurement rather than a promise.
        Some(bytes) => println!(
            "  read over HTTP   {:.2} MiB of a {:.2} GiB model ({:.4}%)",
            sift_core::mib(bytes),
            sift_core::gib(payload),
            bytes as f64 / payload.max(1) as f64 * 100.0
        ),
        None => println!("  read from disk   directory only, payload untouched"),
    }

    match model::infer_moe_shape(&g) {
        Some(shape) => {
            println!("\nmixture of experts");
            println!("  moe layers       {}", shape.moe_layers);
            println!("  experts / layer  {}", shape.experts_per_layer);
            println!("  active / token   {}", shape.experts_per_token);
            println!(
                "  activation       {:.2}% of expert weights per token",
                shape.activation_ratio() * 100.0
            );
            println!(
                "  one expert       {:.2} MiB (gate + up + down)",
                sift_core::mib(shape.bytes_per_expert)
            );
            println!(
                "  expert weights   {:.2} GiB total",
                sift_core::gib(shape.total_expert_bytes())
            );

            // Show the access pattern that motivates a repacked container.
            if let Ok(r) = model::expert_ranges(&g, 0, 0) {
                println!(
                    "\n  layer 0 expert 0 spans 3 ranges, contiguous: {}",
                    if r.is_contiguous() {
                        "yes"
                    } else {
                        "no -> 3 scattered reads"
                    }
                );
            }
        }
        None => println!("\ndense model (no stacked expert tensors)"),
    }

    if list_tensors {
        println!("\ntensors");
        for t in &g.tensors {
            println!(
                "  {:<44} {:<8} {:>14?}  {:>12}",
                t.name,
                t.dtype.name(),
                t.dims,
                t.size_bytes()
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "?".into())
            );
        }
    }

    Ok(())
}

fn cmd_plan(src: &Source, hit_rate: f64) -> Result<()> {
    let (g, _) = src.read_gguf()?;
    let shape = model::infer_moe_shape(&g)
        .context("this model has no stacked expert tensors; planning targets MoE models")?;

    // Everything that is not a routed expert is read on every token regardless of routing.
    let trunk_bytes = g
        .total_tensor_bytes()
        .saturating_sub(shape.total_expert_bytes());

    let traffic = model::TokenTraffic {
        expert_bytes: shape.expert_bytes_per_token(),
        trunk_bytes,
    };

    println!("{}", src.label());
    println!("\nper-token weight traffic");
    println!(
        "  routed experts   {:.3} GB   ({} experts x {} layers)",
        traffic.expert_bytes as f64 / 1e9,
        shape.experts_per_token,
        shape.moe_layers
    );
    println!(
        "  always-active    {:.3} GB   (read every token, no cache helps)",
        trunk_bytes as f64 / 1e9
    );
    println!("  total cold       {:.3} GB", traffic.total() as f64 / 1e9);

    let mem = doctor::measure_memory_bandwidth(256 << 20, 4);
    println!("\nceilings at hit rate {:.0}%", hit_rate * 100.0);
    println!(
        "  resident  @ {:>6.1} GB/s measured memory : {:>6.1} tok/s",
        mem.gb_per_sec,
        traffic.tokens_per_sec(mem.gb_per_sec * 1e9, hit_rate)
    );
    println!(
        "  streamed  @ {:>6.1} GB/s typical NVMe    : {:>6.1} tok/s",
        6.15,
        traffic.tokens_per_sec(6.15e9, hit_rate)
    );
    println!(
        "\n  These are roofline ceilings, not predictions. Real MoE decode lands at\n  \
         25-35% of them because expert gather is scattered. Measure, do not assume."
    );

    Ok(())
}
