//! `sift` — will this model run on your machine, and how fast?

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sift_core::doctor::{self, AccelMemory, MachineFacts};
use sift_core::model;
use std::path::PathBuf;

mod fit;
mod source;
use source::Source;

/// Memory reserved for the OS when nothing better is known.
const OS_RESERVE_BYTES: u64 = 4 * sift_core::GIB;

/// Context length `fit` sizes the KV cache for unless told otherwise.
///
/// Roughly where engines start. Deliberately not the model's trained maximum: sizing for
/// Qwen3's 262k would report that nothing fits, which is true and useless as advice.
const DEFAULT_CONTEXT_TOKENS: u64 = 4096;

/// Memory a model may actually occupy on this machine.
///
/// Not physical RAM: the OS, the compositor and the KV cache all take a share, and on
/// Apple Silicon the GPU wired limit binds before physical RAM does.
///
/// Three sources, in descending order of how much they are worth trusting:
///
/// 1. A platform-reported accelerator ceiling. Hard, and it binds first where it exists.
/// 2. What the OS says is available right now, which on Linux and Windows is a real
///    figure and accounts for whatever else is running.
/// 3. Physical RAM minus a flat reserve. An estimate, and the weakest of the three —
///    which is exactly why `sift doctor` measures the machine rather than stopping here.
fn usable_ram(facts: &MachineFacts) -> u64 {
    if let Some(ceiling) = facts.accel_memory.bytes() {
        return ceiling;
    }
    let estimate = facts.ram_bytes.saturating_sub(OS_RESERVE_BYTES);
    match facts.available_bytes {
        // Take the lower of the two. Available memory can briefly exceed RAM minus the
        // reserve on an idle machine, and advising against that headroom would tell
        // someone a model fits when the first other application to start makes it not.
        Some(available) => estimate.min(available),
        None => estimate,
    }
}

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

    /// Which quantization of a model should you download for this machine?
    Fit {
        /// A HuggingFace repo, e.g. `unsloth/Qwen3-30B-A3B-GGUF`.
        repo: String,
        /// Context length to size the KV cache for, in tokens.
        ///
        /// Defaults to 4096, which is roughly what engines start at. Not the model's
        /// trained maximum: sizing for 262k would report that almost nothing fits, which
        /// is true and useless. Raise it to see what your actual workload costs.
        #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS)]
        ctx: u64,
    },

    /// Which engine should run this model, and what to type.
    Route {
        /// A .gguf path, an `org/repo`, an `org/repo:QUANT`, or a URL.
        model: String,
        /// Pick a quantization when `model` is a repo.
        #[arg(long)]
        quant: Option<String>,
    },

    /// List the inference engines installed on this machine.
    Engines,

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
        Command::Fit { repo, ctx } => cmd_fit(&repo, ctx),
        Command::Route { model, quant } => cmd_route(&Source::resolve(&model, quant.as_deref())?),
        Command::Engines => cmd_engines(),
    }
}

fn cmd_fit(repo: &str, context_tokens: u64) -> Result<()> {
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    // Fewer iterations than `doctor` uses: this is one input among many, and the user is
    // waiting on network round-trips anyway.
    let mem = doctor::measure_memory_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3);
    let installed = sift_core::engine::detect_installed();

    println!(
        "machine: {}, {:.1} GiB RAM, {:.1} GiB usable, {:.0} GB/s memory",
        facts.model.as_deref().unwrap_or("unknown"),
        sift_core::gib(facts.ram_bytes),
        sift_core::gib(usable),
        mem.gb_per_sec
    );
    println!("reading headers from {repo} without downloading…");

    let cands = fit::evaluate(repo, mem.gb_per_sec * 1e9, usable, context_tokens)?;
    fit::report(repo, &cands, usable, &installed, context_tokens)
}

fn cmd_route(src: &Source) -> Result<()> {
    let (g, _) = src.read_gguf()?;
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    let installed = sift_core::engine::detect_installed();

    let size = g.total_tensor_bytes();
    let rec = sift_core::engine::route(size, sift_core::engine::Format::Gguf, usable, &installed);

    println!("{}", src.label());
    println!("  size             {:.2} GiB", sift_core::gib(size));
    println!("  usable memory    {:.2} GiB", sift_core::gib(usable));

    match &rec.engine {
        Some(e) => {
            println!(
                "\n  use {} {}",
                e.name,
                if rec.installed {
                    "(installed)"
                } else {
                    "(NOT installed)"
                }
            );
            println!("  because {}", rec.reason);
            if !rec.installed {
                println!("  install: {}", e.install);
            }
        }
        None => println!("\n  no suitable engine: {}", rec.reason),
    }

    // Naming what to avoid matters as much as naming what to use: an engine that thrashes
    // looks like it is working, which is how people lose an afternoon.
    if !rec.avoid.is_empty() {
        println!("\n  installed, but do not use for this model:");
        for (e, why) in &rec.avoid {
            println!("    {:<12} {why}", e.name);
        }
    }
    if !rec.also_suitable.is_empty() {
        let names: Vec<&str> = rec.also_suitable.iter().map(|e| e.name).collect();
        println!("\n  also suitable if installed: {}", names.join(", "));
    }
    Ok(())
}

fn cmd_engines() -> Result<()> {
    let installed = sift_core::engine::detect_installed();
    println!("known engines\n");
    for e in sift_core::engine::ENGINES {
        match installed.iter().find(|i| i.engine.id == e.id) {
            Some(i) => println!("  [x] {:<12} {}", e.name, i.found_at.display()),
            None => println!("  [ ] {:<12} {}", e.name, e.install),
        }
    }
    println!("\n  Detection fails soft: a missing binary means `not found here`, never");
    println!("  `not installed`. Install paths change, and asserting absence would be");
    println!("  wrong in a way you could not debug.");
    Ok(())
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
                "available_bytes": facts.available_bytes,
                "accel_memory": facts.accel_memory,
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
    if let Some(b) = facts.available_bytes {
        println!("  available now    {:.1} GiB", sift_core::gib(b));
    }
    match facts.accel_memory {
        AccelMemory::Limited { bytes } => {
            println!("  gpu wired limit  {:.1} GiB", sift_core::gib(bytes))
        }
        AccelMemory::PlatformDefault => println!(
            "  gpu wired limit  unset -> macOS default, well below physical RAM.\n\
             {:>19}This, not your RAM, is what an 11 GB model hits first.\n\
             {:>19}Raise with: sudo sysctl iogpu.wired_limit_mb=<MB>",
            "", ""
        ),
        // Say it plainly. A tool that silently substitutes a guess here is the reason
        // competing tools report a 4 GB card as an 8 GB one.
        AccelMemory::Unknown => println!(
            "  accelerator      not measured on this platform.\n\
             {:>19}sift is using host memory only; if you have a discrete GPU, its\n\
             {:>19}VRAM is not accounted for and `fits` will be conservative.",
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

    match model::infer_moe_shape(&g.shape()) {
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
                "  one expert       {:.2} MiB mean (gate + up + down)",
                sift_core::mib(shape.mean_bytes_per_expert())
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
    let shape = model::infer_moe_shape(&g.shape())
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
