//! `sift` — will this model run on your machine, and how fast?

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sift_core::doctor::{self, AccelMemory, MachineFacts};
use sift_core::model;
use std::path::PathBuf;

mod bench;
mod fit;
mod ls;
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
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
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
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// Which engine should run this model, and what to type.
    Route {
        /// A .gguf path, an `org/repo`, an `org/repo:QUANT`, or a URL.
        model: String,
        /// Pick a quantization when `model` is a repo.
        #[arg(long)]
        quant: Option<String>,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// Every local model, across LM Studio, Ollama, colibri and ~/.sift.
    Ls {
        /// Context length to size the KV cache for, in tokens.
        #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS)]
        ctx: u64,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// Measure a real engine, and record what it achieved.
    ///
    /// `sift` never runs a model. This drives one that does, so the estimates have
    /// something to be checked against.
    Bench {
        /// Which engine to drive. Defaults to whichever is serving.
        #[arg(long, value_enum, default_value_t = bench::Target::Auto)]
        engine: bench::Target,
        /// Model id as the engine reports it. Defaults to whatever is loaded.
        #[arg(long)]
        model: Option<String>,
        /// Tokens to generate per run.
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Measured runs, after a discarded warm-up.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Prompt to send.
        #[arg(long, default_value = "Explain what a mixture-of-experts layer does.")]
        prompt: String,
        /// LM Studio server port.
        #[arg(long, default_value_t = bench::lmstudio::DEFAULT_PORT)]
        lms_port: u16,
        /// Ollama server port.
        #[arg(long, default_value_t = bench::ollama::DEFAULT_PORT)]
        ollama_port: u16,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// List the inference engines installed on this machine.
    Engines {
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
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
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Doctor { disk_sample, json } => cmd_doctor(disk_sample, json),
        Command::Inspect {
            model,
            quant,
            tensors,
            json,
        } => cmd_inspect(&Source::resolve(&model, quant.as_deref())?, tensors, json),
        Command::Plan {
            model,
            quant,
            hit_rate,
            json,
        } => cmd_plan(&Source::resolve(&model, quant.as_deref())?, hit_rate, json),
        Command::Fit { repo, ctx, json } => cmd_fit(&repo, ctx, json),
        Command::Route { model, quant, json } => {
            cmd_route(&Source::resolve(&model, quant.as_deref())?, json)
        }
        Command::Ls { ctx, json } => cmd_ls(ctx, json),
        Command::Bench {
            engine,
            model,
            max_tokens,
            repeats,
            prompt,
            lms_port,
            ollama_port,
            json,
        } => cmd_bench(
            engine,
            model,
            max_tokens,
            repeats,
            &prompt,
            lms_port,
            ollama_port,
            json,
        ),
        Command::Engines { json } => cmd_engines(json),
    }
}

fn cmd_fit(repo: &str, context_tokens: u64, json: bool) -> Result<()> {
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    // Read bandwidth, not copy: decode streams weights in and writes back a small
    // activation, so a copy benchmark measures the wrong access pattern. Fewer iterations
    // than `doctor` uses — this is one input among many and the user is waiting on network
    // round-trips anyway.
    let mem = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3);
    let installed = sift_core::engine::detect_installed();

    // Progress chatter goes to stderr under --json, so stdout stays a single parseable
    // document. A CLI that another tool shells out to must not interleave the two.
    if json {
        eprintln!("reading headers from {repo} without downloading…");
    } else {
        println!(
            "machine: {}, {:.1} GiB RAM, {:.1} GiB usable, {:.0} GB/s memory",
            facts.model.as_deref().unwrap_or("unknown"),
            sift_core::gib(facts.ram_bytes),
            sift_core::gib(usable),
            mem.gb_per_sec
        );
        println!("reading headers from {repo} without downloading…");
    }

    let cands = fit::evaluate(repo, mem.gb_per_sec * 1e9, usable, context_tokens)?;
    if json {
        return fit::report_json(repo, &cands, &facts, usable, mem.gb_per_sec, context_tokens);
    }
    fit::report(repo, &cands, usable, &installed, context_tokens)
}

fn cmd_route(src: &Source, json: bool) -> Result<()> {
    let (shape, _) = src.read_shape()?;
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    let installed = sift_core::engine::detect_installed();

    // Route on the actual format. mlx-lm reads safetensors and llama.cpp does not, so
    // assuming GGUF here would recommend an engine that cannot open the file.
    let format = src.format();
    let size = shape.total_tensor_bytes();
    let rec = sift_core::engine::route(size, format, usable, &installed);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "size_bytes": size,
                "usable_memory_bytes": usable,
                "format": format,
                "regime": sift_core::engine::Regime::classify(size, usable),
                "engine": rec.engine.as_ref().map(|e| serde_json::json!({
                    "id": e.id,
                    "name": e.name,
                    "installed": rec.installed,
                    "install": e.install,
                })),
                "reason": rec.reason,
                // Named separately from `engine` because "do not use this one" is a
                // distinct claim from "use that one", and a consumer acting on only the
                // first would still send someone at an engine that thrashes.
                "avoid": rec.avoid.iter().map(|(e, why)| serde_json::json!({
                    "id": e.id,
                    "name": e.name,
                    "why": why,
                })).collect::<Vec<_>>(),
                "also_suitable": rec.also_suitable.iter().map(|e| serde_json::json!({
                    "id": e.id,
                    "name": e.name,
                    "install": e.install,
                })).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

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

fn cmd_ls(context_tokens: u64, json: bool) -> Result<()> {
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    let mem = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3);

    let models = ls::discover();
    let rows = ls::evaluate(models, mem.gb_per_sec * 1e9, usable, context_tokens);
    ls::report(&rows, context_tokens, json)
}

#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    engine: bench::Target,
    model: Option<String>,
    max_tokens: u32,
    repeats: usize,
    prompt: &str,
    lms_port: u16,
    ollama_port: u16,
    json: bool,
) -> Result<()> {
    let target = bench::resolve(engine, lms_port, ollama_port)?;

    let (runs, engine_name) = match target {
        bench::Target::LmStudio => {
            let model = match model {
                Some(m) => m,
                None => bench::lmstudio::loaded_models(lms_port)?
                    .first()
                    .cloned()
                    .context("no model is loaded; run `lms load <model>` first")?,
            };
            (
                bench::lmstudio::measure(lms_port, &model, prompt, max_tokens, repeats)?,
                "LM Studio",
            )
        }
        bench::Target::Ollama => {
            let model = match model {
                Some(m) => m,
                None => bench::ollama::local_models(ollama_port)?
                    .first()
                    .cloned()
                    .context("Ollama has no local models; run `ollama pull <model>` first")?,
            };
            (
                bench::ollama::measure(ollama_port, &model, prompt, max_tokens, repeats)?,
                "Ollama",
            )
        }
        bench::Target::Auto => unreachable!("resolve never returns Auto"),
    };

    let median = bench::median_tps(&runs).context("no runs completed")?;

    // Recorded before printing: a measurement that reaches the terminal and not the log is
    // one that cannot replace an estimate later, which is the whole reason to take it.
    let recorded = bench::record(&runs, &stamp());

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engine": runs.first().map(|r| r.engine),
                "model": runs.first().map(|r| r.model.clone()),
                "median_tokens_per_sec": median,
                "runs": runs,
                "recorded_to": recorded.as_ref().ok().map(|p| p.display().to_string()),
            }))?
        );
        return Ok(());
    }

    println!("{engine_name}: {}", runs[0].model);
    println!("  {:>3}  {:>10}  {:>9}", "run", "tok/s", "tokens");
    for (i, r) in runs.iter().enumerate() {
        println!(
            "  {:>3}  {:>10.2}  {:>9}",
            i + 1,
            r.tokens_per_sec,
            r.completion_tokens
        );
    }
    println!(
        "
  median {median:.2} tok/s"
    );
    if !runs[0].engine_timed {
        // Say which clock produced the number. LM Studio's API reports no decode duration,
        // so this includes prompt processing and HTTP overhead and reads slightly low.
        println!("  wall clock, including prompt processing — this engine reports no decode time");
    }
    match &recorded {
        Ok(path) => println!("  recorded to {}", path.display()),
        Err(e) => println!("  not recorded: {e}"),
    }
    Ok(())
}

/// A UTC timestamp for the bench log.
///
/// Derived from the wall clock rather than a date crate: the log needs runs to be
/// orderable and attributable, and seconds since the epoch does both without adding a
/// dependency to a binary that otherwise has none at runtime.
fn stamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("{}", d.as_secs()),
        Err(_) => "0".to_string(),
    }
}

fn cmd_engines(json: bool) -> Result<()> {
    let installed = sift_core::engine::detect_installed();

    if json {
        let engines: Vec<_> = sift_core::engine::ENGINES
            .iter()
            .map(|e| {
                let found = installed.iter().find(|i| i.engine.id == e.id);
                serde_json::json!({
                    "id": e.id,
                    "name": e.name,
                    "formats": e.formats,
                    "oversized": e.oversized,
                    "sweet_spot": e.sweet_spot,
                    "install": e.install,
                    "found": found.is_some(),
                    "found_at": found.map(|i| i.found_at.display().to_string()),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engines": engines,
                // Consumers should not read `found: false` as "absent". Detection probes
                // known install paths, and those change.
                "detection": "best effort; false means not found here, never not installed",
            }))?
        );
        return Ok(());
    }

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
    // The read figure is the one that sets the decode ceiling; the copy figure is kept
    // alongside it because seeing both is what makes the difference legible.
    let read = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 4);

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
            "memory_copy": mem,
            "memory_read": read,
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
    println!("  read, all cores  {:.1} GB/s", read.gb_per_sec);
    println!("  STREAM copy      {:.1} GB/s", mem.gb_per_sec);
    // Both are printed because the difference between them is what made the estimates
    // wrong. Decode streams weights in and writes back a small activation, so the read
    // figure is the ceiling; a copy moves a byte each way and a single-threaded one cannot
    // saturate an Apple Silicon bus at all.
    println!(
        "  -> resident ceiling for a model reading 1.1 GB/token: {:.0} tok/s",
        read.gb_per_sec * 1e9 / 1.1e9
    );
    if read.is_implausible() || mem.is_implausible() {
        println!(
            "\n  warning: a figure above {:.0} GB/s is not a memory measurement. Something\n  \
             other than the bus was timed — usually an invariant loop the optimiser lifted.",
            sift_core::doctor::IMPLAUSIBLE_MEMORY_GB_S
        );
    }

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

fn cmd_inspect(src: &Source, list_tensors: bool, json: bool) -> Result<()> {
    let (g, fetched) = src.read_gguf()?;

    // Payload size is authoritative for remote files: the server's Content-Length covers
    // the whole file, but only the directory was read.
    let payload = g.total_tensor_bytes();

    if json {
        let shape = g.shape();
        let moe = model::infer_moe_shape(&shape).map(|m| {
            serde_json::json!({
                "layers": m.moe_layers,
                "experts_per_layer": m.experts_per_layer,
                "experts_per_token": m.experts_per_token,
                "activation_ratio": m.activation_ratio(),
                "mean_bytes_per_expert": m.mean_bytes_per_expert(),
                "total_expert_bytes": m.total_expert_bytes(),
                "expert_bytes_per_token": m.expert_bytes_per_token(),
            })
        });
        let kv = model::infer_kv_shape(&shape).map(|k| {
            serde_json::json!({
                "layers": k.layers,
                "kv_heads": k.kv_heads,
                "head_dim": k.head_dim,
                "train_context": k.train_context,
                "bytes_at_4096_f16": k.bytes_at(4096, model::KV_F16_BYTES),
            })
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "gguf_version": g.version,
                "architecture": g.architecture(),
                "tensor_count": g.tensors.len(),
                "tensor_payload_bytes": payload,
                "parameters": shape.total_parameters(),
                "bits_per_weight": shape.bits_per_weight(),
                // The headline claim, as a measurement rather than a promise. Null for a
                // local file, where nothing was transferred at all.
                "bytes_read_over_http": fetched,
                "moe": moe,
                "kv_cache": kv,
                "tensors": list_tensors.then(|| g.tensors.iter().map(|t| serde_json::json!({
                    "name": t.name,
                    "dtype": t.dtype.name(),
                    "dims": t.dims,
                    "size_bytes": t.size_bytes(),
                })).collect::<Vec<_>>()),
            }))?
        );
        return Ok(());
    }

    println!("{}", src.label());
    println!("  gguf version     {}", g.version);
    println!(
        "  architecture     {}",
        g.architecture().unwrap_or("unknown")
    );
    println!("  tensors          {}", g.tensors.len());
    println!("  tensor payload   {:.2} GiB", sift_core::gib(payload));
    match fetched {
        // Zero bytes over a remote source means the cached header was revalidated with a
        // conditional request and the server answered 304. Worth saying outright: "0.00 MiB
        // read" is true but reads like a bug.
        Some(0) => println!("  read over HTTP   nothing — cached header still current (304)"),
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

fn cmd_plan(src: &Source, hit_rate: f64, json: bool) -> Result<()> {
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

    if json {
        let mem = doctor::measure_memory_bandwidth(256 << 20, 4);
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "hit_rate": hit_rate,
                "per_token_bytes": {
                    "routed_experts": traffic.expert_bytes,
                    "always_active": trunk_bytes,
                    "total": traffic.total(),
                },
                "measured_memory_gb_per_sec": mem.gb_per_sec,
                "ceilings_tokens_per_sec": {
                    "resident": traffic.tokens_per_sec(mem.gb_per_sec * 1e9, hit_rate),
                    "streamed_typical_nvme": traffic.tokens_per_sec(6.15e9, hit_rate),
                },
                // Stated in the payload, not only in the prose, so a consumer that never
                // reads the human output cannot mistake a roofline for a prediction.
                "note": "roofline ceilings, not predictions; real MoE decode lands at 25-35% of them",
            }))?
        );
        return Ok(());
    }

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
