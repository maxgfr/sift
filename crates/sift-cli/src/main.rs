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

/// A warning to print when the machine is short of memory right now, if it is.
///
/// Every estimate `sift` prints assumes a machine at rest: memory that "fits" is free to
/// take, and the bandwidth probe measures the bus rather than the pager. Under pressure
/// both are optimistic, and saying so is cheaper than a user finding out an hour in.
fn pressure_warning(facts: &MachineFacts) -> Option<&'static str> {
    use sift_core::platform::MemoryPressure;
    match facts.memory_pressure? {
        MemoryPressure::Normal => None,
        MemoryPressure::Warning => Some(
            "warning: the OS reports memory pressure right now (swapping or compressing).\n  \
             fits and tok/s assume a machine at rest; close something, then run this again.",
        ),
        MemoryPressure::Critical => Some(
            "warning: the OS reports CRITICAL memory pressure right now. Nothing measured\n  \
             here is reliable until memory is freed, and loading a model may get it killed.",
        ),
    }
}

/// Memory left to the OS when nothing better is known: a quarter of RAM, between 1 and
/// 4 GiB.
///
/// A flat 4 GiB was right for a 16 GiB laptop and absurd at either end: a 4 GiB machine
/// came out with nothing usable at all, and the reserve did not grow on a 64 GiB box.
fn os_reserve_bytes(ram_bytes: u64) -> u64 {
    (ram_bytes / 4).clamp(sift_core::GIB, 4 * sift_core::GIB)
}

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
/// 1. A platform-reported ceiling on host memory. Hard, and it binds first where it exists.
/// 2. What the OS says is available right now, which on Linux and Windows is a real
///    figure and accounts for whatever else is running.
/// 3. Physical RAM minus a flat reserve. An estimate, and the weakest of the three —
///    which is exactly why `sift doctor` measures the machine rather than stopping here.
///
/// A discrete GPU's VRAM is **not** one of the three. It is a separate pool, not a ceiling
/// on what the host may allocate, so adding it here would tell someone with a 24 GB card
/// and 16 GB of RAM that a 20 GB model fits in memory. `doctor` reports it and says so.
fn usable_ram(facts: &MachineFacts) -> u64 {
    if let Some(ceiling) = facts.accel_memory.host_ceiling_bytes() {
        return ceiling;
    }
    let estimate = facts
        .ram_bytes
        .saturating_sub(os_reserve_bytes(facts.ram_bytes));
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
        ///
        /// An `org/repo:QUANT` suffix sizes a single quantization instead of the whole
        /// repo, e.g. `unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M`.
        repo: String,
        /// Context length to size the KV cache for, in tokens.
        ///
        /// Defaults to 4096, which is roughly what engines start at. Not the model's
        /// trained maximum: sizing for 262k would report that almost nothing fits, which
        /// is true and useless. Raise it to see what your actual workload costs.
        #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS, value_parser = clap::value_parser!(u64).range(1..))]
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
        /// Context length to size the KV cache for, in tokens, as `fit` does.
        #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS, value_parser = clap::value_parser!(u64).range(1..))]
        ctx: u64,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },

    /// Every local model, across LM Studio, Ollama, colibri and ~/.sift.
    Ls {
        /// Context length to size the KV cache for, in tokens.
        #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS, value_parser = clap::value_parser!(u64).range(1..))]
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
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..))]
        max_tokens: u32,
        /// Measured runs, after a discarded warm-up.
        #[arg(long, default_value_t = 3, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
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

    /// Inspect or empty the cache of fetched model headers.
    Cache {
        /// Delete every cached header.
        #[arg(long)]
        clear: bool,
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
        #[arg(long, default_value_t = 0.0, value_parser = parse_fraction)]
        hit_rate: f64,
        /// Emit JSON instead of a human-readable report.
        #[arg(long)]
        json: bool,
    },
}

/// A fraction from 0 to 1, as `--hit-rate` takes. NaN and anything outside are refused
/// rather than clamped: a 200% hit rate is a typo, and printing ceilings for it is a lie.
fn parse_fraction(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("`{s}` is not a number"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("{s} is not between 0 and 1"));
    }
    Ok(v)
}

impl Command {
    /// Whether this invocation asked for JSON.
    fn json(&self) -> bool {
        match self {
            Command::Doctor { json, .. }
            | Command::Inspect { json, .. }
            | Command::Fit { json, .. }
            | Command::Route { json, .. }
            | Command::Ls { json, .. }
            | Command::Bench { json, .. }
            | Command::Engines { json }
            | Command::Cache { json, .. }
            | Command::Plan { json, .. } => *json,
        }
    }
}

fn main() {
    let command = Cli::parse().command;
    let json = command.json();
    if let Err(e) = run(command) {
        // Under --json stdout stays one parseable document even on failure, so a script
        // reading it gets an error object instead of nothing.
        if json {
            println!("{}", serde_json::json!({ "error": format!("{e:#}") }));
        }
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

fn run(command: Command) -> Result<()> {
    match command {
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
        Command::Route {
            model,
            quant,
            ctx,
            json,
        } => cmd_route(&Source::resolve(&model, quant.as_deref())?, ctx, json),
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
        Command::Cache { clear, json } => cmd_cache(clear, json),
    }
}

/// Report what the header cache holds, and empty it on request.
///
/// The cache is validated rather than expiring — a `304` proves an entry is still the file
/// on the server — so nothing here evicts on age. What it needed was a bound and a way to
/// see it, since `~/.sift/cache` otherwise grows quietly with every file ever inspected.
fn cmd_cache(clear: bool, json: bool) -> Result<()> {
    let dir = sift_core::cache::dir();
    let stats = if clear {
        sift_core::cache::clear().context("could not empty the cache")?
    } else {
        sift_core::cache::stats().context("could not read the cache")?
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "directory": dir.as_ref().map(|d| d.display().to_string()),
                // Under `--clear` these are what was removed, not what remains. Named so a
                // consumer cannot read one as the other.
                "cleared": clear,
                "entries": stats.entries,
                "bytes": stats.bytes,
                "enabled": dir.is_some(),
            }))?
        );
        return Ok(());
    }

    match &dir {
        Some(d) => println!("cache  {}", d.display()),
        // The only way `dir()` is None with HOME set.
        None => println!("cache  disabled by SIFT_NO_CACHE"),
    }
    if clear {
        println!(
            "  cleared {} entries, {:.2} MiB freed",
            stats.entries,
            sift_core::mib(stats.bytes)
        );
    } else {
        println!(
            "  {} entries, {:.2} MiB",
            stats.entries,
            sift_core::mib(stats.bytes)
        );
        println!(
            "\n  Entries are validated by ETag, never expired: a 304 proves a cached header\n  \
             is still the file on the server, so nothing here goes stale. Empty it with\n  \
             `sift cache --clear` when you want the space back."
        );
    }
    Ok(())
}

fn cmd_fit(reference: &str, context_tokens: u64, json: bool) -> Result<()> {
    // `fit repo:QUANT` asks about one quantization. The suffix is stripped here rather than
    // reaching the hub: it used to be sent as part of the repo name, and HuggingFace answers
    // a 401 for a repo that does not exist, so a valid request read as an auth failure. A
    // HuggingFace file URL — the `blob` form a browser shows, or `resolve` — narrows the
    // sweep to that one file, at the revision the URL names.
    let parsed = source::RepoRef::parse(reference, None)?;
    let (target, quant) = match &parsed {
        source::RepoRef::Repo { repo, quant } => (
            fit::Target {
                repo,
                quant: quant.as_deref(),
                ..Default::default()
            },
            quant.as_deref(),
        ),
        source::RepoRef::HubFile(f) => (
            fit::Target {
                repo: &f.repo,
                revision: Some(&f.revision),
                only: Some(&f.path),
                quant: None,
            },
            None,
        ),
        // `fit` sweeps a repo's quantizations. A file already on disk has one, and
        // sending its path to the hub as a repo name produced a 401.
        source::RepoRef::Path(p) => anyhow::bail!(
            "{} is a local file; `fit` compares the quantizations a HuggingFace repo offers.\n\
             For a file you have:  sift inspect {0}   or   sift route {0}",
            p.display()
        ),
        source::RepoRef::Url(u) => anyhow::bail!(
            "{u} is not a HuggingFace repo; `fit` compares the quantizations a repo offers.\n\
             For one file:  sift inspect {u}   or   sift route {u}"
        ),
    };
    let repo = target.repo;

    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    // Read bandwidth, not copy: decode streams weights in and writes back a small
    // activation, so a copy benchmark measures the wrong access pattern. Fewer iterations
    // than `doctor` uses — this is one input among many — and measured on its own thread
    // while the listing waits on the network, instead of before it.
    let bandwidth =
        std::thread::spawn(|| doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3));
    let installed = sift_core::engine::detect_installed();

    // Progress chatter goes to stderr under --json, so stdout stays a single parseable
    // document. A CLI that another tool shells out to must not interleave the two.
    if json {
        eprintln!("reading headers from {repo} without downloading…");
    } else {
        println!("reading headers from {repo} without downloading…");
    }
    // The probe overlaps the listing, which is all network wait, and is finished before
    // the headers are parsed: parsing on eight threads alongside it read the bus low.
    let listed = fit::list(target);
    let mem = bandwidth
        .join()
        .map_err(|_| anyhow::anyhow!("the bandwidth probe panicked"))?
        .context("measuring memory bandwidth")?;
    let swept = fit::read(listed?)?;
    if !json {
        println!(
            "machine: {}, {:.1} GiB RAM, {:.1} GiB usable{}, {:.0} GB/s memory",
            facts.model.as_deref().unwrap_or("unknown"),
            sift_core::gib(facts.ram_bytes),
            sift_core::gib(usable),
            if facts.accel_memory.is_estimate() {
                " (estimate)"
            } else {
                ""
            },
            mem.gb_per_sec
        );
    }
    if let Some(w) = pressure_warning(&facts) {
        if json {
            eprintln!("  {w}");
        } else {
            println!("  {w}");
        }
    }
    let cands = fit::score(swept, mem.gb_per_sec * 1e9, usable, context_tokens);
    // The same answer, and the same exit code, whether or not --json was asked for.
    if cands.is_empty() {
        anyhow::bail!("no readable weight files in `{repo}`");
    }
    // Narrow after the sweep, not before it: the hub listing is what knows which labels
    // exist, so an unmatched suffix can be answered with the ones that do.
    let cands = match quant {
        Some(q) => fit::narrow(cands, q)?,
        None => cands,
    };
    if json {
        let machine = fit::Machine {
            facts: &facts,
            usable_ram: usable,
            installed: &installed,
            memory_gb_per_sec: mem.gb_per_sec,
        };
        return fit::report_json(repo, &cands, &machine, context_tokens);
    }
    fit::report(repo, &cands, usable, &installed, context_tokens)
}

fn cmd_route(src: &Source, context_tokens: u64, json: bool) -> Result<()> {
    let loaded = src.read_shape()?;
    let shape = loaded.shape;
    let facts = MachineFacts::collect();
    let usable = usable_ram(&facts);
    let installed = sift_core::engine::detect_installed();

    // Route on the actual format. mlx-lm reads safetensors and llama.cpp does not, so
    // assuming GGUF here would recommend an engine that cannot open the file.
    let format = loaded.format;
    let size = shape.total_tensor_bytes();
    // Route on weights plus the KV cache, as `fit` does. Routing on the file alone sent a
    // model to an in-memory engine that `fit` had just said does not fit at this context.
    let kv = model::infer_kv_shape(&shape).map(|k| k.bytes_at(context_tokens, model::KV_F16_BYTES));
    let footprint = size.saturating_add(kv.unwrap_or(0));
    let rec = sift_core::engine::route(footprint, format, usable, &installed);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "size_bytes": size,
                "context_tokens": context_tokens,
                // Null when the metadata cannot size it, rather than a zero that reads as
                // "no cache".
                "kv_cache_bytes": kv,
                "footprint_bytes": footprint,
                "usable_memory_bytes": usable,
                "format": format,
                "regime": sift_core::engine::Regime::classify(footprint, usable),
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
    match kv {
        Some(kv) => println!(
            "  kv cache         {:.2} GiB at {context_tokens} tokens (f16), counted",
            sift_core::gib(kv)
        ),
        None => println!("  kv cache         not sizeable from this model's metadata"),
    }
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
    let mem = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3)
        .context("measuring memory bandwidth")?;

    let models = ls::discover();
    let rows = ls::evaluate(models, mem.gb_per_sec * 1e9, usable, context_tokens);
    ls::report(&rows, context_tokens, json)?;
    if let Some(w) = pressure_warning(&facts) {
        eprintln!("\n  {w}");
    }
    Ok(())
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
                    .context("no text model is loaded; run `lms load <model>` first")?,
            };
            (
                bench::lmstudio::measure(lms_port, &model, prompt, max_tokens, repeats)?,
                "LM Studio",
            )
        }
        bench::Target::Ollama => {
            let model = match model {
                Some(m) => m,
                None => match bench::ollama::loaded_models(ollama_port)?.first() {
                    Some(m) => m.clone(),
                    None => {
                        let local = bench::ollama::local_models(ollama_port).unwrap_or_default();
                        anyhow::bail!(
                            "no model is loaded in Ollama; load one with `ollama run <model>` or \
                             pass --model. Pulled here: {}",
                            if local.is_empty() {
                                "none".to_string()
                            } else {
                                local.join(", ")
                            }
                        );
                    }
                },
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
    let recorded = bench::record(&runs, stamp());

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engine": runs.first().map(|r| r.engine),
                "model": runs.first().map(|r| r.model.clone()),
                "median_tokens_per_sec": median,
                "runs": runs,
                "recorded_to": recorded.as_ref().ok().map(|p| p.display().to_string()),
                // Why it was not recorded, so a script does not read a null path as success.
                "recorded_error": recorded.as_ref().err().map(|e| format!("{e:#}")),
            }))?
        );
        return Ok(());
    }

    println!("{engine_name}: {}", runs[0].model);
    println!(
        "  {:>3}  {:>10}  {:>9}  {:>6}",
        "run", "tok/s", "tokens", "clock"
    );
    for (i, r) in runs.iter().enumerate() {
        println!(
            "  {:>3}  {:>10.2}  {:>9}  {:>6}",
            i + 1,
            r.tokens_per_sec,
            r.completion_tokens,
            if r.engine_timed { "engine" } else { "wall" }
        );
    }
    println!(
        "
  median {median:.2} tok/s"
    );
    if runs.iter().any(|r| !r.engine_timed) {
        // Say which clock produced the number. A wall-clock run includes prompt processing
        // and HTTP overhead and reads slightly low.
        println!(
            "  wall-clock runs include prompt processing — the engine reported no decode time"
        );
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
fn stamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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
    // Refused before anything is measured: a missing file used to print twelve "skipped"
    // lines after the memory probes and exit 0.
    if let Some(path) = disk_sample.as_ref().filter(|p| !p.is_file()) {
        anyhow::bail!("--disk-sample {}: no such file", path.display());
    }
    let facts = MachineFacts::collect_with_vram();
    let usable = usable_ram(&facts);

    // 256 MiB per buffer, enough to overflow every cache level so we measure DRAM.
    let mem = doctor::measure_memory_bandwidth(256 << 20, 8);
    // The read figure is the one that sets the decode ceiling; the copy figure is kept
    // alongside it because seeing both is what makes the difference legible.
    let read = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 4)
        .context("measuring memory bandwidth")?;

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
                "cgroup_limit_bytes": facts.cgroup_limit_bytes,
                "memory_pressure": facts.memory_pressure,
                "usable_memory_bytes": usable,
                // True when the figure above rests on sift's estimate of an OS default the
                // platform does not publish, rather than on a value it reported.
                "usable_memory_is_estimate": facts.accel_memory.is_estimate(),
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
    if let Some(limit) = facts.cgroup_limit_bytes {
        println!(
            "  container limit  {:.1} GiB (cgroup memory limit; RAM above is capped to it)",
            sift_core::gib(limit)
        );
    }
    if let Some(b) = facts.available_bytes {
        println!("  available now    {:.1} GiB", sift_core::gib(b));
    }
    match facts.accel_memory {
        AccelMemory::Limited { bytes } => {
            println!("  gpu wired limit  {:.1} GiB", sift_core::gib(bytes))
        }
        AccelMemory::PlatformDefault { estimated_bytes } => println!(
            "  gpu wired limit  unset -> macOS default, ~{:.1} GiB (estimate: Apple does\n\
             {:>19}not publish it; about 2/3 of RAM up to 36 GiB, 3/4 above). This, not\n\
             {:>19}your RAM, is what a model hits first.\n\
             {:>19}Raise with: sudo sysctl iogpu.wired_limit_mb=<MB>",
            sift_core::gib(estimated_bytes),
            "",
            "",
            ""
        ),
        AccelMemory::Integrated { bytes, vendor } => println!(
            "  integrated gpu   {:.1} GiB reserved ({vendor})\n\
             {:>19}A slice of system RAM, not memory of its own: nothing is added to\n\
             {:>19}what fits.",
            sift_core::gib(bytes),
            "",
            ""
        ),
        // Reported, and reported as a *separate pool*. Folding it into usable memory would
        // be the mistake that has competing tools telling a 16 GB laptop with a 24 GB card
        // that a 20 GB model fits in RAM.
        AccelMemory::Discrete { bytes, vendor } => println!(
            "  gpu vram         {:.1} GiB ({vendor})\n\
             {:>19}Not counted toward `fits`: this is a separate pool, not a ceiling\n\
             {:>19}on host memory. sift plans what fits in RAM, so on this machine it\n\
             {:>19}is conservative for anything you offload to the card.",
            sift_core::gib(bytes),
            "",
            "",
            ""
        ),
        // Say it plainly. A tool that silently substitutes a guess here is the reason
        // competing tools report a 4 GB card as an 8 GB one.
        AccelMemory::Unknown => println!(
            "  accelerator      none found.\n\
             {:>19}sift is using host memory only; if you have a discrete GPU that\n\
             {:>19}was not detected, its VRAM is not accounted for and `fits` will\n\
             {:>19}be conservative.",
            "", "", ""
        ),
    }

    if let Some(p) = facts.memory_pressure {
        println!(
            "  memory pressure  {}",
            match p {
                sift_core::platform::MemoryPressure::Normal => "normal",
                sift_core::platform::MemoryPressure::Warning => "WARNING (swapping or compressing)",
                sift_core::platform::MemoryPressure::Critical => "CRITICAL",
            }
        );
    }
    println!(
        "  usable for a model {:.1} GiB{}",
        sift_core::gib(usable),
        if facts.accel_memory.is_estimate() {
            " (estimate)"
        } else {
            ""
        }
    );

    if let Some(w) = pressure_warning(&facts) {
        println!("\n  {w}");
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
        println!("  Use a file this machine has NOT read recently. Uncached reads stop new");
        println!("  caching but cannot evict resident pages, so a warm file reports RAM.");
    } else {
        println!("\ndisk, cold random read");
        println!(
            "  {:>8}  {:>7}  {:>9}  {:>10}  {:>8}",
            "block", "threads", "GB/s", "ms/read", "cached"
        );
        for s in &disk {
            println!(
                "  {:>7}K  {:>7}  {:>9.2}  {:>10.3}  {:>8}{}",
                s.block_bytes / 1024,
                s.threads,
                s.gb_per_sec,
                s.ms_per_read,
                match s.resident_before {
                    Some(f) => format!("{:.0}%", f * 100.0),
                    None => "?".to_string(),
                },
                if s.looks_cached() {
                    "   <- page cache, not disk"
                } else {
                    ""
                }
            );
        }

        // Each row's residency is measured before it runs, so a sweep that warms its own
        // sample — on a platform where reads cannot bypass the cache — flags the rows it
        // warmed instead of letting them pass as disk.
        let cached = disk.iter().filter(|s| s.looks_cached()).count();
        // Take the best *trustworthy* sample. A cached row is not a disk measurement, and
        // quoting it would inflate every downstream ceiling.
        let best = disk
            .iter()
            .filter(|s| !s.looks_cached())
            .fold(0.0f64, |a, s| a.max(s.gb_per_sec));

        if cached > 0 {
            println!(
                "\n  {cached} of {} samples read from memory, not disk: more than {:.0}% of the\n  \
                 file was already cached, or the rate passed {:.0} GB/s, which no consumer\n  \
                 NVMe reaches. Uncached reads stop new caching but cannot evict. For a real\n  \
                 cold number, reboot or use a file this machine has not read.",
                disk.len(),
                sift_core::doctor::RESIDENT_SAMPLE_LIMIT * 100.0,
                sift_core::doctor::IMPLAUSIBLE_DISK_GB_S
            );
        }

        if best > 0.0 {
            println!(
                "\n  best trustworthy {:.2} GB/s -> streamed ceiling at 1.1 GB/token: {:.1} tok/s",
                best,
                best * 1e9 / 1.1e9
            );
            // Read against read. Streaming weights off disk is a read workload, so quoting
            // the STREAM *copy* figure here compares two different rulers and understates
            // the gap — it printed 10x on a machine where the honest ratio is 17x, and the
            // README documents 19x from the read figure. Mixing rulers is the mistake this
            // whole module exists to avoid; it should not survive in its own summary line.
            println!(
                "  RAM is {:.0}x faster than this disk. That ratio is why residency matters.",
                read.gb_per_sec / best
            );
        } else {
            println!("\n  no trustworthy cold sample; every read hit the page cache.");
        }
    }

    Ok(())
}

fn cmd_inspect(src: &Source, list_tensors: bool, json: bool) -> Result<()> {
    // GGUF carries a format version and per-tensor dtypes that safetensors does not, so
    // those lines appear only where they exist rather than being filled with a placeholder.
    let source::Loaded {
        shape,
        fetched,
        format,
        gguf,
    } = src.read_shape()?;

    // Payload size is authoritative for remote files: the server's Content-Length covers
    // the whole file, but only the directory was read.
    let payload = shape.total_tensor_bytes();

    if json {
        let moe = model::infer_moe_shape(&shape).map(|m| {
            serde_json::json!({
                "layers": m.moe_layers,
                "experts_per_layer": m.experts_per_layer,
                "experts_per_token": m.experts_per_token,
                "experts_per_token_assumed": m.experts_per_token_assumed,
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
                "full_layers": k.full_layers,
                "sliding_layers": k.sliding_layers,
                "recurrent_layers": k.recurrent_layers,
                "window": k.window,
                "train_context": k.train_context,
                "bytes_at_4096_f16": k.bytes_at(4096, model::KV_F16_BYTES),
            })
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "format": format,
                "gguf_version": gguf.as_ref().map(|g| g.version),
                // Set when this file is one part of a split model, so everything below
                // describes that part only.
                "split": gguf.as_ref().and_then(|g| g.split()).map(|(index, count)| {
                    serde_json::json!({ "index": index, "count": count })
                }),
                "architecture": shape.architecture(),
                "tensor_count": shape.tensors().len(),
                // Tensors whose dtype this build cannot size, left out of every total.
                "dropped_tensors": shape.dropped_tensors,
                "tensor_payload_bytes": payload,
                "parameters": shape.total_parameters(),
                "bits_per_weight": shape.bits_per_weight(),
                // The headline claim, as a measurement rather than a promise. Null for a
                // local file, where nothing was transferred at all.
                "bytes_read_over_http": fetched.map(|f| f.total()),
                // Of which the repo's `config.json`, for a safetensors repo. Split out so
                // a consumer can tell "the header was cached" from "659 bytes of the model
                // were read": the two are different claims.
                "config_json_bytes_read_over_http": fetched.map(|f| f.config),
                "moe": moe,
                "kv_cache": kv,
                "tensors": list_tensors.then(|| shape.tensors().iter().map(|t| serde_json::json!({
                    "name": t.name,
                    "dims": t.dims,
                    "size_bytes": t.size_bytes,
                })).collect::<Vec<_>>()),
            }))?
        );
        return Ok(());
    }

    println!("{}", src.label());
    if let Some((index, count)) = gguf.as_ref().and_then(|g| g.split()) {
        println!(
            "  warning          shard {index} of {count}: every figure below covers this part\n\
             {:>19}only. `sift fit` on the repo sums every part of a split model.",
            ""
        );
    }
    if let Some(g) = &gguf {
        println!("  gguf version     {}", g.version);
    } else {
        println!(
            "  format           {}",
            match format {
                sift_core::engine::Format::Mlx => "mlx (safetensors)",
                _ => "safetensors",
            }
        );
    }
    println!(
        "  architecture     {}",
        shape.architecture().unwrap_or("unknown")
    );
    println!("  tensors          {}", shape.tensors().len());
    if shape.dropped_tensors > 0 {
        println!(
            "  unsized tensors  {} of a dtype this build does not know — left out of the\n\
             {:>19}payload, parameter count and bits per weight below",
            shape.dropped_tensors, ""
        );
    }
    println!("  tensor payload   {:.2} GiB", sift_core::gib(payload));
    match fetched {
        // Zero bytes over a remote source means the cached header was revalidated with a
        // conditional request and the server answered 304. Worth saying outright: "0.00 MiB
        // read" is true but reads like a bug.
        Some(f) if f.header == 0 => {
            println!("  read over HTTP   nothing — cached header still current (304)")
        }
        // The headline claim of this tool, stated as a measurement rather than a promise.
        Some(f) => println!(
            "  read over HTTP   {:.2} MiB of a {:.2} GiB model ({:.4}%)",
            sift_core::mib(f.header),
            sift_core::gib(payload),
            f.header as f64 / payload.max(1) as f64 * 100.0
        ),
        None => println!("  read from disk   directory only, payload untouched"),
    }
    // The config is not part of the model, so it gets its own line rather than inflating
    // the figure above.
    if let Some(f) = fetched.filter(|f| f.config > 0) {
        println!(
            "  config.json      {} bytes, for the architecture",
            f.config
        );
    }

    match model::infer_moe_shape(&shape) {
        Some(shape) => {
            println!("\nmixture of experts");
            println!("  moe layers       {}", shape.moe_layers);
            println!("  experts / layer  {}", shape.experts_per_layer);
            println!(
                "  active / token   {}{}",
                shape.experts_per_token,
                if shape.experts_per_token_assumed {
                    " (not stated by the model; assumed)"
                } else {
                    ""
                }
            );
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

            // Show the access pattern that motivates a repacked container. GGUF only:
            // byte ranges are addressable within a file, and this is the format whose
            // offsets we hold.
            if let Ok(r) = gguf
                .as_ref()
                .context("expert ranges need a GGUF file")
                .and_then(|g| Ok(model::expert_ranges(g, 0, 0)?))
            {
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

    // The same facts `--json` reports under `kv_cache`, so the two views cannot disagree
    // about whether the model's metadata was enough to size it. Shown at 4k, the context
    // engines start at, with the trained maximum alongside because the two can differ by
    // 64x and the cache scales linearly between them.
    match model::infer_kv_shape(&shape) {
        // Mamba, RWKV: every layer recurrent. Zero is the answer, not an unknown.
        Some(kv) if kv.layers == 0 => println!(
            "\nkv cache         none: all {} layers are recurrent, with a fixed-size state",
            kv.recurrent_layers
        ),
        Some(kv) => {
            println!("\nkv cache");
            println!("  layers           {}", kv.layers);
            if kv.is_mixed() {
                // Spelled out because it is the difference between the cache this model
                // actually keeps and the one a uniform formula would charge it for.
                let mut parts = vec![format!("{} full", kv.full_layers)];
                if kv.sliding_layers > 0 {
                    parts.push(format!(
                        "{} sliding (window {})",
                        kv.sliding_layers,
                        kv.window.unwrap_or(0)
                    ));
                }
                if kv.recurrent_layers > 0 {
                    parts.push(format!("{} linear, no KV", kv.recurrent_layers));
                }
                println!("  attention        {}", parts.join(" / "));
            }
            println!("  kv heads         {}", kv.kv_heads);
            println!("  head dim         {}", kv.head_dim);
            println!(
                "  at 4096 tokens   {:.2} GiB f16",
                sift_core::gib(kv.bytes_at(4096, model::KV_F16_BYTES))
            );
            match kv.train_context {
                Some(ctx) => println!(
                    "  at trained max   {:.2} GiB f16 ({ctx} tokens)",
                    sift_core::gib(kv.bytes_at(ctx, model::KV_F16_BYTES))
                ),
                None => println!("  trained context  not stated"),
            }
        }
        None => println!(
            "\nkv cache         not sizeable: layers, kv heads or head width are not stated —\n                 \
             a bare safetensors file or URL has no config.json to read them from —\n                 \
             so `fit` counts weights only for this model"
        ),
    }

    if list_tensors {
        println!("\ntensors");
        for t in shape.tensors() {
            println!("  {:<52} {:>14?}  {:>12}", t.name, t.dims, t.size_bytes);
        }
    }

    Ok(())
}

fn cmd_plan(src: &Source, hit_rate: f64, json: bool) -> Result<()> {
    let model_shape = src.read_shape()?.shape;
    // Everything that is not a routed expert is read on every token regardless of routing.
    // A dense model is all trunk: its ceiling is simpler, not absent, and refusing to
    // print it sent people to work it out by hand.
    let (traffic, moe) = model::token_traffic(&model_shape);
    let trunk_bytes = traffic.trunk_bytes;

    if json {
        let mem = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3)
            .context("measuring memory bandwidth")?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": src.label(),
                "hit_rate": hit_rate,
                "is_moe": moe.is_some(),
                "experts_per_token": moe.as_ref().map(|m| m.experts_per_token),
                "experts_per_token_assumed": moe.as_ref().map(|m| m.experts_per_token_assumed),
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
                "note": format!(
                    "roofline ceilings, not predictions; measured decode lands at ~{:.0}% of \
                     the resident figure",
                    if moe.is_some() { fit::MOE_EFFICIENCY } else { fit::DENSE_EFFICIENCY } * 100.0
                ),
            }))?
        );
        return Ok(());
    }

    println!("{}", src.label());
    println!("\nper-token weight traffic");
    match &moe {
        Some(shape) => {
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
        }
        None => println!(
            "  dense model      {:.3} GB   (every weight is read every token; a hit rate\n\
             {:>32}changes nothing without experts to cache)",
            trunk_bytes as f64 / 1e9,
            ""
        ),
    }
    println!("  total cold       {:.3} GB", traffic.total() as f64 / 1e9);
    if let Some(shape) = moe.as_ref().filter(|m| m.experts_per_token_assumed) {
        println!(
            "\n  top-k is not stated in this model's metadata, so {} experts per token is\n  \
             assumed. Routed traffic scales with it: read the figures above as a guess.",
            shape.experts_per_token
        );
    }

    // Read bandwidth, not the copy figure. Decode streams weights in and writes back a
    // small activation, and a copy moves a byte each way — so a copy-based roofline is not
    // merely conservative, it is below what an engine actually reaches. This line used to
    // print 70 tok/s as the "ceiling" for a model LM Studio decodes at 129.
    let mem = doctor::measure_read_bandwidth(doctor::BANDWIDTH_BUF_BYTES, 3)
        .context("measuring memory bandwidth")?;
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
        "\n  These are roofline ceilings, not predictions. The {} model measured on a\n  \
         real engine reached {:.0}% of the resident figure — `sift fit` applies that factor,\n  \
         this command does not. Measure your own with `sift bench`.",
        if moe.is_some() { "one MoE" } else { "dense" },
        if moe.is_some() {
            fit::MOE_EFFICIENCY
        } else {
            fit::DENSE_EFFICIENCY
        } * 100.0
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(ram_gib: u64, accel: AccelMemory) -> MachineFacts {
        MachineFacts {
            ram_bytes: ram_gib * sift_core::GIB,
            available_bytes: None,
            page_bytes: 4096,
            cpus: 1,
            model: None,
            accel_memory: accel,
            cgroup_limit_bytes: None,
            memory_pressure: None,
        }
    }

    #[test]
    fn the_os_reserve_scales_with_ram_between_one_and_four_gib() {
        // Regression: a flat 4 GiB left a 4 GiB machine with nothing usable.
        assert_eq!(
            usable_ram(&facts(4, AccelMemory::Unknown)),
            3 * sift_core::GIB
        );
        assert_eq!(usable_ram(&facts(2, AccelMemory::Unknown)), sift_core::GIB);
        assert_eq!(
            usable_ram(&facts(16, AccelMemory::Unknown)),
            12 * sift_core::GIB
        );
        assert_eq!(
            usable_ram(&facts(64, AccelMemory::Unknown)),
            60 * sift_core::GIB
        );
    }

    fn parses(args: &[&str]) -> bool {
        Cli::try_parse_from(std::iter::once("sift").chain(args.iter().copied())).is_ok()
    }

    #[test]
    fn nonsense_numbers_are_refused_by_the_parser() {
        // Regression: `--ctx 0` reached the KV arithmetic and printed "this model's
        // metadata does not state its attention shape", which was false.
        assert!(!parses(&["fit", "org/repo", "--ctx", "0"]));
        assert!(parses(&["fit", "org/repo", "--ctx", "1"]));
        assert!(!parses(&["route", "org/repo", "--ctx", "0"]));
        assert!(!parses(&["ls", "--ctx", "0"]));
        assert!(!parses(&["plan", "org/repo", "--hit-rate", "2"]));
        assert!(!parses(&["plan", "org/repo", "--hit-rate", "NaN"]));
        assert!(!parses(&["plan", "org/repo", "--hit-rate", "-0.1"]));
        assert!(parses(&["plan", "org/repo", "--hit-rate", "0.5"]));
        assert!(!parses(&["bench", "--repeats", "0"]));
        assert!(!parses(&["bench", "--max-tokens", "0"]));
    }

    #[test]
    fn fit_on_a_local_file_points_elsewhere_without_asking_the_hub() {
        let f = tempfile::NamedTempFile::new().expect("temp file");
        let err = cmd_fit(f.path().to_str().unwrap(), 4096, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("local file"), "{err}");
        assert!(err.contains("sift inspect"), "{err}");
    }

    #[test]
    fn a_missing_disk_sample_is_an_error() {
        let err = cmd_doctor(Some(PathBuf::from("/nonexistent/sift-sample")), true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no such file"), "{err}");
    }

    #[test]
    fn pressure_is_warned_about_only_when_the_os_reports_it() {
        use sift_core::platform::MemoryPressure;
        let mut f = facts(16, AccelMemory::Unknown);
        assert!(pressure_warning(&f).is_none(), "unknown is not a warning");
        f.memory_pressure = Some(MemoryPressure::Normal);
        assert!(pressure_warning(&f).is_none());
        f.memory_pressure = Some(MemoryPressure::Warning);
        assert!(pressure_warning(&f).unwrap().contains("pressure"));
        f.memory_pressure = Some(MemoryPressure::Critical);
        assert!(pressure_warning(&f).unwrap().contains("CRITICAL"));
    }

    #[test]
    fn an_unset_mac_wired_limit_plans_against_the_estimate() {
        let ram = 16 * sift_core::GIB;
        let f = facts(
            16,
            sift_core::platform::apple_wired_limit(Some(0), None, ram),
        );
        assert_eq!(usable_ram(&f), ram / 3 * 2);
    }
}
