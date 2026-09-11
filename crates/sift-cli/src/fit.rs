//! `sift fit` — which quantization of this model should you download?
//!
//! The question everyone answers by downloading 18 GB, discovering it is too slow,
//! deleting it and guessing again. Every input needed to answer it up front is cheap: file
//! sizes come from the hub API, and each file's real header comes from a few kilobytes of
//! range requests.
//!
//! # Why the estimate is honest about MoE
//!
//! The obvious speed model is `bandwidth / file_size`. It is wrong for mixture-of-experts
//! models by roughly the sparsity factor — a 30B-A3B model reads about 3B parameters per
//! token, not 30B — and that error is large enough to send someone at a model that will
//! disappoint them. A competing tool has an open issue for exactly this: a 3× overestimate
//! on Qwen3-30B.
//!
//! So the traffic model splits routed-expert bytes from always-active bytes and scales
//! only the former by top-k. That is what [`sift_core::model::TokenTraffic`] does.

use anyhow::{Context, Result};
use sift_core::engine::{self, Format, Installed, Regime};
use sift_core::hub;
use sift_core::model::{self, hf_url, RemoteFile, TokenTraffic};

/// One evaluated quantization.
#[derive(Debug)]
pub struct Candidate {
    pub label: String,
    /// What to pass `huggingface-cli download` to get this quantization.
    ///
    /// A path for a single file, a `--include` glob for a split set. Naming one shard
    /// would fetch a third of a model that then fails to load, which is a worse outcome
    /// than no hint at all.
    pub download_arg: String,
    pub size_bytes: u64,
    pub traffic: TokenTraffic,
    pub regime: Regime,
    pub est_tok_s: f64,
    pub is_moe: bool,
    /// Mean bits stored per weight, measured from this file's tensor directory.
    pub bits_per_weight: Option<f64>,
    /// KV cache bytes at the context this evaluation assumed.
    pub kv_bytes: u64,
    /// Files this quantization ships as. 1 for the ordinary case.
    pub shard_count: u32,
    /// Weight format, carried per candidate rather than assumed for the repo.
    ///
    /// The engine recommendation depends on it: mlx-lm reads safetensors and llama.cpp does
    /// not. Assuming GGUF here told anyone running `sift fit` on a safetensors repo to open
    /// it with LM Studio, which cannot — a confidently wrong answer of exactly the kind
    /// routing exists to prevent.
    pub format: Format,
}

impl Candidate {
    /// Everything that has to be resident at once: weights plus KV cache.
    ///
    /// The number `fits` is actually about. File size alone answers a question nobody
    /// asked — a 30B model at 4 bits is ~18 GB of weights, and a long context can add more
    /// than that again.
    pub fn footprint_bytes(&self) -> u64 {
        self.size_bytes.saturating_add(self.kv_bytes)
    }
}

/// Bits per weight below which a quantization is damaged enough to warn about.
///
/// Quantization quality does not fall off smoothly. Down to roughly 3 bits a model loses
/// accuracy gradually and stays useful; below that it degrades fast, and the 1- and 2-bit
/// formats are a last resort for models that would otherwise not run at all.
///
/// 3.0 is a judgement, not a measurement, and it is applied as a *warning threshold*
/// rather than a filter: if nothing above it fits, `sift` still names the best option
/// available and says plainly what the user is accepting.
const QUALITY_FLOOR_BPW: f64 = 3.0;

impl Candidate {
    /// Whether this quantization is above the quality floor.
    ///
    /// Unknown precision counts as acceptable: refusing to recommend a file whose bits per
    /// weight could not be computed would silently drop valid options, and a missing
    /// measurement is not evidence of a bad one.
    fn meets_quality_floor(&self) -> bool {
        self.bits_per_weight
            .is_none_or(|bpw| bpw >= QUALITY_FLOOR_BPW)
    }

    /// Ranking key for "which of these should you download".
    ///
    /// Ordered by what actually matters, most significant first:
    ///
    /// 1. **Above the quality floor.** A 3-bit quant that only just fits beats a 1-bit one
    ///    with room to spare. This is the whole point: file size is a bad proxy for
    ///    quality below ~3 bits, and ranking on size alone recommends a damaged model.
    /// 2. **Comfortably resident over tight.** Between two quants that are both good
    ///    enough, take the one that will not fall over once the KV cache grows.
    /// 3. **More bits.** Within one regime, precision is the tiebreak — and bits per
    ///    weight, not file size, because a dynamic quantization can be larger and no more
    ///    precise.
    fn rank(&self) -> (u8, u8, u64) {
        let regime_rank = match self.regime {
            Regime::Fits => 1,
            Regime::Tight => 0,
            Regime::Oversized => 0,
        };
        // Scaled to an integer so the key stays Ord; a millibit of resolution is far finer
        // than the difference between any two real quantizations.
        let bpw_rank = (self.bits_per_weight.unwrap_or(0.0) * 1000.0).max(0.0) as u64;
        (self.meets_quality_floor() as u8, regime_rank, bpw_rank)
    }
}

/// Fraction of the measured read-bandwidth roofline that real dense decode achieves.
///
/// **Calibrated against a measurement, and the calibration changed the number.** LM Studio
/// decoding Qwen3.5-9B Q4_K_M on an M5: 21.03 tok/s median over three runs, 5.616 GB of
/// weight traffic per token, so 118.1 GB/s effective. Read bandwidth measured on the same
/// machine is 119–136 GB/s. That is ~0.90 of the ceiling, not the 0.80 this used to carry.
///
/// The old figure was not merely stale, it was measured against the wrong ruler: a
/// single-threaded STREAM *copy*. Decode is read-dominated and one core cannot saturate an
/// Apple Silicon bus, so that benchmark understated the machine and 0.80 was silently
/// absorbing the error. See [`sift_core::doctor::measure_read_bandwidth`].
///
/// Re-measured when [`MOE_EFFICIENCY`] was calibrated: 21.11 tok/s, which reproduces the
/// original 21.03 to within 0.4%, against a bandwidth reading of 135.4 GB/s that session —
/// arithmetically 0.88. Left at 0.90, because the difference is smaller than the spread of
/// the ruler itself: the same machine reads 135 GB/s idle and 115 GB/s with a model
/// resident, and `fit` measures live, so a second digit here would be false precision.
pub const DENSE_EFFICIENCY: f64 = 0.90;

/// The same fraction for mixture-of-experts decode.
///
/// **Calibrated, and the calibration was the largest correction this tool has made.** LM
/// Studio decoding OLMoE-1B-7B Q4_K_M on an M5: 129.10 tok/s median over three 415-token
/// runs, 0.799 GB of weight traffic per token, so 103.1 GB/s effective against 135.4 GB/s
/// of read bandwidth — median of seven runs, 132.1 to 136.0, machine idle — so **0.76 of
/// the ceiling.**
///
/// Taking it exposed a bug in the ruler first. `sift ls` was reporting 306 tok/s for the
/// model measured here at 129, because [`sift_core::doctor::measure_read_bandwidth`] would
/// intermittently return 372 GB/s on a bus that peaks near 153. Calibrating against that
/// would have produced a confidently wrong constant, which is the argument for checking a
/// prediction against a real engine rather than against a benchmark of your own.
///
/// The figure it replaces was 0.28, so every MoE estimate was low by 2.8×: `sift ls` called
/// that model 46 tok/s where the engine delivers 129.
///
/// The guess was wrong because its premise was. "Expert gather is scattered, so MoE decode
/// cannot stream" is true of the *addresses* and false of the *bytes*: one expert here is
/// 3.81 MB of contiguous weights, and eight of them per layer is eight large sequential
/// reads, not a random walk. Scattered megabytes stream at nearly the same rate as
/// sequential ones. A model with far smaller experts would gather less efficiently, and
/// this constant would then be optimistic — which is the honest limit of one data point.
///
/// Measured the same way as [`DENSE_EFFICIENCY`], against the same ruler in the same
/// session, so the two are comparable: dense re-measured at 21.11 tok/s that day, 118.6
/// GB/s effective, 0.88 of the same 135.4. MoE decode is therefore ~87% as efficient as
/// dense on this machine, not ~31% as the old pair of constants claimed.
///
/// Both constants are fitted against `sift`'s own traffic model, which counts the token
/// embedding table as read every token when only one row of it is. That bias is small
/// (~7% here) and is absorbed by the constant rather than left to cancel by luck — which
/// is also why these two numbers must always be re-derived together if that model changes.
pub const MOE_EFFICIENCY: f64 = 0.76;

/// Evaluate every quantization a repo offers, at a given context length.
pub fn evaluate(
    repo: &str,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Result<Vec<Candidate>> {
    let files = hub::list_weights(repo)?;
    let (whole, sets) = hub::group_shards(&files);

    // Safetensors keeps its architecture numbers in a separate `config.json`, so fetch it
    // once for the whole repo rather than per candidate. `None` means the KV cache cannot
    // be sized and `fits` covers weights only — reported, not assumed away.
    let config = if files.iter().any(|f| f.is_safetensors())
        || sets.iter().any(|s| s.base.is_safetensors())
    {
        hub::fetch_config(repo).map(|(cfg, _)| cfg)
    } else {
        None
    };

    // Everything to evaluate, as one list, so the workers below need no per-kind logic.
    let work: Vec<Work> = whole
        .iter()
        .map(Work::Whole)
        .chain(sets.iter().map(Work::Split))
        .collect();

    // The sweep is entirely network-bound — 25 quantizations meant 25 serial round-trips,
    // and a fully split repo like Qwen3-235B is 72 of them. Threads spend their lives
    // blocked on a socket, so the useful count is set by politeness to the hub rather than
    // by cores; 8 is enough to hide latency without looking like a scraper.
    let workers = work.len().clamp(1, 8);

    let queue = std::sync::Mutex::new(work.into_iter());
    let out = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some(item) = queue.lock().expect("queue lock").next() else {
                    return;
                };
                if let Some(c) = evaluate_one(
                    repo,
                    &item,
                    config.as_ref(),
                    mem_bytes_per_sec,
                    usable_ram,
                    context_tokens,
                ) {
                    out.lock().expect("results lock").push(c);
                }
            });
        }
    });

    let mut out = out.into_inner().expect("results lock");
    // Sort here rather than relying on arrival order, which is now nondeterministic.
    out.sort_by_key(|c| c.size_bytes);
    Ok(out)
}

/// One unit of work for the sweep.
enum Work<'a> {
    Whole(&'a hub::RepoFile),
    Split(&'a hub::ShardSet),
}

/// Evaluate one quantization, whether it ships as one file or several.
///
/// Returns `None` when the headers could not be read, having said why on stderr. A wrong
/// row in this table is worse than a missing one, so nothing is guessed at.
fn evaluate_one(
    repo: &str,
    item: &Work,
    config: Option<&serde_json::Value>,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Option<Candidate> {
    match item {
        Work::Whole(f) => {
            let shape = match read_shape(repo, std::slice::from_ref(&f.path), config) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  skipped {}: {e}", f.quant_label());
                    return None;
                }
            };
            let size = f.size.unwrap_or_else(|| shape.total_tensor_bytes());
            Some(build(
                f.quant_label(),
                f.path.clone(),
                size,
                &shape,
                format_of(&f.path),
                mem_bytes_per_sec,
                usable_ram,
                context_tokens,
            ))
        }
        Work::Split(set) => {
            if !set.is_complete() {
                eprintln!(
                    "  skipped {}: {} of {} shards present in the repo",
                    set.base.quant_label(),
                    set.parts.len(),
                    set.expected
                );
                return None;
            }

            // Read every part, not just the first. Each shard carries only its own slice
            // of the tensor directory, so asking part 1 for the model's size or expert
            // layout gives an answer that is confidently a third of the truth.
            let paths: Vec<String> = set.parts.iter().map(|p| p.path.clone()).collect();
            let shape = match read_shape(repo, &paths, config) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  skipped {}: {e}", set.base.quant_label());
                    return None;
                }
            };
            let size = set.size().unwrap_or_else(|| shape.total_tensor_bytes());
            Some(build(
                set.base.quant_label(),
                shard_glob(&set.base.path),
                size,
                &shape,
                format_of(&set.base.path),
                mem_bytes_per_sec,
                usable_ram,
                context_tokens,
            ))
        }
    }
}

/// A `--include` pattern matching every part of a split model.
///
/// Built from the set's base path, so `Q4_K_M/Model-Q4_K_M.gguf` becomes
/// `--include "Q4_K_M/Model-Q4_K_M-*.gguf"`. Quoted because the shell would otherwise
/// expand the glob against the local directory before the tool ever sees it.
///
/// The extension is carried through rather than assumed: a safetensors set given a
/// `.gguf` pattern matches nothing, and the user finds out only after the command runs.
fn shard_glob(base_path: &str) -> String {
    for ext in [".gguf", ".safetensors"] {
        if let Some(stem) = base_path.strip_suffix(ext) {
            return format!("--include \"{stem}-*{ext}\"");
        }
    }
    format!("--include \"{base_path}-*\"")
}

/// Read every part's header and merge them into one shape.
///
/// Dispatches on extension. Both formats are read the same way — a few kilobytes of range
/// requests, no payload — but they carry their architecture numbers in different places:
/// GGUF in its own header, safetensors in the repo's `config.json`, which the caller has
/// already fetched once for the whole sweep.
fn read_shape(
    repo: &str,
    paths: &[String],
    config: Option<&serde_json::Value>,
) -> Result<model::ModelShape> {
    let first = paths.first().context("a model with no files")?;

    if first.ends_with(".safetensors") {
        let mut tensors = Vec::new();
        for p in paths {
            let mut remote = RemoteFile::new(hf_url(repo, p));
            let st =
                model::Safetensors::parse(&mut remote).with_context(|| format!("reading {p}"))?;
            tensors.extend(st.tensors);
        }
        let facts = config
            .map(model::safetensors::facts_from_config)
            .unwrap_or_default();
        return Ok(model::ModelShape::new(facts, tensors, paths.len() as u32));
    }

    let mut parts = Vec::with_capacity(paths.len());
    for p in paths {
        let mut remote = RemoteFile::new(hf_url(repo, p));
        parts.push(
            sift_core::model::Gguf::parse(&mut remote).with_context(|| format!("reading {p}"))?,
        );
    }
    model::ModelShape::sharded(&parts).context("no parts to merge")
}

/// Keep only the candidates whose quantization matches `quant`.
///
/// `sift fit repo:Q4_K_M` asks a narrower question than `sift fit repo` — does *this* file
/// fit, and at what speed — so the table shrinks to it rather than the suffix being sent to
/// the hub as part of the repo name, which is what happened before and produced a `401`.
/// Matches the same way `route` and `inspect` pick a file: exact label first, then a
/// substring so `Q4` finds `Q4_K_M`, case-insensitively throughout.
pub fn narrow(cands: Vec<Candidate>, quant: &str) -> Result<Vec<Candidate>> {
    let want = quant.to_ascii_uppercase();
    let exact: Vec<&Candidate> = cands
        .iter()
        .filter(|c| c.label.to_ascii_uppercase() == want)
        .collect();
    let keep: Vec<bool> = if exact.is_empty() {
        cands
            .iter()
            .map(|c| c.label.to_ascii_uppercase().contains(&want))
            .collect()
    } else {
        cands
            .iter()
            .map(|c| c.label.to_ascii_uppercase() == want)
            .collect()
    };

    if !keep.iter().any(|&k| k) {
        let available: Vec<&str> = cands.iter().map(|c| c.label.as_str()).collect();
        anyhow::bail!(
            "no quantization matching `{quant}` in this repo. Available: {}",
            available.join(", ")
        );
    }

    Ok(cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect())
}

/// Turn a model's shape into an evaluated candidate.
///
/// Shared by the single-file and split paths so the two cannot drift. A split model that
/// scored differently from the same weights in one file would be a bug nobody would spot.
/// The weight format a repo file is in, from its extension.
///
/// The same rule `Source::format` applies, kept to one line here because the sweep sees
/// repo paths rather than resolved sources.
fn format_of(path: &str) -> Format {
    if path.ends_with(".safetensors") {
        Format::Safetensors
    } else {
        Format::Gguf
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    label: String,
    download_arg: String,
    size: u64,
    shape: &model::ModelShape,
    format: Format,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Candidate {
    let (traffic, is_moe) = match model::infer_moe_shape(shape) {
        Some(moe) => (
            TokenTraffic {
                expert_bytes: moe.expert_bytes_per_token(),
                trunk_bytes: shape
                    .total_tensor_bytes()
                    .saturating_sub(moe.total_expert_bytes()),
            },
            true,
        ),
        // Dense: every weight is read every token.
        None => (
            TokenTraffic {
                expert_bytes: 0,
                trunk_bytes: shape.total_tensor_bytes(),
            },
            false,
        ),
    };

    // A model that fits only with an empty context is a model that fails an hour in.
    // Classify on weights plus cache, not on the file size.
    let kv_bytes = model::infer_kv_shape(shape)
        .map(|kv| kv.bytes_at(context_tokens, model::KV_F16_BYTES))
        .unwrap_or(0);

    let regime = Regime::classify(size.saturating_add(kv_bytes), usable_ram);
    let efficiency = if is_moe {
        MOE_EFFICIENCY
    } else {
        DENSE_EFFICIENCY
    };

    // Only a resident model runs at memory speed. Past the boundary the bottleneck
    // becomes the disk, and pretending otherwise is how a router misleads people.
    let est_tok_s = match regime {
        Regime::Fits | Regime::Tight => traffic.tokens_per_sec(mem_bytes_per_sec * efficiency, 0.0),
        Regime::Oversized => f64::NAN,
    };

    Candidate {
        label,
        download_arg,
        shard_count: shape.shard_count,
        size_bytes: size,
        traffic,
        regime,
        est_tok_s,
        is_moe,
        bits_per_weight: shape.bits_per_weight(),
        kv_bytes,
        format,
    }
}

/// The candidate `report` would recommend, if any.
///
/// Shared by both output paths so the human table and the JSON document can never
/// disagree about which quantization to download.
pub fn recommended(cands: &[Candidate]) -> Option<&Candidate> {
    cands
        .iter()
        .filter(|c| c.regime != Regime::Oversized)
        .max_by_key(|c| c.rank())
}

/// Emit the same evaluation as [`report`], as one JSON document on stdout.
pub fn report_json(
    repo: &str,
    cands: &[Candidate],
    facts: &sift_core::doctor::MachineFacts,
    usable_ram: u64,
    memory_gb_per_sec: f64,
    context_tokens: u64,
) -> Result<()> {
    let rows: Vec<_> = cands
        .iter()
        .map(|c| {
            serde_json::json!({
                "quant": c.label,
                "size_bytes": c.size_bytes,
                "kv_cache_bytes": c.kv_bytes,
                "footprint_bytes": c.footprint_bytes(),
                "regime": c.regime,
                "bits_per_weight": c.bits_per_weight,
                // Named `below_quality_floor` rather than `damaged` so the field says what
                // was measured, not what to conclude from it.
                "below_quality_floor": !c.meets_quality_floor(),
                "bytes_per_token": c.traffic.total(),
                "is_moe": c.is_moe,
                "shard_count": c.shard_count,
                // Null rather than 0 when disk-bound: a consumer that plots this must not
                // read "we did not estimate" as "zero tokens per second".
                "estimated_tokens_per_sec": (!c.est_tok_s.is_nan()).then_some(c.est_tok_s),
                "download_arg": c.download_arg,
            })
        })
        .collect();

    let best = recommended(cands);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "repo": repo,
            "machine": {
                "model": facts.model,
                "ram_bytes": facts.ram_bytes,
                "usable_memory_bytes": usable_ram,
                "measured_memory_gb_per_sec": memory_gb_per_sec,
                "accel_memory": facts.accel_memory,
            },
            "context_tokens": context_tokens,
            "quality_floor_bits_per_weight": QUALITY_FLOOR_BPW,
            "candidates": rows,
            "recommended": best.map(|c| serde_json::json!({
                "quant": c.label,
                "download_arg": c.download_arg,
                "below_quality_floor": !c.meets_quality_floor(),
            })),
            "estimates": {
                "dense_efficiency": DENSE_EFFICIENCY,
                "moe_efficiency": MOE_EFFICIENCY,
                "note": "tokens per second are estimates from measured memory bandwidth, not measurements",
            },
        }))?
    );
    Ok(())
}

/// Print the table and a recommendation.
pub fn report(
    repo: &str,
    cands: &[Candidate],
    usable_ram: u64,
    installed: &[Installed],
    context_tokens: u64,
) -> Result<()> {
    if cands.is_empty() {
        anyhow::bail!("no readable GGUF files in `{repo}`");
    }

    // State the assumption before the table, because `fits` is only meaningful relative to
    // it. The same model fits at 4k and does not at 128k, and a table that hides which one
    // it answered is worse than no table.
    let kv = cands.iter().map(|c| c.kv_bytes).max().unwrap_or(0);
    if kv > 0 {
        println!(
            "\n  fits assumes {} tokens of context: {:.2} GB of f16 KV cache, counted\n  \
             alongside the weights. Change it with --ctx.",
            context_tokens,
            kv as f64 / 1e9
        );
    } else {
        println!(
            "\n  KV cache not counted: this model's metadata does not state its attention\n  \
             shape, so `fits` covers weights only and is optimistic at long context."
        );
    }

    println!(
        "\n  {:<14} {:>9} {:>8} {:>6} {:>11} {:>10}",
        "quant", "size", "fits", "bpw", "GB/token", "est tok/s"
    );
    for c in cands {
        let fits = match c.regime {
            Regime::Fits => "yes",
            Regime::Tight => "tight",
            Regime::Oversized => "no",
        };
        let tps = if c.est_tok_s.is_nan() {
            "  disk-bound".to_string()
        } else {
            format!("{:>10.0}", c.est_tok_s)
        };
        let bpw = match c.bits_per_weight {
            Some(b) => format!("{b:>6.2}"),
            None => format!("{:>6}", "?"),
        };
        // Annotations go after the numbers, never inside the label: a label that grows to
        // "UD-Q2_K_XL (2 parts)" overflows its column and unaligns every row below it.
        let mut notes = vec![if c.is_moe { "moe" } else { "dense" }.to_string()];
        if c.shard_count > 1 {
            notes.push(format!("{} parts", c.shard_count));
        }
        if !c.meets_quality_floor() {
            notes.push("damaged".to_string());
        }

        println!(
            "  {:<14} {:>6.2} GB {:>8} {} {:>11.3} {}  {}",
            c.label,
            c.size_bytes as f64 / 1e9,
            fits,
            bpw,
            // Worth showing: a MoE row's GB/token is far below its file size, and that gap
            // is the whole reason these models are fast. Hiding it makes the table look
            // like an arithmetic error.
            c.traffic.total() as f64 / 1e9,
            tps,
            notes.join(", ")
        );
    }

    // Rank on quality first, size never. See `Candidate::rank`: on Qwen3-30B-A3B, picking
    // the largest file that fits elects a 1-bit quant over a 3-bit one barely larger.
    let best = recommended(cands);

    match best {
        Some(c) => {
            // Route on the footprint, so an engine is not recommended for a model that
            // only fits with an empty context — and on the candidate's own format, so a
            // safetensors repo is not sent to an engine that only reads GGUF.
            let rec = engine::route(c.footprint_bytes(), c.format, usable_ram, installed);
            println!("\n  recommended: {}", c.label);
            if let Some(e) = &rec.engine {
                println!("    engine     {} — {}", e.name, rec.reason);
                if !rec.installed {
                    println!("    not installed: {}", e.install);
                }
            }
            // Say it out loud when the best available option is a damaged one. The user is
            // about to spend an hour downloading it, and "it fits" is not the same claim
            // as "it is worth running".
            if !c.meets_quality_floor() {
                println!(
                    "\n    warning    at {:.2} bits per weight this is below the {:.1}-bit\n\
                     {:>15}floor where quantization stops degrading gracefully. Nothing\n\
                     {:>15}better fits here. Expect noticeably worse output than the same\n\
                     {:>15}model at 4 bits — consider a smaller model instead.",
                    c.bits_per_weight.unwrap_or(0.0),
                    QUALITY_FLOOR_BPW,
                    "",
                    "",
                    ""
                );
            }
            println!(
                "    download   huggingface-cli download {repo} {}",
                c.download_arg
            );
        }
        None => {
            println!(
                "\n  nothing here fits in {:.1} GiB of usable memory.",
                sift_core::gib(usable_ram)
            );
            let smallest = cands.first().context("no candidates")?;
            let rec = engine::route(
                smallest.footprint_bytes(),
                smallest.format,
                usable_ram,
                installed,
            );
            if let Some(e) = &rec.engine {
                println!(
                    "    the smallest option is {} at {:.2} GB.",
                    smallest.label,
                    smallest.size_bytes as f64 / 1e9
                );
                println!("    {} can still run it: {}", e.name, rec.reason);
                if !rec.installed {
                    println!("    not installed: {}", e.install);
                }
            }
            for (e, why) in &rec.avoid {
                println!("    avoid {}: {why}", e.name);
            }
        }
    }

    println!(
        "\n  tok/s figures are estimates from measured memory bandwidth, not measurements.\n  \
         MoE rows assume {:.0}% of roofline, dense rows {:.0}%. Run `sift bench` for real numbers.",
        MOE_EFFICIENCY * 100.0,
        DENSE_EFFICIENCY * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(label: &str, gb: f64, bpw: Option<f64>, regime: Regime) -> Candidate {
        let size_bytes = (gb * 1e9) as u64;
        Candidate {
            label: label.into(),
            download_arg: format!("{label}.gguf"),
            size_bytes,
            traffic: TokenTraffic {
                expert_bytes: 0,
                trunk_bytes: size_bytes,
            },
            regime,
            est_tok_s: 10.0,
            is_moe: false,
            bits_per_weight: bpw,
            // The regime is supplied directly by these tests, so the cache is already
            // accounted for in whatever the caller passed.
            kv_bytes: 0,
            shard_count: 1,
            format: Format::Gguf,
        }
    }

    /// Pick the way `report` does.
    fn pick(cands: &[Candidate]) -> &Candidate {
        cands
            .iter()
            .filter(|c| c.regime != Regime::Oversized)
            .max_by_key(|c| c.rank())
            .expect("a candidate")
    }

    #[test]
    fn a_three_bit_quant_that_is_tight_beats_a_one_bit_quant_that_fits() {
        // The bug this exists to prevent, with the real Qwen3-30B-A3B numbers from the
        // README. Ranking on file size elects UD-IQ1_S because it is the only row marked
        // "yes" — and hands the user a model damaged past the point of being worth an
        // 9 GB download when a usable one was 5 GB away.
        let cands = vec![
            candidate("UD-IQ1_S", 9.04, Some(2.37), Regime::Fits),
            candidate("Q2_K", 11.26, Some(2.95), Regime::Tight),
            candidate("Q3_K_M", 14.71, Some(3.86), Regime::Tight),
            candidate("Q4_K_M", 18.56, Some(4.87), Regime::Oversized),
        ];
        assert_eq!(pick(&cands).label, "Q3_K_M");
    }

    #[test]
    fn among_quants_that_all_clear_the_floor_the_safe_fit_wins_over_the_tight_one() {
        // Both are good enough to run, so the tiebreak stops being quality and becomes
        // whether it will survive a growing KV cache.
        let cands = vec![
            candidate("Q3_K_M", 14.0, Some(3.86), Regime::Fits),
            candidate("Q4_K_M", 18.0, Some(4.87), Regime::Tight),
        ];
        assert_eq!(pick(&cands).label, "Q3_K_M");
    }

    #[test]
    fn within_one_regime_more_bits_wins() {
        let cands = vec![
            candidate("Q4_K_M", 18.0, Some(4.87), Regime::Fits),
            candidate("Q6_K", 25.0, Some(6.56), Regime::Fits),
            candidate("Q3_K_M", 14.0, Some(3.86), Regime::Fits),
        ];
        assert_eq!(pick(&cands).label, "Q6_K");
    }

    #[test]
    fn bits_per_weight_decides_not_file_size() {
        // A dynamic quantization can be the larger file and the less precise one, because
        // it spends its extra bytes on a few sensitive layers rather than uniformly. Size
        // would pick the wrong one; bits per weight does not.
        let cands = vec![
            candidate("UD-Q2_K_XL", 12.0, Some(2.80), Regime::Fits),
            candidate("Q3_K_S", 11.5, Some(3.41), Regime::Fits),
        ];
        let best = pick(&cands);
        assert_eq!(best.label, "Q3_K_S");
        assert!(
            best.size_bytes < cands[0].size_bytes,
            "the smaller file won"
        );
    }

    #[test]
    fn when_nothing_clears_the_floor_the_best_available_is_still_named() {
        // Refusing to answer is not more honest than answering with a warning. The user
        // has this machine and wants this model; say which is least bad and why.
        let cands = vec![
            candidate("IQ1_S", 6.0, Some(1.78), Regime::Fits),
            candidate("IQ2_XXS", 8.0, Some(2.10), Regime::Fits),
        ];
        let best = pick(&cands);
        assert_eq!(best.label, "IQ2_XXS");
        assert!(
            !best.meets_quality_floor(),
            "and it must be flagged as such"
        );
    }

    #[test]
    fn unknown_precision_is_not_treated_as_damaged() {
        // A file whose bits per weight could not be computed is an unmeasured one, not a
        // bad one. Excluding it would silently drop valid options — the same failure as
        // guessing a number we do not have.
        let c = candidate("MYSTERY", 10.0, None, Regime::Fits);
        assert!(c.meets_quality_floor());
    }

    #[test]
    fn the_floor_sits_exactly_at_three_bits() {
        assert!(candidate("x", 1.0, Some(3.0), Regime::Fits).meets_quality_floor());
        assert!(!candidate("x", 1.0, Some(2.999), Regime::Fits).meets_quality_floor());
    }

    #[test]
    fn an_oversized_quant_is_never_recommended_however_good_it_is() {
        let cands = vec![
            candidate("Q3_K_M", 14.0, Some(3.86), Regime::Fits),
            candidate("F16", 60.0, Some(16.0), Regime::Oversized),
        ];
        assert_eq!(pick(&cands).label, "Q3_K_M");
    }

    fn labels(cands: &[Candidate]) -> Vec<&str> {
        cands.iter().map(|c| c.label.as_str()).collect()
    }

    fn three_quants() -> Vec<Candidate> {
        vec![
            candidate("Q4_K_M", 18.0, Some(4.8), Regime::Fits),
            candidate("Q4_K_S", 17.0, Some(4.5), Regime::Fits),
            candidate("Q8_0", 32.0, Some(8.5), Regime::Oversized),
        ]
    }

    #[test]
    fn an_exact_quant_narrows_to_that_one_file() {
        let kept = narrow(three_quants(), "Q4_K_M").unwrap();
        assert_eq!(labels(&kept), ["Q4_K_M"]);
    }

    #[test]
    fn a_partial_quant_keeps_every_match_rather_than_guessing_one() {
        // `Q4` is ambiguous between two files here, and the table is the right place to
        // show both — picking silently would hide the choice.
        let kept = narrow(three_quants(), "q4").unwrap();
        assert_eq!(labels(&kept), ["Q4_K_M", "Q4_K_S"]);
    }

    #[test]
    fn an_exact_match_beats_a_substring_that_would_also_match() {
        // `Q4_K_M` is also a substring of `UD-Q4_K_M`; the exact file wins outright.
        let cands = vec![
            candidate("UD-Q4_K_M", 19.0, Some(4.9), Regime::Fits),
            candidate("Q4_K_M", 18.0, Some(4.8), Regime::Fits),
        ];
        assert_eq!(labels(&narrow(cands, "Q4_K_M").unwrap()), ["Q4_K_M"]);
    }

    #[test]
    fn an_unknown_quant_lists_what_the_repo_actually_has() {
        let err = narrow(three_quants(), "IQ2_XXS").unwrap_err().to_string();
        assert!(err.contains("IQ2_XXS"), "{err}");
        assert!(err.contains("Q4_K_M") && err.contains("Q8_0"), "{err}");
    }
}

/// The two efficiency constants, pinned to the measurements they came from.
///
/// Both are hand-set numbers that silently multiply every tok/s figure the tool prints, so
/// each is held against the run it was fitted to. Anyone editing one has to come back here
/// and say which measurement replaced it — which is the point, since the last time one of
/// these moved on a hunch it shipped a 2.8× error for months.
#[cfg(test)]
mod calibration_tests {
    use super::*;

    /// Read bandwidth on the calibration machine, idle: median of seven runs spanning
    /// 132.1 to 136.0 GB/s.
    const MEASURED_READ_BYTES_PER_SEC: f64 = 135.4e9;

    /// What `sift` predicts for a model of `bytes_per_token` at that bandwidth.
    fn predicted(bytes_per_token: u64, efficiency: f64) -> f64 {
        let traffic = TokenTraffic {
            expert_bytes: 0,
            trunk_bytes: bytes_per_token,
        };
        traffic.tokens_per_sec(MEASURED_READ_BYTES_PER_SEC * efficiency, 0.0)
    }

    #[test]
    fn the_moe_factor_predicts_what_lm_studio_actually_did() {
        // OLMoE-1B-7B Q4_K_M, LM Studio on an M5: 129.10 tok/s median over three runs of
        // 415 tokens each, against 798,615,552 bytes of weight traffic per token as
        // `sift plan` computes it.
        let predicted = predicted(798_615_552, MOE_EFFICIENCY);
        let measured = 129.10;
        let error = (predicted - measured).abs() / measured;
        assert!(
            error < 0.05,
            "predicted {predicted:.1} tok/s against a measured {measured:.1}: {:.1}% out",
            error * 100.0
        );
    }

    #[test]
    fn the_dense_factor_still_predicts_the_run_it_was_fitted_to() {
        // Qwen3.5-9B Q4_K_M, same machine, same engine: 21.03 tok/s when the factor was
        // set, 21.11 when it was re-checked alongside the MoE measurement.
        let predicted = predicted(5_616_076_800, DENSE_EFFICIENCY);
        let error = (predicted - 21.07).abs() / 21.07;
        assert!(
            error < 0.05,
            "predicted {predicted:.2} tok/s against a measured 21.07: {:.1}% out",
            error * 100.0
        );
    }

    #[test]
    fn moe_decode_is_not_assumed_to_be_a_fraction_of_dense() {
        // The premise behind the old 0.28 was that scattered expert gather cannot stream.
        // Measurement says otherwise: one expert is megabytes of contiguous weights, and
        // MoE lands within ~15% of dense efficiency rather than a third of it. This test
        // exists so that reverting to a "MoE is much slower" intuition fails loudly.
        //
        // Checked at compile time rather than at test time: both operands are constants,
        // so a violation should stop the build rather than wait for someone to run tests.
        const {
            assert!(
                MOE_EFFICIENCY > DENSE_EFFICIENCY * 0.7,
                "MoE efficiency is implausibly far below dense; the one measurement \
                 taken says MoE reaches ~87% of it, not a third"
            );
            assert!(
                MOE_EFFICIENCY <= DENSE_EFFICIENCY,
                "scattered gather cannot beat streaming"
            );
        }
    }

    #[test]
    fn neither_factor_claims_more_than_the_machine_has() {
        // A factor above 1.0 would mean decode outruns the memory bus, which is not a
        // calibration but a broken ruler — the exact failure that produced 369 GB/s in
        // `doctor` and 138%-of-machine in the first dense fit.
        for f in [DENSE_EFFICIENCY, MOE_EFFICIENCY] {
            assert!(f > 0.0 && f <= 1.0, "efficiency {f} is not a fraction");
        }
    }
}

#[cfg(test)]
mod download_tests {
    use super::*;

    #[test]
    fn a_split_model_downloads_every_part_not_just_the_first() {
        // Naming one shard fetches a third of a model that then fails to load — worse
        // than giving no hint at all.
        let g = shard_glob("Q4_K_M/Qwen3-235B-A22B-Q4_K_M.gguf");
        assert_eq!(g, "--include \"Q4_K_M/Qwen3-235B-A22B-Q4_K_M-*.gguf\"");
        assert!(g.contains('"'), "must be quoted against shell expansion");
    }

    #[test]
    fn a_safetensors_set_keeps_its_own_extension() {
        // A `.gguf` pattern on a safetensors set matches nothing, and the user finds out
        // only after the download command runs and fetches zero files.
        assert_eq!(
            shard_glob("model.safetensors"),
            "--include \"model-*.safetensors\""
        );
    }

    #[test]
    fn a_base_path_without_a_known_extension_still_yields_a_usable_pattern() {
        assert_eq!(shard_glob("Model-Q8_0"), "--include \"Model-Q8_0-*\"");
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use sift_core::engine::Installed;

    #[test]
    fn a_safetensors_repo_is_not_routed_to_a_gguf_only_engine() {
        // The bug this pins, found by running `sift fit allenai/OLMoE-1B-7B-0924-Instruct`
        // on a real machine: the sweep hardcoded GGUF when routing, so a safetensors repo
        // was answered with "use LM Studio", which cannot open one. `route` had already
        // been taught to dispatch on format; `fit` had its own copy of the call and had
        // not.
        let installed: Vec<Installed> = engine::ENGINES
            .iter()
            .filter(|e| e.id == "lm-studio" || e.id == "mlx")
            .map(|e| Installed {
                engine: e.clone(),
                found_at: "/test".into(),
            })
            .collect();

        let rec = engine::route(4 << 30, Format::Safetensors, 12 << 30, &installed);
        let chosen = rec.engine.as_ref().expect("something must be recommended");
        assert!(
            chosen.formats.contains(&Format::Safetensors),
            "recommended {} which cannot read safetensors",
            chosen.name
        );
    }

    #[test]
    fn format_comes_from_the_file_extension_not_the_repo() {
        // A repo can hold both. Deciding once for the whole sweep would mislabel whichever
        // kind is in the minority.
        assert_eq!(
            format_of("model-00001-of-00003.safetensors"),
            Format::Safetensors
        );
        assert_eq!(format_of("Q4_K_M/Model-Q4_K_M.gguf"), Format::Gguf);
        // Anything unrecognised stays GGUF, which is what the hub listing is overwhelmingly
        // made of, and is the format every local engine reads.
        assert_eq!(format_of("model.bin"), Format::Gguf);
    }
}
