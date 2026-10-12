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
use sift_core::model::{self, hf_url_at, RemoteFile, TokenTraffic};

/// One evaluated quantization.
#[derive(Debug)]
pub struct Candidate {
    pub label: String,
    /// The file this row describes, or the path a split set would have as one file.
    pub path: String,
    /// What to pass `hf download` to get this quantization.
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
    /// MoE only: top-k was not stated, so the traffic figure rests on an assumed 8.
    pub top_k_assumed: bool,
    /// Mean bits stored per weight, measured from this file's tensor directory.
    pub bits_per_weight: Option<f64>,
    /// KV cache bytes at the context this evaluation assumed.
    pub kv_bytes: u64,
    /// Whether the metadata was enough to size the KV cache at all. A sized cache can be
    /// zero (an all-recurrent model); an unsized one is unknown, not zero.
    pub kv_known: bool,
    /// The context the model was trained for, when it says so.
    pub train_context: Option<u64>,
    /// Files this quantization ships as. 1 for the ordinary case.
    pub shard_count: u32,
    /// Tensors of a dtype this build cannot size, left out of `bits_per_weight`.
    pub dropped_tensors: u64,
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

/// What to evaluate: a repo, optionally at a revision, optionally one file of it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Target<'a> {
    pub repo: &'a str,
    /// Branch, tag or commit. `None` is the default branch.
    pub revision: Option<&'a str>,
    /// Evaluate only the model stored at this path, or the split set it belongs to.
    pub only: Option<&'a str>,
    /// Evaluate only the quantizations this selects. See [`narrow`].
    pub quant: Option<&'a str>,
}

/// A quantization whose headers have been read, waiting to be scored against a machine.
pub struct Swept {
    label: String,
    path: String,
    download_arg: String,
    size: u64,
    shape: model::ModelShape,
    format: Format,
}

/// One model to read: a single file, or every part of a split set.
struct Unit {
    label: String,
    path: String,
    download_arg: String,
    size: Option<u64>,
    parts: Vec<String>,
}

/// A parsed header, of whichever format the part is in.
enum Header {
    Gguf(model::Gguf),
    Safetensors(model::Safetensors),
}

/// What a repo offers, listed but not yet read.
pub struct Listed<'a> {
    repo: &'a str,
    revision: String,
    units: Vec<Unit>,
    config: Option<serde_json::Value>,
}

/// List a repo and decide what to read, without reading a single header.
///
/// Together with [`read`], this reads every quantization a repo offers without
/// downloading any of it. Both are network-bound and built to overlap: the repo's
/// `config.json` is fetched while the listing is, and every *part* of every model is its
/// own unit of work in [`read`]. One unit per model left a 2-part BF16 set — or
/// DeepSeek-V3's 163 shards — on a single worker while the other seven sat idle.
///
/// This half is pure network wait: the listing and `config.json` round-trips. `fit` runs its bandwidth
/// probe during this phase and not during [`read`], whose header parsing would compete
/// with the probe for the memory bus and make the machine look slower than it is.
pub fn list(target: Target) -> Result<Listed> {
    let repo = target.repo;
    // The config is fetched at the revision asked for, alongside the listing rather than
    // after it: a safetensors repo needs it, and it costs one small request either way.
    let (listing, config) = std::thread::scope(|scope| {
        let config = scope.spawn(|| hub::fetch_config(repo, target.revision.unwrap_or("main")));
        let listing = hub::list_weights(repo, target.revision);
        (listing, config.join().ok().flatten().map(|(cfg, _)| cfg))
    });
    let listing = listing?;
    // Pin every header fetch to the commit the listing describes.
    let revision = listing.revision(target.revision).to_string();
    let listing_indexes = listing.indexes;
    let mut files = listing.files;
    if let Some(only) = target.only {
        let base = |p: &str| hub::shard_position(p).map(|(b, _, _)| b);
        let wanted = base(only).unwrap_or_else(|| only.to_string());
        files.retain(|f| f.path == only || base(&f.path).as_deref() == Some(wanted.as_str()));
        if files.is_empty() {
            anyhow::bail!("`{only}` is not a weight file in `{repo}`");
        }
    }
    let indexed = hub::fetch_indexes(repo, &revision, &listing_indexes, &files);
    let (whole, sets) = hub::group_models(&files, &indexed);
    // Only safetensors keeps its architecture numbers in `config.json`; a GGUF carries its
    // own, and a stray config beside it must not override them.
    let config = config.filter(|_| {
        files.iter().any(|f| f.is_safetensors()) || sets.iter().any(|s| s.base.is_safetensors())
    });

    let (mut units, skipped) = plan_units(&whole, &sets);
    for why in skipped {
        eprintln!("  skipped {why}");
    }

    // `repo:QUANT` narrows before a single header is read: the labels come from file
    // names, and reading 25 headers to show one row was most of the time `fit` took.
    if let Some(q) = target.quant {
        let labels: Vec<&str> = units.iter().map(|u| u.label.as_str()).collect();
        let keep = matching(&labels, q)?;
        let mut keep = keep.into_iter();
        units.retain(|_| keep.next().unwrap_or(false));
    }

    Ok(Listed {
        repo,
        revision,
        units,
        config,
    })
}

/// Read every listed model's headers, one part per unit of work.
pub fn read(listed: Listed) -> Result<Vec<Swept>> {
    let Listed {
        repo,
        revision,
        units,
        config,
    } = listed;
    let jobs: Vec<(usize, usize)> = units
        .iter()
        .enumerate()
        .flat_map(|(u, unit)| (0..unit.parts.len()).map(move |p| (u, p)))
        .collect();
    let slots: Vec<Vec<std::sync::OnceLock<Result<Header>>>> = units
        .iter()
        .map(|u| u.parts.iter().map(|_| std::sync::OnceLock::new()).collect())
        .collect();

    // Threads spend their lives blocked on a socket, so the useful count is set by
    // politeness to the hub rather than by cores; 8 hides latency without looking like a
    // scraper.
    let workers = jobs.len().clamp(1, 8);
    let queue = std::sync::Mutex::new(jobs.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some((u, p)) = queue.lock().expect("queue lock").next() else {
                    return;
                };
                let header = read_part(repo, &revision, &units[u].parts[p], p);
                let _ = slots[u][p].set(header);
            });
        }
    });

    let mut out = Vec::with_capacity(units.len());
    for (unit, parts) in units.into_iter().zip(slots) {
        match assemble(parts, config.as_ref()) {
            Ok(shape) if not_a_model(&shape).is_some() => eprintln!(
                "  skipped {}: {}",
                unit.path,
                not_a_model(&shape).unwrap_or_default()
            ),
            Ok(shape) => out.push(Swept {
                size: unit.size.unwrap_or_else(|| shape.total_tensor_bytes()),
                format: engine::format_of(&unit.path, config.as_ref()),
                label: unit.label,
                path: unit.path,
                download_arg: unit.download_arg,
                shape,
            }),
            Err(e) => eprintln!("  skipped {}: {e:#}", unit.label),
        }
    }
    Ok(out)
}

/// Score swept quantizations against this machine, at a given context length.
pub fn score(
    swept: Vec<Swept>,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = swept
        .into_iter()
        .map(|s| {
            build(
                s.label,
                s.path,
                s.download_arg,
                s.size,
                &s.shape,
                s.format,
                mem_bytes_per_sec,
                usable_ram,
                context_tokens,
            )
        })
        .collect();
    out.sort_by_key(|c| c.size_bytes);
    disambiguate(&mut out);
    out
}

/// Give every row a label of its own.
///
/// Two files can carry the same quantization tag — `BF16/Model-BF16.gguf` beside a
/// `Model-BF16.gguf`, or two safetensors sets both called `model`. Shown twice, `BF16`
/// leaves the user to guess which one was recommended; the rows that collide are labelled
/// by their path instead.
fn disambiguate(cands: &mut [Candidate]) {
    use std::collections::HashMap;
    let mut seen: HashMap<String, usize> = HashMap::new();
    for c in cands.iter() {
        *seen.entry(c.label.to_ascii_uppercase()).or_default() += 1;
    }
    for c in cands.iter_mut() {
        if seen[&c.label.to_ascii_uppercase()] > 1 {
            c.label = c
                .path
                .strip_suffix(".gguf")
                .or_else(|| c.path.strip_suffix(".safetensors"))
                .unwrap_or(&c.path)
                .to_string();
        }
    }
}

/// Read one part's header. Payload is never fetched.
///
/// Both formats are read the same way — a few kilobytes of range requests — but sized
/// differently up front: a safetensors header is 8 bytes of length and then exactly the
/// header, and only the first part of a split GGUF carries the tokenizer.
fn read_part(repo: &str, revision: &str, path: &str, index: usize) -> Result<Header> {
    let url = hf_url_at(repo, revision, path);
    if path.ends_with(".safetensors") {
        let mut remote = RemoteFile::new(url).with_initial_fetch(8);
        let st =
            model::Safetensors::parse(&mut remote).with_context(|| format!("reading {path}"))?;
        remote.persist();
        return Ok(Header::Safetensors(st));
    }
    let mut remote = RemoteFile::new(url);
    if index > 0 {
        remote = remote.with_initial_fetch(64 << 10);
    }
    let g = model::Gguf::parse(&mut remote).with_context(|| format!("reading {path}"))?;
    remote.persist();
    Ok(Header::Gguf(g))
}

/// Merge one model's parts, in order, into its shape.
///
/// GGUF carries its architecture numbers in its own first part; safetensors in the repo's
/// `config.json`, fetched once for the whole sweep.
fn assemble(
    parts: Vec<std::sync::OnceLock<Result<Header>>>,
    config: Option<&serde_json::Value>,
) -> Result<model::ModelShape> {
    let mut ggufs = Vec::new();
    let mut safetensors = Vec::new();
    for part in parts {
        match part.into_inner().context("a part was never read")?? {
            Header::Gguf(g) => ggufs.push(g),
            Header::Safetensors(st) => safetensors.push(st),
        }
    }
    if !safetensors.is_empty() {
        return Ok(model::safetensors::model_shape(safetensors, config));
    }
    model::ModelShape::sharded(&ggufs).context("a model with no parts")
}

/// Turn grouped files into the models to read, and say what was left out and why.
///
/// Two things are left out. A split set missing parts is not a model that can load. And a
/// safetensors checkpoint in a subdirectory, when the repo has one at its root, is an
/// adjunct or an alternative — MiMo-V2.6's `audio_tokenizer/` and `dflash/` draft model,
/// gpt-oss's `original/` — and listing it as a quantization got a 1.9 GB audio tokenizer
/// recommended. GGUF quants in subdirectories (`BF16/`) are quants and stay.
fn plan_units(whole: &[hub::RepoFile], sets: &[hub::ShardSet]) -> (Vec<Unit>, Vec<String>) {
    let mut units: Vec<Unit> = whole
        .iter()
        .map(|f| Unit {
            label: f.quant_label(),
            path: f.path.clone(),
            download_arg: f.path.clone(),
            size: f.size,
            parts: vec![f.path.clone()],
        })
        .collect();
    let mut skipped = Vec::new();
    for set in sets {
        if !set.is_complete() {
            skipped.push(format!(
                "{}: the repo does not hold all {} parts of it",
                set.base.quant_label(),
                set.expected
            ));
            continue;
        }
        // Every part, not just the first. Each shard carries only its own slice of the
        // tensor directory, so asking part 1 for the model's size or expert layout gives
        // an answer that is confidently a third of the truth.
        let parts: Vec<String> = set.parts.iter().map(|p| p.path.clone()).collect();
        units.push(Unit {
            label: set.base.quant_label(),
            path: set.base.path.clone(),
            download_arg: download_glob(&set.base.path, &parts),
            size: set.size(),
            parts,
        });
    }

    let st_root = units
        .iter()
        .any(|u| u.path.ends_with(".safetensors") && !u.path.contains('/'));
    if st_root {
        units.retain(|u| {
            let adjunct = u.path.ends_with(".safetensors") && u.path.contains('/');
            if adjunct {
                skipped.push(format!(
                    "{}: a checkpoint in a subdirectory beside the repo's own model",
                    u.path
                ));
            }
            !adjunct
        });
    }
    (units, skipped)
}

/// An `--include` pattern matching every part of a multi-file model, and nothing else.
///
/// The set's own name gives it when its parts follow the `-0000x-of-0000N` convention.
/// Parts an index named however their publisher liked get the longest prefix they share.
fn download_glob(base_path: &str, parts: &[String]) -> String {
    if let [only] = parts {
        return only.clone();
    }
    let named = shard_glob(base_path);
    let stem_ext = [".gguf", ".safetensors"]
        .iter()
        .find_map(|ext| base_path.strip_suffix(ext).map(|stem| (stem, *ext)));
    if let Some((stem, ext)) = stem_ext {
        let prefix = format!("{stem}-");
        if parts
            .iter()
            .all(|p| p.starts_with(&prefix) && p.ends_with(ext))
        {
            return named;
        }
    }
    let first = parts.first().map(String::as_str).unwrap_or(base_path);
    let mut common = first.len();
    for p in parts {
        common = common.min(
            first
                .bytes()
                .zip(p.bytes())
                .take_while(|(a, b)| a == b)
                .count(),
        );
    }
    let ext = std::path::Path::new(first)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    format!("--include \"{}*{ext}\"", &first[..common])
}

/// Why a file that parsed is still not a model to recommend, if it is not.
///
/// The filename filter catches `mmproj-*` and `eagle3-*`; this catches the same things
/// shipped under any other name, by what their own headers say they are: a multimodal
/// projector (`clip`), or a speculative-decoding head (EAGLE, DFlash, Medusa) that only
/// runs beside the model it was trained for.
fn not_a_model(shape: &model::ModelShape) -> Option<&'static str> {
    match shape.architecture()? {
        "clip" => Some("a vision or audio projector, not a model"),
        "eagle" | "eagle3" | "dflash" | "medusa" => {
            Some("a speculative-decoding head, which runs only beside its model")
        }
        _ => None,
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

/// Keep only the candidates whose quantization matches `quant`.
///
/// `sift fit repo:Q4_K_M` asks a narrower question than `sift fit repo` — does *this* file
/// fit, and at what speed — so the table shrinks to it rather than the suffix being sent to
/// the hub as part of the repo name, which is what happened before and produced a `401`.
/// Matches the same way `route` and `inspect` pick a file: exact label first, then a
/// substring so `Q4` finds `Q4_K_M`, case-insensitively throughout.
pub fn narrow(cands: Vec<Candidate>, quant: &str) -> Result<Vec<Candidate>> {
    if cands.is_empty() {
        anyhow::bail!("no readable weight files in this repo to match `{quant}` against");
    }
    let labels: Vec<&str> = cands.iter().map(|c| c.label.as_str()).collect();
    let keep = matching(&labels, quant)?;
    Ok(cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect())
}

/// Which of `labels` a quantization request selects: the exact label if any matches, else
/// every label containing it, case-insensitively throughout.
///
/// Shared by the sweep, which narrows before reading a single header, and [`narrow`], so
/// the two cannot select differently.
fn matching(labels: &[&str], quant: &str) -> Result<Vec<bool>> {
    let want = quant.to_ascii_uppercase();
    let exact = labels.iter().any(|l| l.to_ascii_uppercase() == want);
    let keep: Vec<bool> = labels
        .iter()
        .map(|l| {
            let l = l.to_ascii_uppercase();
            if exact {
                l == want
            } else {
                l.contains(&want)
            }
        })
        .collect();
    if !keep.iter().any(|&k| k) {
        anyhow::bail!(
            "no quantization matching `{quant}` in this repo. Available: {}",
            labels.join(", ")
        );
    }
    Ok(keep)
}

/// Turn a model's shape into an evaluated candidate.
///
/// Shared by the single-file and split paths so the two cannot drift. A split model that
/// scored differently from the same weights in one file would be a bug nobody would spot.
#[allow(clippy::too_many_arguments)]
fn build(
    label: String,
    path: String,
    download_arg: String,
    size: u64,
    shape: &model::ModelShape,
    format: Format,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Candidate {
    let moe = model::infer_moe_shape(shape);
    let is_moe = moe.is_some();
    let top_k_assumed = moe.as_ref().is_some_and(|m| m.experts_per_token_assumed);
    let traffic = match &moe {
        Some(moe) => TokenTraffic {
            expert_bytes: moe.expert_bytes_per_token(),
            trunk_bytes: shape
                .total_tensor_bytes()
                .saturating_sub(moe.total_expert_bytes()),
        },
        // Dense: every weight is read every token.
        None => TokenTraffic {
            expert_bytes: 0,
            trunk_bytes: shape.total_tensor_bytes(),
        },
    };

    // A model that fits only with an empty context is a model that fails an hour in.
    // Classify on weights plus cache, not on the file size.
    let kv_shape = model::infer_kv_shape(shape);
    let kv_bytes = kv_shape
        .as_ref()
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
        path,
        download_arg,
        shard_count: shape.shard_count,
        dropped_tensors: shape.dropped_tensors,
        size_bytes: size,
        traffic,
        regime,
        est_tok_s,
        is_moe,
        top_k_assumed,
        bits_per_weight: shape.bits_per_weight(),
        kv_bytes,
        kv_known: kv_shape.is_some(),
        train_context: kv_shape.and_then(|kv| kv.train_context),
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
/// The command that downloads a candidate.
///
/// `hf`, the HuggingFace CLI that replaced the deprecated `huggingface-cli`.
pub fn download_command(repo: &str, c: &Candidate) -> String {
    format!("hf download {repo} {}", c.download_arg)
}

/// Whether the context asked for runs past what the model was trained on.
fn beyond_training(cands: &[Candidate], context_tokens: u64) -> Option<u64> {
    cands
        .iter()
        .filter_map(|c| c.train_context)
        .min()
        .filter(|&t| context_tokens > t)
}

#[allow(clippy::too_many_arguments)]
pub fn report_json(
    repo: &str,
    cands: &[Candidate],
    facts: &sift_core::doctor::MachineFacts,
    usable_ram: u64,
    installed: &[Installed],
    memory_gb_per_sec: f64,
    context_tokens: u64,
) -> Result<()> {
    let rows: Vec<_> = cands
        .iter()
        .map(|c| {
            serde_json::json!({
                "quant": c.label,
                "size_bytes": c.size_bytes,
                // Null when the metadata cannot size it: "unknown" is not "zero".
                "kv_cache_bytes": c.kv_known.then_some(c.kv_bytes),
                "footprint_bytes": c.footprint_bytes(),
                "regime": c.regime,
                "bits_per_weight": c.bits_per_weight,
                // Named `below_quality_floor` rather than `damaged` so the field says what
                // was measured, not what to conclude from it.
                "below_quality_floor": !c.meets_quality_floor(),
                "bytes_per_token": c.traffic.total(),
                "is_moe": c.is_moe,
                "experts_per_token_assumed": c.top_k_assumed,
                "shard_count": c.shard_count,
                "dropped_tensors": c.dropped_tensors,
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
            "context_exceeds_training": beyond_training(cands, context_tokens).is_some(),
            "recommended": best.map(|c| {
                let rec = engine::route(c.footprint_bytes(), c.format, usable_ram, installed);
                serde_json::json!({
                    "quant": c.label,
                    "download_arg": c.download_arg,
                    "download_command": download_command(repo, c),
                    "below_quality_floor": !c.meets_quality_floor(),
                    "engine": rec.engine.as_ref().map(|e| e.id),
                    "installed": rec.installed,
                    "reason": rec.reason,
                })
            }),
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
        anyhow::bail!("no readable weight files in `{repo}`");
    }

    // State the assumption before the table, because `fits` is only meaningful relative to
    // it. The same model fits at 4k and does not at 128k, and a table that hides which one
    // it answered is worse than no table. Whether the cache was sized is a fact about the
    // metadata, not about the number: a cache sized at zero is still sized.
    let kv = cands.iter().map(|c| c.kv_bytes).max().unwrap_or(0);
    if cands.iter().any(|c| c.kv_known) {
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
    if let Some(trained) = beyond_training(cands, context_tokens) {
        println!(
            "  warning: {context_tokens} tokens is past the {trained} this model was trained for;\n  \
             quality past that point depends on how the engine extends the context."
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
        if c.top_k_assumed {
            notes.push(format!(
                "top-k {} assumed",
                model::ASSUMED_EXPERTS_PER_TOKEN
            ));
        }
        if c.shard_count > 1 {
            notes.push(format!("{} parts", c.shard_count));
        }
        if !c.meets_quality_floor() {
            notes.push("damaged".to_string());
        }
        if c.dropped_tensors > 0 {
            notes.push(format!("{} unsized tensors", c.dropped_tensors));
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
            println!("    download   {}", download_command(repo, c));
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
            path: format!("M-{label}.gguf"),
            download_arg: format!("{label}.gguf"),
            size_bytes,
            traffic: TokenTraffic {
                expert_bytes: 0,
                trunk_bytes: size_bytes,
            },
            regime,
            est_tok_s: 10.0,
            is_moe: false,
            top_k_assumed: false,
            bits_per_weight: bpw,
            // The regime is supplied directly by these tests, so the cache is already
            // accounted for in whatever the caller passed.
            kv_bytes: 0,
            kv_known: true,
            train_context: None,
            shard_count: 1,
            dropped_tensors: 0,
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
    fn colliding_labels_are_replaced_by_their_paths() {
        // gemma-3 GGUF repos ship `gemma-3-4b-it-BF16.gguf` beside `mmproj-BF16.gguf`;
        // two rows both called BF16 leave the user guessing which was recommended.
        let mut a = candidate("BF16", 8.0, Some(16.0), Regime::Oversized);
        a.path = "BF16/m-BF16.gguf".into();
        let mut b = candidate("BF16", 8.0, Some(16.0), Regime::Oversized);
        b.path = "m-BF16.gguf".into();
        let mut cands = vec![a, b, candidate("Q4_K_M", 2.0, Some(4.8), Regime::Fits)];
        disambiguate(&mut cands);
        assert_eq!(labels(&cands), ["BF16/m-BF16", "m-BF16", "Q4_K_M"]);
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
mod sweep_tests {
    use super::*;
    use std::sync::OnceLock;

    #[test]
    fn heads_and_projectors_are_not_models_whatever_their_files_are_called() {
        for arch in ["clip", "eagle", "eagle3", "dflash", "medusa"] {
            let facts = model::ArchFacts {
                architecture: Some(arch.into()),
                ..Default::default()
            };
            let shape = model::ModelShape::new(facts, Vec::new(), 1);
            assert!(not_a_model(&shape).is_some(), "{arch}");
        }
        let llm = model::ModelShape::new(
            model::ArchFacts {
                architecture: Some("gpt-oss".into()),
                ..Default::default()
            },
            Vec::new(),
            1,
        );
        assert!(not_a_model(&llm).is_none());
    }

    fn file(path: &str, size: u64) -> hub::RepoFile {
        hub::RepoFile {
            path: path.into(),
            size: Some(size),
        }
    }

    #[test]
    fn subdirectory_checkpoints_beside_a_root_model_are_not_candidates() {
        // MiMo-V2.6 ships an audio tokenizer and a draft model in subdirectories. Listed as
        // quantizations, the 1.9 GB audio tokenizer "fit" and was recommended.
        let files = vec![
            file("model_pp0_ep0_shard0.safetensors", 5),
            file("model_pp0_ep1_shard0.safetensors", 5),
            file("audio_tokenizer/model.safetensors", 2),
            file("dflash/dflash_draft_model.safetensors", 1),
        ];
        let indexed = vec![(
            "model.safetensors.index.json".to_string(),
            vec![
                "model_pp0_ep0_shard0.safetensors".to_string(),
                "model_pp0_ep1_shard0.safetensors".to_string(),
            ],
        )];
        let (whole, sets) = hub::group_models(&files, &indexed);
        let (units, skipped) = plan_units(&whole, &sets);
        let paths: Vec<&str> = units.iter().map(|u| u.path.as_str()).collect();
        assert_eq!(paths, ["model.safetensors"]);
        assert_eq!(skipped.len(), 2);
        // The download pattern must match every part, whatever the parts are called.
        assert_eq!(
            units[0].download_arg,
            "--include \"model_pp0_ep*.safetensors\""
        );
    }

    #[test]
    fn a_one_part_index_downloads_its_one_file() {
        assert_eq!(
            download_glob("model.safetensors", &["model.safetensors".to_string()]),
            "model.safetensors"
        );
    }

    #[test]
    fn gguf_quants_in_subdirectories_stay_candidates() {
        // Unsloth keeps a split BF16 in `BF16/` beside the root quants; it is a quant of
        // the same model, not an adjunct.
        let files = vec![
            file("M-Q4_K_M.gguf", 5),
            file("BF16/M-BF16-00001-of-00002.gguf", 5),
            file("BF16/M-BF16-00002-of-00002.gguf", 5),
        ];
        let (whole, sets) = hub::group_models(&files, &[]);
        let (units, skipped) = plan_units(&whole, &sets);
        assert_eq!(units.len(), 2);
        assert!(skipped.is_empty());
        assert_eq!(units[1].download_arg, "--include \"BF16/M-BF16-*.gguf\"");
    }

    fn safetensors(name: &str) -> model::Safetensors {
        let header =
            format!(r#"{{"{name}":{{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        model::Safetensors::parse(&mut std::io::Cursor::new(bytes)).expect("parse")
    }

    fn slot(h: Result<Header>) -> OnceLock<Result<Header>> {
        let s = OnceLock::new();
        let _ = s.set(h);
        s
    }

    #[test]
    fn parts_read_separately_merge_into_one_model() {
        let parts = vec![
            slot(Ok(Header::Safetensors(safetensors("a.weight")))),
            slot(Ok(Header::Safetensors(safetensors("b.weight")))),
        ];
        let shape = assemble(parts, None).expect("merged");
        assert_eq!(shape.tensors().len(), 2);
        assert_eq!(shape.shard_count, 2);
    }

    #[test]
    fn one_unreadable_part_makes_the_whole_model_unreadable() {
        // Two parts of three is a model that cannot load; reporting it would be a row
        // that is wrong by a third.
        let parts = vec![
            slot(Ok(Header::Safetensors(safetensors("a.weight")))),
            slot(Err(anyhow::anyhow!("HTTP 401"))),
        ];
        match assemble(parts, None) {
            Err(e) => assert!(e.to_string().contains("401"), "{e}"),
            Ok(_) => panic!("a model with an unreadable part was assembled"),
        }
        assert!(assemble(vec![OnceLock::new()], None).is_err(), "never read");
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

        // On a Mac: off one, nothing runs safetensors, which is its own test in `engine`.
        let rec = engine::route_on(
            engine::Host::AppleSilicon,
            4 << 30,
            Format::Safetensors,
            12 << 30,
            &installed,
        );
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
            engine::format_of("model-00001-of-00003.safetensors", None),
            Format::Safetensors
        );
        assert_eq!(
            engine::format_of("Q4_K_M/Model-Q4_K_M.gguf", None),
            Format::Gguf
        );
        // Anything unrecognised stays GGUF, which is what the hub listing is overwhelmingly
        // made of, and is the format every local engine reads.
        assert_eq!(engine::format_of("model.bin", None), Format::Gguf);
    }

    #[test]
    fn an_mlx_quantized_safetensors_repo_is_mlx() {
        let cfg = serde_json::json!({"quantization": {"bits": 4, "group_size": 64}});
        assert_eq!(
            engine::format_of("model.safetensors", Some(&cfg)),
            Format::Mlx
        );
        // The same config on a GGUF file says nothing about the GGUF.
        assert_eq!(engine::format_of("m.gguf", Some(&cfg)), Format::Gguf);
    }
}
