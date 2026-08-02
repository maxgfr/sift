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

/// Fraction of the memory-bandwidth roofline that real decode achieves.
///
/// Measured, not guessed: LM Studio on an M5 runs Qwen3.5-9B at 20.91 tok/s against
/// ~128 GB/s of effective traffic on a 153.6 GB/s bus. Dense resident decode lands close to
/// the roofline; MoE decode is lower because expert gather is scattered.
///
/// This is the crudest part of the tool and it is labelled as such wherever it is printed.
/// `sift bench` exists to replace it with measurement.
const DENSE_EFFICIENCY: f64 = 0.80;
const MOE_EFFICIENCY: f64 = 0.35;

/// Evaluate every quantization a repo offers, at a given context length.
pub fn evaluate(
    repo: &str,
    mem_bytes_per_sec: f64,
    usable_ram: u64,
    context_tokens: u64,
) -> Result<Vec<Candidate>> {
    let files = hub::list_gguf(repo)?;
    let (whole, sets) = hub::group_shards(&files);

    let mut out = Vec::new();

    for f in &whole {
        // A file whose header we cannot read is skipped with a note rather than guessed
        // at: a wrong row in this table is worse than a missing one.
        let g = match read_header(repo, &f.path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  skipped {}: {e}", f.quant_label());
                continue;
            }
        };
        let shape = g.shape();
        let size = f.size.unwrap_or_else(|| shape.total_tensor_bytes());
        out.push(build(
            f.quant_label(),
            f.path.clone(),
            size,
            &shape,
            mem_bytes_per_sec,
            usable_ram,
            context_tokens,
        ));
    }

    for set in &sets {
        // Read every part, not just the first. Each shard carries only its own slice of
        // the tensor directory, so asking part 1 for the model's size or expert layout
        // gives an answer that is confidently a third of the truth.
        if !set.is_complete() {
            eprintln!(
                "  skipped {}: {} of {} shards present in the repo",
                set.base.quant_label(),
                set.parts.len(),
                set.expected
            );
            continue;
        }

        let mut parts = Vec::with_capacity(set.parts.len());
        let mut failed = None;
        for p in &set.parts {
            match read_header(repo, &p.path) {
                Ok(g) => parts.push(g),
                Err(e) => {
                    failed = Some(format!("{}: {e}", p.path));
                    break;
                }
            }
        }
        if let Some(why) = failed {
            eprintln!("  skipped {}: {why}", set.base.quant_label());
            continue;
        }

        let Some(shape) = model::ModelShape::sharded(&parts) else {
            continue;
        };
        let size = set.size().unwrap_or_else(|| shape.total_tensor_bytes());
        out.push(build(
            set.base.quant_label(),
            shard_glob(&set.base.path),
            size,
            &shape,
            mem_bytes_per_sec,
            usable_ram,
            context_tokens,
        ));
    }

    out.sort_by_key(|c| c.size_bytes);
    Ok(out)
}

/// A `--include` pattern matching every part of a split model.
///
/// Built from the set's base path, so `Q4_K_M/Model-Q4_K_M.gguf` becomes
/// `--include "Q4_K_M/Model-Q4_K_M-*.gguf"`. Quoted because the shell would otherwise
/// expand the glob against the local directory before the tool ever sees it.
fn shard_glob(base_path: &str) -> String {
    let stem = base_path.strip_suffix(".gguf").unwrap_or(base_path);
    format!("--include \"{stem}-*.gguf\"")
}

/// Read one file's header over range requests, touching no payload.
fn read_header(repo: &str, path: &str) -> Result<sift_core::model::Gguf> {
    let mut remote = RemoteFile::new(hf_url(repo, path));
    Ok(sift_core::model::Gguf::parse(&mut remote)?)
}

/// Turn a model's shape into an evaluated candidate.
///
/// Shared by the single-file and split paths so the two cannot drift. A split model that
/// scored differently from the same weights in one file would be a bug nobody would spot.
#[allow(clippy::too_many_arguments)]
fn build(
    label: String,
    download_arg: String,
    size: u64,
    shape: &model::ModelShape,
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
    }
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
    let best = cands
        .iter()
        .filter(|c| c.regime != Regime::Oversized)
        .max_by_key(|c| c.rank());

    match best {
        Some(c) => {
            let format = Format::Gguf;
            // Route on the footprint, so an engine is not recommended for a model that
            // only fits with an empty context.
            let rec = engine::route(c.footprint_bytes(), format, usable_ram, installed);
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
                Format::Gguf,
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
    fn a_base_path_without_the_extension_still_yields_a_usable_pattern() {
        assert_eq!(shard_glob("Model-Q8_0"), "--include \"Model-Q8_0-*.gguf\"");
    }
}
