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
use sift_core::hub::{self, RepoFile};
use sift_core::model::{self, hf_url, RemoteFile, TokenTraffic};

/// One evaluated quantization.
pub struct Candidate {
    pub label: String,
    pub file: String,
    pub size_bytes: u64,
    pub traffic: TokenTraffic,
    pub regime: Regime,
    pub est_tok_s: f64,
    pub is_moe: bool,
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

/// Evaluate every quantization a repo offers.
pub fn evaluate(repo: &str, mem_bytes_per_sec: f64, usable_ram: u64) -> Result<Vec<Candidate>> {
    let files = hub::list_gguf(repo)?;
    let whole: Vec<&RepoFile> = files.iter().filter(|f| !f.is_shard()).collect();

    let mut out = Vec::new();
    for f in whole {
        // A file whose header we cannot read is skipped with a note rather than guessed
        // at: a wrong row in this table is worse than a missing one.
        let mut remote = RemoteFile::new(hf_url(repo, &f.path));
        let g = match sift_core::model::Gguf::parse(&mut remote) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  skipped {}: {e}", f.quant_label());
                continue;
            }
        };

        let size = f.size.unwrap_or_else(|| g.total_tensor_bytes());
        let (traffic, is_moe) = match model::infer_moe_shape(&g) {
            Some(shape) => (
                TokenTraffic {
                    expert_bytes: shape.expert_bytes_per_token(),
                    trunk_bytes: g
                        .total_tensor_bytes()
                        .saturating_sub(shape.total_expert_bytes()),
                },
                true,
            ),
            // Dense: every weight is read every token.
            None => (
                TokenTraffic {
                    expert_bytes: 0,
                    trunk_bytes: g.total_tensor_bytes(),
                },
                false,
            ),
        };

        let regime = Regime::classify(size, usable_ram);
        let efficiency = if is_moe {
            MOE_EFFICIENCY
        } else {
            DENSE_EFFICIENCY
        };

        // Only a resident model runs at memory speed. Past the boundary the bottleneck
        // becomes the disk, and pretending otherwise is how a router misleads people.
        let est_tok_s = match regime {
            Regime::Fits | Regime::Tight => {
                traffic.tokens_per_sec(mem_bytes_per_sec * efficiency, 0.0)
            }
            Regime::Oversized => f64::NAN,
        };

        out.push(Candidate {
            label: f.quant_label(),
            file: f.path.clone(),
            size_bytes: size,
            traffic,
            regime,
            est_tok_s,
            is_moe,
        });
    }

    out.sort_by_key(|c| c.size_bytes);
    Ok(out)
}

/// Print the table and a recommendation.
pub fn report(
    repo: &str,
    cands: &[Candidate],
    usable_ram: u64,
    installed: &[Installed],
) -> Result<()> {
    if cands.is_empty() {
        anyhow::bail!("no readable GGUF files in `{repo}`");
    }

    println!(
        "\n  {:<14} {:>9} {:>8} {:>11} {:>10}",
        "quant", "size", "fits", "GB/token", "est tok/s"
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
        println!(
            "  {:<14} {:>6.2} GB {:>8} {:>11.3} {}  {}",
            c.label,
            c.size_bytes as f64 / 1e9,
            fits,
            c.traffic.total() as f64 / 1e9,
            tps,
            // Worth showing: a MoE row's GB/token is far below its file size, and that gap
            // is the whole reason these models are fast. Hiding it makes the table look
            // like an arithmetic error.
            if c.is_moe { "moe" } else { "dense" }
        );
    }

    // Recommend the largest that still fits: more bits is better quality, and the biggest
    // resident option is the best trade available on this machine.
    let best = cands
        .iter()
        .filter(|c| c.regime == Regime::Fits)
        .max_by_key(|c| c.size_bytes)
        .or_else(|| {
            cands
                .iter()
                .filter(|c| c.regime == Regime::Tight)
                .max_by_key(|c| c.size_bytes)
        });

    match best {
        Some(c) => {
            let format = Format::Gguf;
            let rec = engine::route(c.size_bytes, format, usable_ram, installed);
            println!("\n  recommended: {}", c.label);
            if let Some(e) = &rec.engine {
                println!("    engine     {} — {}", e.name, rec.reason);
                if !rec.installed {
                    println!("    not installed: {}", e.install);
                }
            }
            println!("    download   huggingface-cli download {repo} {}", c.file);
        }
        None => {
            println!(
                "\n  nothing here fits in {:.1} GiB of usable memory.",
                sift_core::gib(usable_ram)
            );
            let smallest = cands.first().context("no candidates")?;
            let rec = engine::route(smallest.size_bytes, Format::Gguf, usable_ram, installed);
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
