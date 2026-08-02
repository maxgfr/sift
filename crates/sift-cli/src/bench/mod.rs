//! `sift bench` — measure a real engine, so the estimates can be checked against it.
//!
//! `sift` never runs a model. This drives an engine that does, and records what it
//! achieved next to what `sift fit` predicted. The point is not to rank engines; it is to
//! find out where the estimator is wrong.
//!
//! That has already paid for itself once. `fit` predicted 12.9 tok/s for a model LM Studio
//! runs at 21.03, and chasing the gap turned up three broken bandwidth measurements rather
//! than any inefficiency in the engine — see [`sift_core::doctor::measure_read_bandwidth`].
//!
//! # What makes a comparison fair
//!
//! Decode throughput is the number under test, so it must be isolated from everything
//! else. Three rules, each of which someone has published a wrong number by breaking:
//!
//! - **Warm up first.** The first request pays model load and graph construction. It is
//!   not decode speed and must not be averaged into it.
//! - **Fix the token count and the temperature.** Comparing runs that generated different
//!   numbers of tokens compares two different workloads.
//! - **Keep every run.** A single outlier is informative — thermal throttling, a
//!   background process — and hiding it inside a mean is how a benchmark becomes
//!   unfalsifiable.
//!
//! # Why results are persisted
//!
//! Two hand-set efficiency factors decide every tok/s figure `sift` prints, and both are
//! now fitted to runs taken here. A measurement kept only in a terminal scrollback cannot
//! replace a constant later, so every run is appended to `~/.sift/bench.jsonl`.
//!
//! That log is what caught the larger of the two errors. The MoE factor was a guess of 0.28
//! reasoned from "expert gather is scattered"; a single OLMoE run against it read 129 tok/s
//! where the estimator said 46. See [`crate::fit::MOE_EFFICIENCY`].

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

pub mod lmstudio;
pub mod ollama;

/// One decode measurement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Run {
    /// Which engine produced it.
    pub engine: &'static str,
    /// Model identifier as the engine reported it.
    pub model: String,
    /// Tokens the engine said it generated.
    pub completion_tokens: u32,
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Seconds spent decoding.
    pub seconds: f64,
    /// Decode throughput, completion tokens per second.
    pub tokens_per_sec: f64,
    /// Whether `seconds` came from the engine's own accounting or from the wall clock.
    ///
    /// Recorded because the two are not the same measurement: engine timing excludes
    /// prompt processing and HTTP overhead, which on a short generation is a double-digit
    /// percentage. A consumer comparing across engines needs to know which it has.
    pub engine_timed: bool,
}

/// Median tokens per second across runs.
///
/// Median, not mean: one thermal stall should not drag the headline number, and with a
/// handful of runs the median is the more honest summary.
pub fn median_tps(runs: &[Run]) -> Option<f64> {
    if runs.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = runs.iter().map(|r| r.tokens_per_sec).collect();
    v.sort_by(|a, b| a.partial_cmp(b).expect("throughput is never NaN"));
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    })
}

/// Which engine to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    /// Whichever of the two is serving. Errors if both are, rather than picking one.
    Auto,
    LmStudio,
    Ollama,
}

/// Where measurements accumulate.
fn log_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".sift/bench.jsonl"))
}

/// Append runs to `~/.sift/bench.jsonl`, one JSON object per line.
///
/// JSON Lines rather than a single document: appending is a write with no read, so a
/// crash mid-run cannot corrupt earlier measurements, and the file stays greppable.
///
/// Best effort. A measurement that was taken and printed should not be lost to an error
/// message about a log file, but a silent failure would be worse — the caller is told.
pub fn record(runs: &[Run], stamp: &str) -> Result<PathBuf> {
    use std::io::Write;

    let path = log_path().context("HOME is not set, so there is nowhere to record results")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;

    for r in runs {
        let mut line = serde_json::to_value(r)?;
        line["recorded_at"] = serde_json::Value::String(stamp.to_string());
        writeln!(f, "{line}")?;
    }
    Ok(path)
}

/// Pick an engine to drive.
///
/// Refuses to choose when both are serving. Silently preferring one would attribute a
/// measurement to the wrong engine, and that is a number someone would go on to publish.
pub fn resolve(target: Target, lms_port: u16, ollama_port: u16) -> Result<Target> {
    match target {
        Target::LmStudio | Target::Ollama => Ok(target),
        Target::Auto => {
            let lms = lmstudio::server_is_up(lms_port);
            let oll = ollama::server_is_up(ollama_port);
            match (lms, oll) {
                (true, false) => Ok(Target::LmStudio),
                (false, true) => Ok(Target::Ollama),
                (true, true) => bail!(
                    "both LM Studio (:{lms_port}) and Ollama (:{ollama_port}) are serving.\n  \
                     Name one with --engine lm-studio or --engine ollama; guessing here \
                     would label the measurement with the wrong engine."
                ),
                (false, false) => bail!(
                    "no engine is serving.\n  \
                     LM Studio:  lms server start && lms load <model>\n  \
                     Ollama:     ollama serve && ollama run <model>"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tps: f64) -> Run {
        Run {
            engine: "test",
            model: "m".into(),
            completion_tokens: 100,
            prompt_tokens: 10,
            seconds: 100.0 / tps,
            tokens_per_sec: tps,
            engine_timed: true,
        }
    }

    #[test]
    fn median_of_an_odd_count_is_the_middle_value() {
        let runs = [run(10.0), run(30.0), run(20.0)];
        assert_eq!(median_tps(&runs), Some(20.0));
    }

    #[test]
    fn median_of_an_even_count_averages_the_middle_pair() {
        let runs = [run(10.0), run(20.0), run(30.0), run(40.0)];
        assert_eq!(median_tps(&runs), Some(25.0));
    }

    #[test]
    fn a_single_outlier_does_not_move_the_median() {
        // The point of using a median: one thermal stall must not become the headline.
        let clean = [run(50.0), run(51.0), run(52.0)];
        let stalled = [run(50.0), run(51.0), run(2.0)];
        assert_eq!(median_tps(&clean), Some(51.0));
        assert_eq!(median_tps(&stalled), Some(50.0));
    }

    #[test]
    fn no_runs_yields_no_median_rather_than_zero() {
        // Returning 0.0 would read as "measured, and it was slow".
        assert_eq!(median_tps(&[]), None);
    }

    #[test]
    fn an_explicit_engine_is_honoured_without_probing() {
        // Ports that nothing is listening on: an explicit choice must not be second-guessed
        // by a reachability check.
        assert_eq!(
            resolve(Target::Ollama, 1, 2).expect("explicit"),
            Target::Ollama
        );
        assert_eq!(
            resolve(Target::LmStudio, 1, 2).expect("explicit"),
            Target::LmStudio
        );
    }

    #[test]
    fn auto_with_nothing_serving_explains_how_to_start_one() {
        // Ports 1 and 2 are privileged and will not have an LLM server on them.
        let err = resolve(Target::Auto, 1, 2).expect_err("nothing is serving");
        let msg = err.to_string();
        assert!(msg.contains("lms server start"), "{msg}");
        assert!(msg.contains("ollama serve"), "{msg}");
    }
}
