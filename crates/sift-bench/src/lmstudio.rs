//! Driving LM Studio's OpenAI-compatible server to get a comparable number.
//!
//! HTTP goes through `curl` rather than a client crate. This is a benchmark harness whose
//! job is to be trusted, and a two-line subprocess call has less surface to be wrong in
//! than a TLS stack we would never otherwise depend on.
//!
//! # What makes a comparison fair
//!
//! Decode throughput is the number under test, so it must be isolated from everything
//! else. Three rules, each of which someone has published a wrong number by breaking:
//!
//! - **Warm up first.** The first request pays model load and graph construction. It is
//!   not decode speed and must not be averaged into it.
//! - **Report prefill and decode separately.** Time-to-first-token is dominated by prompt
//!   processing, which scales with prompt length; mixing them lets a short prompt
//!   masquerade as a fast engine.
//! - **Fix the token count.** Comparing runs that generated different numbers of tokens
//!   compares two different workloads.

use anyhow::{bail, Context, Result};
use std::process::Command;
use std::time::Instant;

/// Default port for LM Studio's local server.
pub const DEFAULT_PORT: u16 = 1234;

/// One decode measurement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Run {
    /// Model identifier as the server reported it.
    pub model: String,
    /// Tokens the server said it generated.
    pub completion_tokens: u32,
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Total wall-clock seconds for the request.
    pub seconds: f64,
    /// Decode throughput, completion tokens per second.
    pub tokens_per_sec: f64,
}

/// Is the LM Studio server reachable?
pub fn server_is_up(port: u16) -> bool {
    Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "--max-time",
            "2",
            &format!("http://127.0.0.1:{port}/v1/models"),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Models the server currently has loaded and ready.
pub fn loaded_models(port: u16) -> Result<Vec<String>> {
    let out = Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "5",
            &format!("http://127.0.0.1:{port}/v1/models"),
        ])
        .output()
        .context("running curl")?;

    if !out.status.success() {
        bail!("server did not respond on port {port}; start it with `lms server start`");
    }

    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing /v1/models response")?;

    Ok(json["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default())
}

/// Send one completion request and measure it.
///
/// `max_tokens` is enforced rather than suggested, and the temperature is pinned to zero,
/// so two engines asked the same question do the same amount of work.
pub fn measure_once(port: u16, model: &str, prompt: &str, max_tokens: u32) -> Result<Run> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });

    let started = Instant::now();
    let out = Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "600",
            "-H",
            "Content-Type: application/json",
            "-d",
            &body.to_string(),
            &format!("http://127.0.0.1:{port}/v1/chat/completions"),
        ])
        .output()
        .context("running curl")?;
    let seconds = started.elapsed().as_secs_f64();

    if !out.status.success() {
        bail!(
            "completion request failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing completion response")?;

    if let Some(err) = json.get("error") {
        bail!("server returned an error: {err}");
    }

    // Trust the server's own token accounting over any local tokenizer guess; a mismatch
    // between its tokenizer and ours would silently skew every rate we report.
    let usage = &json["usage"];
    let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;

    if completion_tokens == 0 {
        bail!("server reported zero completion tokens; nothing was measured");
    }

    Ok(Run {
        model: json["model"].as_str().unwrap_or(model).to_string(),
        completion_tokens,
        prompt_tokens,
        seconds,
        tokens_per_sec: completion_tokens as f64 / seconds,
    })
}

/// Warm up, then measure `repeats` times and return every run.
///
/// Every run is returned rather than an average: a single outlier is informative (thermal
/// throttling, a background process), and hiding it inside a mean is how benchmarks
/// become unfalsifiable.
pub fn measure(
    port: u16,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    repeats: usize,
) -> Result<Vec<Run>> {
    // Discarded on purpose: this pays model load and graph construction.
    let _ =
        measure_once(port, model, prompt, 8.min(max_tokens)).context("warm-up request failed")?;

    let mut runs = Vec::with_capacity(repeats);
    for i in 0..repeats {
        runs.push(
            measure_once(port, model, prompt, max_tokens)
                .with_context(|| format!("run {} of {repeats}", i + 1))?,
        );
    }
    Ok(runs)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tps: f64) -> Run {
        Run {
            model: "m".into(),
            completion_tokens: 100,
            prompt_tokens: 10,
            seconds: 100.0 / tps,
            tokens_per_sec: tps,
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
}
