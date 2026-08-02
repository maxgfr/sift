//! Driving LM Studio's OpenAI-compatible server to get a comparable number.
//!
//! HTTP goes through `curl` rather than a client crate. This is a benchmark harness whose
//! job is to be trusted, and a two-line subprocess call has less surface to be wrong in
//! than a TLS stack we would never otherwise depend on.
//!
//! The OpenAI shape reports token counts but no decode duration, so every figure here is
//! wall clock and includes prompt processing and HTTP overhead. [`super::ollama`] gets the
//! engine's own decode timing and is the more accurate of the two; the difference is
//! recorded on each [`Run`] rather than smoothed over.
//!
//! See [`super`] for what makes the comparison fair.

use anyhow::{bail, Context, Result};
use std::process::Command;
use std::time::Instant;

use super::Run;

/// Default port for LM Studio's local server.
pub const DEFAULT_PORT: u16 = 1234;

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
        engine: "lm-studio",
        model: json["model"].as_str().unwrap_or(model).to_string(),
        completion_tokens,
        prompt_tokens,
        seconds,
        tokens_per_sec: completion_tokens as f64 / seconds,
        // The OpenAI shape reports token counts but no decode duration, so this is wall
        // clock — prompt processing and HTTP overhead included. Ollama's native API does
        // report it, which is why that driver is not a copy of this one.
        engine_timed: false,
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
