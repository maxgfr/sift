//! Driving Ollama's native API to get a comparable number.
//!
//! Ollama also speaks the OpenAI shape at `/v1/chat/completions`, and using it would let
//! this file be a copy of [`super::lmstudio`]. It is deliberately not.
//!
//! The native `/api/generate` endpoint reports `eval_count` and `eval_duration` — the
//! tokens decoded and the nanoseconds spent decoding them, as the runtime itself counted.
//! That excludes prompt processing, HTTP overhead and the time `curl` spends starting up,
//! all of which a wall-clock measurement folds into the rate. On a short generation that
//! overhead is a double-digit percentage.
//!
//! So: engine-reported timing where the engine reports it, wall clock only where it does
//! not. Comparing the two across engines is a small unfairness, and it is named here rather
//! than hidden — Ollama's figure is the more accurate of the pair, so a comparison flatters
//! LM Studio slightly rather than the reverse.

use anyhow::{bail, Context, Result};
use std::process::Command;
use std::time::Instant;

use super::Run;

/// Default port for the Ollama server.
pub const DEFAULT_PORT: u16 = 11434;

/// Is the Ollama server reachable?
pub fn server_is_up(port: u16) -> bool {
    Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "--max-time",
            "2",
            &format!("http://127.0.0.1:{port}/api/tags"),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Models Ollama has pulled locally.
pub fn local_models(port: u16) -> Result<Vec<String>> {
    let out = Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "5",
            &format!("http://127.0.0.1:{port}/api/tags"),
        ])
        .output()
        .context("running curl")?;

    if !out.status.success() {
        bail!("Ollama did not respond on port {port}; start it with `ollama serve`");
    }

    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing /api/tags response")?;

    Ok(json["models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default())
}

/// Send one generation request and measure it.
///
/// `num_predict` caps the generation and the temperature is pinned to zero, so two engines
/// asked the same question do the same amount of work.
pub fn measure_once(port: u16, model: &str, prompt: &str, max_tokens: u32) -> Result<Run> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": { "num_predict": max_tokens, "temperature": 0.0 },
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
            &format!("http://127.0.0.1:{port}/api/generate"),
        ])
        .output()
        .context("running curl")?;
    let wall_seconds = started.elapsed().as_secs_f64();

    if !out.status.success() {
        bail!(
            "generate request failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing generate response")?;

    if let Some(err) = json.get("error") {
        bail!("Ollama returned an error: {err}");
    }

    let completion_tokens = json["eval_count"].as_u64().unwrap_or(0) as u32;
    if completion_tokens == 0 {
        bail!("Ollama reported zero eval tokens; nothing was measured");
    }

    // `eval_duration` is decode time in nanoseconds, excluding prompt processing. Fall
    // back to wall clock if it is absent, and say which was used rather than presenting
    // two different measurements as one.
    let eval_ns = json["eval_duration"].as_u64().unwrap_or(0);
    let (seconds, engine_timed) = if eval_ns > 0 {
        (eval_ns as f64 / 1e9, true)
    } else {
        (wall_seconds, false)
    };

    Ok(Run {
        engine: "ollama",
        model: json["model"].as_str().unwrap_or(model).to_string(),
        completion_tokens,
        prompt_tokens: json["prompt_eval_count"].as_u64().unwrap_or(0) as u32,
        seconds,
        tokens_per_sec: completion_tokens as f64 / seconds,
        engine_timed,
    })
}

/// Warm up, then measure `repeats` times and return every run.
pub fn measure(
    port: u16,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    repeats: usize,
) -> Result<Vec<Run>> {
    // Discarded on purpose: the first request pays model load. Ollama also unloads after
    // an idle timeout, so this is not merely a cold-start formality.
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
