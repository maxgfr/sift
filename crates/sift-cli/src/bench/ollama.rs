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
///
/// `--fail`, so a web server that happens to be on the port and answers 404 is not taken
/// for Ollama.
pub fn server_is_up(port: u16) -> bool {
    Command::new("curl")
        .args([
            "-sS",
            "--fail",
            "-o",
            "/dev/null",
            "--max-time",
            "2",
            &format!("http://127.0.0.1:{port}/api/tags"),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Models Ollama has loaded right now.
///
/// `/api/ps`, not `/api/tags`: the first pulled model is whatever sorts first — on this
/// machine an embedding model that cannot generate at all — while a loaded one is the
/// model the user is actually running.
pub fn loaded_models(port: u16) -> Result<Vec<String>> {
    Ok(model_names(&get_json(port, "/api/ps")?))
}

/// Models Ollama has pulled locally, for telling the user what they could load.
pub fn local_models(port: u16) -> Result<Vec<String>> {
    Ok(model_names(&get_json(port, "/api/tags")?))
}

fn model_names(json: &serde_json::Value) -> Vec<String> {
    json["models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn get_json(port: u16, path: &str) -> Result<serde_json::Value> {
    let out = Command::new("curl")
        .args([
            "-sS",
            "--fail",
            "--max-time",
            "5",
            &format!("http://127.0.0.1:{port}{path}"),
        ])
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("Ollama did not respond on port {port}; start it with `ollama serve`");
    }
    serde_json::from_slice(&out.stdout).with_context(|| format!("parsing {path}"))
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
            "-sS",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_models_are_read_from_ps() {
        let ps = serde_json::json!({"models": [{"name": "qwen3:4b", "model": "qwen3:4b"}]});
        assert_eq!(model_names(&ps), ["qwen3:4b"]);
        assert!(model_names(&serde_json::json!({"models": []})).is_empty());
    }
}
