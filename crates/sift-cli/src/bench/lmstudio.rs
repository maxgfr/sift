//! Driving LM Studio's local server to get a comparable number.
//!
//! HTTP goes through `curl` rather than a client crate. This is a benchmark harness whose
//! job is to be trusted, and a two-line subprocess call has less surface to be wrong in
//! than a TLS stack we would never otherwise depend on.
//!
//! LM Studio's own REST API (`/api/v0`) is used where it answers: it says which models are
//! actually loaded and of what kind, and it reports the engine's own generation time and
//! tokens per second, which exclude prompt processing and HTTP overhead. The
//! OpenAI-compatible `/v1` endpoints are the fallback for builds without it, and give
//! wall-clock figures only — the difference is recorded on each [`Run`] rather than
//! smoothed over.
//!
//! See [`super`] for what makes the comparison fair.

use anyhow::{bail, Context, Result};
use std::process::Command;
use std::time::Instant;

use super::Run;

/// Default port for LM Studio's local server.
pub const DEFAULT_PORT: u16 = 1234;

/// Is the LM Studio server reachable?
///
/// `--fail`, so a web server that happens to be on the port and answers 404 is not taken
/// for LM Studio.
pub fn server_is_up(port: u16) -> bool {
    Command::new("curl")
        .args([
            "-sS",
            "--fail",
            "-o",
            "/dev/null",
            "--max-time",
            "2",
            &format!("http://127.0.0.1:{port}/v1/models"),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Models the server has loaded and that generate text.
///
/// `/api/v0/models` says which are loaded and what kind each is. `/v1/models` lists every
/// model LM Studio knows, loaded or not and embeddings included, so taking its first entry
/// benchmarked whatever sorted first — sometimes a model that was not loaded at all.
pub fn loaded_models(port: u16) -> Result<Vec<String>> {
    if let Ok(json) = get_json(port, "/api/v0/models") {
        return Ok(loaded_text_models(&json));
    }
    let json = get_json(port, "/v1/models")
        .context("server did not respond; start it with `lms server start`")?;
    Ok(json["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default())
}

/// The loaded, text-generating models in an `/api/v0/models` answer.
fn loaded_text_models(json: &serde_json::Value) -> Vec<String> {
    json["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|m| m["state"].as_str() == Some("loaded"))
                .filter(|m| matches!(m["type"].as_str(), Some("llm" | "vlm")))
                .filter_map(|m| m["id"].as_str().map(str::to_owned))
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
        bail!("{path} answered with an error");
    }
    serde_json::from_slice(&out.stdout).with_context(|| format!("parsing {path}"))
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

    // The native endpoint first, for the engine's own timing; the OpenAI one if this
    // build of LM Studio does not have it.
    let mut last_err = None;
    for path in ["/api/v0/chat/completions", "/v1/chat/completions"] {
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
                &format!("http://127.0.0.1:{port}{path}"),
            ])
            .output()
            .context("running curl")?;
        let wall = started.elapsed().as_secs_f64();
        if !out.status.success() {
            last_err = Some(anyhow::anyhow!(
                "completion request failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
            continue;
        }
        let json: serde_json::Value = match serde_json::from_slice(&out.stdout) {
            Ok(j) => j,
            Err(e) => {
                last_err = Some(anyhow::anyhow!("parsing {path} response: {e}"));
                continue;
            }
        };
        match run_from(&json, model, wall) {
            Ok(run) => return Ok(run),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no completion endpoint answered")))
}

/// Read a run out of a completion response, preferring the engine's own timing.
///
/// `/api/v0` responses carry `stats.generation_time` — seconds spent generating, after the
/// prompt — which is the decode time this benchmark is after. Without it the wall clock is
/// all there is, prompt processing and HTTP overhead included.
fn run_from(json: &serde_json::Value, model: &str, wall_seconds: f64) -> Result<Run> {
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

    let engine_seconds = json["stats"]["generation_time"]
        .as_f64()
        .filter(|&t| t > 0.0);
    let (seconds, engine_timed) = match engine_seconds {
        Some(t) => (t, true),
        None => (wall_seconds, false),
    };
    Ok(Run {
        engine: "lm-studio",
        model: json["model"].as_str().unwrap_or(model).to_string(),
        completion_tokens,
        prompt_tokens,
        seconds,
        tokens_per_sec: completion_tokens as f64 / seconds,
        engine_timed,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loaded_text_models_are_candidates() {
        // Regression: the first entry of /v1/models was benchmarked, loaded or not, and
        // embedding models were in that list too.
        let json = serde_json::json!({"data": [
            {"id": "text-embedding-nomic", "type": "embeddings", "state": "loaded"},
            {"id": "qwen3-4b", "type": "llm", "state": "not-loaded"},
            {"id": "gemma-3-4b", "type": "vlm", "state": "loaded"},
            {"id": "qwen3.5-9b", "type": "llm", "state": "loaded"},
        ]});
        assert_eq!(loaded_text_models(&json), ["gemma-3-4b", "qwen3.5-9b"]);
    }

    #[test]
    fn the_engines_own_generation_time_is_preferred_over_the_wall_clock() {
        let native = serde_json::json!({
            "model": "m", "usage": {"completion_tokens": 100, "prompt_tokens": 10},
            "stats": {"tokens_per_second": 50.0, "generation_time": 2.0},
        });
        let run = run_from(&native, "m", 3.0).unwrap();
        assert!(run.engine_timed);
        assert_eq!(run.tokens_per_sec, 50.0);

        let openai = serde_json::json!({
            "model": "m", "usage": {"completion_tokens": 100, "prompt_tokens": 10},
        });
        let run = run_from(&openai, "m", 4.0).unwrap();
        assert!(!run.engine_timed);
        assert_eq!(run.tokens_per_sec, 25.0);
    }

    #[test]
    fn a_response_with_no_tokens_measured_nothing() {
        let json = serde_json::json!({"usage": {"completion_tokens": 0}});
        assert!(run_from(&json, "m", 1.0).is_err());
    }
}
