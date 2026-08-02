//! Which engine should run this model.
//!
//! The runtime landscape moves faster than anyone can track by hand. llama.cpp sits under
//! most of it, but *the fastest path to a given chip* has flipped repeatedly — Ollama left
//! llama.cpp for its own engine, then adopted MLX on Apple Silicon; MLX leads llama.cpp
//! below ~14B and converges above ~27B; colibri appeared and took the tera-parameter
//! streaming case that nothing else served.
//!
//! Routing is therefore not a static opinion but a lookup against measured facts, and this
//! module is deliberately **a table**. Adding an engine should be a data change, because
//! the churn is the reason the tool exists — absorbing it must be trivial.
//!
//! Two rules the routing must never break:
//!
//! - **Only recommend what is installed**, or say plainly that it is not.
//! - **Never recommend something that will thrash.** One confidently wrong recommendation
//!   costs more trust than ten correct ones earn.

use std::path::{Path, PathBuf};

/// Weight formats an engine can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Gguf,
    Mlx,
    Safetensors,
}

/// What an engine does when a model does not fit in memory.
///
/// This is the axis that actually decides the recommendation, and the one no existing tool
/// surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Oversized {
    /// Refuses to load, or fails allocating.
    Refuses,
    /// Loads, then pages against the OS and collapses. The worst outcome, because it looks
    /// like it worked.
    Thrashes,
    /// Streams weights from disk by design. Slow, but honest and bounded.
    Streams,
}

/// How a model's size compares to what the machine can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Comfortably resident.
    Fits,
    /// Near the boundary. It may load, but context growth will push it over.
    Tight,
    /// Past what the machine can hold.
    Oversized,
}

impl Regime {
    /// Classify a model against usable memory.
    ///
    /// `usable` is not physical RAM: the OS, the window server and the KV cache all take a
    /// share, and on Apple Silicon the GPU wired limit binds first.
    pub fn classify(model_bytes: u64, usable: u64) -> Self {
        if usable == 0 {
            return Regime::Oversized;
        }
        let ratio = model_bytes as f64 / usable as f64;
        if ratio < 0.8 {
            Regime::Fits
        } else if ratio < 1.2 {
            Regime::Tight
        } else {
            Regime::Oversized
        }
    }
}

/// A known inference engine.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Engine {
    /// Stable identifier, e.g. `lm-studio`.
    pub id: &'static str,
    /// Name as a human would write it.
    pub name: &'static str,
    /// Formats it can load.
    pub formats: &'static [Format],
    /// Behaviour when the model exceeds memory.
    pub oversized: Oversized,
    /// Where it is at its best, in one clause.
    pub sweet_spot: &'static str,
    /// How to install it, when it is absent.
    pub install: &'static str,
}

/// An engine found on this machine.
#[derive(Debug, Clone)]
pub struct Installed {
    pub engine: Engine,
    /// Where it was found, for the user to verify our detection.
    pub found_at: PathBuf,
}

/// The registry.
///
/// Adding an engine here is the entire change: no routing logic to touch. Ordered by how
/// generally applicable each is, which breaks ties when several are equally suitable.
pub const ENGINES: &[Engine] = &[
    Engine {
        id: "lm-studio",
        name: "LM Studio",
        formats: &[Format::Gguf, Format::Mlx],
        oversized: Oversized::Thrashes,
        sweet_spot:
            "models that fit; on Apple Silicon its MLX runtime is near the memory-bandwidth limit",
        install: "https://lmstudio.ai",
    },
    Engine {
        id: "ollama",
        name: "Ollama",
        formats: &[Format::Gguf],
        oversized: Oversized::Thrashes,
        sweet_spot: "models that fit, with the simplest workflow",
        install: "https://ollama.com/download",
    },
    Engine {
        id: "llama.cpp",
        name: "llama.cpp",
        formats: &[Format::Gguf],
        oversized: Oversized::Thrashes,
        sweet_spot: "models that fit, with the most control over offload and cache types",
        install: "brew install llama.cpp",
    },
    Engine {
        id: "mlx",
        name: "mlx-lm",
        formats: &[Format::Mlx, Format::Safetensors],
        oversized: Oversized::Refuses,
        sweet_spot: "Apple Silicon, models that fit; fastest below ~14B",
        install: "pip install mlx-lm",
    },
    Engine {
        id: "colibri",
        name: "colibri",
        formats: &[Format::Gguf],
        oversized: Oversized::Streams,
        sweet_spot: "models far larger than memory, streaming experts from disk",
        install: "https://github.com/JustVugg/colibri",
    },
];

/// Detect which engines are present.
///
/// Detection is best-effort and **fails soft**: a missing binary means "not found here",
/// never "not installed". Install paths change, and asserting absence on a failed probe
/// would make the tool wrong in a way users cannot debug.
pub fn detect_installed() -> Vec<Installed> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut found = Vec::new();

    for engine in ENGINES {
        if let Some(path) = candidate_paths(engine.id, home.as_deref())
            .unwrap_or_default()
            .into_iter()
            .find(|p| p.exists())
        {
            found.push(Installed {
                engine: engine.clone(),
                found_at: path,
            });
        }
    }
    found
}

/// Where each engine might be, most specific first.
///
/// **Every entry must be something that can run a model** — a binary or an application
/// bundle — and never a data directory. `~/.ollama` was once on this list and is the reason
/// the rule is written down: `ollama pull` creates it, uninstalling Ollama leaves it
/// behind, and `sift` went on reporting an engine that was gone and routing models to it.
/// Detection failing soft means an absent engine reads as "not found here"; it does not
/// license reporting one that is not there.
///
/// Separated from [`detect_installed`] so the list itself is testable without a filesystem
/// that happens to have these programs on it.
///
/// `None` means no rule exists for this id — distinct from `Some(vec![])`, which means
/// there is a rule and this machine matched none of it. Without that distinction an engine
/// added to [`ENGINES`] and forgotten here would be undetectable, and the only symptom
/// would be `(NOT installed)` printed on a machine that has it.
fn candidate_paths(id: &str, home: Option<&Path>) -> Option<Vec<PathBuf>> {
    let paths: Vec<PathBuf> = match id {
        "lm-studio" => home
            .iter()
            .map(|h| h.join(".lmstudio/bin/lms"))
            .chain(["/Applications/LM Studio.app".into()])
            .collect(),
        "ollama" => which("ollama")
            .into_iter()
            .chain([
                PathBuf::from("/Applications/Ollama.app"),
                PathBuf::from("/usr/local/bin/ollama"),
                PathBuf::from("/opt/homebrew/bin/ollama"),
            ])
            // Windows installs per-user, off `PATH` for processes that started before it.
            // Absent elsewhere, so this simply drops out rather than needing a `cfg`.
            .chain(local_app_data().map(|d| d.join("Programs/Ollama/ollama.exe")))
            .collect(),
        "llama.cpp" => which("llama-server")
            .into_iter()
            .chain(which("llama-cli"))
            .collect(),
        "mlx" => which("mlx_lm.generate").into_iter().collect(),
        "colibri" => which("colibri").into_iter().collect(),
        _ => return None,
    };
    Some(paths)
}

/// Locate an executable on `PATH`.
///
/// Also tries the `.exe` spelling, since Windows installers put the binary on `PATH` under
/// a name this lookup would otherwise miss — the same false negative for every engine.
fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .flat_map(|d| [d.join(bin), d.join(format!("{bin}.exe"))])
        .find(|p| p.is_file())
}

/// `%LOCALAPPDATA%`, where Windows installers put per-user programs.
///
/// Returns `None` elsewhere, so the Windows-only paths simply drop out of the candidate
/// list on Unix rather than needing a `cfg` here.
fn local_app_data() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

/// A routing recommendation.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The engine to use, if any is suitable and present.
    pub engine: Option<Engine>,
    /// Whether it is installed here.
    pub installed: bool,
    /// Why this one, in a sentence a user can check.
    pub reason: String,
    /// Engines that could run it but are absent.
    pub also_suitable: Vec<Engine>,
    /// Engines that are installed but would handle this badly.
    pub avoid: Vec<(Engine, &'static str)>,
}

/// Recommend an engine for a model of `model_bytes` in `format`, given `usable` memory.
pub fn route(
    model_bytes: u64,
    format: Format,
    usable: u64,
    installed: &[Installed],
) -> Recommendation {
    let regime = Regime::classify(model_bytes, usable);

    let is_installed = |e: &Engine| installed.iter().any(|i| i.engine.id == e.id);

    // Suitable means: reads this format, and does not fall over in this regime.
    let suitable: Vec<&Engine> = ENGINES
        .iter()
        .filter(|e| e.formats.contains(&format))
        .filter(|e| match regime {
            Regime::Fits | Regime::Tight => true,
            Regime::Oversized => e.oversized == Oversized::Streams,
        })
        .collect();

    // Naming what to avoid is as valuable as naming what to use: an engine that thrashes
    // looks like it is working, which is precisely why users lose hours to it.
    let avoid: Vec<(Engine, &'static str)> = if regime == Regime::Oversized {
        ENGINES
            .iter()
            .filter(|e| e.formats.contains(&format))
            .filter(|e| e.oversized != Oversized::Streams)
            .filter(|e| is_installed(e))
            .map(|e| {
                (
                    e.clone(),
                    match e.oversized {
                        Oversized::Refuses => "will refuse to load a model this size",
                        _ => "will load, then page against the OS and slow to a crawl",
                    },
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // Prefer something present over something merely suitable — a recommendation the user
    // cannot act on is not a recommendation.
    let chosen = suitable
        .iter()
        .find(|e| is_installed(e))
        .or(suitable.first())
        .map(|e| (*e).clone());

    let reason = match (&chosen, regime) {
        (None, Regime::Oversized) => format!(
            "at {:.1} GiB this exceeds what the machine can hold, and no installed engine \
             streams weights from disk",
            crate::gib(model_bytes)
        ),
        (None, _) => "no known engine reads this format".to_string(),
        (Some(e), Regime::Fits) => format!("it fits in memory, and {}", e.sweet_spot),
        (Some(e), Regime::Tight) => format!(
            "it only just fits — {} — so expect trouble once the KV cache grows",
            e.sweet_spot
        ),
        (Some(e), Regime::Oversized) => format!(
            "it exceeds memory, and {} is the one that {}",
            e.name, e.sweet_spot
        ),
    };

    Recommendation {
        installed: chosen.as_ref().is_some_and(is_installed),
        also_suitable: suitable
            .iter()
            .filter(|e| !is_installed(e))
            .filter(|e| chosen.as_ref().is_none_or(|c| c.id != e.id))
            .map(|e| (*e).clone())
            .collect(),
        engine: chosen,
        reason,
        avoid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn installed(ids: &[&str]) -> Vec<Installed> {
        ENGINES
            .iter()
            .filter(|e| ids.contains(&e.id))
            .map(|e| Installed {
                engine: e.clone(),
                found_at: PathBuf::from("/test"),
            })
            .collect()
    }

    #[test]
    fn regimes_split_at_the_documented_boundaries() {
        let usable = 12 * GIB;
        assert_eq!(Regime::classify(5 * GIB, usable), Regime::Fits);
        assert_eq!(Regime::classify(11 * GIB, usable), Regime::Tight);
        assert_eq!(Regime::classify(30 * GIB, usable), Regime::Oversized);
    }

    #[test]
    fn zero_usable_memory_means_nothing_fits() {
        assert_eq!(Regime::classify(1, 0), Regime::Oversized);
    }

    #[test]
    fn a_model_that_fits_goes_to_an_installed_engine() {
        let r = route(5 * GIB, Format::Gguf, 12 * GIB, &installed(&["lm-studio"]));
        assert_eq!(r.engine.as_ref().unwrap().id, "lm-studio");
        assert!(r.installed);
        assert!(r.avoid.is_empty(), "nothing to avoid when the model fits");
    }

    #[test]
    fn an_oversized_model_never_routes_to_an_engine_that_thrashes() {
        // The rule that matters most: thrashing looks like working, so recommending it is
        // worse than recommending nothing.
        let r = route(
            60 * GIB,
            Format::Gguf,
            12 * GIB,
            &installed(&["lm-studio", "ollama", "colibri"]),
        );
        assert_eq!(r.engine.as_ref().unwrap().id, "colibri");
        assert_eq!(r.engine.as_ref().unwrap().oversized, Oversized::Streams);
    }

    #[test]
    fn an_oversized_model_names_the_installed_engines_to_avoid() {
        let r = route(
            60 * GIB,
            Format::Gguf,
            12 * GIB,
            &installed(&["lm-studio", "ollama"]),
        );
        let ids: Vec<&str> = r.avoid.iter().map(|(e, _)| e.id).collect();
        assert!(ids.contains(&"lm-studio"));
        assert!(ids.contains(&"ollama"));
        assert!(
            r.avoid.iter().all(|(_, why)| !why.is_empty()),
            "every avoidance must say why"
        );
    }

    #[test]
    fn only_installed_engines_are_listed_as_things_to_avoid() {
        // Warning about software the user does not have is noise.
        let r = route(60 * GIB, Format::Gguf, 12 * GIB, &installed(&[]));
        assert!(r.avoid.is_empty());
    }

    #[test]
    fn an_uninstalled_but_suitable_engine_is_still_suggested() {
        let r = route(60 * GIB, Format::Gguf, 12 * GIB, &installed(&["lm-studio"]));
        assert_eq!(r.engine.as_ref().unwrap().id, "colibri");
        assert!(!r.installed, "colibri is not installed in this scenario");
        assert!(!r.engine.as_ref().unwrap().install.is_empty());
    }

    #[test]
    fn format_is_respected() {
        // MLX weights do not load under llama.cpp, however well it would otherwise fit.
        let r = route(5 * GIB, Format::Mlx, 12 * GIB, &installed(&["llama.cpp"]));
        let id = r.engine.as_ref().unwrap().id;
        assert!(id == "lm-studio" || id == "mlx", "got {id}");
    }

    #[test]
    fn a_tight_fit_is_flagged_rather_than_called_fine() {
        let r = route(11 * GIB, Format::Gguf, 12 * GIB, &installed(&["ollama"]));
        assert!(
            r.reason.contains("only just fits"),
            "the warning must reach the user: {}",
            r.reason
        );
    }

    #[test]
    fn every_registry_entry_is_complete() {
        // The registry is meant to be edited by contributors; a half-filled row would ship
        // an engine that cannot be recommended or installed.
        for e in ENGINES {
            assert!(
                !e.id.is_empty() && !e.name.is_empty(),
                "{} incomplete",
                e.id
            );
            assert!(!e.formats.is_empty(), "{} declares no formats", e.id);
            assert!(!e.sweet_spot.is_empty(), "{} has no sweet spot", e.id);
            assert!(!e.install.is_empty(), "{} has no install hint", e.id);
        }
    }

    #[test]
    fn at_least_one_engine_can_handle_oversized_models() {
        // If this ever fails, every oversized recommendation silently becomes "nothing".
        assert!(ENGINES.iter().any(|e| e.oversized == Oversized::Streams));
    }

    #[test]
    fn a_data_directory_is_never_treated_as_an_installed_engine() {
        // The regression this pins. `~/.ollama` holds pulled models; it survives an
        // uninstall, and `ollama pull` on a machine that later removes Ollama leaves it
        // there forever. Detecting on it reported an engine that could not run anything —
        // and `route` then recommended it over one that was actually present.
        let home = PathBuf::from("/home/someone");
        let paths = candidate_paths("ollama", Some(&home)).expect("ollama has a rule");
        assert!(
            !paths.iter().any(|p| p.ends_with(".ollama")),
            "a data directory is not an engine: {paths:?}"
        );
    }

    #[test]
    fn every_candidate_path_is_something_that_can_run_a_model() {
        // The general form of the rule, applied to every engine so a new row cannot
        // reintroduce the same mistake under a different name. A runnable candidate is a
        // binary or an application bundle — never a dotfile directory.
        let home = PathBuf::from("/home/someone");
        for engine in ENGINES {
            for path in candidate_paths(engine.id, Some(&home)).unwrap_or_default() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                assert!(
                    !name.starts_with('.'),
                    "{}: {} is a data directory, not a program",
                    engine.id,
                    path.display()
                );
            }
        }
    }

    #[test]
    fn every_engine_in_the_registry_is_looked_for_somewhere() {
        // A registry entry with no detection rule can never be reported as installed, so
        // `route` would print "(NOT installed)" on a machine that has it and send the user
        // to an install page for software they already run. Asserted on the rule existing
        // rather than on paths being found, so the test says the same thing on a CI runner
        // with none of these programs as on a developer machine with all of them.
        let home = PathBuf::from("/home/someone");
        for engine in ENGINES {
            assert!(
                candidate_paths(engine.id, Some(&home)).is_some(),
                "{} is in the registry but nothing looks for it",
                engine.id
            );
        }
        assert_eq!(candidate_paths("not-an-engine", Some(&home)), None);
    }
}
