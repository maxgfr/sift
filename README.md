# sift

**Will this model fit and run fast on your machine? Answered before you download it.**

```console
$ sift fit unsloth/Qwen3-30B-A3B-GGUF
machine: Mac17,2, 16.0 GiB RAM, 12.0 GiB usable, 123 GB/s memory
reading headers from unsloth/Qwen3-30B-A3B-GGUF without downloading…

  fits assumes 4096 tokens of context: 0.40 GB of f16 KV cache, counted
  alongside the weights. Change it with --ctx.

  quant               size     fits    bpw    GB/token  est tok/s
  UD-IQ1_S         9.04 GB      yes   2.37       1.427         24  moe, damaged
  Q2_K            11.26 GB    tight   2.91       1.399         25  moe, damaged
  Q3_K_M          14.71 GB    tight   3.85       1.753         20  moe
  Q4_K_M          18.56 GB       no   4.86       2.095   disk-bound  moe
  Q8_0            32.48 GB       no   8.51       3.600   disk-bound  moe

  recommended: Q3_K_M
```

Twenty-five quantizations evaluated. **Nothing downloaded.**

Today that answer costs an hour: pull 18 GB, find out it's too slow, delete, guess again.

## Two ways this table is not the obvious one

**It does not recommend the biggest file that fits.** `UD-IQ1_S` is the only row marked
`yes`, and it is the wrong answer: at 2.37 bits per weight the model is damaged. Bits per
weight is *measured from each file's own tensor directory* rather than looked up from its
name — so it is right about a quant family invented this morning, and it distinguishes two
files both labelled `Q4_K_M` that a dynamic quantization made genuinely different.

**`fits` counts the KV cache.** Raise the context and the answer changes, because it should:

```console
$ sift fit unsloth/Qwen3-30B-A3B-GGUF --ctx 65536
  fits assumes 65536 tokens of context: 6.44 GB of f16 KV cache…
  nothing here fits in 12.0 GiB of usable memory.
```

Same machine, same model, opposite advice. A tool that answers only the 4k question and
does not say so is not being conservative, it is being wrong quietly.

## No bundled model list

`sift` reads the **real GGUF header of the real file at the real URL** over HTTP range
requests. A model uploaded ten minutes ago works exactly like one from last year.

Measured on a 17.28 GiB model: **15 MiB read, 0.0848% of the file.**

This matters because the alternative — shipping a catalog — rots. The most popular tool in
this space compiles a model list into its binary, and its top user complaint is that it
recommends two-year-old models as perfect matches.

## Not a runtime. A companion to yours.

`sift` never runs a model. It tells you which one to get and which engine should run it.

```console
$ sift route ~/.sift/models/olmoe-q4km.gguf
  size             3.92 GiB
  usable memory    12.00 GiB
  use LM Studio (installed)
  because it fits in memory
```

Routing knows what to **avoid**, not just what to use — an engine that thrashes looks like
it's working, which is how people lose an afternoon:

```console
  installed, but do not use for this model:
    LM Studio    will load, then page against the OS and slow to a crawl
```

Adding an engine is a row in a table (`crates/sift-core/src/engine.rs`), not a code change.
That's deliberate: the runtime landscape churns, and absorbing churn has to be trivial.

## Why not a website?

Browsers cannot measure your machine. The Device Memory API deliberately anonymizes what
it reports to prevent fingerprinting — one popular web tool tells a 24 GB M4 Pro it has
8 GB of VRAM.

`sift` runs a real memory-bandwidth probe and a real cold-disk read. That is the difference
between an estimate and an answer, and it cannot be done from a browser tab.

## Why you might not want sift

- **You only use LM Studio and only run models that fit.** Its built-in compatibility
  badges are good enough. Use those.
- **You already know your hardware and want VRAM arithmetic.** A web calculator is a form
  and needs no install.
- **You want a curated shortlist without thinking.** [llmfit](https://github.com/AlexsJones/llmfit)
  ships one and has a nice TUI.
- **You want the memory estimate and nothing else.**
  [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) does remote GGUF parsing
  well, and did it first. `sift` differs by measuring your machine instead of asking you to
  type in your FLOPS and memory bandwidth, and by recommending an engine.

## Install

```sh
brew install maxgfr/tap/sift
```

macOS and Linux, Apple Silicon / arm64 / x86-64. Homebrew has a `sift` of its own — a grep
alternative — so install it tap-qualified as above; the two cannot be linked at once.

On Windows, take `sift-windows-x64.exe` from the
[latest release](https://github.com/maxgfr/sift/releases/latest).

Or `cargo build --release` — Rust 1.85+, no runtime dependencies beyond `curl`.

## Commands

```
sift fit <hf-repo> [--ctx N]    which quantization to download, and why
sift ls [--ctx N]               every local model, across every engine
sift route <model>              which engine should run it
sift inspect <path|hf-repo>     model shape, local or remote, no download
sift plan <model>               per-token traffic and speed ceilings
sift doctor [--disk-sample F]   measure this machine
sift engines                    which runtimes are installed here
```

`<model>` accepts a path, `org/repo`, `org/repo:QUANT`, or a URL.
Every command takes `--json`.

`sift ls` is the one view across tools that cannot see each other — LM Studio, Ollama,
colibri and `~/.sift` in one table, each scored against this machine:

```console
$ sift ls
  engine       model                                    size    fits    bpw  est tok/s
  LM Studio    …Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf  5.63 GB  yes   5.02   21  dense
  sift         olmoe-q4km.gguf                          4.21 GB  yes   4.87   46  moe
```

Headers are cached in `~/.sift/cache/` and revalidated by ETag, so a repeated sweep costs
one small conditional request per file instead of the bytes again — 3.42s to 0.68s on a
13-quant repo. A `304` proves the cached header is still the file on the server; nothing is
served on trust. `SIFT_NO_CACHE=1` switches it off.

## Measured on an Apple M5, 16 GB

Reproduce with the commands; don't take the table.

**LM Studio baseline** (`sift-bench baseline`), Qwen3.5-9B Q4_K_M resident on MLX-NAX:
**21.03 tok/s** median, under 1% spread across runs. A dense model re-reads every weight
per token, so that's 118.1 GB/s effective against 119–136 GB/s of measured read bandwidth —
about **90% of what this machine actually delivers**, and ~77% of the M5's 153.6 GB/s
paper figure.

That gap between measured and paper bandwidth is itself the argument for measuring. And
90% of the real ceiling is why `sift` doesn't try to be a runtime: there is almost no
headroom left in kernel work.

**Read granularity costs 9×.** Same volume, cold, varying only block size:

| block | 1 thread | 8 threads |
|---|---|---|
| 64 KiB | **0.75 GB/s** | 3.19 GB/s |
| 12 MiB | 6.92 GB/s | 16.70 GB/s |

RAM reads at ~128 GB/s against ~6.9 GB/s from cold disk — **RAM is roughly 19× faster than
the SSD**, and that ratio is why "does it fit" is the question that matters.

### On honest disk numbers

`F_NOCACHE` prevents *new* caching but cannot evict resident pages, so a file the machine
touched recently reads at memory speed and looks like a spectacular SSD. `sift doctor`
flags any sample above 20 GB/s as a page-cache hit and excludes it, and warns when a sweep
has warmed its own sample.

At least one established project in this space discarded four of its own published results
after finding it had measured RAM.

## Accuracy: predicted versus measured

Every tool in this space publishes flattering numbers. Here is the one that matters, and
the two ways this one was wrong before it was right.

| model | predicted | measured | error |
|---|---|---|---|
| Qwen3.5-9B Q4_K_M, dense, LM Studio on M5 | 21 tok/s | **21.03 tok/s** | under 1% |

One model, one machine, one engine. That is a calibration point, not a validation — but it
is a real one, and getting it took discarding two measurements that looked fine.

**The first ruler measured the wrong access pattern.** `sift` sized the ceiling with a
STREAM *copy*. Decode streams weights in and writes back a small activation; a copy moves a
byte each way. The efficiency factor of 0.80 was silently absorbing that mismatch.

**The second ruler measured latency, not bandwidth.** The obvious fix — sum a buffer
single-threaded — reported 40 GB/s on a machine whose bus delivers three times that. One
accumulator is a serial dependency chain, and one core cannot saturate an Apple Silicon
bus. Four accumulators across all cores reads 119–136 GB/s.

Then the loop got fast enough to be impossible: **369 GB/s**, on a bus that tops out near
153. Reading the same immutable buffer into the same accumulators is loop-invariant, so
LLVM hoisted the whole sweep out and timed one pass as if it were four. Each pass now
depends on the last, and `sift doctor` refuses any figure above 1200 GB/s outright.

With an honest ruler, LM Studio decodes at **118.1 GB/s effective against 119–136 GB/s
measured** — around 90% of the machine, which is now the dense efficiency factor.

**The MoE factor is not calibrated.** It was rescaled to preserve the predictions the old
basis gave, so switching rulers did not silently move every MoE estimate — but rescaling a
guess leaves a guess. MoE rows are marked as estimates and this is the open half of the
work.

What `sift` gets right that a naive `bandwidth / file_size` model does not: **MoE sparsity**.
A 30B-A3B model reads about 3B parameters per token, not 30B. Getting this wrong is a ~16×
error on a 128-expert top-8 model, and it is a live open issue in a competing tool.

Treat MoE estimates as ordering hints. Treat the dense one as calibrated on exactly one
machine.

## Status

Early, and honest about it. See [TODO.md](TODO.md) for what is deliberately *not* built.

- Works and tested: measurement, GGUF reading (local and remote, single-file and split),
  MoE traffic, KV cache sizing, quality-first recommendation, routing.
- macOS, Linux and Windows. The measurement layer has a real implementation per OS behind
  one seam, so nothing outside `sift_core::platform` carries a `cfg`.
- Accelerator memory is only read on macOS, where `iogpu.wired_limit_mb` is a hard ceiling
  on host memory. Discrete VRAM is a different quantity and is **not** counted, so `fits`
  is conservative on a Linux or Windows box with a dedicated GPU. `sift doctor` says so
  rather than substituting a guess.
- 123 tests, clippy clean on all three.

## Prior art

Named because a router that pretends to be the only tool is not trustworthy as a router.

- [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) — remote GGUF header
  parsing, and it did it first.
- [llmfit](https://github.com/AlexsJones/llmfit) — hardware detection and model scoring.
- [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com),
  [llama.cpp](https://github.com/ggml-org/llama.cpp),
  [colibri](https://github.com/JustVugg/colibri) — the engines this recommends.
- Apple's *[LLM in a Flash](https://arxiv.org/abs/2312.11514)* — the read-granularity
  result reproduced above.

## Licence

MIT.
