# sift

**Will this model fit and run fast on your machine? Answered before you download it.**

```console
$ sift fit unsloth/Qwen3-30B-A3B-GGUF
machine: Mac17,2, 16.0 GiB RAM, 12.0 GiB usable, 89 GB/s memory
reading headers from unsloth/Qwen3-30B-A3B-GGUF without downloading…

  quant               size     fits    GB/token  est tok/s
  UD-IQ1_S         9.04 GB      yes       1.427         22  moe
  Q2_K            11.26 GB    tight       1.399         22  moe
  Q3_K_M          14.71 GB    tight       1.753         18  moe
  Q4_K_M          18.56 GB       no       2.095   disk-bound  moe
  Q8_0            32.48 GB       no       3.600   disk-bound  moe
```

Twenty-five quantizations evaluated. **Nothing downloaded.**

Today that answer costs an hour: pull 18 GB, find out it's too slow, delete, guess again.

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

Or `cargo build --release` — Rust 1.85+, no runtime dependencies beyond `curl`.

## Commands

```
sift fit <hf-repo>              which quantization to download, and why
sift route <model>              which engine should run it
sift inspect <path|hf-repo>     model shape, local or remote, no download
sift plan <model>               per-token traffic and speed ceilings
sift doctor [--disk-sample F]   measure this machine
sift engines                    which runtimes are installed here
```

`<model>` accepts a path, `org/repo`, `org/repo:QUANT`, or a URL.

## Measured on an Apple M5, 16 GB

Reproduce with the commands; don't take the table.

**LM Studio baseline** (`sift-bench baseline`), Qwen3.5-9B Q4_K_M resident on MLX-NAX:
**20.91 tok/s** median, under 1% spread across runs. A dense model re-reads every weight
per token, so that's ~128 GB/s effective against the M5's 153.6 GB/s ceiling — about **83%
of what the memory bus can physically deliver.**

That number is why `sift` doesn't try to be a runtime. There is almost no headroom in
kernel work.

**Read granularity costs 9×.** Same volume, cold, varying only block size:

| block | 1 thread | 8 threads |
|---|---|---|
| 64 KiB | **0.75 GB/s** | 3.19 GB/s |
| 12 MiB | 6.92 GB/s | 16.70 GB/s |

RAM measures ~95 GB/s against ~6.9 GB/s from cold disk — **RAM is roughly 14× faster than
the SSD**, and that ratio is why "does it fit" is the question that matters.

### On honest disk numbers

`F_NOCACHE` prevents *new* caching but cannot evict resident pages, so a file the machine
touched recently reads at memory speed and looks like a spectacular SSD. `sift doctor`
flags any sample above 20 GB/s as a page-cache hit and excludes it, and warns when a sweep
has warmed its own sample.

At least one established project in this space discarded four of its own published results
after finding it had measured RAM.

## Accuracy

The tok/s figures from `sift fit` are **estimates**: measured memory bandwidth times an
efficiency factor (80% dense, 35% MoE). They are labelled as estimates everywhere they
appear.

What `sift` gets right that a naive `bandwidth / file_size` model does not: **MoE sparsity**.
A 30B-A3B model reads about 3B parameters per token, not 30B. Getting this wrong is a ~16×
error on a 128-expert top-8 model, and it is a live open issue in a competing tool.

A predicted-versus-measured table is the next thing this README needs. Until it exists,
treat the estimates as ordering hints rather than promises.

## Status

Early, and honest about it. See [TODO.md](TODO.md) for what is deliberately *not* built.

- Works and tested: measurement, GGUF reading (local and remote), MoE traffic, routing.
- macOS, Linux and Windows. The measurement layer has a real implementation per OS behind
  one seam, so nothing outside `sift_core::platform` carries a `cfg`.
- Accelerator memory is only read on macOS, where `iogpu.wired_limit_mb` is a hard ceiling
  on host memory. Discrete VRAM is a different quantity and is **not** counted, so `fits`
  is conservative on a Linux or Windows box with a dedicated GPU. `sift doctor` says so
  rather than substituting a guess.
- 88 tests, clippy clean on all three.

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
