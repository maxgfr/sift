# sift

**Compile the model to the machine.**

Every local-inference tool works the same way: a publisher picks a quantization, you
download that file, you find out how fast it runs, and if it's too slow you download a
different one. The model knows nothing about your machine.

That is why the same failure keeps appearing. Cross your RAM limit and throughput falls
off a cliff — llama.cpp thrashes, and streaming engines report drops like 0.63 → 0.07
tok/s *while their cache hit rate improves*. The OS pages behind your back and the engine
has no say.

`sift` inverts it: **you declare your machine and your target, and it decides what stays
resident.** The resident-set size becomes a number you set, not one that emerges from
kernel paging behaviour.

> **Status: early.** The measurement and model-inspection layers work and are tested. The
> inference engine is not written yet. Numbers below are real measurements from this
> repository's own tools; nothing here is projected or copied from a marketing page.

## Not another Ollama

| | Ollama / LM Studio | `sift` |
|---|---|---|
| You give it | a quant file you picked | your machine and a target |
| Model fits in RAM | fast — **use them** | comparable at best |
| Model is ~1.2× your RAM | thrashes | the whole point |
| Model is 3× your RAM | refuses | runs |
| Choosing the quant | guess, download, retry | measured and compiled |
| Resident set size | emergent | a number you set |
| Shape | app / daemon | library first, CLI on top |

If `sift` loses to LM Studio on a model that fits, that goes in this README as a table
row. The claim is about the case they cannot serve; overclaiming past it would destroy the
only thing this project has.

## What works today

```
sift doctor  [--disk-sample FILE]   measure RAM, memory bandwidth, cold disk, GPU limit
sift inspect MODEL [--tensors]      read a model's shape without loading its weights
sift plan    MODEL [--hit-rate R]   per-token traffic and the resulting speed ceilings
sift-bench probe | models           what's installed, and which regime each model is in
```

## Measured on an Apple M5, 16 GB

Produced by `sift doctor` on `Mac17,2`. Reproduce with the command, don't take the table.

**Read granularity dominates.** Same total volume, cold, varying only block size:

| block | 1 thread | 8 threads |
|---|---|---|
| 64 KiB | **0.75 GB/s** | 3.19 GB/s |
| 1 MiB | 4.03 | 7.26 |
| 4 MiB | 6.82 | 10.26 |
| 12 MiB | 6.92 | 16.70 |

A **9× penalty** for reading in small pieces rather than large ones. This reproduces
Apple's *LLM in a Flash* result on current hardware, and it is the entire argument for
laying experts out contiguously: in stock GGUF, using one expert costs three reads
scattered across the file.

Memory streams at **~95 GB/s** on this machine, against ~6.9 GB/s from cold disk. **RAM is
roughly 14× faster than the SSD**, and that single ratio governs every design decision
here.

### Baseline: LM Studio, and why kernels are not the lever

Measured with `sift-bench baseline` against LM Studio's own server, MLX-NAX runtime,
Qwen3.5-9B Q4_K_M (6.10 GiB, dense, fully resident), 128 tokens per run after a discarded
warm-up:

| run | seconds | tok/s |
|---|---|---|
| 1 | 6.15 | 20.80 |
| 2 | 6.12 | 20.93 |
| 3 | 6.12 | 20.91 |

**Median 20.91 tok/s**, and the spread across runs is under 1%.

A dense model re-reads every weight per token, so ~6.1 GB moves each token. At 20.91 tok/s
that is roughly **128 GB/s of effective bandwidth against the M5's 153.6 GB/s ceiling** —
LM Studio is running at ~83% of what the memory bus can physically deliver.

That is the most useful thing we have measured, and it is inconvenient: **there is almost
no headroom in kernel optimisation.** A faster matmul cannot help when the bus is the
limit. The only remaining lever is to move *fewer bytes per token* — which is what MoE
sparsity (a token touches 6–12% of expert weights) and speculative decoding actually do.

It also sets an honest expectation: on a dense model that fits, `sift` will not beat LM
Studio, and this README will keep saying so.

### On honest disk numbers

`F_NOCACHE` prevents *new* caching but cannot evict pages already resident, so a file the
machine touched recently reads at memory speed and looks like a spectacular SSD. `sift
doctor` flags any sample above 20 GB/s as a page-cache hit and excludes it, and warns when
a sweep has warmed its own sample. Cold-read discipline is not optional: at least one
established project in this space discarded four of its own published results after
discovering it had measured RAM.

## Design

Three ideas, each answering a measurement rather than an intuition.

**1. Bits should follow access cost, not just importance.** Uniform quantization spends
bits evenly; importance-aware quantization spends them where they matter. Both miss half
the equation — a weight in RAM is free to store, a weight on SSD costs 14× to move. So the
always-resident trunk gets high precision, hot experts get more bits than a uniform quant
would give them, and the cold tail — precisely the weights that cost bandwidth — gets
squeezed. At equal total size this can beat a uniform quant.

**2. Never let the OS decide.** Pin what fits with a measured margin, serve the rest
through explicit `pread` with `F_NOCACHE` so the cold tier cannot grow into the pinned set.

**3. The cache floor is not what it looks like.** Below one token's working set, a
demand-only expert cache holds nothing. Running the *next* layer's router on the *current*
layer's hidden state means an expert only has to survive one attention rather than one
whole token, which is what makes a small resident footprint viable at all. It is exact by
construction: the real router still decides, so logits are unchanged.

## Prior art

This is a crowded field and none of it is ours. Named credit, because the measurements
below shaped the design more than any paper did:

- **Apple, *LLM in a Flash*** (arXiv:2312.11514) — windowing, row-column bundling, and the
  read-granularity result this repo reproduces.
- **[flash-moe](https://github.com/danveloper/flash-moe)** — ~90 documented experiments on
  Apple Silicon, most of them negative, with reasons. Including: SSD DMA and GPU compute
  contend for the same memory controller, so prefetch during GPU work nets zero.
- **[WASTE](https://github.com/sqliteai/waste)** — Kimi K3 (2.78 T) on a 64 GB MacBook,
  and a `LEARNED.md` that preserves wrong beliefs next to their refutations.
- **[colibri](https://github.com/JustVugg/colibri)**,
  **[moe-stream](https://github.com/GOBA-AI-Labs/moe-stream)**,
  **[mbolt](https://github.com/doramirdor/mbolt)** — expert streaming, three-mode
  residency, and profile-guided expert layout respectively.
- **[moe-offload-findings](https://github.com/GaelicThunder/moe-offload-findings)** — nine
  measured results, six of them negative, that removed more from this design than they
  added.

## Building

```sh
cargo build --release
cargo test
```

Requires Rust 1.85+. macOS is the primary target; the measurement layer is Unix-portable.

## Licence

MIT OR Apache-2.0, at your option.
