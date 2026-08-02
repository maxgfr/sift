# TODO

## Deliberately not built

These are scope decisions, not gaps. They exist here so anyone landing on the repo can see
the boundary without reading the source.

- **An inference engine.** LM Studio runs Qwen3.5-9B on this machine at ~83% of the memory
  bus. There is almost no headroom in kernel work, and the runtime landscape churns past
  itself every quarter. `sift` recommends engines instead of becoming one.
- **Downloading models.** `sift` prints the exact command for whichever tool owns the job.
  Nothing to maintain around resume, checksums, auth or LFS.
- **A bundled model catalog.** The whole point is reading the real header of the real file.
  A catalog would rot, and catalog rot is the top complaint against the most popular tool
  in this space.
- **Training, fine-tuning, quantizing.** Other tools do these well.

## Next

Roughly in order of how much they improve the answer.

### Correctness

- [x] **Recommendation quality, not just size.** Ranking is now quality-first: bits per
      weight **measured from each file's own tensor directory**, not a lookup table keyed
      on the filename. A table has to be taught every new quant family and is wrong until
      someone updates it; this arithmetic works on a quantization invented this morning,
      and it also catches what a name cannot — two files both labelled `Q4_K_M` genuinely
      differ under dynamic quantization. Below 3.0 bpw a row is marked `damaged` and is
      only recommended when nothing better fits, with a warning saying so.
      On Qwen3-30B-A3B this moves the answer from `UD-IQ1_S` (2.37 bpw) to `Q3_K_M` (3.85).
- [ ] **Predicted vs measured table.** The single most valuable thing this repo could
      publish. Run `sift fit`, then run the model, then record the error — including where
      the estimator is wrong. Every competitor publishes only flattering numbers.
- [x] **KV cache in the fit calculation.** `fits` now weighs weights plus cache, sized
      from the model's real attention geometry — `block_count`, `head_count_kv`, head
      width — at a context you set with `--ctx` (4096 by default, roughly where engines
      start). Decisively, not marginally: on a 16 GB machine Qwen3-30B-A3B UD-IQ1_S reads
      `yes` at 4k and `no` at 64k, where 6.44 GB of f16 cache is more than half the budget.
      Reading `head_count` instead of `head_count_kv` would overstate it 8x under
      grouped-query attention, so there is a test pinning that.
- [x] **Split-shard models.** Every part's header is read and merged through
      `model::ModelShape`, because a shard carries only its own slice of the tensor
      directory — asking part 1 for the model's size gives an answer confidently a third
      of the truth. `unsloth/Qwen3-235B-A22B-GGUF` is 72 files, **none** of them whole:
      19 complete shard sets that `sift` was previously silent on. An incomplete set is
      reported rather than summed, since two of three shards add up to a plausible number
      that is wrong by a third.

### Reach

- [x] **Linux support.** `/proc/meminfo` (`MemAvailable`, never `MemFree`),
      `posix_fadvise(DONTNEED)` after `fsync` for cold reads.
- [x] **Windows support.** `GlobalMemoryStatusEx` (not
      `GetPhysicallyInstalledSystemMemory` — it reads SMBIOS and fails on VMs),
      `FILE_FLAG_NO_BUFFERING`. `seek_read` moves the shared file pointer, unlike Unix
      `pread`, so every reader thread now holds its own handle.
- [x] Structure both behind one seam with per-OS modules, so the rest of the crate never
      sees a `cfg`. See `crates/sift-core/src/platform/`.
- [ ] **Accelerator memory off macOS.** NVML and amdgpu sysfs on Linux, DXGI
      `QueryVideoMemoryInfo` on Windows. Deliberately not faked in the meantime: those
      report discrete VRAM, which is not a ceiling on host memory the way Apple's wired
      limit is, so folding them into one number would be wrong rather than incomplete.
- [ ] **`sift ls`** — every local model across LM Studio, Ollama, colibri and `~/.sift`,
      with regime and predicted speed. One view across tools that do not know about each
      other.
- [ ] **`sift bench`** — fold `sift-bench` into the main binary, drive Ollama as well as
      LM Studio, and persist results so measurements replace estimates over time.
- [ ] **`--json` on every command.** Being the thing another tool shells out to is how a
      CLI becomes a reference rather than a repo.

### Polish

- [ ] Cache fetched headers in `~/.sift/cache/` keyed by ETag. A `fit` sweep currently
      re-reads on every invocation and takes ~2 minutes for 25 quants.
- [ ] Parallelise the `fit` sweep. It is entirely network-bound and entirely serial.
- [ ] Safetensors support, so non-GGUF repos are not a dead end.
- [x] Homebrew formula in `maxgfr/homebrew-tap` plus a release workflow building
      macOS/Linux/Windows binaries. Cron hour 19 UTC is free; 0–18 are taken.
- [ ] Windows on ARM (`aarch64-pc-windows-msvc`). The `windows-11-arm` runner exists, so
      this is one matrix row — held back only because nothing here has been run on that
      hardware, and shipping an unverified binary is worse than shipping none.

## Known wrong

Things currently shipped that are known to be imperfect, recorded rather than hidden.

- **Efficiency factors are hand-set** (80% dense, 35% MoE) from a single machine's
  measurements. They should come from `sift bench` data, per architecture.
- **Ollama detection matches `~/.ollama`**, a data directory, so it can report Ollama as
  present after an uninstall. Fails soft by design, but it is a false positive.
- **The `fit` sweep warms nothing and caches nothing**, so it is slower than it needs to be
  by roughly the number of quants.
