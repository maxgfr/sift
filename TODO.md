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

- [ ] **Recommendation quality, not just size.** `fit` currently recommends the largest
      file that fits, which on Qwen3-30B-A3B picks a 1-bit quant over a 3-bit one that is
      barely larger. Size is a bad proxy for quality below ~3 bits. Needs a quality prior
      per quant family, and probably a "smallest quant I would not warn you about" floor.
- [ ] **Predicted vs measured table.** The single most valuable thing this repo could
      publish. Run `sift fit`, then run the model, then record the error — including where
      the estimator is wrong. Every competitor publishes only flattering numbers.
- [ ] **KV cache in the fit calculation.** Context length changes what fits, sometimes
      decisively. Right now `fits` ignores it, which makes long-context users' answers
      optimistic.
- [ ] **Split-shard models.** Currently skipped entirely. Multi-part GGUFs are how the
      largest models ship, so the tool is silent on exactly the cases where the question
      is hardest.

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
- [ ] Homebrew formula in `maxgfr/homebrew-tap` plus a release workflow building
      macOS/Linux/Windows binaries. Cron hour 19 UTC is free; 0–18 are taken.

## Known wrong

Things currently shipped that are known to be imperfect, recorded rather than hidden.

- **Efficiency factors are hand-set** (80% dense, 35% MoE) from a single machine's
  measurements. They should come from `sift bench` data, per architecture.
- **Ollama detection matches `~/.ollama`**, a data directory, so it can report Ollama as
  present after an uninstall. Fails soft by design, but it is a false positive.
- **The `fit` sweep warms nothing and caches nothing**, so it is slower than it needs to be
  by roughly the number of quants.
