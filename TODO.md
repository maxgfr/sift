# TODO

## Deliberately not built

These are scope decisions, not gaps. They exist here so anyone landing on the repo can see
the boundary without reading the source.

- **An inference engine.** LM Studio runs Qwen3.5-9B on this machine at ~90% of its
  *measured* read bandwidth. There is almost no headroom in kernel work, and the runtime
  landscape churns past itself every quarter. `sift` recommends engines instead of becoming
  one.
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
- [x] **Predicted vs measured table.** Published in the README, dense half done: Qwen3.5-9B
      Q4_K_M predicted 21 tok/s, measured 21.03 on LM Studio. Getting there meant throwing
      away two measurements that looked fine — a STREAM copy that measured the wrong access
      pattern, and a single-threaded read that measured latency and reported 40 GB/s on a
      bus delivering three times that. Then a third that was impossible: 369 GB/s, from an
      invariant loop the optimiser hoisted. All three are written up rather than hidden.
- [ ] **Calibrate the MoE efficiency factor.** The open half. It was rescaled to preserve
      the predictions the old copy-based basis gave, so switching rulers moved nothing
      silently — but rescaling a guess leaves a guess. Needs an MoE model measured on a real
      engine, the way the dense one was.
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
- [x] **`sift ls`** — every local model across LM Studio, Ollama, colibri and `~/.sift`,
      with regime and predicted speed. Ollama needed its manifests read rather than its
      blob directory listed: the blobs are `sha256-<digest>` with no extension and no
      indication of which is a model, which a projector and which a template.
- [ ] **`sift bench`** — fold `sift-bench` into the main binary, drive Ollama as well as
      LM Studio, and persist results so measurements replace estimates over time.
- [x] **`--json` on every command.** Under `--json`, progress chatter goes to stderr so
      stdout stays one parseable document.

### Polish

- [x] Cache fetched headers in `~/.sift/cache/` keyed by ETag. Validated, never expiring:
      a `304 Not Modified` proves the bytes on disk are the bytes on the server, which
      matters here because the whole claim is reading the *real* header of the *real* file.
      The trap was which validator to send — HuggingFace redirects to a signed CDN URL
      whose `etag` describes an object that expires and whose signature encodes the byte
      range, so sending it back gets the whole body again. `x-linked-etag` on the 302 is
      the file's stable hash and answers 304 before the redirect is even followed.
      A repeated sweep drops from 3.42s to 0.68s.
- [x] Parallelise the `fit` sweep. Eight workers over a shared queue: 72 files in 22
      seconds, against a serial run that had not finished after five minutes.
- [ ] Safetensors support, so non-GGUF repos are not a dead end.
- [x] Homebrew formula in `maxgfr/homebrew-tap` plus a release workflow building
      macOS/Linux/Windows binaries. Cron hour 19 UTC is free; 0–18 are taken.
- [ ] Windows on ARM (`aarch64-pc-windows-msvc`). The `windows-11-arm` runner exists, so
      this is one matrix row — held back only because nothing here has been run on that
      hardware, and shipping an unverified binary is worse than shipping none.

## Known wrong

Things currently shipped that are known to be imperfect, recorded rather than hidden.

- **The MoE efficiency factor is still a guess** (28%). The dense one (90%) is now
  calibrated against a real measurement on one machine — one machine, one model, one
  engine, so treat it as a calibration point rather than a validated constant.
- **Ollama detection matches `~/.ollama`**, a data directory, so it can report Ollama as
  present after an uninstall. Fails soft by design, but it is a false positive.
- **The header cache never evicts.** Entries are small and validated, so a stale one
  cannot produce a wrong answer, but `~/.sift/cache/` grows without bound. `SIFT_NO_CACHE`
  disables it; there is no `sift cache clear` yet.
