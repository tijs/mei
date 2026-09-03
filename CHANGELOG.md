# Changelog

All notable changes to Mei are documented here.

## [0.2.0-alpha.1] - 2026-09-03

Source-first preview release candidate. Ships the verified Qwen3.8 / Gemma 4 /
Qwen3.8-Heretic runtime work on top of the 0.1.0 Ornith release and bumps the
runtime version metadata. **Not tagged, not pushed, not published** — public
release requires explicit user authorization.

### Added

- Qwen3.8-27B, Gemma 4 26B-A4B and Qwen3.8-Uncensored (Heretic) MLX
  loadability/acceptance evidence on the generic profile; the staged
  `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`
  and `orcarouter/Qwen3.8-27B-Uncensored-MLX` 4-bit checkpoints are loadable
  and acceptance-gated (see `configs/model-lineup.json` for per-model status,
  exact revisions, digests, quant settings and measured results).
- Model-aware disposable on-disk KV default extended to Gemma 4 bundles
  (`model_type` `gemma4`/`gemma4_text`): exact-repeat prefixes never restore on
  the in-memory-only paged tier for this family, so with `--cache-reuse` on and
  no explicit `--kv-cache-dir` they now default to a disposable on-disk cache
  under the OS temp directory, matching the existing dense
  qwen3_5/qwen3_8 default. Explicit `--kv-cache-dir` always wins;
  `--cache-reuse false` keeps caching fully disabled; the MoE/Ornith
  `qwen3_5_moe` family is untouched. `needsDiskKVTier` /
  `diskKVRequiredModelTypes` replace the former
  `denseQwen35NeedsDiskKVTier` / `denseQwen35KVUnsafeModelTypes` (deprecated
  aliases kept).
- Reader-facing release metadata: README logo (`assets/mei-logo.png`),
  `mei --version` now reports `0.2.0-alpha.1`, an explicit staging allowlist
  and script (`configs/release-allowlist.json`,
  `scripts/stage_release_candidate.sh`) and this release-notes file
  (`docs/RELEASE-0.2.0-alpha.1.md`).
- Gemma 4 chunked-prefill default 64 → 256, arch-scoped
  (`ModelOptimizationProfile.prefill256ModelTypes` = `gemma4`/`gemma4_text`;
  Ornith stays 512, all other models stay 64, explicit `--prefill-step-size`
  always wins, malformed metadata stays 64). Measured 2026-09-03 on
  `mlx-community/gemma-4-26b-a4b-it-4bit`: 30k fresh fill 266.5/265.3/266.2
  pps (3 cold repeats) vs ~139 baseline (+91%), peak 27.23 GB unchanged,
  30k loaded decode unchanged (~7.4 t/s), acceptance pass-set identical to
  the 64 baseline (only the pre-existing user-gated Gemma string-args tool
  schema fails). The 30k-decode row closes the last measurable GGUF A/B gap
  (GGUF 37.08 t/s, MLX 5.0x slower — dense/rotating-attention cost, recorded
  as a measured constraint). Note: this change postdates the staged
  v0.2.0-alpha.1 snapshot (2026-09-03T14:27Z); any future staging refresh
  must re-run `scripts/stage_release_candidate.sh`. Evidence:
  `artifacts/gemma4-prefill-step-sweep-20260903.md`.

### Fixed

- Raw `/v1/completions` crash on the VLM-routed Qwen3.8-27B-4bit checkpoint
  (deterministic `SmallVector out of range`, vmlx `mlx/c/array.cpp:335`, at ANY
  length): `MLXVLM.Qwen35.prepare` requires batch-first tokens; the raw
  completions path delivered a 1-D token array. Raw and chat `LMInput` sites
  now emit `[1,T]` via `expandedDimensions(axis: 0)` — safe for both the VLM
  prepare and the rank-safe LLM default prepare. The 4-bit raw path re-gated
  PASS on the full matrix (acceptance/streaming/tool/coding/KV/long-context).
- Gemma 4 fresh chat-prefill SIGTRAP (rank-3 embedding slice in
  `Gemma4.prepare`, any prompt length): chat `LMInput` built 1-D `[T]` tokens
  and the VLM prepare embedded them directly; chat sites now emit batch-first
  `[1,T]` like the raw path and the vmlx cache-restore rebuild.
- Gemma 4 growing-transcript prefix-extension reuse (cached_tokens 0 →
  786/824 tokens restored on the disk tier; cache-ON == cache-OFF
  byte-identical outputs): vmlx fork commit `318a4e68` admits standalone
  rotating-window cache topologies to the gen-suffix-stripped boundary store.
  **This fork commit is NOT yet pushed** — it is consumed via a local SwiftPM
  edit; `Package.swift`/`Package.resolved` stay pinned to the remote revision
  `91fed8be`. Pure-source builds therefore lack this fix until the fork
  `main` advances (blocker, user action).

### Known blockers

Full list with measured evidence in `docs/RELEASE-0.2.0-alpha.1.md`; summary:

- Fork commit `318a4e68` (Gemma4 reuse fix + cache-fetch diagnostics) is
  un-pushed; external/source-only builds resolve `91fed8be` without it.
- Gemma 4 tool calls emit string-typed JSON arguments; the strict-schema tool
  gate FAILS on this family (user go/no-go).
- Qwen3.8-27B decode is below the 30 t/s primary target: 4-bit 15.66 t/s
  (sd 0.060, 3 cold repeats, peak 18.9 GB), 5-bit parity artifact 13.11 t/s —
  hardware ceiling accepted and recorded in the plan (2026-09-02); 4-bit raw
  path is fixed and re-gated.
- Ornith-1.5-35B reaches >= 30 t/s only with the env-gated fused gate/up cache
  disabled (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, user-gated); MTP /
  speculative decode stays out of scope.

### Release status

Source-first preview candidate. The validated runtime target is macOS 15+ on
Apple Silicon with local MLX/Metal. Model weights are never bundled (see
`docs/RELEASE-0.2.0-alpha.1.md` for the weight-separation boundary). No
tag/release was created; publishing requires explicit user authorization.

## [0.1.0] - 2026-09-02

Initial public release for Apple Silicon.

### Added

- Native Swift/MLX OpenAI-compatible server for one local model per process.
- Automatic `auto|generic|ornith` runtime profile selection from model metadata.
- Validated Ornith profile: aligned checkpoint support, prefill step 512, and
  Ornith-only fused gate/up cache disablement.
- Conservative generic profile with experimental compiled decode, rotating KV
  quantization, bounded windows, and SSM anchors disabled by default.
- In-process and optional disk-tier KV/prefix reuse.
- Scoped disposable-cache cleanup and a 20 GiB free-space launch guard.
- Local vMLX patch queue pinned to an immutable upstream revision.
- Focused unit tests and release provenance.

### Changed

- Generic-profile safety default: dense Qwen3.5/Qwen3.8-lineage checkpoints
  (`model_type` `qwen3_5`/`qwen3_5_text`) crash the in-memory-only paged KV
  cache tier (vmlx `array.cpp:335`, crash trigger isolated by a bounded 2x2
  on 2026-09-02), so with `--cache-reuse` on and no explicit `--kv-cache-dir`
  they now default to a disposable on-disk cache under the OS temp
  directory. Explicit `--kv-cache-dir` always wins; `--cache-reuse false`
  keeps caching fully disabled; the MoE/Ornith `qwen3_5_moe` family is
  untouched.

### Release status

This is an initial public, source-first release. The validated runtime target
is macOS 15+ on Apple Silicon with the aligned Ornith 1.5 35B checkpoint. The
model is not bundled. Full model/GPU acceptance requires the local MLX/Metal
runtime and is not reproduced by every CI runner.

[0.1.0]: https://github.com/tijs/mei/releases/tag/v0.1.0