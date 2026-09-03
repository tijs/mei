# Changelog

All notable changes to Mei are documented here.

## [Unreleased]

### Changed

- Generic-profile disk-KV safety default extended to Gemma 4 bundles:
  `model_type` `gemma4`/`gemma4_text` never restore exact-repeat prefixes on
  the in-memory-only paged tier (cached=0; the disk tier restored 6173/6174
  in the 2026-09-03 Gemma 4 26B matrix), so with `--cache-reuse` on and no
  explicit `--kv-cache-dir` they now default to a disposable on-disk cache
  under the OS temp directory, matching the existing dense
  qwen3_5/qwen3_8 default. Explicit `--kv-cache-dir` always wins;
  `--cache-reuse false` keeps caching fully disabled; the MoE/Ornith
  `qwen3_5_moe` family is untouched. `denseQwen35NeedsDiskKVTier` /
  `denseQwen35KVUnsafeModelTypes` renamed to `needsDiskKVTier` /
  `diskKVRequiredModelTypes` (old names kept as deprecated aliases).

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
