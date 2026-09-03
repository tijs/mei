# Qwen3.8-27B-4bit raw-completions crash: root cause + fix (todo 0b87b76a#7 blocker) — 2026-09-03

## Blocker restated

`probe-mei-qwen38-4bit-20260902T222245Z.json` (previous tick): raw `/v1/completions`
deterministically killed the server with `Fatal error: SmallVector out of range`
at vmlx `mlx/c/array.cpp:335` for the mlx-community 4-bit checkpoint at ANY length
(11/60/30k/65k tokens), while chat/completions and the Mei-produced 5-bit
checkpoint (raw 30k + 65k) passed on the SAME binary. Suspected delta: the 4-bit
bundle's bundled vision sidecar (`333 vision_tower.*` tensors + `vision_config`
+ processor files).

## Root cause (proven, disassembly + source)

1. **Loader routing**: `ModelFactoryRegistry` (vmlx MLXLMCommon/ModelFactory.swift)
   registers the VLM factory FIRST, LLM factory second. The mlx-community 4-bit
   bundle carries `vision_config` + `processor_config.json` +
   `preprocessor_config.json` + `video_preprocessor_config.json`, so
   `loadModelContainer` instantiates **MLXVLM.Qwen35** (multimodal class).
   The Mei-produced 5-bit bundle has none of those files -> VLM factory rejects
   it -> **MLXLLM.Qwen35Model** (plain LLM).
2. **Crash site**: MLXVLM `Qwen35.prepare(_:cache:windowSize:)` line ~3300 runs
   `let promptTokenCount = inputIds.dim(1)` UNCONDITIONALLY on
   `input.text.tokens`. Fresh requests (cache miss) deliver 1-D `[T]` tokens;
   `dim(1)` on a 1-D array raises SmallVector out of range (array.cpp:335).
   Verified in the crash report + disassembly:
   `0x101714514: ldr x20, [sp,#0xb8]` (inputIds) / `0x101714518: bl MLXArray.dim`
   with `w0=1`, at `Qwen35.prepare +0x9A8`.
3. **Precedent**: the runtime ALREADY knows this failure mode —
   vmlx Evaluate.swift "Rebuild inputForPrepare with tokens shaped as `[1, T]`"
   reshapes to 2-D on the cache-restore path precisely because "the Qwen3.5 VLM
   `Qwen35Language.LanguageModel` which reads `inputs.dim(1)` ... crash with
   MLX's `SmallVector out of range` ... when fed a 1D tensor". The raw
   completions path never received the same treatment. Chat requests evidently
   land 2-D in practice (cache-restore rebuild; all 10 chat acceptance legs
   passed) — raw does not.

## Fix (Mei, minimal)

`Sources/MeiCore/Engine.swift` `completionRunLocked`: emit the raw token array as
`[1, T]` via `MLXArray(tokens).expandedDimensions(axis: 0)` instead of 1-D.
Safe for both model classes: the VLM prepare reads `dim(1)` on the batch axis;
the LLM default prepare flattens and only rejects batch > 1. Same shape the
cache-restore path already uses. No config, profile, or default changes.

## Re-gate (same binary family, proven safe config: generic, cap 65536,
prefill step 64, kv-bits none, disposable disk KV, port 8024)

New binary sha256(16) `d4fc502152ebd161` (previous parity binary
`a0bd8367aa5cb3cb`; only Mei source change is the raw-path token shape).

| Check | Before fix | After fix | Artifact |
|---|---|---|---|
| raw completions, 10-token prompt | CRASH 0.013s | PASS: prefill 16.0 pps, decode 15.9 t/s, active 18.66 GB | this md (curl) |
| raw completions, ~120-token prompt | CRASH (ladder 60 tok, 0.006s) | PASS | same |
| probe_mei chat legs (models/status/plain/tools/parity/KV reuse/cache growing) | 10/10 | 10/10 | probe-mei-qwen38-4bit-fix-r1-20260903T001034Z.json |
| probe_long_context 30k raw (fresh) | CRASH x2 (0.37s) | PASS: 30,000 tok @ 56.1 pps, decode 11.9 t/s @30k, peak 24.71 GB; +30,001-tok restore leg (823 cached, 70,933 pps) | probe-longctx-qwen38-4bit-fix-20260903T001306Z.json |
| probe_mei context_exact_cap 65536 raw | BLOCKED (crash) | PASS: 65,536 tok with 30,001 cached-token reuse, 86.3 pps effective, peak **31.70 GB < 32 GB** | probe-mei-qwen38-4bit-fix-full-20260903T002230Z.json |
| probe_mei context_over_cap 65537 rejected | BLOCKED (server dead) | PASS: HTTP 400 "request exceeded context cap: 65537 > 65536" | same |
| probe_mei FULL | 10/12 | **12/12, status passed** | same |
| Heretic 4-bit raw (same VLM-routed bundle) | untested (no raw legs in its 10/10) | PASS: 11/21/2-token raw, 16.0-17.1 t/s, peak ~19 GB, server alive | this md (curl) |
| 5-bit raw regression (LLM-path bundle) | raw 30k/65k PASS on old binary | PASS: 11-token 13.6 t/s + 1,681-token 12.5 t/s (peak 22.15 GB), server alive | this md (curl) |

Memory note: the 65,536-token raw prefill peaked at 31.70 GB — the earlier
linear extrapolation (~37 GB) was pessimistic; the real 4-bit full-cap fill
FITS the 32 GB machine (5-bit's 34.64 GB peak at cap did not).

## Status

- Qwen3.8-27B-4bit common-matrix BLOCKER CLEARED: raw completions work at
  every length, full 12/12 acceptance incl. context-cap gates, 30k long
  context survives, decode speed unchanged (15.5-16.0 t/s short).
- Todo 0b87b76a#7 stays OPEN (Gemma/Heretic/Ornith legs pending).
- Defaults unchanged (5-bit remains the documented full-parity artifact).
- Commit: 2026-09-03 tick (Source change + artifacts).