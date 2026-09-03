# Gemma4-26B 30k-decode lever matrix (todo 0b87b76a#12 leg) — 2026-09-04

Unit: continuing optimization loop (#12) — attack the top open Gemma target
queued by the 2026-09-03 worker: 30k-loaded decode is ~7.4 t/s MLX vs 37.08 t/s
llama.cpp (+5.0x). Queued candidates: windowed-attention gate, compiled-decode
on gemma4 shape, upstream KV-quant re-check.

Window: clean — no foreign inference processes before/during any leg (gate
re-checked per leg), no benchmark ports in use, Mei-owned port 8024, logs in
Mei-owned disposable dirs `mei-runtime-gemma4-30k-{bl,kv8,cdec}/`,
local-model-bench READ-ONLY.

Config (proven safe, identical to the recorded family rows): generic profile,
prefill-step 256, context cap 65536, kv-bits none (except the kv8 leg),
disposable disk KV, model mlx-community/gemma-4-26b-a4b-it-4bit staged at
mei-models, release binary at mei-build (main 23811db, vmlx local 318a4e68).
Each leg: fresh disposable KV cache removed before start, cold server,
probe_long_context --lengths 30000 --max-tokens 32 (fresh fill + reuse decode).

## Measured

| leg | fresh 30k decode t/s | reuse decode t/s | 30k prefill pps | peak mem |
|---|---|---|---|---|
| baseline (this window, r1) | 7.099 | 7.395 | 229.2 | 27.23 GB |
| recorded family (2026-09-03, pref256 r3 + rows) | 7.431 / 7.16–7.51 | 7.354 | 266.2 | 27.23 GB |
| kv-bits 8 (r1) | — CRASH first decode step | — | ~112 s fill then died | — |
| compiled-decode true (r1) | 7.391 | 7.537 | 266.0 | 27.23 GB |
| compiled-decode true (r2) | 7.554 | 7.499 | 266.0 | 27.23 GB |
| compiled-decode true (r3) | 7.564 | 7.524 | 266.1 | 27.23 GB |
| cdec mean (n=3) | 7.503 (sd 0.096) | 7.520 (sd 0.019) | 266.0 | — |

Artifacts: artifacts/probe-30k-baseline-20260904-leg1.json,
artifacts/probe-30k-kv8-20260903T221424Z.json,
artifacts/probe-30k-cdec-20260903T{221646,221953,222214}Z.json.
Server stage dumps: mei-runtime-gemma4-30k-{bl,kv8,cdec}/logs/server.log.

## KV-quant re-check (kv-bits 8): CRASH — blocker, vmlx-side

kv-bits 8 engages (listening line shows "kv-bits 8"), 30k fill completes in
normal time (~112 s), then the FIRST decode step dies:

```
MLXLMCommon/KVCache.swift:1209: Fatal error: `update` was called on
`QuantizedRotatingKVCache`. Use `updateQuantized` instead.
```

Root cause: Gemma4's 25 rotating/sliding layers (ring=1024, RotatingKVCache)
are promoted to QuantizedRotatingKVCache by the kv-quant path, but the
Gemma4Text decode path still calls `cache.update(...)` — the `updateQuantized`
branch is never selected for this topology. Same family as the Qwen3.8 kv-quant
unavailability (KVCache.swift:911), different throw site. Requires a vmlx-side
model-path change; Mei config surface cannot fix it. kv-quant re-check
ANSWERED: unavailable on gemma4 without vmlx work.

## Compiled-decode on gemma4 shape: SILENT NO-OP — blocker, vmlx-side

3 cold repeats with --compiled-decode true: stage profile shows ZERO
`decode.compiled_forward` rows and per-token `decode.model_forward` (1.84–1.89
ms) identical to eager, `decode.async_eval_submit` 134–140 ms/step. The engaged
signature (seen in the Qwen cdec runs: `model_forward count=1` trace row +
`compiled_forward count=32`) never appears. vmlx's promote guard
(Evaluate.swift, `allPromotable`) silently returns for the gemma4 mixed
paged/kv/rotating topology → eager fallback, no error, no speed change. The
small cdec-vs-baseline deltas (+1–5%) sit inside the recorded family spread
(7.16–7.51) and are not a claim.

## Windowed-attention gate: already active; no valid configuration

The architecture's sliding window is ALREADY implemented in MLX: topology
log "layers=30 kvLayers=5 rotatingLayers=25" — 25 sliding layers attend over a
1024-ring (RotatingKVCache, window mask via makeMask), 5 full-attention layers
attend over all keys (architecturally required). `--max-kv-window` only caps
the rotating ring; any value < 1024 degrades the architecture window, and it
does not touch the 5 full layers. No headroom via this gate. Candidate
answered: the residual 30k decode cost is the 5 full-attention layers over 30k
keys + per-step GPU eval; correctly windowed layers are not the bottleneck.

## Root-cause profile of the 30k cliff

Per decode token at 30k (both baseline and cdec legs, MLXPress profile):
`decode.async_eval_submit` avg 134–140 ms vs `decode.model_forward` avg 1.9 ms
CPU graph-build. The ~135 ms/step is GPU execution of the 30k-wide decode
graph, NOT Swift/graph overhead and NOT the disk-KV tier (identical numbers
with in-memory attempt, 2026-09-03). Same signature as the accepted Qwen3.8
bandwidth record but worse amplitude per full-attention layer (~21 ms/layer × 5
at 30k vs ~1.4 ms/layer × 16 for Qwen3.8): llama.cpp's gemma4 path executes the
same architectural attention at 37 t/s, so the gap is a vmlx SDPA/long-KV
kernel-efficiency matter, not a hardware ceiling.

## Verdict

All Mei-side config levers for the Gemma 30k-decode gap are now measured:
kv-quant crashes, compiled decode silently no-ops, window gate has no valid
setup. Next levers are vmlx-internal (Gemma4Text `updateQuantized` path;
`allPromotable`/compiled promotion for the mixed paged/rotating topology;
long-KV SDPA kernel efficiency) — outside autonomous worker scope. #12 stays
OPEN; recorded as measured blocker with exact throw sites and signature rows.
Next queued distinct unit: Ornith-35B (primary) prefill-step sweep
(64/128/256/512) vs the current 512 default — the Gemma sweep showed +91%
prefill at 256, and Ornith 30k prefill (400 pps MLX vs 510.6 GGUF, -28%) is
the one Ornith row losing to the reference; decode already beats GGUF (+8.7%
short, +18% @30k).