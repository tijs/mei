# Ornith-35B first-generation +12.2 GB materialization — root cause found & eliminated (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`).
Machine: Sulaco M1 Max, 32 GB unified (34,359,738,368 B). Server: Mei release
binary (vmlx-swift pinned `aeb5e21c`), port 8024. Model:
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`
(19,509,024,201 B payload, qwen3_5_moe, 40 layers, 10 KV / 30 Mamba, 128 experts).

## Question (from the 2026-09-02 00:14 artifact)

Both backing stores show an IDENTICAL fixed +12.2 GB active-memory step on the
FIRST generation that never releases (anonymous 24.28→36.49 GB; aligned
19.55→31.76 GB; unchanged at ctx 4096 vs 65536 and at limit 24/30 GB;
vmmap: 120×128 MB + 40×256 MB weight-sized MTLBuffer regions). What is it?

## Instrument

`OSAURUS_MLX_MALLOC_TRACE=1 OSAURUS_MLX_MALLOC_TRACE_BYTES=16777216`
(env-gated allocator tracer already in the pinned fork) on the ALIGNED path
(file-backed NoCopy, zero realigned copies at load). Backtraces are
symbolicated C+++Swift through the release binary.

## Root cause — PROVEN by backtrace

During the FIRST forward pass (prefill) the model allocates:

- **40 × 256 MiB (10.00 GiB)** — one per layer
- **80 × 16 MiB (1.25 GiB)** — two per layer (fused scales/biases)
- total **11.25 GiB**, retained for process lifetime (module properties).

Demangled backtrace of a 256 MiB event:

```
#5 MLXLMCommon.SwitchGLU.ensureFusedGateUp()          ← lazy concat cache
#6 SwitchGLU.callAsFunction(preDownScores:)
#7 MLXLLM.Qwen35SparseMoeBlock.callAsFunction
#8 Qwen35DecoderLayer.callAsFunction
#9 Qwen35TextModelInner.callAsFunctionCapturing
... (TokenIterator.step)
```

This is the **SwitchGLU lazy fused gate+up gatherQuantizedMM cache**
(`Libraries/MLXLMCommon/SwitchLayers.swift`, `ensureFusedGateUp()`): on first
forward it concatenates gate+up quantized weights/scales/biases along the
output axis, evaluates the concat eagerly (`MLX.eval`), and keeps it forever.
Measured checkpoint arithmetic confirms the size: gate+up weight =
268 MB/layer × 40 = 10.74 GB (+ ~1.2 GB scales/biases) ≈ 12.0 GB, under the
upstream 512 MiB/layer default cache limit
(`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES` / `BENCH_NO_FUSED_GATE_UP` gates), so
the fuse applies. The 9B proxy (dense qwen35) has no switch_mlp → no fuse.

## Fix (upstream env gate, zero Mei code change)

`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` → `ensureFusedGateUp` bails at the
cacheLimit check; decode falls back to the original two-call
`gatherQuantizedMM` path (stock upstream behavior). Verified by trace:
ZERO 256/16 MiB fuse events on the fuse-off legs.

## A/B (same binary; same request "Reply with exactly: ready", temp 0, max_tokens 64)

| leg | post-load active | gen active (max) | prefill 15 tok | decode t/s (engine) | notes |
|---|---|---|---|---|---|
| ORIGINAL + fuse ON  (prior worker row) | 24.28 GB | 36.49 GB | 117.6 s | **4.02** | over 32 GB, swap thrash |
| ALIGNED + fuse ON   (trace control, this run) | 19.55 GB | 31.77 GB | 40.2 s | **0.094** | over/at 32 GB, file-page thrash |
| ORIGINAL + fuse OFF (3 repeats) | 24.28 GB | **24.54 GB** | ~0.3 s | **49.8** (48.4/50.4/50.5) | fits; content `ready` |
| ALIGNED + fuse OFF  (3 repeats) | 19.55 GB | **20.27 GB** | 0.31 s | **55.0** (55.0/55.0/55.1) | **fits; primary config** |

The +12.2 GB was the ENTIRE measured slowdown: with it gone, prefill drops
117.6/40.2 s → 0.3 s and decode 4.02/0.094 → ~55 t/s on BOTH backing stores.

## Correctness gates — ALIGNED + fuse-off, ctx 65536, kv-cache-dir set (probe_mei.py)

`artifacts/acceptance-ornith35B-fuseoff-aligned-65536-diskcache-20260902T0530Z.json`
**STATUS: PASSED** — all 12 gates:

- `models_identity`, `mei_status`, `plain_completion` (55.9 t/s, active 19.82 GB)
- `parity_stream_vs_nonstream` (streaming/non-streaming parity)
- `tool_nonstreaming` + `tool_streaming` (add_numbers `{"a":15,"b":27}`,
  ~55.6 t/s, TTFT 2.2 s / 0.78 s cache-hit)
- `cache_repeat_1/2` (repeat2 restored 6165/6170 cached tokens, prefill
  75472 pps), `cache_growing_turn1` + `cache_growing_turn2_reuses_slot`
  (turn2 restored 785/790 cached tokens, prefill 3972 pps)
- `context_exact_cap` (65536 ok) + `context_over_cap_rejected` (65537 rejected)
- Peak during the 65536-token context prefill: **25.73 GB < 32 GB**.

## KV-reuse path finding (why `pagedEnabled=false`)

The qwen35 hybrid cache (30 Mamba layers) is **paged-incompatible** —
`CacheCoordinator.setPagedIncompatible(true)` fires on first hybrid slot
admission (`cacheCannotUsePagedCoordinatorRestore`), so prefix reuse rides the
**disk tier** (`TQDiskSerializer`) only. Without `MEI_KV_CACHE_DIR` the
growing-transcript gate fails (cached_tokens=0) — a CONFIG gap, not a model
regression (9B cells that passed carried `--kv-cache-dir`). With the Mei-owned
`kv-cache-35b-fuseoff` dir set, KV reuse works (`ssmHits=3` observed; disk-tier
restores). The 16-block paged pool stays off — expected for hybrid.

## MLXPress router finding (secondary)

`[MLXPressRouter] enabled=false … symbol=missing` in the logs: the advisor only
resolves the C symbol when `enabled==true` (options default to `.disabled`
because Mei constructs `LoadConfiguration(useMmapSafetensors:)` with the public
init that defaults `jangPress: .disabled`, and the 35B MLX `config.json` lacks
`num_experts` so `.auto` can't classify it routed). `symbol=missing` is
therefore cosmetic. `MLXPRESS=70` env would bypass both gates for the mmap
path; per-expert routed region names DO match the C++ registry regex
(`language_model.model.layers.N.mlp.switch_mlp.*` stacked). Not needed for the
fit now — file-backed decode runs at 55 t/s with resident pages.

## Behavior note (not a gate failure)

temp 0 + `emit-reasoning=true`: creative prompts ("limerick about 42") can burn
the full 64-token budget in `reasoning_content` with empty `content`
(finish=length). The acceptance prompts ("Reply with exactly: ready", tool
prompts) produce visible content + valid tool calls. Engine t/s includes
reasoning tokens.

## Verdict

- **Root cause: PROVEN** — first-forward `SwitchGLU.ensureFusedGateUp` concat
  cache (+11.25 GiB traced), retained forever, on both backing stores.
- **35B NOW FITS 32 GB in a usable configuration**: aligned backing store +
  `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` + `MEI_KV_CACHE_DIR` set →
  post-load 19.55 GB, generation active 20.27 GB, peak 25.73 GB at 65k ctx,
  **55.0 t/s decode (3 clean repeats)**, full acceptance PASSED (incl. tool
  calls, streaming parity, KV reuse, cap gates). Goal ≥30 t/s: exceeded on
  short decode.
- Long-context (>65k) survival and 3× long-context throughput remain unproven
  on 35B (deferred); the engine limits here are now memory headroom, not the
  working set.
- Ornith remains primary. NOT exhausted — next candidates: (1) long-context /
  KV-reuse cell runs on the fitted 35B config; (2) MLXPress `MLXPRESS=70`
  cold-routing A/B on the aligned path (Lever B); (3) compiled-decode gate on
  35B now that decode is clean. No Qwen3.8/Laguna transition (per escalation
  order — no exhaustion condition met).

## Artifacts & commands

- Trace control: `artifacts/ornith-35B-trace-r1-20260902T0300Z.json` (PASS
  412.5 s, max active 31.77 GB); trace events summary
  `artifacts/ornith-35B-fuse-trace-events-20260902.json`.
- Fuse-off aligned r1: `artifacts/ornith-35B-fuseoff-aligned-r1-20260902T0400Z.json`
  (PASS 1.07 s; prefill 341 ms, 55.1 t/s, active 19.82 GB).
- Acceptance (4096-cap, cache-gate artifacts): `...-20260902T0430Z.json` /
  (65536 no-kvdir): `...-20260902T0500Z.json` / (65536 + disk, FULL PASS):
  `...-20260902T0530Z.json`.
- 3× repeats: `ornith-35B-fuseoff-aligned-3x-repeat-20260902T0600Z.json`
  (reasoning-only limerick row), `ornith-35B-fuseoff-aligned-3x-repeat-ready-20260902T0610Z.json`
  (55.0/55.0/55.1), `ornith-35B-fuseoff-original-r1-20260902T0630Z.json`
  (48.4/50.4/50.5, active 24.54 GB).
- Server (both legs): `MEI_MODEL_DIR=<dir> MEI_CONTEXT_CAP=65536
  MEI_MEMORY_LIMIT_BYTES=30000000000 MEI_LOAD_MMAP=true MEI_KV_CACHE_DIR=…/kv-cache-35b-fuseoff
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 [MLX_SAFETENSORS_MMAP_DEBUG=1
  MLXPRESS_DEBUG=1 MLXPRESS_GENERATION_PROFILE=1
  OSAURUS_MLX_MALLOC_TRACE=1 OSAURUS_MLX_MALLOC_TRACE_BYTES=16777216]
  bash scripts/start_mei_server.sh`; probes via mem35b/trace probes + probe_mei.py.
- Aligned repack dir: `~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned`
  (tool `tools/align_safetensors.py`, committed 3317d1b).

No Mei source code changed. Committed: this artifact, trace summary, probe
JSONs, model-lineup status update.