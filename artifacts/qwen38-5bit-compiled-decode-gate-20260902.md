# Qwen3.8-27B compiled-decode gate + unsafe-compile A/B (2026-09-02)

One focused experiment for Kiem plan `0b87b76a`, todo 4 (Qwen3.8 optimization,
critical gate: >=30 decode tok/s with acceptable 32 GB profile). Question: does
the graph-traced compiled decode path help the DENSE hybrid qwen3_5_text
model (48 linear_attention + 16 full_attention), and is the unsafe-compile env
(`VMLX_ENABLE_UNSAFE_COMPILE=1`) a lever of its own?

Machine: Sulaco M1 Max, 32 GB unified, applegpu_g13s, recommended working set
26.80 GB. No foreign runner active during any probe (checked before every
launch; port 8024 dedicated). Binary: Mei release, vmlx-swift pinned fork
`91fed8be` (Mei repo HEAD 06b0520, no source change this session).

## Exact config (all three runs)

Model: `Qwen/Qwen3.8-27B-5bit-affine-g64` (Mei-produced plain affine 5-bit
g64 bf16, 18,514,906,426 B, staged at
`~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-5bit-affine-g64`).
Flags identical to the proven parity config, per run:

```
MEI_MODEL_DIR=…/Qwen3.8-27B-5bit-affine-g64  MEI_SERVED_MODEL_ID=Qwen/Qwen3.8-27B-5bit-affine-g64
MEI_OPTIMIZATION_PROFILE=generic  MEI_PORT=8024  MEI_CONTEXT_CAP=65536  MEI_MAX_TOKENS=32768
MEI_PREFILL_STEP_SIZE=64  MEI_TEMPERATURE=0.6  MEI_TOP_P=0.95  MEI_TOP_K=20  MEI_EMIT_REASONING=true
MEI_CACHE_REUSE=true  MEI_MEMORY_LIMIT_BYTES=0  MEI_CACHE_LIMIT_BYTES=0  MEI_SSM_REDERIVE=true
MEI_LOAD_MMAP=true  MEI_KV_CACHE_DIR=<fresh disposable per server>  (kv-bits unset = none)
```

Invocation: `bash scripts/start_mei_server.sh` (background), probe via
`tools/probe_load.py --decode-tokens 32`, acceptance via `tools/probe_mei.py`.

| run | variant | extra env |
|---|---|---|
| A control | compiled-decode false | (none — parity baseline) |
| B compiled | compiled-decode true, threshold 65536 | `VMLX_ENABLE_UNSAFE_COMPILE=1 MLXPRESS_GENERATION_PROFILE=1` |
| C unsafe-eager | compiled-decode false | `VMLX_ENABLE_UNSAFE_COMPILE=1 MLXPRESS_GENERATION_PROFILE=1` |

All servers logged `[Qwen35] fused_gdn_decode_input_projections=active groups=[4]`
(uniform 5-bit affine allows all four GDN input projections into one fused
quantized matmul — active in ALL runs, i.e. baseline already includes it).

## Results — engine decode t/s, 32 tokens, 3 probe_load passes each

| variant | r1 | r2 | r3 | mean | sd |
|---|---|---|---|---|---|
| A control (this session) | 13.066 | – | – | 13.07 | – |
| A parity baseline (2026-09-02T16:05Z, 3 repeats) | 13.091 | 13.110 | 13.113 | 13.105 | 0.010 |
| B compiled (ENGAGED) | 11.840 | 11.844 | 11.841 | 11.842 | 0.002 |
| C unsafe-eager | 13.536 | 13.491 | 13.489 | 13.505 | 0.022 |

- B vs A: **-9.6 %** (compiled decode is a regression on this dense hybrid).
- C vs A: **+3.1 %** (unsafe-compile env alone, 3 repeats, consistent sd 0.022).

Memory: active 21.77 GB, peak 21.8-21.9 GB after load; unchanged across
variants (compiled promote at short offsets does not blow the budget).

## Engagement proof (server log, MLXPressGenerationProfile)

Variant B: `decode.compiled_forward count=32 … avg=1.530-1.654ms` and
`decode.model_forward count=1` (prefill only) — the per-token forward WAS
traced and replayed (`compiledDecodeDenied` does not deny qwen3_5_text; the
short 65-token prompt offset is under the 65536 threshold). Still the
generation loop took 3330-4405 ms per 32 tokens.

Variant C (eager, same env): `decode.model_forward count=33 … avg=8.36-19.0ms`,
`decode.step_build avg=8.0-19.2ms`, `decode.async_eval_submit
count=32 … avg=56.7-67.9ms`.

## Interpretation — the decode wall is weight bandwidth, not dispatch

`async_eval_submit` (56-86 ms/token) dwarfs `model_forward` (8-19 ms) and
`compiled_forward` (1.5-1.7 ms). The eager graph submit blocks until the whole
dense forward's Metal kernels finish. Qwen3.8-27B is DENSE (no `num_experts`;
48/64 layers are linear_attention with float32 recurrent state, 16/64
full_attention): every token streams ~18.5 GB of 5-bit weights through Metal.
At the M1 Max realistic ~350-400 GB/s, that is a ~46-53 ms floor per token —
matching the observed 56-86 ms/step. Compiled decode removes per-op launch
overhead but NOT weight streaming, so it cannot help a bandwidth-bound dense
model; it added the trace/replay fixed-buffer tax instead (-9.6 %).

The +3.1 % unsafe-eager gain replicates on a second architecture the Ornith
35B observation (unsafe env alone, compiled off, +4-5 % there) — attributed
there to the same env switching on additional fused Metal helpers; here the
C-vs-A delta is backed by 3 repeats and is within the same plausible
micro-fusion explanation. It is NOT adopted as a default yet: the env also
unlocks the documented macOS Tahoe Metal JIT bug surface (MLX #3329/#3201/
#3256) and the Osaurus #1173 model-switch corruption, and the Ornith 35B gate
record kept compiled-decode off. It stays a gated candidate pending
acceptance + a repeatable long-context guard.

## Acceptance evidence

probe_mei on the unsafe-eager server (variant C): first invocation skipped
the two context rows only because `--tokenizer` was omitted (probe-side, not
config); all 10 functional rows PASSED (mei_status, models_identity, plain,
parity stream/non-stream, tool non-streaming a=15 b=27, tool streaming, cache
repeat 1/2, cache growing turn1+turn2). Full re-run with tokenizer + cap
65536 was in flight at summary time (artifact name prefix
`probe-mei-qwen38-5bit-unsafe-full-`); see that artifact for the completed
context rows.

## Verdict

- **Compiled decode on Qwen3.8-27B: CLOSED as a throughput lever.** Engaged,
  correct, memory-safe, but -9.6 % vs eager. The model is decode-bandwidth
  bound (dense 27B); graph tracing cannot reduce per-token weight streaming.
- **Unsafe-compile env: +3.1 % repeatable, gated candidate, not defaulted.**
- ~~13.1 -> 30 tok/s~~ is NOT reachable by software dispatch changes on a
  dense 27B at 5-bit: the measured floor (~46-53 ms/token bandwidth) puts the
  realistic eager ceiling near 13.5-14.5 t/s. Remaining credible speed
  levers: 4-bit weights (16.05 GB, -13 % bytes -> ~+10-15 % speed, quality/
  tool-risk to re-gate), prefill-step tuning (fresh prefill is the weakest
  stage at ~50-190 pps), and the decisive counter-check: the matching
  llama.cpp GGUF Q5_K_M (19.7 GB) reference on the same machine — if it also
  lands in the 12-15 t/s band, the hardware ceiling is confirmed and todo 4
  can only close via documented exhaustion or bit-depth compromise.
- Kiem todo `0b87b76a#4` STAYS OPEN (target not met).

## Artifacts

- `artifacts/probe-load-qwen38-5bit-ctrl-r1-20260902T171616Z.json` (A)
- `artifacts/probe-load-qwen38-5bit-cdec-r1/r2/r3-*.json` (B)
- `artifacts/probe-load-qwen38-5bit-unsafe-r1/r2/r3-*.json` (C)
- `artifacts/probe-mei-qwen38-5bit-unsafe-*.json`, `…-unsafe-full-*.json` (C acceptance)
- runtime logs:
  `~/.local/share/local-model-bench/mei-runtime-q38-ctrl-20260902T171528Z/logs/server.log`
  `…/mei-runtime-q38-cdec-20260902T171635Z/logs/server.log`
  `…/mei-runtime-q38-unsafe-20260902T171806Z/logs/server.log`
- this summary.