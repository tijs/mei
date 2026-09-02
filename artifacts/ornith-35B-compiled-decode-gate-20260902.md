# Ornith-35B compiled-decode gate — 35B MoE, 30k engage / 80k guard (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`, todo 1,
item (1) on the long-context artifact's remaining list: "compiled-decode gate on
35B (targets the recurrent-scan decode cost observed at 80k)"). Question: does
the graph-traced compiled decode path help the actual 35B MoE (qwen3_5_moe),
and is the default-off gate justified?

Machine: Sulaco M1 Max, 32 GB unified, device applegpu_g13s, recommended
working set 26.80 GB. Server: Mei release binary (vmlx-swift pinned
`aeb5e21c` + patches 0001-0005), port 8024. GGUF :8017 not running this window
(foreign; untouched). No foreign runner active during any probe.

## Exact variant & flags

- Model: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`
  (19,509,024,201 B, qwen3_5_moe, 40 layers, 10 KV / 30 Mamba, 128 experts);
  aligned repack backing store; `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`;
  `MEI_CONTEXT_CAP=90000 MEI_MEMORY_LIMIT_BYTES=30000000000`; fresh
  `MEI_KV_CACHE_DIR` per server; `--prefill-step-size 512` (default);
  `--compiled-decode true --compiled-decode-threshold 30000`;
  `VMLX_ENABLE_UNSAFE_COMPILE=1 MLXPRESS_GENERATION_PROFILE=1`.
- Probes: `probe_long_context.py --lengths 30000 80000 --max-tokens 32`, three
  repeats on the same server (`-lc-` = r1 cold, `-r2`, `-r3`), 12/12 rows PASS
  every run.

```bash
env MEI_MODEL_DIR=…/Ornith-1.5-35B-A3B-MLX-4bit-aligned MEI_CONTEXT_CAP=90000 \
  MEI_MEMORY_LIMIT_BYTES=30000000000 MEI_KV_CACHE_DIR=…/kv-cache-35b-cdec30k-unsafe \
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 MEI_COMPILED_DECODE=true \
  MEI_COMPILED_DECODE_THRESHOLD=30000 VMLX_ENABLE_UNSAFE_COMPILE=1 \
  MLXPRESS_GENERATION_PROFILE=1 bash scripts/start_mei_server.sh
```

## Engagement semantics (why this threshold)

- The iterator engages compiled decode only when
  `enableCompiledDecode && !compiledDecodeDenied(for: model)` AND
  `promptOffset <= compiledDecodeMaxPromptOffset` (Evaluate.swift:2113-2128).
  Above the threshold it prints `[compiled-decode] skipped promote+trace at
  offset N > threshold M; eager decode` and stays eager — the guard exists
  because the promote+trace materializes the whole prefill KV/SSM into
  FIXED-size Compilable buffers and records the attention graph (a multi-minute
  tax at 45K on hybrid qwen3_5, per the code comment).
- `compiledDecodeDenied` does NOT deny qwen3_5_moe (deny-list: deepseekv4,
  hy3/hunyuan, laguna, qwen4exp, minimax-gated).
- **Critical found gate**: `setupCompiledDecode` bails at
  `HardwareInfo.isCompiledDecodeSupported`, which is FALSE unless
  `VMLX_ENABLE_UNSAFE_COMPILE=1` (or `MLXPRESS_ENABLE_UNSAFE_COMPILE=1`) —
  disable-by-default because of the macOS Tahoe Metal JIT bug (MLX #3329/
  #3201/#3256: compiled eval returns zero → crash) + the Osaurus #1173
  model-switch corruption. WITHOUT that env, the "compiled" run is silent
  pure-eager (verified: a control run with the same flags but no env printed
  eager profile stages and — after reading this gate — is discarded as
  vacuous). With the env, engagement is PROVEN by the server log:
  `decode.compiled_forward count=32 total=69.8ms avg=2.183ms` (vs eager
  `decode.model_forward avg≈6.2ms`) and the absence of a skip line on the
  30000-offset row.

So with threshold 30000: fill_30000_fresh (offset 30000) RUNS COMPILED;
fill_30000_cache_reuse (offset 30001) is guard-skipped to eager; both 80k rows
(offset ≥ 80000) are guard-skipped to eager.

## Results — 3 clean repeats (engine decode t/s, 32 tokens, temp 0)

| row | decode path | r1 | r2 | r3 | mean |
|---|---|---|---|---|---|
| fill_30000_fresh | COMPILED | 36.42 | 40.22 | 40.13 | 38.9 |
| fill_30000_cache_reuse | EAGER (skip) | 49.97 | 50.27 | 49.85 | 50.0 |
| fill_80000_fresh | EAGER (skip) | 35.69 | 35.49 | 35.85 | 35.7 |
| fill_80000_cache_reuse | EAGER (skip) | 35.83 | 35.88 | 35.76 | 35.8 |

Prior-session eager baseline (no unsafe env, same config):
fill_30000_fresh 46.7 (47.5/46.5/46.1), fill_30000_reuse 47.9 (47.7/48.0/47.9),
80k ~35.2–35.4 (mean 35.3).

Memory: peak 26.48 GB on the 30k phase, 26.62 GB on the 80k phase — compiled
promote at 30k did NOT blow the budget (no fixed-buffer blowup at this length);
post-load 19.55 GB unchanged. All 12/12 probe rows PASS (no crash, no
divergent-output failure, KV reuse rows still restore 30000/80000). The
documented Metal JIT crash did not manifest on this run.

## Interpretation

1. **Compiled decode at 30k on the 35B MoE is SLOWER than eager on the very
   requests it engages** (38.9 mean vs 47.5 eager mean). The per-request
   promote+trace (rebuild fixed Compilable buffers + (re)record the graph after
   each fresh 30k prefill) lands inside the generation window of the triggering
   request (~0.3-0.5 s per 32-token row). No steady-state compiled gain could
   be isolated because every compiled row pays the tax.
2. The guard-skipped EAGER rows under `VMLX_ENABLE_UNSAFE_COMPILE=1` ran
   49.85–50.27 t/s at 30k vs 47.5–47.9 last session (+4–5%) — attributed to
   the OTHER `HardwareInfo.isCompiledDecodeSupported`-gated fused helpers
   (Qwen35 shapeless micro-fusions) that the same env switches on, but NOT
   proven: could be session variance. Needs a dedicated 3x A/B
   (`VMLX_ENABLE_UNSAFE_COMPILE=1` alone, compiled-decode false) before any
   claim.
3. At 80k the compiled path never engages (threshold guard) and per the guard's
   own design LONGER contexts make the promote tax worse (multi-minute at 45K
   on the 9B hybrid; our 30k tax already ≈ +0.3-0.5 s per request). The
   55→35 t/s decode falloff at growing context is NOT an iterator-dispatch
   overhead the compiled path removes — it is the recurrent/attention cost per
   step (the GDN decode path is already a fused compiled step in eager mode).
4. The compiled-decode default-off gate is now backed by 35B evidence and
   STAYS default-off (`MEI_COMPILED_DECODE=false` / env unset), exactly as the
   plan requires ("preserve default-off compiled decode until the actual 35B
   model gates pass" — the gate passed load/correctness but failed throughput).

## Verdict

- **Compiled-decode gate on 35B: CLOSED as a throughput lever.** Engaged,
   correct, memory-safe at 30k; no steady-state win (compiled rows 36–40 t/s,
   eager 47.5–50.3); long-context (80k) compiled is design-gated and, per the
   threshold guard + measured per-request tax, would only be slower. Keep
   default-off. The one new thread worth a dedicated A/B later: the +4–5% on
   ALL rows when `VMLX_ENABLE_UNSAFE_COMPILE=1` is set (unproven; needs 3x
   repeats with compiled-decode false).
- Ornith remains primary, NOT exhausted. Remaining credible experiments:
  100k survival (peak extrapolates ≈28.8 GB at 512-step — inside budget,
  untested), the unsafe-compile-env A/B above, MLXPress `MLXPRESS=70` A/B.
  No Qwen3.8/Laguna transition (no exhaustion condition met).
- No Mei source/config change; the tree stays as found (the uncommitted
  `scripts/run_measurement_cycle.sh` diff remains the coordinating session's).
  The empty-engagement control run (`-cdec30k-lc-…` and `-cdec-verify-…`
  JSONs) is preserved as the falsification record for the
  `isCompiledDecodeSupported` env gate.

## Artifacts

- `artifacts/ornith-35B-cdec30k-lc-20260902T035835Z.json` — control WITHOUT
  `VMLX_ENABLE_UNSAFE_COMPILE` (vacuous: compiled never engaged; kept as
  falsification record)
- `artifacts/ornith-35B-cdec-verify-20260902T040725Z.json` — profile control
- `artifacts/ornith-35B-cdec-unsafe-lc-20260902T041052Z.json` (r1, engaged,
  PASS 12/12)
- `artifacts/ornith-35B-cdec-unsafe-r2-20260902T041938Z.json` (r2, PASS)
- `artifacts/ornith-35B-cdec-unsafe-r3-20260902T042556Z.json` (r3, PASS)
- this summary; server log
  `~/.local/share/local-model-bench/mei-runtime/logs/server.log` (contains the
  `[compiled-decode]` engage/skip lines and `decode.compiled_forward` stages)