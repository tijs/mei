# Ornith-35B `VMLX_ENABLE_UNSAFE_COMPILE=1` dedicated A/B — compiled decode OFF (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`, todo 1
remaining item): the unproven +4–5% thread left by the compiled-decode gate
session (e218ec3) — guard-skipped EAGER rows under `VMLX_ENABLE_UNSAFE_COMPILE=1`
ran 49.85–50.27 t/s at 30k vs 47.5–47.9 in the baseline session, attributed to
the same env flag's OTHER gated helpers (shapeless micro-fusions gated by
`HardwareInfo.isCompiledDecodeSupported`). This run isolates that effect:
**`VMLX_ENABLE_UNSAFE_COMPILE=1` with `MEI_COMPILED_DECODE=false`**, 3 clean
repeats per leg, fresh and cache-reused context metrics separated.

## Pin note (IMPORTANT — mid-session repository move)

The repository advanced underneath this session (external commits): commit
`b169dce` ("refactor: consume vmlx fork directly", 07:55Z) changed
`Package.resolved` from `aeb5e21c` to the fork `91fed8be`, which additionally
contains decode-path commits NOT in the night pin (`e1f64fec` GDN grouped
input-projection fusion + MLA bf16 decode SDPA, `32e683a6` quantized-module
activation dtype, `ab09d363` compiled-decode region guard, …). Consequences:

- The **100K survival experiment ran on `aeb5e21c`** (launched 07:40Z, before
  b169dce landed; verified identical 80k numbers 35.2–35.3 vs the aeb5e21c
  night baseline). Its numbers are therefore directly comparable to the
  80k/90k proof session.
- **THIS A/B ran entirely on `91fed8be`** (both legs launched after the pin
  move). Cross-pin comparison to the night 47.5–47.9 numbers is INVALID
  (the fork's GDN fusion shifts absolute decode itself). The A/B verdict is
  therefore strictly **within-pin**: same binary, same model, same machine,
  only the env flag differs.
- The fork's GDN fusion is confirmed live on both legs via server log:
  `[Qwen35] fused_gdn_decode_input_projections=active groups=[4]`.

## Exact variant & flags

- Model: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ pinned HF revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`, aligned repack backing store.
- Server (both legs): `MEI_CONTEXT_CAP=90000 MEI_MEMORY_LIMIT_BYTES=30000000000
  MEI_KV_CACHE_DIR=…/kv-cache-exp-unsafe{A,B}-<TS> (fresh, cold, disposable)
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`; `--compiled-decode false` (default),
  `--prefill-step-size 512`, `--load-mmap true`.
- Leg A: `VMLX_ENABLE_UNSAFE_COMPILE=1`. Leg B: env absent.
  (`MLXPRESS_ENABLE_UNSAFE_COMPILE` alias also unset in B.)
- Probes: `probe_long_context.py --lengths 30000 80000 --max-tokens 32`,
  three repeats per leg on the same server.
- All 24/24 rows PASSED (12/12 per leg; KV-reuse `full_prefix_cached` gates
  PASS, no crash, memory normal).

## Results — 3 clean repeats each leg (engine decode t/s, 32 tokens, temp 0)

| row | Leg A `env=1` (mean ± sd) | Leg B env off (mean ± sd) | Δ A/B |
|---|---|---|---|
| fill_30000_fresh | **51.14 ± 0.57** (51.80/50.82/50.80) | 48.43 ± 0.19 (48.21/48.58/48.51) | **+5.6%** |
| fill_30000_cache_reuse | **52.27 ± 0.04** (52.27/52.22/52.31) | 49.73 ± 0.11 (49.79/49.60/49.79) | **+5.1%** |
| fill_80000_fresh | 35.88 ± 0.39 (36.03/36.18/35.44) | 35.45 ± 0.10 (35.48/35.33/35.54) | +1.2% |
| fill_80000_cache_reuse | 36.00 ± 0.01 (35.99/36.01/36.00) | 35.72 ± 0.10 (35.82/35.73/35.62) | +0.8% |

Memory: post-load 19.55 GB both legs; peak 26.64 GB (A) / 26.79 GB (B) during
80k phase — no regression from the env flag; both < 30 GB limit.

## Interpretation

1. **The +4–5% hint is REPRODUCED and now PROVEN at 30k context**: +5.1–5.6%
   on fresh AND cache-reused 30k rows, legs non-overlapping (A 50.80–51.80 vs
   B 48.21–48.58). Mechanism: `VMLX_ENABLE_UNSAFE_COMPILE=1` switches on
   `isCompiledDecodeSupported`-gated shapeless micro-fusion helpers (the same
   flag also arms full compiled decode — which was OFF here, so the effect is
   the micro-fusions alone).
2. **At 80k the effect shrinks to ~+1%** (35.88 vs 35.45 fresh; 36.00 vs 35.72
   reused) — consistent with the recurrent-scan decode cost dominating long
   context; the fusion gains are swallowed by the SSM scan.
3. The absolute B-leg 30k numbers (48.4–49.7) sit above the aeb5e21c night
   baseline (46.7–47.9) because of the fork's decode commits (GDN fusion) —
   NOT a claim about those commits; the forks delta was not A/B'd this session.

## Verdict

- **`VMLX_ENABLE_UNSAFE_COMPILE=1` alone (compiled decode off) is a PROVEN
  +5% throughput lever at ≤30k context on the 35B** (3×3 clean repeats, fresh
  and reused metrics separated), negligible at 80k. Correctness gates all
  passed; no crash in 12/12 env=1 rows this session.
- Despite the win, **default stays OFF**: the flag is named unsafe upstream
  (Metal JIT compile corruption crash MLX#3329/#3201/#3256 + model-switch
  corruption #1173; Mei is single-model so #1173 is mitigated, but the Metal
  JIT crash class remains a documented residual risk), and the plan rule
  requires a complete correctness/performance gate + non-default-off decision
  by the user for ANY generic compiler engagement. This artifact is the
  evidence a future opt-in (env or profile) would cite. Do NOT silently
  enable it as default.
- Ornith remains primary and NOT exhausted; with this A/B done, all remaining
  plan-listed 35B candidates (MLXPress=70, 100k survival, unsafe-compile A/B)
  are now closed or proven this session. Next credible step would be a
  source-backed low-risk vMLX allocator or recurrent-scan experiment with a
  clear acceptance gate (none currently staged) or the user's go/no-go on the
  env opt-in. No Qwen3.8/Laguna transition — no exhaustion condition met.
- No Mei source change this iteration.

## Artifacts

- Leg A: `ornith-35B-unsafeA-r1-20260902T0816Z.json`, `-r2-…T0823Z.json`,
  `-r3-…T0830Z.json` (PASS 4/4 each)
- Leg B: `ornith-35B-unsafeB-r1-20260902T0840Z.json`, `-r2-…T0847Z.json`,
  `-r3-…T0854Z.json` (PASS 4/4 each)
- this summary; server logs in `~/.local/share/local-model-bench/mei-runtime/logs/server.log`