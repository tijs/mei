# Ornith-35B prefill-step sweep 2 — finer steps 64/128 vs the 512 default (2026-09-04)

Unit: plan 0b87b76a, todo #12 (continuing optimization loop) leg — queued by the
2026-09-04 Gemma lever-matrix tick ("Ornith-35B (primary) prefill-step sweep
(64/128/256/512) vs the current 512 default; Ornith 30k prefill 400 pps MLX vs
510.6 GGUF, -28%, is the one Ornith row losing to the reference").

Question: can a chunked-prefill step BELOW the measured range (256 was the
smallest tested on 2026-09-02) lift fresh-prefill throughput toward the
llama.cpp reference? Gemma4 showed +91% prefill at 256 vs its baseline, so the
direction was worth testing on the qwen3_5_moe hybrid path.

Window: clean — no foreign inference processes before/during either leg, no
benchmark ports in use, Mei-owned port 8024, per-leg disposable runtime dirs
`mei-runtime-ornith35-step{64,128}/`, local-model-bench READ-ONLY.

Config (identical to the 2026-09-02 curve for direct comparability): release
binary at mei-build (Sources == main 23811db, HEAD 0fd00cf artifacts-only;
vmlx fork pinned 91fed8be), optimization profile auto→ornith, prefill step
<leg>, context cap 90000, memory limit 30000000000, VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0,
compiled-decode false, load-mmap true, kv-bits none, max-kv-window 0,
ssm-anchor-boundaries 0, fresh disposable disk KV per leg, model
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ 19504d912fa8fc7622bf6b1de3db5d5d890b1f02
from the aligned repack (payload-bit-identical per MEI_ALIGN_MANIFEST.json).
One `probe_long_context.py` run per leg: `--lengths 30000 80000 --max-tokens 32`.

## New rows (r1 each — exploratory; both are strict regressions, no 3x repeat warranted)

| step | 30k fresh pps | 30k peak | 80k fresh pps | 80k peak | 80k decode t/s | status |
|---|---|---|---|---|---|---|
| 64  | **206.1** | 22.02 GB | **232.4** | 25.09 GB | 35.0–35.7 | 12/12 PASS, slowest |
| 128 | 326.5 | 22.02 GB | 355.8 | 25.09 GB | 34.8–35.6 | 12/12 PASS |

## Full curve (new + historical rows from ornith-35B-prefill-step-tuning-20260902.md)

| step | 30k fresh pps | 30k peak | 80k fresh pps | 80k peak |
|---|---|---|---|---|
| 64   | 206.1 | 22.02 GB | 232.4 | 25.09 GB |
| 128  | 326.5 | 22.02 GB | 355.8 | 25.09 GB |
| 256  | 371    | 21.83 GB | 401    | 24.92 GB |
| 512  | ~392   | (26.62 GB @80k) | ~411   | **26.62 GB** |
| 1024 | 403    | 24.24 GB | 311    | 30.26 GB |
| 2048 | 409    | 27.51 GB | 223    | 33.83 GB |

Decode is step-invariant (30k 47–49 t/s, 80k 34.8–35.7 t/s across ALL rows) —
the step gate moves prefill only.

## Findings

1. **Monotonic beam-down confirmed below 512.** 30k fresh pps: 206 (64) → 327
   (128) → 371 (256) → 392 (512); 80k: 232 → 356 → 401 → ~411. Every step
   below 512 strictly reduces Ornith fresh-prefill throughput; the boundary
   cost (per-chunk `MLX.eval(cache)` + `Memory.clearCache()` sync; 30k/64 = 469
   chunks vs 59 at 512) dominates any per-chunk memory saving. The Gemma result
   does not transfer: qwen3_5_moe hybrid (GDN fused kernel + MoE) pays more per
   boundary than gemma4's dense path.
2. **Peak memory still falls with step** (30k: 22.02 GB at 64/128 vs 21.83 at
   256; 80k: 25.09 at 64/128 vs 24.92 at 256) — the memory advantage of small
   steps is real but < 0.2 GB and comes at a −47% to −16% prefill penalty. Not
   a trade any scenario wants while disk KV is available.
3. **512 remains the proven optimum** (best 80k pps, decode identical, peak
   26.62 GB inside the 30 GB limit). No Mei config change; the tree is left as
   found. The remaining Ornith 30k-prefill deficit vs llama.cpp (510.6 pps) is
   now bounded on both sides: bigger chunks (1024/2048) win at 30k but blow the
   80k memory budget; smaller chunks lose everywhere. The plateau is the
   ～400 pps memory-budget optimum already recorded 2026-09-02.

## Artifacts

- artifacts/ornith-35B-step64-lc-20260903T232957UTC.json (PASS 12/12)
- artifacts/ornith-35B-step128-lc-20260903T234353UTC.json (PASS 12/12)
- this summary; per-leg server logs in
  ~/.local/share/local-model-bench/mei-runtime-ornith35-step{64,128}/logs/
  (disposable kv-cache dirs removed post-leg; disk back to 39 GiB free).

## Verdict

Queued candidate ANSWERED: no step below 512 helps Ornith prefill; 512 stays
default. #12 stays OPEN. Next queued distinct unit (per plan order):
todo #13 — wire the isolated Mei MLX backend into local-model-bench
(dedicated launch/stop scripts, port, logs, model staging, bench configs),
reading its AGENTS.md first; #12 remains available for a future vmlx-side
lever (long-KV SDPA / compiled promotion) that is out of autonomous scope.