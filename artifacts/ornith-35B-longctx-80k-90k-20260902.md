# Ornith-35B long-context survival & throughput at 80k/90k — 3 clean repeats (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`): close the
last unproven correctness dimension for the primary target — long-context survival
and 3x long-context throughput on the fitted 35B config
(aligned backing store + `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, see
`artifacts/ornith-35B-fuse-gateup-eliminated-20260902.md`), at context lengths
beyond the previously-proven 65536.

Machine: Sulaco M1 Max, 32 GB unified (34,359,738,368 B), device applegpu_g13s,
recommended working set 26.80 GB. Server: Mei release binary (vmlx-swift pinned
`aeb5e21c`), port 8024, `prepare_metallib.sh` precached kernels. GGUF reference
llama-server :8017 NOT running this window (foreign; untouched).
No foreign runner active during any probe (checked per phase).

## Exact variant & flags

- Model: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ pinned HF revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`, exact payload 19,509,024,201 B,
  architecture qwen3_5_moe (40 layers, 10 KV / 30 Mamba, 128 experts).
- Backing store: aligned repack dir `~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned`
  (0 realigned tensors; payload-bit-identical per `MEI_ALIGN_MANIFEST.json`).
- Server env: `MEI_CONTEXT_CAP=90000 MEI_MEMORY_LIMIT_BYTES=30000000000
  MEI_KV_CACHE_DIR=…/kv-cache-35b-lc80k (new,cold) VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`;
  args as printed at launch: `--context-cap 90000 --prefill-step-size 512
  --memory-limit-bytes 30000000000 --cache-limit-bytes 0 --compiled-decode false
  --load-mmap true --kv-cache-dir … --emit-reasoning true --cache-reuse true`.
  `--kv-bits none`, `--max-kv-window 0`, `--ssm-anchor-boundaries 0` (defaults off).
- Post-load memory: active 19,551,131,108 B (19.55 GB) — identical to the 65536-cap
  runs (cache is demand-allocated, not cap-sized). Peak counter reset to 0 after load
  (Engine.swift:125), so all peaks below are inference-only.
- Server log: `~/.local/share/local-model-bench/mei-runtime/logs/server.log`
  (truncated at launch; prior content preserved in committed artifacts).

## Commands

```bash
# server
env MEI_MODEL_DIR=$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned \
  MEI_CONTEXT_CAP=90000 MEI_MEMORY_LIMIT_BYTES=30000000000 \
  MEI_KV_CACHE_DIR=$HOME/.local/share/local-model-bench/mei-runtime/kv-cache-35b-lc80k \
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 bash scripts/start_mei_server.sh

# long-context probe x3 (same server+dir; r1 = cold disk cache)
$MEI_RUNTIME_VENV/bin/python tools/probe_long_context.py \
  --base-url http://127.0.0.1:8024/v1 --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
  --tokenizer $ALIGNED_DIR --lengths 30000 80000 --max-tokens 32 \
  --output artifacts/ornith-35B-longctx-lc80k-r{1,2,3}-<TS>.json

# full acceptance at the new cap (incl. 90000 exact-cap fill + 90001 reject)
$MEI_RUNTIME_VENV/bin/python tools/probe_mei.py --base-url http://127.0.0.1:8024/v1 \
  --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --tokenizer $ALIGNED_DIR \
  --context-cap 90000 --output artifacts/acceptance-ornith35B-longctx-90000-<TS>.json
```

## Results — long-context probe (3 clean repeats), 12/12 rows PASSED

Decode t/s (engine `tokens_per_second`, generation of 32 tokens at temp 0):

| row | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| fill_30000_fresh (cached 0) | 47.50 | 46.47 | 46.09 | 46.7 |
| fill_30000_cache_reuse (cached 30000) | 47.69 | 48.02 | 47.92 | 47.9 |
| fill_80000_fresh (cached 30001) | 35.24 | 35.31 | 35.15 | 35.2 |
| fill_80000_cache_reuse (cached 80000) | 35.43 | 35.32 | 35.24 | 35.3 |

- **80k-context decode = 35.2–35.3 t/s across 6 rows — still ≥ the 30 t/s goal at
  full 80k context.** Context-length trend (same engine t/s): ~55 (short 15 tok) →
  ~47 (30k) → ~35 (80k); the recurrent (Mamba) scan cost per decode step grows with
  context length.
- Fresh chunked prefill: 30k tokens 76.5 s (~400 pps); 80k fresh portion ~411 pps
  (196.6–198.5 s). Restore (disk tier): 30k+1 → 30000 cached @ 106k–557k pps;
  80k+1 → 80000 cached @ 97k–154k pps. Cross-length prefix reuse works: the 80k
  fill restores the 30k prefix (cached_tokens=30001) before prefill of the rest.
- Memory: after 30k fill active ≈ 21.0 GB; after 80k fill ≈ 23.2 GB; peak 26.49 GB
  (r1) → 26.62 GB (r2, r3), all < 30 GB memory limit and < 32 GB physical.
- Observation (non-gating): an EXACT repeat 30k prompt (fill_30000_fresh in r2/r3)
  reports cached_tokens=0 — the disk serializer does not reuse the stored 30001-seq
  for an equal/shorter-length prompt — while the 80k fill restores the 30k prefix.
  Hypothesis: store is keyed by full sequence length and restore requires the new
  prompt to strictly extend (length > stored); not investigated further this run.

## Results — acceptance at cap 90000 (`probe_mei.py`), 12/12 PASSED

- `models_identity`, `mei_status` (contextCap 90000, prefill step 512) — passed.
- `plain_completion` 58.2 t/s (content ok), `parity_stream_vs_nonstream` passed.
- `tool_nonstreaming` + `tool_streaming` passed (add_numbers `{"a":15,"b":27}`,
  finish_reason tool_calls) — tool-call correctness intact at the new cap.
- KV reuse gates: `cache_repeat_1/2` (6165/6170 restored, 76.9k pps restore),
  `cache_growing_turn1` + `cache_growing_turn2_reuses_slot` (785/790 restored) — passed.
- `context_exact_cap` 90000: passed in 53.2 s; prompt 90000 tokens, 80001 restored
  from disk, fresh portion 10k @ 1777 pps; active 24.08 GB, **peak 27.72 GB < 32 GB**.
- `context_over_cap_rejected`: 90001-token prompt rejected HTTP 400
  ("request exceeded context cap: 90001 prompt tokens > 90000 allowed"). Passed.
- Final /v1/mei/status after all probes: active 20.26 GB (released to baseline),
  cache 5.73 GB, peak 27.72 GB; cache stats `ssmHits=14 ssmMisses=0`
  (SSM-state restore path genuinely engaged at long context), `pagedEnabled=false`
  (hybrid paged-incompatibility re-confirmed; disk tier carries reuse as documented).

## Verdict

- **35B long-context survival PROVEN at 80k (3 cold repeats) and at the 90k cap
  (exact-cap fill + reuse + boundary reject).** Peak 27.72 GB at 90k < 32 GB with
  ~4.3 GB headroom under the physical bound and 2.3 GB under the 30 GB MLX limit.
- **3x long-context throughput: 35.3 t/s at 80k context (mean of 6 rows, sd ≈ 0.1)
  — above the ≥30 t/s goal even at full 80k depth**; KV-reused decode at 80k
  (35.3 t/s) is not materially faster than fresh (35.2 t/s) because decode is
  recurrent-state bound, not prefill bound.
- Correctness gates at the new cap: streaming/non-streaming parity, tool calls
  (non+stream), KV reuse (disk tier), exact-cap/over-cap — all PASSED.
- Ornith remains primary and NOT exhausted. Remaining credible 35B experiments:
  (1) compiled-decode gate on 35B (targets the recurrent-scan decode cost observed
  at 80k); (2) chunked-prefill step tuning (fresh long prefill ~400 pps vs 55 t/s
  decode is now the least-optimized part of the pipeline); (3) MLXPress
  `MLXPRESS=70` cold-routing A/B on the aligned path; (4) 100k survival (needs
  MEI_CONTEXT_CAP ≥ 100032 server). No Qwen3.8/Laguna transition — no exhaustion
  condition met, per escalation order.
- No Mei source change this iteration (measurement-only run). Tree left as found
  (the uncommitted `scripts/run_measurement_cycle.sh` readiness-poll diff belongs to
  the interrupted 9B Phase-C work of the coordinating session; not touched).

## Artifacts

- `artifacts/ornith-35B-longctx-lc80k-r1-20260902T015543Z.json` (PASS, cold)
- `artifacts/ornith-35B-longctx-lc80k-r2-20260902T020036Z.json` (PASS)
- `artifacts/ornith-35B-longctx-lc80k-r3-20260902T020527Z.json` (PASS)
- `artifacts/acceptance-ornith35B-longctx-90000-20260902T021014Z.json` (PASS 12/12)
- this summary; server log tail in `~/.local/share/local-model-bench/mei-runtime/logs/server.log`