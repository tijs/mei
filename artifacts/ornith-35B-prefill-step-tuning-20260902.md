# Ornith-35B chunked-prefill step tuning — 256/512/1024/2048 A/B (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`, todo 1,
item on the previous iteration's remaining list: "chunked-prefill step tuning
— fresh long prefill ~400 pps is the least-optimized stage"). Question: does
`--prefill-step-size` (the `LLMModel.prepare` chunk window) move fresh-prefill
throughput and/or memory on the fitted 35B config, and is 512 the right default
for the 32 GB bound?

Machine: Sulaco M1 Max, 32 GB unified (34,359,738,368 B), device applegpu_g13s,
recommended working set 26.80 GB. Server: Mei release binary (vmlx-swift pinned
`aeb5e21c` + patch series 0001-0005), port 8024, `prepare_metallib.sh`
precached kernels. GGUF llama-server :8017 NOT running this window (foreign;
untouched). No foreign runner active during any leg (checked per phase).

## Exact variant & flags (identical across legs except the step size)

- Model: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ pinned HF revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`, exact payload 19,509,024,201 B,
  architecture qwen3_5_moe (40 layers, 10 KV / 30 Mamba, 128 experts).
- Backing store: aligned repack dir
  `~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned`
  (0 realigned tensors; payload-bit-identical per `MEI_ALIGN_MANIFEST.json`).
- Server env (leg 1024/256/2048): `MEI_CONTEXT_CAP=90000
  MEI_MEMORY_LIMIT_BYTES=30000000000 MEI_KV_CACHE_DIR=<fresh dir per leg>
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 MEI_PREFILL_STEP_SIZE=<leg>`;
  args as printed at launch: `--compiled-decode false --load-mmap true
  --kv-bits none --max-kv-window 0 --ssm-anchor-boundaries 0` (defaults off).
- Baseline leg 512 = the proven configuration from
  `artifacts/ornith-35B-longctx-80k-90k-20260902.md` (3 clean repeats at cap
  90000, acceptance 12/12); NOT re-run this iteration — its numbers are reused
  as history per the no-duplicate-measurement rule.
- One `probe_long_context.py` run per leg: `--lengths 30000 80000 --max-tokens
  32` (rows: fresh + strict-extension cache_reuse per length), fresh disk
  kv-cache dir per leg.

```bash
env MEI_MODEL_DIR=…/Ornith-1.5-35B-A3B-MLX-4bit-aligned MEI_CONTEXT_CAP=90000 \
  MEI_MEMORY_LIMIT_BYTES=30000000000 \
  MEI_KV_CACHE_DIR=…/kv-cache-35b-step$STEP VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 \
  MEI_PREFILL_STEP_SIZE=$STEP bash scripts/start_mei_server.sh
$MEI_RUNTIME_VENV/bin/python tools/probe_long_context.py --base-url http://127.0.0.1:8024/v1 \
  --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --tokenizer $ALIGNED_DIR \
  --lengths 30000 80000 --max-tokens 32 --output artifacts/ornith-35B-step$STEP-lc-<TS>.json
```

## Results — 12/12 rows PASSED on every completed leg (256, 1024, 2048-clean)

Fresh-prefill throughput (engine `prompt_tokens_per_second`) and inference
peak (`mei_memory_peak_bytes`, peak counter reset after load):

| step | 30k fresh pps | 30k peak | 80k fresh pps | 80k peak | 80k decode t/s | note |
|---|---|---|---|---|---|---|
| 256  | 371 | 21.83 GB | 401 | **24.92 GB** | 35.3–35.4 | slightly slower, lowest peak |
| 512  | ~392 (76.5 s) | (prior) | **~411 (197 s)** | **26.62 GB** | 35.2–35.4 | proven baseline, 3 clean repeats |
| 1024 | 403 | 24.24 GB | 311 | 30.26 GB | 33.1–35.0 | peak > 30 GB limit |
| 2048 | **409** | 27.51 GB | 223 | **33.83 GB** | 35.0–35.3 | 30k fastest; 80k peak > 32 GB phys |

(512 peak/pps from the baseline artifact; decode 46–48 t/s at 30k on all
legs — decode unaffected by step size at 30k; at 80k only 1024 dipped to
33.1, the rest 35.0–35.4.)

Key rows:

- 2048 at 30k fresh: 75.3 s, 408.8 pps, peak 27.51 GB — the FASTEST 30k
  prefill of all legs, i.e. larger chunks DO amortize boundary overhead (fewer
  `MLX.eval(cache)` + `Memory.clearCache()` syncs per prompt).
- 2048 at 80k fresh: 363.0 s, 222.7 pps, peak 33.83 GB — active memory at fill
  end only 23.16 GB (same as every other leg); the peak is the per-chunk
  single-forward footprint (chunk × prefix attention/mask + `[B,T,Hv,Dv]` GDN
  output + MoE gathers), which grows with chunk size — linear in chunk, and it
  is what crosses the physical bound.
- 1024 at 80k: 262 s, 311 pps, peak 30.26 GB — over the configured 30 GB MLX
  limit; the first leg where memory pressure degraded prefill.
- Decode is resilient: 33.1–35.4 t/s at 80k across ALL legs including the
  over-32 GB one (swap pressure hits prefill, not single-token decode).

## Findings

1. **512 is already the right default for 30–90k contexts on 32 GB.** The
   ~400 pps fresh-prefill plateau is NOT boundary/overhead-bound (disproven:
   2048 was fastest at short context). It is MEMORY-bound: the per-chunk
   forward footprint scales with chunk size at fixed context, so the step that
   fits under the peak-memory budget is the step that runs at full speed.
   Peak vs step at 80k is nearly linear: 24.92 (256) → 26.62 (512) → 30.26
   (1024) → 33.83 (2048) GB.
2. An adaptive step (large chunk while context is short, small chunk near the
   cap) would recover the 30k-fastest behavior without the 80k blowup, but the
   total time for a full 80k fill is dominated by the long-tail chunks — the
   expected gain is small (~consistency, not a step-change). Not worth a fork
   change on this evidence; recorded as a candidate only.
3. Mechanism note (from fork code, not instrumented): the GDN layers
   (`MLXLLM/Models/GatedDelta.swift`) use ONE fused Metal kernel
   (`gated_delta_step*`, grid (32, Dv, B·Hv)) that iterates the whole chunk
   sequentially inside the kernel with per-thread register state; the masked
   (prefill) kernel is used when `createSSMMask` supplies a mask. The
   per-chunk sequential depth and the `[B,T,Hv,Dv]` output both scale with
   chunk size. The 2048-collapse is consistent with per-chunk memory
   (33.83 GB > 32 GB physical → swap), not with the kernel being slower per
   token in isolation (30k row: 409 pps is FASTER than 512's 392).

## Environment incident (reported, cleaned up)

- During the first 2048 attempt the data volume ran to 100% (116 MiB free)
  from accumulated `mei-runtime/kv-cache-*` dirs (43 GB `kv-cache-sweep` from
  the interrupted 9B Phase-C + 8.1 GB `kv-cache-35b-lc80k` + per-cell dirs +
  this iteration's legs). Symptoms: probe write `OSError: [Errno 28]`, and a
  `swift build` "disk I/O error ... build.db" that resolved on retry. The
  first 2048 row is therefore CONFOUNDED and discarded. After deleting this
  iteration's own dirs (step256/1024/2048) 8.4 GiB is free again. Historical
  dirs (`kv-cache-sweep`, cell dirs, `kv-cache-35b-lc80k`) were NOT touched.
- Recommendation for the orchestrator: prune or archive old 9B/cell kv-cache
  dirs; each 35B probe leg persists ~6.5 GB on disk.

## Verdict

- **Chunked-prefill step tuning: RESOLVED.** `MEI_PREFILL_STEP_SIZE` stays at
  the default 512 (optimal under the 32 GB / 30 GB-limit bounds; 256 is the
  memory-safe alternative with −1.7 GB peak at −2% prefill pps; ≥1024 degrades
  or collapses at 80k via per-chunk memory). No Mei configuration or code
  change this iteration — the tree is left as found (the uncommitted
  `scripts/run_measurement_cycle.sh` readiness-poll diff remains the
  coordinating session's).
- Fresh-prefill ~400 pps at 80k is now understood as the memory-budget
  optimum for the 35B hybrid chunked path, not an overhead bug. The listed
  "chunked-prefill step tuning" candidate is closed with evidence; remaining
  credible 35B experiments: compiled-decode gate on 35B (targets the
  55→35 t/s decode falloff at long context), MLXPress `MLXPRESS=70` A/B,
  100k survival (peak extrapolates to ~28.8 GB at 512-step — inside budget,
  untested). No Qwen3.8/Laguna transition — no exhaustion condition met.

## Artifacts

- `artifacts/ornith-35B-step1024-lc-20260902T031529Z.json` (PASS 12/12)
- `artifacts/ornith-35B-step256-lc-20260902T032400Z.json` (PASS 12/12)
- `artifacts/ornith-35B-step2048-clean-20260902.json` (PASS 12/12; renamed
  from the first-attempt filename — see incident above)
- this summary; server logs in `~/.local/share/local-model-bench/mei-runtime/logs/`