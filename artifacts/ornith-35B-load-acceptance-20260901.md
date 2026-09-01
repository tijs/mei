# Ornith-35B MLX load / acceptance / memory evidence — first clean window (2026-09-01)

Session gate: clean Metal window (no foreign inference owners; ports 8017/8024 free;
reclaimable free+inactive ≈ 14.2 GB pre-load, 22.8 GB post-stop; RAM 32.0 GB).

## Outcome: LOADABLE + ACCEPTANCE PASSED; PERF-MEMORY-BOUND (not clean-decode measurable)

The primary `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` (qwen3_5_moe, official plain 4-bit
affine/group-64, ~19.5 GB weights, 4 shards) **loaded and served** under Mei's pinned
vmlx path and **passed the full load probe** for the first time this plan:
status, hello completion, and short decode all `passed`.

Artifact: `artifacts/load-35B-20260901T163924Z.json` (probe exit 0).

## Model identity (exact)
- served id: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
- topology: layers=40, kvLayers=10, mambaLayers=30, companion=ssm, restore=disk-backed
- prefix cache: paged in-memory, no disk (`MEI_KV_CACHE_DIR` unset)
- device: `applegpu_g13s`, physical memory 34,359,738,368 B (32.0 GB)
- memory after load: active 24,278,561,174 B (24.28 GB), peak 0
- config flags (all default-off preserved): compiled-decode false, kv-bits none,
  max-kv-window 0, ssm-anchor-boundaries 0; prefill-step 512; cache-reuse true;
  emit-reasoning true; memory-limit-bytes 0 (engine default 24,051,816,857 B);
  working set recommended 26,800,603,136 B (26.8 GB).

## Measured rows (engine-reported; FIRST clean 35B decode evidence)
- hello: prompt 15 tok; completion 33 tok; **tokens_per_second 3.455**;
  prefill_ms 78083.6 (78.1 s → 0.192 pps); generate_ms 9550.9; cached_tokens 0.
- short_decode: prompt 23 tok; completion 32 tok; **tokens_per_second 1.225**;
  prefill_ms 99513.4 (99.5 s → 0.231 pps); generate_ms 26119.6; cached_tokens 0.

## Memory blocker / measured constraint
During generation the **MLX active working set rose to 36,687,287,872 B (36.7 GB),
peak 36,750,876,284 B — ABOVE the 32.0 GB physical** (and multi-GB above the 26.8 GB
recommended working set). The MLX memory limit (24.05 GB) also sits BELOW the post-load
active footprint (24.28 GB), the documented hang/alloc-wait failure mode. Prefill of even
15–23 tokens takes 78–100 s (0.19–0.23 prefill tok/s) and decode is 1.2–3.5 tok/s:
**unified-memory pressure, not a clean-decode rate.** Goal is ≥30 decode tok/s.

## Verdict (honest, no invented perf claim)
- **Loadability: YES** (first confirmed clean 35B load + serve in this plan).
- **Correctness/acceptance: PASSED** (model id, plain completion run).
- **Performance optimization: BLOCKED by measured unified-memory constraint** — not a
  code defect (arch dispatch loaded fine); a working-set/resource limit. No ≥3 clean
  repeat perf rows are claimable for 35B on this 32 GB machine with its Q4 working set.
- Per plan gate, optimization proceeds on the **Ornith-9B proxy** on the identical Mei
  text path; 35B status updated in `configs/model-lineup.json` from
  `blocked-on-memory` → `loadable-accept-passed-perf-memory-bound`.

## Not tested (deferred, memory-bound)
Tool calls, streaming/non-streaming parity, KV-reuse, long-context (30K/80K) for 35B —
each would require multi-GB extra transient buffers on an already-over-commit window.