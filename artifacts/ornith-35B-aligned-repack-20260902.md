# Ornith-35B aligned-mmap repack experiment — memory & decode A/B (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`).
Machine: Sulaco M1 Max, 32 GB unified (34,359,738,368 B). Server: Mei release
binary (vmlx-swift pinned `aeb5e21c`), port 8024, context-cap 65536, prefill
step 512, kv-bits none, compiled-decode false, cache-reuse true,
ssm-rederive true.

## Hypothesis (from the load log + code reading)

The C++ mmap safetensors loader (`Source/Cmlx/mlx/mlx/io/safetensors.cpp`,
`MLX_SAFETENSORS_MMAP=1`) zero-copy maps only *naturally aligned* tensors
(`absolute_offset % dtype_size == 0`) and memcpy's every other tensor into
MLX/Metal-backed RAM ("realigned ... without shard fallback"). The log for the
first clean 35B run printed a truncated `realigned 282 un...`; the 9B logs
printed `realigned 927 unaligned tensor(s) (5038040064 bytes)` = the ENTIRE
9B payload. Suspicion: the 35B also realigned nearly its whole payload,
materializing ~19 GB of anonymous copies whose pages cannot be reclaimed,
which is a large part of the 36.7 GB generation working set.

## Measured checkpoint alignment (exact audit, 2026-09-01)

`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ 19504d912fa8fc7622bf6b1de3db5d5d890b1f02,
4 shards, 1,757 tensors, payload 19,508,787,456 B (19.509 GB):

| shard | data_start % 8 | unaligned tensors | unaligned bytes |
|---|---|---|---|
| 00001 (5.344 GB) | 2 | 137 (all U32) | 4.750 GB |
| 00002 (5.368 GB) | 5 | 352 BF16 + 146 U32 | 5.368 GB |
| 00003 (5.368 GB) | 5 | 358 BF16 + 146 U32 | 5.368 GB |
| 00004 (3.428 GB) | 5 | 199 BF16 + 83 U32 | 3.428 GB |
| **total** | | **1,421 / 1,757 (80.9%)** | **18,914,762,368 B (18.915 GB, 97.0% of payload)** |

Log line reproduced on the original dir: `realigned 282+137+504+498 =
1421 unaligned tensor(s)` — matches the audit exactly.

Root cause: the checkpoint writer packed U32 (4-bit weights) and BF16
(scales/biases) back-to-back after a header whose `8 + header_len` is not a
multiple of 8 (data_start % 8 ∈ {2, 5}), so ~all absolute tensor offsets are
misaligned for their dtype.

## Fix tested (Mei-owned, rollback-safe, zero payload rewrite)

`tools/align_safetensors.py`: for each shard, insert `P` JSON-whitespace bytes
before the final `}` of the header so `data_start % 8 == 0`
(P = 6, 3, 3, 3). Safetensors `data_offsets` are relative to the data segment,
so relative offsets are unchanged and every tensor becomes naturally aligned
(audit: 0 unaligned in all 4 shards). Payload bytes verified bit-identical
(per-shard sha256 of the payload region). Output:
`~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned/`
with `MEI_ALIGN_MANIFEST.json` (31 s for 21.5 GB).

## A/B results — ALIGNED vs ORIGINAL (same binary, same flags, limit 24.05 GB then 30 GB)

Same request: `Reply with exactly: ready`, temp 0, max_tokens 64
(usage: prompt ≈15 tok, completion 35 tok).

| metric | ORIGINAL (anonymous copies) | ALIGNED (file-backed NoCopy) |
|---|---|---|
| realigned at load | 1,421 tensors / 18.915 GB | 0 / 0 GB |
| post-load active | 24,278,561,174 B (24.28 GB) | 19,551,131,108 B (19.55 GB) |
| prefill_ms (15 tok) | 117,590 (117.6 s) | 36,712 (36.7 s) |
| generate_ms (35 tok) | 8,710 (4.02 tok/s) | 357,887 (0.098 tok/s) |
| generation active/peak | 36.49 / 36.69 GB | 31.76 / 31.96 GB |
| active after request (t+30s) | 36.49 GB (never releases) | 31.76 GB (never releases) |

Memory-limit control: raising `--memory-limit-bytes` 24.05 → 30 GB changed
nothing (identical tps and peaks on both paths) → the limit is not the
throttle.

Context-cap control: ctx 65536 → 4096 on ORIGINAL changed nothing
(post-load 24.28 GB, generation delta still +12.22 GB, same tps) → the fixed
+12.2 GB is NOT context-sized.

vmmap (ORIGINAL, after one generation): physical footprint 29.3 GB with
10.9 GB already swapped; IOAccelerator (MTLBuffer) 29.2 GB across 1,523
regions — 120 × 128 MB + 40 × 256 MB + 2 × 242.5 MB ≈ 25.6 GB of
weight-sized regions, plus ~3.6 GB of 2–16 MB regions.

## Interpretation

1. **Realignment hypothesis CONFIRMED quantitatively.** 97% of the 35B
   payload is copied into anonymous RAM today; the aligned repack removes it:
   post-load active 24.28 → 19.55 GB and prefill 117.6 → 36.7 s (3.2×).
2. **File-backed NoCopy decode is pathological in this fork (~40× slower).**
   With the aligned file-backed weights the GPU/Metal path decodes at
   0.098 tok/s vs 4.02 tok/s for the anonymous path (engine-reported). The
   fork's intended mitigation for file-backed weights — the MLXPress router
   that advises hot/cold regions (`mlx_safetensors_mmap_advise_routed/…`) —
   logs `[MLXPressRouter] enabled=false … symbol=missing` on this path, i.e.
   not wired for Mei's qwen3_5_moe load. Net: aligned is a net loss today.
3. **A fixed ~12.22 GB materialization appears on the first generation on
   BOTH backing stores and never releases** (24.28→36.5 / 19.55→31.8; same at
   ctx 4096; same at limit 30 GB). vmmap suggests it is MTLBuffer weight
   regions (128/256 MB granularity), i.e. the per-expert / per-layer weight
   materialization itself (quantized wrap and/or f16→bf16 conversions and/or
   MoE staging), not KV or Mamba state (KV at ctx 65536 ≈ 1.34 GB; mamba
   conv/state tensors are small, conv1d [8192,4,1] per linear-attn layer).
   This is what keeps the anonymous path at 36.5 GB > 32 GB physical, and it
   must be identified precisely next (allocator-level trace) before any
   "fits in 32 GB" claim.

## Verdict

- Load-memory lever (realign copies): identified, quantified, and removed —
  the repack is a safe Mei-owned repro tool and is kept.
- **35B still does NOT fit 32 GB in a usable configuration**: fast path is
  36.5 GB working set (over physical), low-memory path decodes at 0.098 t/s.
- Not exhausted: the fixed +12.2 GB materialization is the next target
  (trace + eliminate), and the MLXPress hot/cold router is a designed
  lever for the file-backed path. Ornith remains the primary; no
  Qwen3.8/Laguna transition (documented per escalation order).

## Artifacts & commands

- Audit: `python3 /tmp/align_safetensors.py` + inline shard audit (this doc).
- Repack: `python3 tools/align_safetensors.py <src> <dst-aligned>` (31 s;
  manifest `MEI_ALIGN_MANIFEST.json`).
- Server (both legs): `MEI_MODEL_DIR=<dir> MEI_CONTEXT_CAP=65536
  MEI_LOAD_MMAP=true [MEI_MEMORY_LIMIT_BYTES=30000000000]
  MLXPRESS_GENERATION_PROFILE=1 bash scripts/start_mei_server.sh`
- Probes: `/tmp/mem35b_probe2.py` (stage-windowed memory sampler reusing
  `tools/probe_mei.py` helpers) → `artifacts/ornith-35B-aligned-repack-stage2-20260901T224547Z.json`
  (partial: killed during r2; r1 complete), plus inline probes in this
  session's terminal commands.
- No Mei code was changed. Commit: tool + this artifact only.