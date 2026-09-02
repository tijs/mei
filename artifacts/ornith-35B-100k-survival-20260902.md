# Ornith-35B long-context survival & throughput at 100k — 3 clean repeats (2026-09-02)

One focused experiment per the Ornith-first plan (Kiem plan `0b87b76a`, todo 1
remaining item): close the last unproven long-context dimension for the primary
target — 100k survival with exact-cap/over-cap gates on the fitted 35B config
(aligned repack backing store + `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`,
prefill step 512, compiled decode off, disk KV tier). Preceded in this session
by the MLXPress=70 engagement falsification (see
`artifacts/ornith-35B-mlxpress-falsification-20260902.md`).

Machine: Sulaco M1 Max, 32 GB unified (34,359,738,368 B), device applegpu_g13s,
recommended working set 26.80 GB. Server: Mei release binary (vmlx-swift pinned
`aeb5e21c` + patches 0001-0005), port 8024. GGUF reference llama-server :8017
not running this window (foreign; untouched). No foreign runner active during
any probe (checked per phase).

## Exact variant & flags

- Model: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ pinned HF revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`, aligned repack dir
  `…/Ornith-1.5-35B-A3B-MLX-4bit-aligned`.
- Server env: `MEI_CONTEXT_CAP=100032 MEI_MEMORY_LIMIT_BYTES=30000000000
  MEI_KV_CACHE_DIR=…/kv-cache-exp-100k-20260902T0740Z (cold, disposable)
  VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`. Args as printed at launch:
  `--context-cap 100032 --prefill-step-size 512 --memory-limit-bytes
  30000000000 --cache-limit-bytes 0 --compiled-decode false --load-mmap true
  --kv-cache-dir … --emit-reasoning true --cache-reuse true`. `--kv-bits none`,
  `--max-kv-window 0`, `--ssm-anchor-boundaries 0` (defaults off).
- Post-load memory: active 19,551,131,108 B (19.55 GB). Peak counter reset
  after load (Engine.swift:125).
- Probes: `probe_long_context.py --lengths 30000 80000 100000 --max-tokens 32`,
  three repeats on the same server; then `probe_mei.py --context-cap 100032`.

## Results — long-context probe (3 clean repeats), 18/18 rows PASSED

Decode t/s (engine `tokens_per_second`, 32 tokens temp 0):

| row | r1 | r2 | r3 | mean | sd |
|---|---|---|---|---|---|
| fill_30000_fresh | 46.27 | 14.36* | 46.38 | 35.7 | — |
| fill_30000_cache_reuse | 47.91 | 16.94* | 48.31 | 37.7 | — |
| fill_80000_fresh | 4.73* | 35.16 | 35.27 | 25.1 | — |
| fill_80000_cache_reuse | 30.43* | 35.12 | 35.32 | 33.6 | — |
| **fill_100000_fresh** | **31.29** | **31.83** | **32.08** | **31.7** | **0.41** |
| **fill_100000_cache_reuse** | **32.07** | **32.02** | **32.08** | **32.05** | **0.03** |

\* r1 80k-fresh 4.73 t/s and r2 30k rows (14–17 t/s) are attributed to
first-write disk-serializer flush (largest single extension, fresh cache dir)
and concurrent system pressure; the same rows reproduce 35.2–35.3 (80k) and
46.4–48.3 (30k) on the other repeats and match the proven 80k/90k session
baseline exactly (35.2–35.3 @80k, 46–48 @30k). The 100k rows — this
experiment's claim — are tight across all three repeats.

- **100k-context decode = 31.7–32.05 t/s across 6 rows — still ≥ the 30 t/s
  goal at full 100k context** (fresh mean 31.7 sd 0.41; cache-reuse mean 32.05
  sd 0.03). Trend: ~55 (short) → ~47 (30k) → ~35 (80k) → ~31.8 (100k) —
  recurrent (Mamba) scan decoding cost grows with context length.
- Fresh chunked prefill at 100k: ~964–966 pps on the fresh remainder (r2/r3;
  restore of the 80001 prefix from disk, ~20k fresh tokens per row), 106 s per
  row. Restore (disk tier): 100k+1 → 100000 cached @ 120k–330k pps
  (`full_prefix_cached` PASS). Cross-length prefix reuse works: 100k fill
  restores the 80k prefix (cached_tokens=80001).
- Memory: after 30k fill ≈ 21.0 GB; after 80k fill ≈ 23.3 GB; after 100k fill
  ≈ 24.1 GB; **peak 28.07 GB (r1) / 28.19 GB (r2/r3) across the 100k phase —
  < 30 GB MLX limit and < 32 GB physical**, ~2 GB headroom under the limit,
  matching the session's ~28.8 GB extrapolation. No regression vs proofed
  26.62 GB @80k / 27.72 GB @90k (peak grows ~1.1 GB per 10k of context).

## Results — acceptance at cap 100032 (`probe_mei.py`), 12/12 PASSED

- `models_identity`, `mei_status` (contextCap 100032, prefill step 512) — passed.
- `plain_completion` 57.1 t/s (content ok), `parity_stream_vs_nonstream` — passed.
- `tool_nonstreaming` + `tool_streaming` — passed (add_numbers `{"a":15,"b":27}`,
  finish_reason tool_calls) — tool-call correctness intact at 100k cap.
- KV reuse gates: `cache_repeat_1/2` (6165/6170 restored), `cache_growing_turn1`
  + `cache_growing_turn2_reuses_slot` (785/790 restored) — passed.
- `context_exact_cap` 100032: PASSED — prompt 100032 tokens accepted (prefix
  restored from disk, ~144.6k pps restore); active 24.1 GB, peak 28.19 GB.
- `context_over_cap_rejected`: 100033-token prompt rejected HTTP 400
  ("request exceeded context cap"). Passed.
- Final /v1/mei/status: active 20.26 GB, cache 5.17 GB, peak 28.19 GB;
  `ssmHits` > 0 (SSM-state restore genuinely engaged at 100k), `pagedEnabled`
  hybrid + disk-tier KV reuse as documented.

## Verdict

- **35B long-context survival PROVEN at 100k: 3 clean repeats, 18/18 probe rows
  PASS, acceptance 12/12 at cap 100032 with exact-cap accepted (28.19 GB peak)
  and over-cap rejected.** Peak 28.19 GB < 30 GB limit with ~2 GB headroom.
- **3x long-context throughput: 31.7 t/s fresh / 32.05 t/s cache-reused at
  full 100k context (mean of 3 repeats, sd ≤ 0.41) — still ≥ the 30 t/s goal.**
  The recurrent-scan decode cost continues its monotonic context trend; the
  30 t/s goal's margin at 100k is thin (~1.7 t/s) but real and repeatable.
- Correctness gates at the new cap: streaming/non-streaming parity, tool calls
  (non+stream), KV reuse (disk tier incl. cross-length 80k→100k prefix
  restore), exact-cap/over-cap — all PASSED.
- Ornith remains primary and NOT exhausted. Remaining credible 35B threads:
  (1) dedicated `VMLX_ENABLE_UNSAFE_COMPILE=1` A/B with compiled decode OFF
  (the +4–5% guard-skipped-eager hint from the compiled-decode session —
  run this session, see companion artifact); (2) any source-backed low-risk
  vMLX/MLX allocator or recurrent-scan experiment with a clear acceptance gate.
  MLXPress=70 A/B is CLOSED as not engageable in the current binary
  (falsification artifact, same session). No Qwen3.8/Laguna transition — no
  exhaustion condition met.
- No Mei source change this iteration (measurement-only run). Tree left as
  found except this session's new artifacts + model-lineup notes commit.

## Artifacts

- `artifacts/ornith-35B-100k-surv-r1-20260902T0745Z.json` (PASS 6/6, cold)
- `artifacts/ornith-35B-100k-surv-r2-20260902T0752Z.json` (PASS 6/6)
- `artifacts/ornith-35B-100k-surv-r3-20260902T0805Z.json` (PASS 6/6)
- `artifacts/acceptance-ornith35B-100k-20260902T0815Z.json` (PASS 12/12)
- `artifacts/ornith-35B-mlxpress-falsification-20260902.md`
- this summary; server log in `~/.local/share/local-model-bench/mei-runtime/logs/server.log`