# Qwen3.8-27B llama.cpp GGUF ceiling counter-check — 2026-09-02

Unit: todo 0b87b76a#4 exhaustion evidence + todo 0b87b76a#7 GGUF reference row.
Queued by worker note bb1ba325 ("llama.cpp Qwen3.8-27B-UD-Q5_K_M ceiling
counter-check: inspect local-model-bench gguf.yaml conventions first, use an
uncontended clean window, append-only results").

## Question

Is the Qwen3.8-27B (~27B dense hybrid, 64 layers / 16 full-attention)
30 tok/s decode target on the 32 GB M1 Max reachable by the *reference*
llama.cpp/GGUF engine? Mei's MLX 5-bit affine-g64 leg measures 13.105 t/s
(3x, sd 0.010). If llama.cpp lands in the same single-digit/low-teens band,
both engines confirm a hardware ceiling far below 30 t/s and the todo-4
closure path becomes "explicitly documented exhausted hardware ceiling"
(plan gate), pending the one remaining credible lever (4-bit) being
measured or explicitly declined.

## Artifacts

- `artifacts/gguf-ceiling-qwen38-provenance-20260902.json` — provenance gate (sha256 + meta)
- `artifacts/gguf-ceiling-qwen38-ud-q5-20260902.json` — measurement rows
- `~/.local/share/local-model-bench/mei-runtime-q38-ceiling/logs/llama-ceiling.log`
- Tool: `tools/llama_ceiling.py` (generalized; defaults unchanged = Ornith-9B behavior)

## Provenance (verified)

- GGUF: `unsloth/Qwen3.8-27B-GGUF` revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`,
  file `Qwen3.8-27B-UD-Q5_K_M.gguf`, 19,771,509,664 bytes, blob
  `2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd`.
- Computed sha256 == pinned HF blob digest: **match**.
- GGUF meta: arch `qwen35`, block_count 65 (64 + 1 MTP head,
  nextn_predict_layers=1), context_length 262144, full_attention_interval 4,
  file_type 17, quant_version 2. Run WITHOUT `--spec-type` → single-token
  decode, comparable to Mei's no-MTP baseline.
- Engine: `/opt/homebrew/bin/llama-server`, build 10470 (commit 34af94cd9) —
  the exact documented local-model-bench GGUF engine.
- Config parity: `--ctx-size 65536`, temp/top-p/top-k per
  `configs/Qwen3.8-27B/gguf-unsloth-ud-q5-64k.yaml`; port 8075 (Mei-owned),
  `--parallel 1`, `--no-webui`, `--metrics`.
- Window: no foreign inference process resident at start; per-row
  `contended_during_row=false` (own server excluded by pid).
- Method: Mei `probe_load`-style short decode — prompt "Count from 1 to 10,
  one per line.", max_tokens 32, temp 0, 3 repeats, raw `/v1/completions`
  (timings.predicted_per_second, prefill excluded) + 1 chat row.

## Measured results (3 cold repeats)

| row | decode t/s | prompt t/s | prompt tok | comp tok | RSS after |
|---|---|---|---|---|---|
| short_fresh_r1 | 9.285 | — | 13 | 22 | 24.36 GB |
| short_fresh_r2 | 9.286 | — | 13 | 22 | 24.83 GB |
| short_fresh_r3 | 9.287 | — | 13 | 22 | 24.83 GB |
| short_chat     | 9.294 | 47.4 | 65 | 32 | 24.83 GB |

- fresh decode mean **9.286 t/s**, sd 0.0010 (extremely stable).
- Peak RSS **24.83 GB** — inside 32 GB budget, matches the on-record
  24.8 GB for this config (LEADERBOARD rows).
- Server log confirms Metal path, no CPU fallback, no warnings.

## Comparison vs Mei MLX (same model, same short decode protocol)

| leg | artifact | decode t/s (3x mean) | note |
|---|---|---|---|
| Mei MLX | Qwen3.8-27B-5bit-affine-g64 (Mei-produced, sha-pinned source) | **13.105** (sd 0.010) | committed baseline (parity run) |
| llama.cpp | unsloth UD-Q5_K_M (19.77 GB) | **9.286** (sd 0.001) | this run |
| on record (agentic) | unsloth UD-Q5_K_M | 6.7–7.7 tok/s decode | LOCAL-MODEL-BENCH LEADERBOARD (reasoning-mode overhead) |

MLX is **+41% faster** than llama.cpp on decode for this model. Historic
agentic GGUF decode rates (6.7–7.7 t/s) sit below both raw decode
measurements, consistent with thinking-mode token overhead.

## Interpretation — hardware ceiling for todo #4

- Dense hybrid ~27B: every decode token must stream ~19.77 GB (GGUF) /
  ~16.05–18.5 GB (MLX 5-bit affine) of weights through unified memory
  (melded with the per-token serial Gated DeltaNet state recurrence in 48/64
  layers).
- M1 Max ~350–400 GB/s → absolute floor ≈ 20.2 t/s (GGUF) / 21.6 t/s (MLX
  5-bit). Observed: 9.29 (46% of floor) / 13.1 (61% of floor) — the
  delta-net recurrence + quant dequant cost sits on top of pure bandwidth.
- **Both engines land 9–13 t/s; the 30 t/s target is 2.3–3.2× above the
  fastest measured engine on this hardware.** Single-token decode at 5-bit
  cannot reach 30 t/s for this model on this machine.
- Remaining credible levers (unchanged from note bb1ba325):
  1. **4-bit affine g64** (≈13.5 GB): floor ≈29.6 t/s, believable gain
     +10–15% → ~14.5–15.5 t/s measured. Still < 30. Needs produce + gated
     re-check (quality/tool/streaming/KV/long-context) if adopted.
  2. MTP/speculative: GGUF leg has the MTP head in-model (llama.cpp
     `--spec-type draft-mtp` could plausibly reach ~1.5–2× → 14–18 t/s;
     needs a spec-type validation run, out of scope here, still < 30). The
     Mei MLX artifact has NO MTP head (base source config has no `nextn`
     keys) and policy forbids GGUF-derived MLX conversion → not available
     on the MLX leg without a new source.
  3. Prefill-stage tuning (fresh prefill is the weakest measured stage) —
     affects TTFT, not decode t/s.

## Todo status

- `0b87b76a#4` (Qwen3.8 parity + optimization): **STAYS OPEN**. The GGUF
  counter-check confirms the hardware ceiling on the reference engine, but
  the plan gate requires the 4-bit lever to be measured (or explicitly
  declined with evidence) before "exhausted hardware ceiling" may be
  accepted. Next queued unit: produce Qwen3.8-27B 4-bit affine g64 from the
  pinned source and run a 3-repeat decode A/B vs the 5-bit baseline on the
  proven safe config; if it wins, re-gate acceptance/parity/tool/coding/KV/
  long-context before considering a default switch. If even the ideal
  outcome (~15 t/s) cannot justify the quality risk, record that as the
  documented decision and take the exhaustion-acceptance path.
- `0b87b76a#7` (per-model matrix vs GGUF refs): this run provides the first
  clean Qwen3.8 GGUF reference row (decode + memory); the full per-model
  matrix remains incomplete → stays open.

## Reproducibility

```
Q38_GGUF=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/\
4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q5_K_M.gguf
MEI_RUNTIME_BASE=$HOME/.local/share/local-model-bench/mei-runtime-q38-ceiling \
python3 tools/llama_ceiling.py --gguf "$Q38_GGUF" \
  --alias qwen38-27B-UD-Q5_K_M --mode short --port 8075 --ctx-size 65536 \
  --max-tokens 32 --repeats 3 \
  --gguf-sha256 2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd \
  --gguf-repo unsloth/Qwen3.8-27B-GGUF \
  --gguf-revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --output artifacts/gguf-ceiling-qwen38-ud-q5-20260902.json
```

Local-model-bench was READ-ONLY this run (conventions inspected, config
parity followed, nothing modified; results kept Mei-contained, consistent
with the prior Ornith ceiling).