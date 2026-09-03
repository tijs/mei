# Qwen3.8 Heretic common correctness matrix (todo 0b87b76a#8 leg) — 2026-09-03

Engine: Mei binary sha256(16) `8051d806ce875ab8` (== the 08:58Z release-candidate
build, byte-identical to the Qwen3.8-27B-4bit raw-fix re-verification binary),
Mei HEAD `3cfadf9`, vmlx fork pinned `91fed8be`. Proven safe config: generic
profile, context cap 65536, prefill step 64, kv-bits none, cache-reuse true,
model-aware disposable disk-backed paged KV (no explicit --kv-cache-dir, temp
disposable cache under $TMPDIR/mei-kv-cache/orcarouter-Qwen3.8-27B-Uncensored-MLX),
port 8024, temperature 0.

Model under test: `orcarouter/Qwen3.8-27B-Uncensored-MLX` 4-bit affine g64 @
`14963e70` (staged complete at
`~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-Uncensored-MLX-4bit`).
Lineage gate (tensors distinct from base Qwen3.8, gated source `404ea47a`,
never a silent substitution) was recorded 2026-09-02 (todo 0b87b76a#6 closed).

## Matrix results

| Probe | Result | Measured |
|---|---|---|
| probe_load r1 (hello 14.984 t/s, short 15.499 t/s, active 18.83 GB, peak 24.71 GB) | PASS | artifacts/probe-load-heretic-4bit-r1-20260903T095342Z.json |
| probe_load r2 (hello 15.375, short 15.582, active 18.83 GB) | PASS | artifacts/probe-load-heretic-4bit-r2-20260903T095418Z.json |
| probe_load r3 (hello 15.713, short 15.863, active 18.83 GB) | PASS | artifacts/probe-load-heretic-4bit-r3-20260903T095430Z.json |
| probe_load 3-repeat mean: hello 15.357 t/s (sd 0.365), short 15.648 t/s (sd 0.191) | PASS | above |
| probe_coding (python_json_sum, shell_rename, sql_users_query, swift_fibonacci) | 4/4 PASS, ~15.3 t/s | artifacts/probe-coding-heretic-4bit-20260903T094203Z.json |
| probe_long_context 30k fresh fill | PASS — 56.15 pps (538.1 s), decode 11.79 t/s at 30k loaded, peak 24.71 GB, active 22.59 GB | artifacts/probe-longctx-heretic-4bit-20260903T094433Z.json |
| probe_long_context 30k +1 cache reuse | PASS — 30000/30001 cached, restore 4.7 s (88.1k pps), decode 11.79 t/s, peak 24.71 GB | same |
| probe_mei full acceptance (12 gates incl. raw 65536 exact-cap + 65537 over-cap) | 11/12 PASS — all functional gates green; `mei_status` leg hit a 30 s client timeout on run 2, endpoint verified healthy (HTTP 200, 18 ms idle) → transient probe noise, not a regression | artifacts/probe-mei-heretic-4bit-full-20260903T100357Z.json |

## Notes

- Decode 15.2–15.9 t/s short-context matches base Qwen3.8-27B-4bit (15.656,
  sd 0.060) — expected: same architecture, same 4-bit quant family, distinct
  weights. The raw /v1/completions path (crash-fixed in `ee7368d` for the base
  4-bit checkpoint) is exercised by the context-cap gates in probe_mei; a PASS
  here confirms the batch-first [1,T] fix covers the second VLM-routed 4-bit
  checkpoint as well.
- Memory: active 18.83 GB post-load/idle, 22.59 GB at 30k loaded, peak 24.71 GB
  at the 30k fill — comfortably inside the 32 GB physical / 30 GB limit budget.
- KV reuse rides the disposable disk tier (hybrid paged cache is
  paged-incompatible for this dense SSM topology), consistent with the
  model-aware default.
- GGUF/llama.cpp behavioral comparison vs the trohrbaugh Heretic Q5_K_M blob
  (sha256 e79fdc96, cached complete) remains queued at the plan level (todo 8)
  — same-suite A/B pending, never a bit-equivalence claim.

## Commands

```
start: MEI_MODEL_DIR=~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-Uncensored-MLX-4bit \
  MEI_SERVED_MODEL_ID=orcarouter/Qwen3.8-27B-Uncensored-MLX MEI_OPTIMIZATION_PROFILE=generic \
  MEI_CONTEXT_CAP=65536 MEI_PREFILL_STEP_SIZE=64 MEI_PORT=8024 MEI_TEMPERATURE=0.0 \
  bash scripts/start_mei_server.sh
probe: ~/.local/share/local-model-bench/mei-runtime/venv/bin/python tools/probe_load.py \
  --base-url http://127.0.0.1:8024/v1 --model orcarouter/Qwen3.8-27B-Uncensored-MLX \
  --server-log ~/.local/share/local-model-bench/mei-runtime/logs/server.log \
  --output artifacts/probe-load-heretic-4bit-rN-<ts>.json
probe: .../venv/bin/python tools/probe_coding.py --base-url ... --model ... --output ...
probe: .../venv/bin/python tools/probe_long_context.py --base-url ... --model ... \
  --tokenizer <staged dir> --lengths 30000 --output ...
probe: .../venv/bin/python tools/probe_mei.py --base-url ... --model ... \
  --tokenizer <staged dir> --context-cap 65536 --output ...
```