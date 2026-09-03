# Qwen3.8-27B-4bit common correctness matrix (todo 0b87b76a#7 first leg) — 2026-09-03

Engine: Mei, binary sha256(16) `a0bd8367aa5cb3cb` (== the 5-bit parity binary of
2026-09-02T1605Z), vmlx fork pinned `91fed8be`. Proven safe config: generic
profile, context cap 65536, prefill step 64, kv-bits none, cache-reuse true,
disposable disk-backed paged KV (model-aware default, no explicit flag),
port 8024, temperature 0 for probes.

Model under test: `mlx-community/Qwen3.8-27B-4bit` @ `3e6447f0` (regular 4-bit
affine g64 MLX quant, staged complete). Text tensors are name-identical to the
Mei-produced 5-bit: the ONLY structural delta is the bundled `vision_tower.*`
sidecar (333 tensors, hidden 1152, ~0.3 GB) + `vision_config` in config.json.

## Matrix results

| Probe | Result | Evidence artifact |
|---|---|---|
| probe_load r1 (hello 15.21 t/s, short 15.48 t/s, peak 18.87 GB) | PASS | probe-load-qwen38-4bit-r1-20260902T222235Z.json |
| probe_load r2 (hello 15.60, short 15.61, active 18.83 GB) | PASS | probe-load-qwen38-4bit-r2-20260902T225759Z.json |
| probe_load r3 (hello 15.23, short 15.71, active 18.83 GB) | PASS | probe-load-qwen38-4bit-r3-20260902T225804Z.json |
| probe_mei acceptance: models_identity, mei_status, plain, tool_nonstream, tool_stream, parity_stream_vs_nonstream, cache_growing ×2, cache_repeat ×2 (6212 tok fresh + 6207 cached) | 10/12 PASS | probe-mei-qwen38-4bit-20260902T222245Z.json |
| probe_mei context_exact_cap (65536 raw) | BLOCKED — server crash | same artifact + mei-2026-09-03-002528.ips |
| probe_mei context_over_cap_rejected | BLOCKED (server dead) | same artifact |
| probe_coding (python_json_sum, shell_rename, sql_users_query, swift_fibonacci) | 4/4 PASS, ~15.6 t/s | probe-coding-qwen38-4bit-20260902T225610Z.json |
| probe_long_context 30k raw (fresh server, no restored cache) | BLOCKED — server crash, 2/2 | probe-longctx-qwen38-4bit-{20260902T222605Z, fresh-20260902T225837Z}.json |

Decode throughput matches the 2026-09-02 A/B (mean 15.656 t/s): load rows
15.2-15.7 t/s, coding 15.6 t/s. 4-bit stays the fastest safe Qwen3.8 config;
llama.cpp UD-Q5_K_M reference was 9.286 t/s on the same machine.

## Chat-path context survival (threshold fills, /v1/chat/completions)

| Length (exact tokens) | Result | Prefill pps | Peak GB | Artifact |
|---|---|---|---|---|
| 16000 | PASS | 58.1 | 23.60 | probe-ctx-threshold-qwen38-4bit-16000-*.json |
| 24000 | PASS | 57.0 | 25.52 | probe-ctx-threshold-qwen38-4bit-24000-*.json |
| 27000 | PASS | 56.5 | 26.06 | probe-ctx-threshold-qwen38-4bit-27000-*.json |
| 30000 | PASS | 56.1 | 26.85 | probe-ctx-threshold-qwen38-4bit-30000-*.json |

Tool: `tools/probe_context_threshold.py` (--endpoint chat|completions, exact-token
prompt reuse).

## BLOCKER: raw /v1/completions path deterministically crashes the 4-bit model

`Fatal error: SmallVector out of range` at vmlx `mlx/c/array.cpp:335`, raised
via `MLXArray.dim` ← `Qwen35.prepare(_:cache:windowSize:)` during prefill.
EXC_BREAKPOINT SIGTRAP; crash reports mei-2026-09-03-{002528,002608,005843}.ips
(+1 at 00:59 for the 60-token ladder). Reproducible 4/4:

1. probe_mei context_exact_cap 65536 raw — died 0.74 s after submit.
2. probe_long_context 30k raw, server restored from crashed session — died 0.37 s.
3. probe_long_context 30k raw, TRULY fresh server (disposable KV dir removed
   at launch) — died 0.37 s.
4. raw ladder length=60 raw — died 0.006 s.

So the raw endpoint is broken at ANY length for this checkpoint (60 tokens is
one 64-token chunk — this is not a long-context/memory issue). The chat path is
NOT affected (30k chat fill passes; all acceptance/coding/tool probes pass).
The 5-bit checkpoint passes raw 30k and raw 65k on the same binary, so the
defect is checkpoint-specific, and the only structural difference is the
bundled vision tower + vision_config. Working hypothesis for the fix: the
raw-completions branch instantiates/segments the multimodal Qwen3_5 sidecar and
indexes a vision-adjacent array; the Mei-produced text-only 5-bit (no vision
tensors, no vision_config) never enters that branch. Text-only repack of the
4-bit (strip `vision_tower.*` + `vision_config`) is the natural next
experiment, plus a source-level look at the raw-completions prepare path.

The 65536 chat-path fill was NOT retested (the exact-cap gate uses raw, which
crashes first); extrapolating the fill peak trend (+~3.0 GB per 10k from
16k→30k) to 65k gives ~37 GB > 32 GB physical, mirroring the 5-bit's measured
34.64 GB peak at cap — high-pressure even if the crash is fixed.

## Status

- Qwen3.8-27B-4bit: chat-path correctness (acceptance 10/12, tools, streaming
  parity, KV reuse, coding 4/4) + decode speed confirmed; raw completions
  endpoint and therefore the context-cap/long-context gates are BLOCKED.
- Todo 0b87b76a#7 stays OPEN (blocker recorded; Gemma/Heretic/Ornith legs
  still pending).
- Defaults unchanged: 5-bit remains the documented full-parity artifact; the
  4-bit stays a best-speed candidate with a documented raw-path blocker.

Commands (all from /Users/tijs/projects/mei, venv python for transformers):
```
start: MEI_MODEL_DIR=~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit \
  MEI_SERVED_MODEL_ID=mlx-community/Qwen3.8-27B-4bit MEI_OPTIMIZATION_PROFILE=generic \
  MEI_CONTEXT_CAP=65536 MEI_PREFILL_STEP_SIZE=64 MEI_PORT=8024 scripts/start_mei_server.sh
probe: python3 tools/probe_load.py --base-url http://127.0.0.1:8024/v1 --model mlx-community/Qwen3.8-27B-4bit --server-log ~/.local/share/local-model-bench/mei-runtime/logs/server.log --output artifacts/probe-load-qwen38-4bit-rN-<ts>.json
probe: .venv/bin/python tools/probe_mei.py --base-url ... --model mlx-community/Qwen3.8-27B-4bit --tokenizer <4bit staged dir> --context-cap 65536 --output ... 
probe: python3 tools/probe_coding.py --base-url ... --model mlx-community/Qwen3.8-27B-4bit --output ...
probe: .venv/bin/python tools/probe_long_context.py --base-url ... --model ... --tokenizer <4bit staged dir> --lengths 30000 --output ...
threshold: .venv/bin/python tools/probe_context_threshold.py --base-url ... --model ... --tokenizer <4bit staged dir> [--endpoint completions] --lengths ... --output ...
```

## UPDATE (2026-09-03, raw-path re-verification pass): ALL previously-blocked raw legs now PASS

Engine: Mei binary sha256(16) `8051d806ce875ab8` (== the 08:58Z build; rebuild this tick was byte-identical, no source delta), Mei HEAD `3a282ef`, vmlx fork pinned `91fed8be`, model-aware disposable disk-KV default, port 8024, temp 0. The raw /v1/completions crash (SmallVector OOR array.cpp:335, 4/4 repro) is CONFIRMED FIXED by the Mei-side [1,T] token-shape change (EE7368d, Sources/MeiCore/Engine.swift) — no dependence on the vmlx fork edits.

| Probe | Result | Evidence artifact |
|---|---|---|
| raw /v1/completions smoke (9-token prompt, 60 tok out, temp 0) | PASS — HTTP 200, 15.54 t/s, 4.79 s, peak 18.82 GB | artifacts/raw-smoke-qwen38-4bit-20260903T080352Z.json + raw-smoke-req.json |
| probe_mei full acceptance (12/12): models_identity, mei_status, plain, tool_nonstream, tool_stream (a=15,b=27 int args), parity, cache_repeat 6207/6212, cache_growing 846/851 | PASS | artifacts/probe-mei-qwen38-4bit-rawfix-20260903T080402Z.json |
| probe_mei context_exact_cap — raw 65536 | PASS — prefill 1292.6 s (50.7 pps), 1-token decode, active 27.25 GB, **peak 31.70 GB** (< 32 GB, ~0.3 GB headroom) | same |
| probe_mei context_over_cap_rejected (65537) | PASS — HTTP 400 "request exceeded context cap", 1.3 s | same |
| probe_long_context raw 30k (fresh server, no restored cache) | PASS — fresh fill 30000 tok in 541.4 s (55.8 pps), decode 11.61 t/s; reuse 30000/30001 cached, restore 4.0 s, decode 11.76 t/s; peak 31.70 GB | artifacts/probe-longctx-qwen38-4bit-rawfix-20260903T082621Z.json |

Status: Qwen3.8-27B-4bit common matrix is now COMPLETE on the chat path AND the raw path — all legs green. Decode 15.2–15.7 t/s short-context / 11.6 t/s at 30k; peak 31.70 GB at both 30k raw and 65k exact-cap raw (raw-endpoint prefill carries a heavier workspace than chat fills; still fits 32 GB, high-pressure). 4-bit remains the fastest safe Qwen3.8 config (llama.cpp UD-Q5_K_M reference: 9.286 t/s). Remaining todo-8 gap for this model: none measured; the GGUF/llama.cpp reference comparison for this model is the 9.286 t/s ceiling counter-check (2026-09-02) — a full same-suite A/B stays pending at the plan level.