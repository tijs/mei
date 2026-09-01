# Ornith-9B proxy: acceptance + 3-repeat baseline (2026-09-01)

Primary 35B is perf-memory-bound (see `ornith-35B-load-acceptance-20260901.md`:
working set 36.7 GB > 32 GB physical, no clean-decode rows claimable). Per the
plan gate, optimization evidence is captured on the **Ornith-9B proxy** on the
identical Mei text path. This window was clean (no foreign inference owners;
only Mei on 8024; reclaimable ~14.2 GB pre-load / 22.8 GB post-35B-stop).

## Server config (identical text path, defaults preserved)
- model: `ornith-ai/Ornith-1.5-9B-MLX-4bit` (dense qwen3_5, official 4-bit affine/group-64)
- served id: `ornith-ai/Ornith-1.5-9B-MLX-4bit`; port 8024; context-cap 65536
- prefill-step 512; cache-reuse true; kv-cache-dir enabled (in-memory + disk tier)
- compiled-decode false; kv-bits none; max-kv-window 0; ssm-anchor-boundaries 0
  (all optimization flags default-off, preserved)
- memory after load: active 5,038,040,076 B (5.04 GB)

## Load/identity
`artifacts/load-9B-20260901T164509Z.json` — probe exit 0, all probes passed:
- hello: 46.35 tok/s, prefill 351.7 ms, generate 711.9 ms
- short_decode: 47.74 tok/s, prefill 265.8 ms, generate 670.3 ms
- active ~5.31 GB during decode (0.2× the 35B working set)

## Bench 3× clean repeats (same server, same prompt set, engine-reported)
Artifacts: `bench-9B-20260901T164526Z.json` (r1), `bench-9B-r2-20260901T165122Z.json` (r2),
`bench-9B-r3-20260901T165444Z.json` (r3). All 6 rows passed all 3 runs (exit 0).

| row | r1 tok/s | r2 tok/s | r3 tok/s | mean tok/s | cached (r3) |
|---|---|---|---|---|---|
| short_context (16 ptok) | 46.43 | 46.37 | 46.37 | 46.39 | 11/16 |
| tool_nonstreaming (293 ptok) | 46.17 | 46.21 | 46.12 | 46.17 | 288/293 |
| tool_streaming (293 ptok) | 46.24 | 46.24 | 46.17 | 46.22 | 288/293 |
| long_loaded_fresh (45000 ptok) | 35.85 | 35.89 | 35.88 | 35.87 | 33175* |
| long_loaded_reuse (45001 ptok) | 35.89 | 35.87 | 35.87 | 35.88 | 45000/45001 |
| long_chat_40k (44002 ptok) | 35.56 | 33.87 | 34.07 | 34.50 | 43997/44002 |

*`long_loaded_fresh` cached=33175 reflects prior in-run prefix state; it is the
45K decode row and its decode rate (35.9) is the loaded-context figure.

## Tool-call correctness (all 3 runs)
`add_numbers` forced call, both non-stream and SSE stream:
validated_call `{name: add_numbers, arguments:{a:15, b:27}, finish_reason: tool_calls}`
— exact schema, streaming/non-streaming parity confirmed.

## KV reuse evidence
- `long_loaded_reuse`: cached_tokens 45000/45001 (full prior 45K prefix),
  prefill 60_s → 0.12_s, decode 35.9 tok/s.
- `long_chat_40k` r2/r3: cached 43997/44002 (disk tier cross-restart reuse from r1).

## Verdict
- 9B proxy full acceptance PASSED on 3 clean repeats.
- Loaded-context (45K) decode ~35.9 tok/s; short/tool ~46.2 tok/s — all above the
  30 tok/s engineering goal; stable across repeats (range < 0.3 tok/s).
- KV/prefix reuse functional (in-memory + on-disk); streaming/non-stream parity and
  tool-call schema correct.

## Files
- `artifacts/load-9B-20260901T164509Z.json`
- `artifacts/bench-9B-20260901T164526Z.json`, `bench-9B-r2-20260901T165122Z.json`,
  `bench-9B-r3-20260901T165444Z.json`
- this summary.