# Heretic (Qwen3.8-27B-Uncensored) 30k loaded-decode 3-repeat (todo 0b87b76a#12 leg) — 2026-09-04

Unit: continuing optimization loop (#12) — fill the missing Heretic speed/memory row at
30k loaded context. Heretic's common matrix (2026-09-03) covered acceptance, tool calls,
KV reuse, and a single 30k survival probe (n=1: fresh 11.79 / reuse 11.80 t/s from
probe-longctx-heretic-4bit-20260903T094433Z.json); the #12 loop requires three
representative repeats per model. This leg provides n=3 on the proven Qwen-family safe
config and checks the ceiling-transfer hypothesis from the accepted Qwen3.8-27B base
exhaustion record (plan note 2026-09-02).

Window: clean for all three legs — no foreign inference processes before/during
(re-checked per leg), no benchmark port in use, Mei-owned port 8026, disposable runtime
bases `mei-runtime-heretic-30k-r{1,2,3}`, local-model-bench repo READ-ONLY.

Config (proven safe Qwen-family row, identical for every repeat): generic profile,
prefill-step 64, context cap 65536, kv-bits none, disposable disk KV
(`--kv-cache-dir` under the runtime base, removed before every start), port 8026,
temp 1.0 (mirrors gguf-heretic-q5.yaml), released binary built by
`runner/start_mei_server.sh` at `mei-build/release/mei` from mei main 3253d07;
vmlx-swift resolved from the local checkout `/Users/tijs/projects/vmlx-swift` @
318a4e68 (= pinned fork 91fed8be + the reviewed gen-suffix-boundary cache fix —
build.log references the local path directly). Model:
`orcarouter/Qwen3.8-27B-Uncensored-MLX` subdir 4-bit @ 14963e70f staged at
`mei-models/Qwen3.8-27B-Uncensored-MLX-4bit` (16.05 GB active after load). Each repeat:
fresh disposable KV, cold server, `tools/probe_long_context.py --lengths 30000
--max-tokens 32` (30k fresh fill + strict-extension reuse decode).

Topology (server log): `layers=64 kvLayers=16 mambaLayers=48 companion=ssm
restore=disk-backed` — identical shape to base Qwen3.8-27B (dense 16/64 full attention
+ 48/64 GDN linear-attention, float32 recurrent state).

## Measured (n=3, cold)

| repeat | fresh 30k decode t/s | reuse decode t/s | 30k fresh fill pps | peak mem |
|---|---|---|---|---|
| r1 | 11.928 | 11.837 | 56.16 | 24.71 GB |
| r2 | 11.829 | 11.899 | 56.23 | 24.71 GB |
| r3 | 11.930 | 11.792 | 56.23 | 24.71 GB |
| mean | 11.896 (sd 0.058) | 11.842 (sd 0.054) | 56.21 | — |

All rows `status: passed` with full checks (http_ok, nonempty_completion,
prompt_tokens_match, decode_above_floor); reuse rows cached all 30000 prefix tokens
(cross-turn disk-backed KV reuse intact at depth).

Artifacts: artifacts/probe-longctx-heretic-30k-r{1,2,3}-20260904T05{10,19,29}*Z.json
(committed); server stage dumps `mei-runtime-heretic-30k-r{1,2,3}/logs/{build,server}.log`
(disposable). Worktree note: `swift build --scratch-path mei-build` re-writes
~/projects/mei/Package.resolved dropping the vmlx-swift entry (effectively a local
package via the checked-out fork dir); restored to HEAD (pin 91fed8be) before commit.

## Verdict

Heretic replicates the accepted base-Qwen hardware-ceiling evidence at 30k: 3-repeat
band sd < 0.06 t/s, no regression vs the family, peak 24.71 GB stable. Same topology +
same 4-bit affine g64 quant family as Qwen3.8-27B base, so the accepted exhaustion
record (plan note 2026-09-02: 30 t/s is 1.87x above fastest measured config; pure-stream
floor ~19-20 t/s @ 350-400 GB/s) transfers to Heretic — no distinct Heretic-specific
lever matrix is warranted. Cross-engine 30k comparison: Mei MLX 4-bit 11.90 t/s vs
llama.cpp Q5_K_M 7.65 t/s (gguf-ref-heretic-30k-decode-retest-20260903.json;
+1.55x MLX), the same ratio shape as base Qwen (11.61-11.76 vs 8.30).

#12 stays OPEN (Ornith-35B primary + Gemma rows still active); Heretic's #12 evidence
row is complete.