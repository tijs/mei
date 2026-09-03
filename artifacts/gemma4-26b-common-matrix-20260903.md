# Gemma 4 26B-A4B common correctness matrix (todo 0b87b76a#7 leg) — 2026-09-03

Engine: Mei, binary sha256(16) `d54a029004b019f5` (build 2026-09-03T03:58:41Z,
AFTER the chat-path `[1,T]` token-shape fix; the pre-fix binary `d4fc502152ebd161`
crashes Gemma at the first chat prefill — see BLOCKER-0 below), vmlx fork pinned
`91fed8be`. Proven safe config: generic profile, context cap 65536, prefill
step 64, kv-bits none, cache-reuse true, port 8024, temperature 0 for probes.
KV tier: default config = "paged in-memory + no disk"; the KV-reuse legs were
re-run with the disk tier (`--kv-cache-dir` fresh disposable dir) because
Gemma4 prefix reuse rides the disk tier (see KV-REUSE).

Model under test: `mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464e`
(4-bit affine g64, staged verified 2026-09-02, 15,341,205,776 B raw,
config sha256 419e13a2...). NOTE: this bundle is multimodal — 355
`vision_tower.*` tensors + processor_config.json — so the loader's VLM-first
registry routes it to `MLXVLM.Gemma4`, exactly the class of bundle that
tripped the Qwen3.8 raw-completions crash.

## BLOCKER-0 (FIXED this tick): Gemma4 VLM prepare crashes on 1-D chat tokens

`Fatal error: precondition failure` at `MLXArray.subscript.getter`
(`getItemND`) via `Gemma4.prepare(_:cache:windowSize:)` during the FIRST
fresh prefill of a 19-token chat prompt (crash report
mei-2026-09-03-035628.ips, EXC_BREAKPOINT/SIGTRAP, exit 309). Mechanism:
`Gemma4.prepare` embeds `input.text.tokens` directly (no flatten) — a 1-D
`[T]` array yields a 2-D embedding `[T, hidden]`, `tokenCount =
emb.dim(1)` reads the HIDDEN size, `prefillStepSize=64 < tokenCount` enters
the chunked loop, and `emb[0..., offset..<end, 0...]` slices a 2-D array
with 3 indices → precondition crash. Reproduced on the chat endpoint
(fresh prefill only; cache-restored requests were already `[1,T]` via the
vmlx rebuild). The Mei-produced 5-bit Qwen/Ornith text bundles never
entered this branch (loader default prepare flattens).

Fix (commit, Sources/MeiCore/Engine.swift): both chat LMInput sites now emit
batch-first `[1, T]` tokens via `expandedDimensions(axis: 0)`, matching the
raw completions path (fixed 2026-09-03T00:xx) and the vmlx cache-restore
rebuild. `[1, T]` is rank-safe for every model class (LLM default prepare
flattens; rejects only batch > 1). 52/52 non-Metal unit tests pass; the 5
MeiAcceptanceTests failures are environmental (need a live server on 8024).

## Matrix results (post-fix binary, disk-KV server for reuse legs)

| Probe | Result | Evidence artifact |
|---|---|---|
| probe_load r1 (hello cold 13.4 t/s incl kernel compile; short 51.35 t/s, peak 24.08 GB) | PASS | probe-load-gemma4-r1-20260903T015903Z.json; pre-fix crash evidence: probe-load-gemma4-r1-20260903T015547Z.json (conn refused) + r1-20260903T015619Z.json (server died mid-hello) |
| probe_load r2 (hello warm 30.3 t/s, short 51.43 t/s) | PASS | probe-load-gemma4-r2-20260903T015917Z.json |
| probe_load r3 (hello warm 31.2 t/s, short 51.27 t/s) | PASS | probe-load-gemma4-r3-20260903T015921Z.json |
| decode 3-repeat: 51.35 / 51.43 / 51.27 t/s → mean 51.35 (sd 0.08) | PASS | above; EXCEEDS the 30 tok/s project target on short decode |
| probe_mei models_identity, mei_status, plain ("ready" 30.7 t/s), parity_stream_vs_nonstream (content "parity-ok" both ways) | PASS | probe-mei-gemma4-nocontext-20260903T020944Z.json (default KV) / probe-mei-gemma4-diskkv-20260903T021151Z.json (disk KV) |
| probe_mei tool_nonstreaming, tool_streaming | STRUCTURAL PASS / strict-schema FAIL — Gemma4-native string-typed args `{"a":"15","b":"27"}` (matches 2026-09-01T165909 load finding); the probe asserts int types. Model-faithful output, not a Mei defect; coercion needs a go/no-go. | same artifacts |
| probe_mei cache_repeat 1+2 (6174-token prompt ×2) | default KV: cached=0 both → RED; disk KV: 6173/6174 cached, prefill 37.3 s → 0.18 s → PASS | probe-mei-*nocontext / *diskkv* |
| probe_mei cache_growing_turn2 (prefix extension 793→824 tok) | RED on BOTH tiers: cached_tokens=0 ("growing-transcript reuse failed"). Qwen3.8/Ornith pass the same probe on the disk tier → Gemma4 rotating-layer (25/30) prefix-extension restore gap. BLOCKER, next unit. | same artifacts |
| probe_coding (python_json_sum, shell_rename, sql_users_query, swift_fibonacci) | 4/4 PASS, decode 20.2–50.8 t/s | probe-coding-gemma4-20260903T021258Z.json |
| probe_long_context 30k raw: fresh fill 212.9 s (~141 pps) + reuse request restored 30000/30000 cached in 5.9 s | PASS | probe-longctx-gemma4-30k-20260903T021406Z.json |
| probe_context_threshold 30k chat fill: 209.1 s (~143 pps) | PASS | probe-ctx-threshold-gemma4-30000-20260903T021939Z.json |
| context over-cap 65537 raw → HTTP 400 | PASS | inline probe 2026-09-03T02:2x |
| context exact-cap 65536 raw: prefill 298.1 s (restored 30k disk seed + ~35k fresh), peak **30.13 GB** active 28.57 GB | PASS | probe-ctx-cap-gemma4-65536-20260903T022841Z.json |

## Memory profile (32 GB M1 Max)

- post-load active: 15.34 GB (limit 24.05 GB, cache-limit 32.64 GB).
- short decode peak: 24.08 GB.
- 6.2k prefill (cache_repeat legs): peak 27.58 GB, active 27.05 GB.
- 30k fills: pps 141-143 (no usage memory capture in the long-context probes).
- 65k exact-cap raw: prefill 298.1 s (~220 pps incl 30k disk-seed restore),
  peak 30.13 GB, active 28.57 GB — fits 32 GB physical with ~2-4 GB headroom,
  high-pressure but survival PROVEN at the full context cap.
- Project plan's 5-bit rejection math for Gemma stands (5-bit would exhaust the
  30 GB budget with zero long-context headroom); 4-bit fits with ~2-4 GB
  headroom at the 65k cap.

## Status

- Gemma 4 26B-A4B is now LOADABLE and GENERATES on the generic profile: speed
  gate EXCEEDED (51.35 t/s short decode vs the 30 t/s target), identity/plain/
  parity/coding/long-context(30k)/over-cap gates PASS; 65k cap leg ran last.
- Tool legs are structurally valid but carry Gemma4's native string-typed
  args — strict-schema gate needs an explicit acceptance decision (no
  autonomous coercion).
- KV reuse gaps: (a) default (no-disk) config does NOT reuse at all for
  Gemma4 (model-aware default covers only dense qwen3_5/qwen3_8) — exact-repeat
  reuse requires the disk tier, currently an explicit operator flag; (b)
  prefix-extension (growing transcript) reuse fails even on the disk tier
  (rotating-layer restore gap). Both are recorded blockers.
- Todo 0b87b76a#7 stays OPEN (Gemma leg partial; Heretic/Ornith legs and the
  GGUF comparisons still pending in todo/plan terms).

## Commands

```
build:  swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
launch: ~/.local/share/local-model-bench/mei-build/arm64-apple-macosx/release/mei \
  --model-dir ~/.local/share/local-model-bench/mei-models/gemma-4-26b-a4b-it-4bit \
  --served-model-id mlx-community/gemma-4-26b-a4b-it-4bit --optimization-profile generic \
  --host 127.0.0.1 --port 8024 --context-cap 65536 --prefill-step-size 64 \
  [--kv-cache-dir <fresh disposable dir>]  # required for the KV-reuse legs
probe: python3 tools/probe_load.py --base-url http://127.0.0.1:8024/v1 --model mlx-community/gemma-4-26b-a4b-it-4bit --server-log ~/.local/share/local-model-bench/mei-runtime/logs/server.log --output artifacts/probe-load-gemma4-rN-<ts>.json
probe: .venv/bin/python tools/probe_mei.py --base-url ... --model ... --tokenizer <gemma staged dir> --context-cap 65536 [--skip-context] --output ...
probe: python3 tools/probe_coding.py ... ; .venv/bin/python tools/probe_long_context.py ... --lengths 30000 ...; .venv/bin/python tools/probe_context_threshold.py ... --endpoint chat --lengths 30000 ...
```
## UPDATE (2026-09-03, second worker pass): blocker (a) CLEARED

The exact-repeat KV-reuse gap on the DEFAULT config is fixed as a gated,
model-aware default: the generic-profile disposable disk-KV default
(`ModelOptimizationProfile.diskKVRequiredModelTypes`) now includes
`gemma4`/`gemma4_text` alongside `qwen3_5`/`qwen3_5_text`, so cache-reuse on
with no explicit `--kv-cache-dir` puts Gemma 4 on a disposable OS-temp disk
cache. Verified on the default config (no flag): server line `prefix cache
enabled (paged in-memory + disk at /var/folders/.../T/mei-kv-cache/mlx-community-gemma-4-26b-a4b-it-4bit)`;
probe_mei cache_repeat_2 cached=6173/6174 (38.99 s -> 0.66 s); probe_load
short decode 51.05 t/s; post-load 15.34 GB unchanged. Qwen3.8-4bit regression
on the same binary passed (disposable tier line, probe_load PASS, raw 120-tok
PASS at 15.6 t/s). Remaining recorded Gemma blockers: (b) growing-transcript
prefix-extension restore gap (cached_tokens=0 on both tiers, rotating-layer
25/30 topology — vmlx cache-extension investigation, next unit) and (c) tool
strict-schema string-args (user go/no-go). Evidence:
artifacts/gemma4-default-kv-tier-20260903.md, probe-mei-gemma4-defaultkv-20260903T033902Z.json,
probe-load-gemma4-defaultkv-r1-20260903T033843Z.json,
probe-load-qwen38-4bit-defaultkv-r1-20260903T034120Z.json.
