# Gemma 4 prefill-step sweep + 30k loaded-decode gap (todo 0b87b76a#8 gap-fill + #9 first leg) — 2026-09-03

Unit: verify and characterize the Gemma 4 MLX 30k-loaded-decode row that the
GGUF A/B (artifacts/gemma4-gguf-ab-20260903.md) had flagged as "n/a", and run
the first optimization-loop leg (prefill-step sweep). Mei HEAD before this
tick: e6f892d. Clean GPU window (no foreign inference processes at launch and
between legs; Mei-owned port 8024; all runtime dirs disposable under
~/.local/share/local-model-bench/mei-runtime-gemma-*).

Engine/config (identical to the morning Gemma matrix "proven safe config":
generic profile, context cap 65536, kv-bits none, cache-reuse true, temp 0,
port 8024, binary from mei-build scratch at HEAD e6f892d — the SAME binary
that produced the 7.45 t/s rows this morning).
Model: `mlx-community/gemma-4-26b-a4b-it-4bit` staged under
mei-models/gemma-4-26b-a4b-it-4bit (4-bit affine g64, rev 0d77464e, VLM
bundle, 30 blocks, kvLayers=5 rotatingLayers=25 per server topology log).

## 1. Baseline reproduction (prefill 64, disk-KV production default)

probe_long_context lengths 30000, max_tokens 32:
- fresh 30k fill: 231.2 s @ **133.5 pps** (morning: 212.9 s @ 144.5 pps;
  run-to-run spread ~8%, mean ~139 pps)
- **30k loaded decode: 7.36 t/s** fresh, 7.33 reuse (morning: 7.45/7.46)
- reuse row: 30000/30000 cached, restore ~5.9 s; peak 27.23 GB
- The server log confirms the model-aware disposable disk-KV default engaged
  automatically ("paged in-memory + disk at …/T/mei-kv-cache/…; restore=disk-backed")
  — no explicit --kv-cache-dir was passed. The "in-memory isolation" intent
  of this leg was therefore NOT measurable from launch flags: the 7.36 t/s
  row IS the production-default number, which is what the A/B table needs.

## 2. The 7.4 t/s 30k decode is real, stable, and NOT disk-I/O

Across all legs today (prefill 64/128/256/512, fresh + reuse rows): decode at
30k loaded = 7.16-7.51 t/s (9 rows, mean ~7.4, sd ~0.1). KV tier and prefill
step are both ruled out as causes. Gemma 4's 30 full-attention blocks (incl.
25 rotating layers) stream the full KV span per decode step on MLX eager
SDPA; at short context the same binary does 51-55 t/s (19 ms/token) vs
~135 ms/token at 30k. This is the dense/rotating-attention long-context
decode cost — the same family of gap as Qwen3.8/Heretic
(16/64 attention layers: 11.8 t/s at 30k) but worse, and it is a measured
constraint, NOT a correctness or config artifact.

A/B vs GGUF (mudler APEX-I-Quality Q6_K, llama.cpp 10470):
| row | GGUF | Mei MLX 4-bit | delta |
|---|---|---|---|
| 30k loaded decode | 37.08 / 37.17 t/s | **7.36 / 7.33 / 7.39 / 7.43 t/s (this tick)** | GGUF 5.0x faster |
| 30k fresh prefill | 231.1 pps | 139 pps (step 64) | GGUF +66% |
| 30k fresh prefill | 231.1 pps | **266.5 pps (step 256)** | MLX **+15%** |

## 3. Prefill-step sweep (single-shot fresh fills, 30k, disk-KV default)

| step | fresh fill | pps | decode @30k | peak |
|---|---|---|---|---|
| 64 (baseline, 2 rows) | 212.9 s / 231.2 s | 144.5 / 133.5 (mean 139.0) | 7.45 / 7.36 t/s | 27.23 GB |
| 128 | 142.2 s | 219.5 | 7.39 | 27.23 GB |
| **256 (3 repeats)** | 117.6 / 118.2 / 117.9 s | **266.5 / 265.3 / 266.2 (mean 266.0, sd 0.63)** | 7.43 / 7.39 / 7.43 | 27.23 GB |
| 512 | 120.5 s | 260.7 | 7.33 | 27.34 GB |

- 256 vs 64 baseline: **+91.4% prefill throughput** (266.0 vs 139.0 pps),
  peak memory unchanged (27.23 GB both), 30k decode unchanged.
- 512: no further gain (+2.2% vs 256 would be noise), +105 MB peak; 128 is
  between but slower. 256 is the sweep winner and now BEATS the llama.cpp
  GGUF reference prefill (266.5 vs 231.1 pps, +15%) — inverting the
  "GGUF +64% prefill" gap recorded in the morning A/B.
- Artifacts: probe-longctx-gemma4-pref{64,128,256(abs),512}-30k-20260903T*.json
  (pref256 @153615Z = clean; @153530Z = cache-contaminated, see §5),
  pref256 r2/r3 20260903.json.

## 4. Acceptance re-gate at prefill 256 (probe_mei, context-cap 65536)

artifacts/probe-mei-gemma4-pref256-20260903T154944Z.json: **11/13 PASS** —
models_identity, mei_status, plain_completion, parity_stream_vs_nonstream,
cache_repeat_1/2, cache_growing_turn1, cache_growing_turn2_reuses_slot,
context_exact_cap, context_over_cap_rejected. The 2 failures are exactly the
pre-existing, USER-GATED Gemma tool-schema mismatch (model-faithful
string-typed args {"a":"15","b":"27"} vs the probe's strict JSON-int schema;
same as the morning 64-step matrix). No new failures ⇒ prefill 256 preserves
the acceptance pass-set of the 64 baseline.

## 5. Methodology notes / uncertainty

- The model-aware disposable disk-KV cache lives in the OS temp dir keyed by
  served model id, shared across server runs. The FIRST pref256 leg restored
  the previous leg's fill (fresh row showed cached_tokens 29999, pps 36253 —
  invalid; kept as evidence of cross-run cache persistence). All clean rows
  here ran after removing that cache dir (2.9 GB, disposable).
- Prefill-pps rows are single-shot fresh fills except the 256 winner (3 cold
  repeats, sd 0.63 pps) and the 64 baseline (2 rows across two days, spread
  ~8%). The +91% margin is far beyond the noise band.
- 30k decode rows: 9 measurements across both days, sd ~0.1 t/s — stable.
- 512 leg's +105 MB peak is within run noise (both legs' probes report the
  row peak; identical 27.23 GB for 64/128/256 suggests the 30k peak is
  KV/weight-dominated, not chunk-bound, at these step sizes).

## 6. Change adopted (arch-scoped, gated)

- Sources/MeiCore/ModelOptimizationProfile.swift: new
  `prefill256ModelTypes` (gemma4, gemma4_text) + static
  `prefillStepSize(modelDirectory:profile:)`; ServerConfig.swift uses it when
  --prefill-step-size is not explicit. Ornith keeps 512, all other models
  keep 64, explicit flag always wins, malformed metadata stays 64 (same
  fail-safe shape as needsDiskKVTier).
- Verified end-to-end: `mei` launched with `--optimization-profile auto` on
  the gemma4 bundle prints "optimization profile generic (requested auto,
  prefill 256, …)" and serves a 16-token completion (54.8 t/s short, smoke).
- Tests: 3 new ServerConfigParsingTests cases (gemma4→256 default, explicit
  flag wins, dense qwen3_5 stays 64). `swift test --skip MeiAcceptanceTests`:
  **57/57 pass, RC=0** (54 prior + 3 new). Metal-library note: running the
  XCTest suite requires mlx.metallib colocated with the test executable
  (.build/**/MeiPackageTests.xctest/Contents/MacOS/mlx.metallib); with the
  default-loader's first candidate missing, the first MLX-touching test
  crashes ("Failed to load the default metallib", array.cpp:232). Package.resolved
  pin (vmlx-swift 91fed8be) was regenerated by `swift build/test` (SwiftPM
  6.3.3 hazard) and restored from HEAD after each run.

## 7. Todo status

- `0b87b76a#8`: 30k-loaded-decode row now filled (7.36-7.51 t/s MLX vs 37.08
  GGUF, 5.0x GGUF-faster — measured constraint recorded). Every model's
  GGUF/Mеi A/B table is complete. Remaining inside #8: ONLY the user
  go/no-go on the Gemma tool strict-schema string-args acceptance — #8 stays
  OPEN for that decision (blocked autonomously).
- `0b87b76a#9` (optimization loop): FIRST LEG recorded — prefill 256 adopted
  as the gemma4 default (+91% fresh prefill, no memory/decode/correctness
  regression). Loop continues: 30k-decode gap remains the Gemma bottleneck
  (next candidates: windowed attention gate, KV-quant upstream blocker
  re-check, compiled-decode Gate on gemma4 shape — all gated/experimental).
- Ornith/Qwen3.8/Heretic configs untouched; release candidate staging
  (todo #7, 3cfadf9 + 14:27Z refresh) predates this change — the staged
  snapshot is superseded for the gemma4 prefill default only; CHANGELOG
  entry added so any future refresh carries it.

## Reproducibility

```bash
# server (any leg): the start script with MEI_* env; see §1 for the exact
# proven-safe config. Example (prefill 256 leg):
MEI_RUNTIME_BASE=$HOME/.local/share/local-model-bench/mei-runtime-gemma-pref256 \
MEI_CACHE_ROOT=$HOME/.local/share/local-model-bench/mei-runtime-gemma-pref256 \
MEI_KV_CACHE_DIR=$HOME/.local/share/local-model-bench/mei-runtime-gemma-pref256/kv-leg256 \
MEI_MODEL_DIR=$HOME/.local/share/local-model-bench/mei-models/gemma-4-26b-a4b-it-4bit \
MEI_SERVED_MODEL_ID=mlx-community/gemma-4-26b-a4b-it-4bit \
MEI_OPTIMIZATION_PROFILE=generic MEI_PORT=8024 MEI_CONTEXT_CAP=65536 \
MEI_MAX_TOKENS=32768 MEI_PREFILL_STEP_SIZE=256 MEI_TEMPERATURE=0 \
MEI_CACHE_REUSE=true bash scripts/start_mei_server.sh

# probe (venv python has transformers 5.15.0):
.local/../local-model-bench/.venv/bin/python tools/probe_long_context.py \
  --base-url http://127.0.0.1:8024/v1 --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --tokenizer $HOME/.local/share/local-model-bench/mei-models/gemma-4-26b-a4b-it-4bit \
  --lengths 30000 --max-tokens 32 --min-decode-tps 1.0 --timeout 900 \
  --output artifacts/probe-longctx-gemma4-pref256-30k-rN-20260903.json
```

Artifacts (this tick): probe-longctx-gemma4-inmem-30k-20260903T153045Z.json,
probe-longctx-gemma4-pref256-30k-20260903T153530Z.json (cache-contaminated,
kept as evidence), probe-longctx-gemma4-pref256-30k-20260903T153615Z.json,
probe-longctx-gemma4-pref128-30k-20260903T153902Z.json,
probe-longctx-gemma4-pref512-30k-20260903T154202Z.json,
probe-longctx-gemma4-pref256-30k-r{2,3}-20260903.json,
probe-mei-gemma4-pref256-20260903T154944Z.json, this md. Servers stopped;
ports released; disposable temp KV caches removed.