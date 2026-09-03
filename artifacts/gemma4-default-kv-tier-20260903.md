# Gemma4 model-aware disk-KV default (todo 0b87b76a#7, blocker (a) cleared) — 2026-09-03

Unit: extend the generic-profile model-aware disposable disk-KV default to the
`gemma4` lineage so exact-repeat prefix reuse works without an explicit
`--kv-cache-dir` (recording blocker (a) of the Gemma leg in
artifacts/gemma4-26b-common-matrix-20260903.md).

## Change (rolled-back-safe, model-aware, gated)

- `Sources/MeiCore/ModelOptimizationProfile.swift`: `diskKVRequiredModelTypes`
  = {`qwen3_5`, `qwen3_5_text`, `gemma4`, `gemma4_text`} with
  `needsDiskKVTier(modelDirectory:)`; the old `denseQwen35KVUnsafeModelTypes` /
  `denseQwen35NeedsDiskKVTier` remain as deprecated aliases (public API kept
  compiling; doc-comment `@deprecated` only, no `@available` warnings).
- `Sources/MeiCore/ServerConfig.swift`: parse now calls `needsDiskKVTier`;
  usage/help text and doc comments updated. Behavior contract unchanged:
  fires only when `--cache-reuse` is on (default) AND the operator passed no
  `--kv-cache-dir`; the disposable dir lives under the OS temp dir, sanitized
  per served-model-id; explicit flag always wins; `--cache-reuse false` never
  creates a dir; the MoE/hybrid `qwen3_5_moe` (Ornith) family is untouched.
- Tests: `testGemma4DefaultsDisposableDiskKVWhenReuseOnAndNoExplicitDir`
  (root `model_type: gemma4`) + `testGemma4TextNestedModelTypeDefaultsDisposableDiskKV`
  (nested `text_config.model_type: gemma4_text`, mirrors the staged bundle);
  the old negative gemma4 test flipped to an unrelated `llama` model type;
  qwen3_5/qwen3_5_text/ornith/explicit-wins/cache-reuse-false tests preserved.
- Docs: README KV paragraph + CHANGELOG [Unreleased] entry.

## Verification (Sulaco M1 Max 32 GB, clean Metal window, port 8024)

Binary `arm64-apple-macosx/release/mei` @ scratch ~/.local/share/local-model-bench/mei-build
(build 2026-09-03T05:3x); vmlx fork pinned 91fed8be. Config = proven safe
generic with NO `--kv-cache-dir` (the point of the change):
`--model-dir ~/.local/share/local-model-bench/mei-models/gemma-4-26b-a4b-it-4bit
--served-model-id mlx-community/gemma-4-26b-a4b-it-4bit --optimization-profile
generic --host 127.0.0.1 --port 8024 --context-cap 65536 --prefill-step-size 64`

- Server line (default config): `prefix cache enabled (paged in-memory + disk at
  /var/folders/.../T/mei-kv-cache/mlx-community-gemma-4-26b-a4b-it-4bit);
  topology layers=30 kvLayers=5 rotatingLayers=25 restore=disk-backed` — the
  default fired, disposable dir under OS temp.
- post-load active 15,344,812,572 B (15.34 GB) — identical to the explicit
  disk-tier matrix run.
- probe_load r1 (artifacts/probe-load-gemma4-defaultkv-r1-20260903T033843Z.json):
  short decode 51.05 t/s; peak 24.01 GB — PASS (matrix 51.35/51.43/51.27).
- probe_mei (artifacts/probe-mei-gemma4-defaultkv-20260903T033902Z.json):
  - models_identity / mei_status / plain_completion (51 t/s) / parity_stream_vs_nonstream: PASS
  - cache_repeat_1 (cold 6174-tok prefill, cached=0, 38.99 s): expected cold leg
  - cache_repeat_2 (exact repeat): **cached_tokens=6173/6174, prefill 38.99 s -> 0.66 s — PASS
    on the DEFAULT config** (previously cached=0 on default; identical to the
    explicit disk-tier result) → blocker (a) CLEARED
  - cache_growing_turn2_reuses_slot: cached_tokens=0 → FAIL — blocker (b)
    UNCHANGED (rotating-layer 25/30 prefix-extension restore gap; vmlx-level
    investigation, separate unit)
  - tool_nonstreaming/tool_streaming: strict-schema FAIL on Gemma4-native
    string-typed args {'a':'15','b':'27'} → blocker (c), unchanged; structural
    tool calls are valid; coercion needs explicit user go/no-go.

## Qwen3.8-27B-4bit regression on the same binary/default config

- Server line: disposable disk tier fired (`.../T/mei-kv-cache/mlx-community-Qwen3.8-27B-4bit`,
  topology layers=64 kvLayers=16 mambaLayers=48 companion=ssm) — qwen behavior
  unchanged (qwen3_5 was already in the set).
- probe_load r1 (artifacts/probe-load-qwen38-4bit-defaultkv-r1-20260903T034120Z.json): PASS.
- raw /v1/completions, 17-token prompt, 120 tokens: PASS (15.6 t/s decode,
  correct fibonacci code, no crash) — the 2026-09-03 raw VLM fix (ee7368d)
  holds on this build; response surface is chat-style `message.content`.

## Test suite

`swift test --scratch-path ~/.local/share/local-model-bench/mei-build`:
54/59 pass; the 5 failures are MeiAcceptanceTests needing a live server
(environmental, same 5 as every prior run).

## Status

Blocker (a) of the Gemma leg is cleared; (b) growing-transcript/rotating-layer
restore and (c) tool strict-schema remain recorded blockers. Todo 0b87b76a#7
stays OPEN (Gemma leg partial: (b)+(c) pending; Heretic long-context + Ornith
matrix legs + GGUF comparisons still pending). Next queued: Gemma4
rotating-layer prefix-extension restore investigation (vmlx cache-extension
path) or Heretic long-context leg.

## Commands
```
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
swift test --scratch-path ~/.local/share/local-model-bench/mei-build
<built>/mei --model-dir <gemma dir> --served-model-id mlx-community/gemma-4-26b-a4b-it-4bit \
  --optimization-profile generic --host 127.0.0.1 --port 8024 --context-cap 65536 --prefill-step-size 64
python3 tools/probe_load.py --base-url http://127.0.0.1:8024/v1 --model <id> --output artifacts/probe-load-*.json
~/.local/share/local-model-bench/mei-runtime/venv/bin/python tools/probe_mei.py \
  --base-url http://127.0.0.1:8024/v1 --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --tokenizer <gemma dir> --context-cap 65536 --skip-context --output artifacts/probe-mei-*.json
```