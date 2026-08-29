# Mei (梅)

Mei — after Mei Long, the sleeping dragon dinosaur — is a narrow, native
Swift/MLX OpenAI-compatible inference server for Apple Silicon, built directly
on the pinned `osaurus-ai/vmlx-swift` engine (not the full Osaurus app).

The project's scope is deliberately small: one model per server process, the
OpenAI chat/completions surface this repo's benchmark needs, chunked prefill
on for hybrid architectures, and in-process KV/prefix reuse across turns.
The primary target is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`; the goal is
>= 30 decode tokens/second with correct tool-calling and no long-context
collapse on this 32GB Apple Silicon machine.

## Layout

- `Sources/MeiCore/` — engine, router, HTTP server, OpenAI DTOs
- `Sources/Mei/main.swift` — CLI entry point
- `Tests/MeiTests/` — unit tests + black-box acceptance oracle
  (`MeiAcceptanceTests`: run RED against a missing server, go green once the
  server behaves; enabled via `MEI_ACCEPTANCE_BASE_URL`, default
  `http://127.0.0.1:8024/v1`)
- `tools/probe_mei.py` — standalone acceptance/perf probe (the authoritative
  copy; `local-model-bench/runner/probe_mei.py` mirrors it)
- `scripts/` — stage_model.sh / start_mei_server.sh / stop_mei_server.sh
  (isolated runtime: dedicated port 8024, own logs/build/model staging under
  `~/.local/share/local-model-bench/mei-*`)

## Build

```bash
swift build            # debug
swift test             # unit tests (acceptance tests need a live server)
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
```

`vmlx-swift` is pinned by revision (`aeb5e21c…`, the revision osaurus-ai's own
`Package.resolved` pins) and `Package.resolved` is committed. Re-pinning is a
deliberate decision that must re-run the whole acceptance suite.

## Run

```bash
scripts/stage_model.sh   # downloads the model once (~20GB)
scripts/start_mei_server.sh    # builds + serves on 127.0.0.1:8024
scripts/stop_mei_server.sh
```

or directly:

```bash
swift run mei --model-dir ~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit \
  --served-model-id ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --port 8024 \
  --context-cap 65536 --prefill-step-size 512
```

## Bench integration

The `local-model-bench` harness drives Mei as a first-class
`inference_engine`:

- `configs/Ornith-1.5-35B-A3B/mei.yaml` — launch command, endpoint, settings
- `runner/start_mei_server.sh` / `runner/stop_mei_server.sh` — isolated lifecycle
- `runner/probe_mei.py` — acceptance/parity/tooling gate
- `runner/run_mei_acceptance.py` — config-driven sequential acceptance runs

## Design notes

- **Chunked prefill**: `GenerateParameters.prefillStepSize` defaults to 512
  and is always on — the long-context safeguard for hybrid (GatedDelta)
  architectures; the Python serving wrappers this project replaces lacked it.
- **KV/prefix reuse**: the engine keeps the rendered token sequence + live KV
  of the last request; when the next request's rendered tokens exactly extend
  that sequence (identical system prompt + growing transcript — the standard
  agentic pattern), generation resumes from the cached KV and only the new
  tokens are prefilled. `usage.prompt_tokens_details.cached_tokens` reports
  the reused prefix. Any divergence falls back to a full prefill (always
  correct).
- **Tool calls**: vmlx-swift parses Qwen/Ornith-style `<tool_call>{json}`
  envelopes inside its generate loop; Mei maps `.toolCall` events to OpenAI
  `tool_calls` in both streaming and non-streaming shapes.
- **Reasoning**: Ornith is a Qwen3.5-lineage thinking model; thought text is
  exposed as `reasoning_content` (opt-out via `--emit-reasoning false`) and
  the thinking/visible streams are kept separate.
- **No MTP**: explicitly out of scope (the benchmark's own data shows no
  MTP/speculative win on this hardware).