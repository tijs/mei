# CoCore attached-engine integration

This document records how Mei (0.7.0) interoperates with the
[CoCore](https://github.com/graze-social/cocore) agent's **attached engine**
mode, shipped in
[graze-social/cocore PR #237](https://github.com/graze-social/cocore/pull/237)
("feat(provider): attached OpenAI-compatible engines (mei, llama-server,
mlx_lm) + per-model admission gate", merged as commit
`0151475bf8c98de10a64cab51c23a46dd84a8fe1`). CoCore's authoritative
description of the same feature is `docs/attached-engine.md` in that
repository; this page is the Mei-side mirror: what Mei must serve, what
CoCore proves at startup, what the canaries expect, and what remains
model-specific in Mei 0.7.0. The wire contract Mei implements is
frozen in **[docs/OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md)**; this
page assumes it.

An **attached engine** is a model CoCore does not run itself. You start the
server; CoCore proves it is up, proves what it can do, advertises exactly
that, and proxies jobs to it over loopback HTTP. This is the integration
surface the PR was built around for Mei and its requirements below are the
ones Mei is tested against.

> **CoCore compatibility status.** The released Mei 0.7.0 binary implements
> token-level `response_format` constrained decoding; its exact CoCore
> structured-output canary and full 9-test live acceptance suite passed with
> `mlx-community/Qwen3-4B-4bit` on Sulaco. An unreleased Mei candidate
> (`ea5a67a`) has now passed the same full live acceptance suite 9/9 on Qwen3.6
> text-only, Qwen3.6 vision, and aligned Ornith, including buffered and SSE
> canaries, ordinary text/tool regressions, and fail-closed truncation checks.
> A separate CoCore `AttachedEngine` live probe reported
> `ready=true structured_output=true` for all three; its forced-tool canary
> reported `tool_calls=false` for both Qwen3.6 profiles and `true` for Ornith.
> This does not change the released 0.7.0 binary. CoCore's LaunchAgent is not
> currently online, so provider Register/PDS advertisement of these candidate
> IDs is not yet verified; do not claim the current CoCore agent advertises
> them. The Qwen3.6 vision test used a text-only prompt; image-conditioned
> structured output remains unverified. Plain-text and tool-calling behavior
> is separate from the structured-output gate.
>
> The feature enforces only the documented strict subset and rejects unsupported
> or unsatisfiable schemas before generation. A tokenizer/template preflight
> does not override a failed runtime canary.

## Startup commands

Mei runs as **one server per model process**. For the CoCore recipe on
Apple Silicon (from PR #237):

```bash
brew install tijs/tap/mei                 # mei 0.7.0
mei --model-dir ~/.cache/mei/models/Qwen3-4B-4bit \
    --served-model-id mlx-community/Qwen3-4B-4bit
```

The server listens on `127.0.0.1:8024` by default (`--host`/`--port`
reconfigurable). CoCore waits up to `COCORE_ATTACHED_READY_TIMEOUT` seconds
(default 300) for `GET /v1/models` to answer 2xx, logging progress every
15 s — a native server loading a 20 GB checkpoint cold can take a couple of
minutes. CoCore never starts or restarts the server: `restart()` is a
re-probe, and the serve loop's health tick de-advertises a server that goes
away within one tick.

Without `--served-model-id`, Mei derives the id from the model directory and
prints `mei: no --served-model-id given; serving as <id> (clients must use
this exact id)`. Treat that printed id as authoritative (see below).

## Engine-map key must equal the served model id

CoCore maps **model id → server root**. The map key must **exactly equal the
id Mei advertises** via `GET /v1/models`, which is `config.servedModelID` —
whatever `--served-model-id` was set to (or the derived default). The map
value is the **server root without `/v1`**.

```text
# ~/.cocore/engine-map — one entry per line, `#` comments; the same map can
# be passed as COCORE_ENGINE_MAP="model=http://host:port" (e.g. in a
# LaunchAgent plist). Picked up when the env var is unset.
mlx-community/Qwen3.6-35B-A3B-4bit = http://127.0.0.1:8024
```

Rules CoCore enforces (an `engine-map-invalid` fault, not a fallback):

- The URL is the **server root** (`http://host:port`, optional path prefix);
  CoCore appends `/v1/models` and `/v1/chat/completions` itself. A URL ending
  in `/v1` is **rejected**.
- `http://` only — the client is plain HTTP/1.1, no TLS. A non-loopback host
  is accepted only with a loud warning.
- A model in the map is served by the attached engine and removed from the
  vllm-mlx set; `stub` cannot be remapped; one model, one engine.
- The `<id> = <url>` separator tolerates surrounding whitespace.

Served-id honesty note (CoCore's own wording): `--served-model-id` decides
only the id the network sees. Advertising a text-only repack under the
catalog id is a convenience, not a provenance claim; off-catalog ids are
priced at the uniform rate and route fine.

## Endpoints and transport

Mei serves exactly the surface CoCore's attached engine uses
(`Sources/MeiCore/Router.swift`, `HTTPServer.swift`; frozen in
`docs/OPENAI-COMPATIBILITY.md`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/models` | GET | Identity — `data[].id` is the exact served id (readiness probe) |
| `/v1/chat/completions` | POST | Jobs; streaming (SSE) and non-streaming |
| `/v1/completions` | POST | Legacy text completions — **not** used by CoCore |
| `/healthz`, `/health` | GET | Plain liveness (`{"status":"ok"}`) |

Transport facts:

- Plain **HTTP/1.1**, loopback only by default. CoCore's client
  (`engines/openai_http.rs`) is HTTP/1.1 with chunked-transfer decoding
  (which NIO-based servers like Mei use).
- Streaming is **SSE** (`text/event-stream`), `chat.completion.chunk` deltas,
  `finish_reason`, then `data: [DONE]`.
- CoCore proxies with **`stream_options.include_usage: true`** so receipt
  token counts are the server's own; Mei emits a terminal usage chunk with
  `choices: []` only when that option is set.
- `reasoning_content` and `tool_calls` deltas are forwarded on their own
  channels — Mei emits both (the latter with `index`-fragmented tool-call
  framing).

## The two startup canaries

CoCore maintains **no model-family capability matrix**: the engine proves
what it can do at startup, and only what passes is advertised
(`tool_call_models`, `structured_output_models` on the Register frame).
`COCORE_ATTACHED_SKIP_CANARIES=1` skips both and advertises neither.

### 1. Tool-calling canary (forced)

Exactly what CoCore sends (`tool_canary_body` in `engines/openai_http.rs`):

```json
{
  "model": "<served model id>",
  "messages": [
    {"role": "system", "content": "You are a tool-calling canary. When a tool is forced, return exactly that tool call and no prose."},
    {"role": "user", "content": "Call report_status with status set to ok."}
  ],
  "tools": [{"type": "function", "function": {
    "name": "report_status",
    "description": "Report the tool-calling canary status.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {"status": {"type": "string"}},
      "required": ["status"],
      "additionalProperties": false
    }
  }}],
  "tool_choice": {"type": "function", "function": {"name": "report_status"}},
  "max_tokens": 96,
  "temperature": 0
}
```

- **Mei 0.7.0 passes** this canary. The request uses the standard nested
  OpenAI forced `tool_choice` shape; Mei's `MessageMapping.additionalContext`
  reads both a top-level `name` and the nested `function.name`, so the
  template context receives `tool_choice = "required"` and
  `tool_choice_name = "report_status"` — the canary is actually pinned to
  `report_status`. Covered by `OpenAITypesTests`
  (`testCoCoreForcedToolCanaryPinsReportStatus`).
- Passing requires a real OpenAI-style `message.tool_calls` reply naming
  `report_status` with arguments that parse to exactly `{"status": "ok"}`.
- Pass → the model is advertised in `tool_call_models` and receives jobs that
  carry `tools`.

### 2. Structured-output canary

Also exactly as CoCore sends it (`structured_output_canary_body`):

```json
{
  "model": "<served model id>",
  "messages": [
    {"role": "system", "content": "You are a friendly assistant who always answers in two or three warm, conversational sentences."},
    {"role": "user", "content": "Say hello and tell me how you are doing today."}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "canary_status",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": false
      }
    }
  },
  "max_tokens": 64,
  "temperature": 0
}
```

- The prompt deliberately begs for prose so a server that **silently drops
  `response_format`** answers in sentences and **fails** the canary instead
  of passing by luck.
- **Passing is exact:** the whole `choices[0].message.content` (trimmed) must
  parse as a JSON object with exactly one key, `status`, whose value is the
  string `"ok"` — no prose around it, no extra keys, no embedded JSON inside
  sentences (CoCore's `structured_output_canary_passed`, mirrored in
  `CoCoreCanaryFixture`).
- **Mei 0.7.0 enforces the canary schema.** It decodes `response_format` on
  `/v1/chat/completions`, compiles the supported strict subset before
  generation, and enforces it token-by-token by constrained decoding (see
  [docs/OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md) §4). The exact
  canary request is the fixture `CoCoreCanary.structuredOutputBody(model:)`;
  buffered and streaming response paths are covered by
  `StructuredGenerationPipelineTests`, and the live probes are
  `MeiAcceptanceTests.testCoCoreStructuredOutputCanaryNonStreaming` /
  `…Streaming`.
- **Live evidence:** the full Mei acceptance suite passed **9/9 tests** against
  `mlx-community/Qwen3-4B-4bit`, including both structured-output probes; JSON
  content parsed to exactly `{"status":"ok"}` in buffered and streaming
  modes (raw whitespace is immaterial). Expanded numeric/array schema features
  (`minimum`/`maximum`, `multipleOf`, scalar enums, `minItems`/`maxItems`) also
  passed live buffered + SSE on the same checkpoint. The standalone CoCore
  attached-engine runner from merged commit
  `0151475bf8c98de10a64cab51c23a46dd84a8fe1` also reported readiness, passed
  the tool and structured canaries, and successfully proxied buffered and SSE
  structured jobs.
- **Per-checkpoint, not model-general:** the same exact canary was run
  against Qwen3.6 and Ornith using the 0.7.0 release-candidate feature build;
  all failed closed. Qwen3.6 text-only
  (`Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`, revision
  `693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56`) and its vision sibling
  (`mlx-community/Qwen3.6-35B-A3B-4bit`) returned HTTP 500 `engine_error`
  (`structured output constraint failed: generation stopped before a complete
  JSON value was produced`); Ornith returned HTTP 500 `engine_error`
  (`structured output constraint failed: no token in the vocabulary can
  advance the grammar`). Plain text still works on the Qwen3.6 text-only
  process, so normal serving is unaffected. A failed canary means CoCore
  simply does not advertise the model for schema jobs: these checkpoints must
  stay disabled until a fix lands and a live canary passes. Per-model
  evidence table: [docs/OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md) §8.
- **Still not proven:** full CoCore advisor connection/Register-frame
  capability readback for Mei 0.7.0. The attached-engine result is direct
  client/engine evidence, not a claim that advisor registration was observed.

## Streamed jobs and budgets

CoCore proxies every job as a streaming SSE request with
`stream_options.include_usage`, forwards `reasoning_content` and
`tool_calls` deltas on their channels, and applies the same budgets as its
subprocess engine: **300 s first-token** and **60 s idle**. A Mei cold start
can prefill for tens of seconds on the first real job after the canaries —
that is inside the budgets, but watch them for very long first prefills.

## One model per process

Mei serves exactly one checkpoint per process; running a second model means
a second `mei` instance on its own port, each with its own map entry
(e.g. port 8025 in CoCore's sample `engine-map`). CoCore's admission gate
also assumes single-flight servers: **one generation running + one queued
per model**, then an immediate typed `engine-busy` refusal (ceiling
advertised as `model_capacity: 2`); when every candidate serving a model is
saturated the job is refused up front with HTTP 503 `no-capacity`.

## Security model

- Loopback HTTP/1.1, **no TLS, no authentication** — by design on both
  sides. Mei's compatibility contract lists the missing auth/401/403/429/503
  paths as deliberate local-server deviations.
- CoCore rejects `https://` URLs (the client has no TLS) and warns loudly on
  non-loopback hosts: decrypted prompts would cross the network in the
  clear. Keep the map on `127.0.0.1`.
- A malformed map is a fault (`engine-map-invalid`), never a silent fallback
  to vllm-mlx.

## Troubleshooting checklist

| Symptom | Likely cause / fix |
|---|---|
| `attached-engine-unreachable` fault | Server not running, wrong port, or still cold-loading. Check `GET /v1/models` with curl; raise `COCORE_ATTACHED_READY_TIMEOUT` (default 300 s) for a slow cold start. |
| `engine-map-invalid` fault | Map URL must be the **server root** without `/v1` (a trailing `/v1` is rejected), `http://` only, one `model = url` per line. |
| Readiness passes but jobs 404 | Map URL has a path prefix; CoCore appends `/v1/...` itself, so the prefix must sit before `/v1` (e.g. `http://127.0.0.1:8024/llm`). |
| Model not advertised / model miss | Engine-map key must **exactly equal** the served id. Read it from `GET /v1/models` or the startup line `mei: no --served-model-id given; serving as …`; then `mei --served-model-id <that id>`. |
| `structured-output-unsupported` on schema jobs | Expected for Qwen3.6 and Ornith until their live canaries pass; only `mlx-community/Qwen3-4B-4bit` is verified for structured output in Mei 0.7.0. |
| `engine-busy` refusals | Gate is full (1 running + 1 queued). One model per process — start another `mei` on its own port for more concurrency. |
| Job times out waiting for first token | First-token budget is 300 s. Cold prefill of a 20k-token system+tools prompt can take ~50 s; if it exceeds the budget, the model is not actually ready — watch `mei:` startup logs. |
| Tool canary fails intermittently | `tool_choice` is now pinned to `report_status` in Mei (nested `function.name` extraction); verify with the regression test. Template-level forced-call fidelity still depends on the model. |
| Non-loopback warning in CoCore | The map points at another host; prompts travel in the clear. Keep `127.0.0.1`. |

## Cross-references

- Mei wire contract: **[docs/OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md)**
  (request/response surface, `tool_choice` mapping, streaming usage,
  deferred fields).
- CoCore feature: [graze-social/cocore PR #237](https://github.com/graze-social/cocore/pull/237),
  merged commit `0151475bf8c98de10a64cab51c23a46dd84a8fe1`;
  CoCore-side doc `docs/attached-engine.md`; client/canaries in
  `engines/openai_http.rs`, admission gate in `engines/admission.rs`,
  engine-map parsing in `engines/attached.rs`.
- Mei setup: [README.md](../README.md) (install, `mei pull`, quickstart).