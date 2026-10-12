# Mei 0.8.0 — bounded single-flight scheduling

**Feature release — 2026-10-12**

Mei 0.8.0 adds bounded admission scheduling in front of the strict
single-flight execution slot: a bounded local queue with interactive and
background priority, overload refusals, and end-to-end cancellation.
Execution stays one generation at a time; overlapping generation is not
enabled.

## What shipped

- A bounded waiting queue: `--max-queue-depth` (default 32; 0 = no queue)
  caps how many requests may wait for the execution slot — an over-limit
  arrival gets HTTP 429 `rate_limit_error` / `queue_full` without being
  enqueued — and `--queue-wait-ms` (default 600000; 0 = unlimited) is the
  wait budget — an expired wait gets HTTP 503 / `queue_timeout` without ever
  starting.
- `mei_priority` request classes: `"interactive"` (default when absent) and
  `"background"`; queued interactive requests outrank queued background
  requests, FIFO within a class, and priority never preempts a running
  generation. A queued background request can be overtaken indefinitely by
  continual interactive arrivals — the only bound is `--queue-wait-ms`. The
  unqualified `priority` field used by other OpenAI-compatible servers is
  ignored.
- End-to-end cancellation: a client disconnect removes a queued request
  without running it and stops a running generation, with the execution slot
  released exactly once. A cancelled streaming run ends without a `.finish`
  event, and the request log records `finish: "cancelled"`.
- Deferred SSE headers: streaming responses no longer send their headers
  before the engine produces its first event, so failures before that point
  (admission refusals, decode errors, pre-generation engine failures) are
  delivered as real HTTP statuses instead of an empty 200. Client header
  timeouts must cover queue wait plus time-to-first-token.
- Scheduler observability: `GET /v1/mei/status` gains a `scheduler` block
  (queue depth and priority mix, overload/cancellation counters, slot
  releases, queue-wait and cancellation-to-slot-release timings), and the
  `--request-log` JSONL gains `priority` and `queue_wait_ms` per run.

Full contract: [docs/INFERENCE-SCHEDULING.md](INFERENCE-SCHEDULING.md).

## Verification

The scheduling slice passed the model-free suites and a 10-check live HTTP
integration run against the packaged 0.8.0 binary on
`mlx-community/Qwen3-4B-4bit`. The integration exercised real queue-full HTTP
429-before-SSE behavior, invalid priority errors, standard `priority` ignoring,
disconnect cancellation, and interactive-before-background request-log
readback. This is scheduler/HTTP lifecycle evidence on one checkpoint, not a
model-quality claim and not evidence for overlapping generation.

Structured-output model support is carried from 0.7.1: the runtime source at
commit `ea5a67a` (included in 0.7.1 and 0.8.0) passed the exact buffered and
streaming CoCore canaries on Qwen3.6 text-only, Qwen3.6 vision (text-only
requests), and aligned Ornith, as well as the earlier Qwen3-4B checkpoint.
Image-conditioned structured output on Qwen3.6 vision remains unverified, and
Qwen3-8B has not been retested after the fix.

See the per-checkpoint matrix in
[OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md#structured-output-live-model-matrix)
and [CoCore integration notes](COCORE.md).

## Install

```bash
brew update
brew upgrade tijs/tap/mei
mei --version  # mei 0.8.0
```

Manual Apple Silicon release asset: `mei-0.8.0-macos-arm64.tar.gz` and its
`.sha256` companion on the GitHub release page.
