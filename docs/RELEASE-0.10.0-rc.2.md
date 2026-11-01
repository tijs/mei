# Mei 0.10.0-rc.2 preview

Release candidate with native tool/cache/context admission on the prepared
Groot pack. This is a preview, with separate `mei-preview` distribution.
Stable remains 0.9.0 until the recorded stable-promotion gates are complete.

This preview adds a payload-preserving text-bundle preparation tool for audited
Qwen3.5-family models. It removes unused vision/MTP tensors, makes supported
mixed-precision metadata explicit and rejects conflicting declarations. The
source stays unchanged; deterministic manifests record every retained tensor.
See [the preparation guide](../tools/export_qwen_text.md).

Automatic architecture policy names now report `qwen35-moe`, so derivatives
such as Groot are not labeled Ornith. Existing named model profiles, legacy
Swift aliases and numeric defaults remain unchanged. No extra CLI setting is
required. The vmlx pin remains f6510fc384bcddb5db9efc96e1dd5ab3afc49346.

Native probe result merging now protects its verdict from HTTP/detail-field
collisions. Exact-context tokenizer loading matches the benchmark, with bounded
metadata-selected overrides. The existing raw tool-history replay gate remains.
All 26 exporter/probe Python tests pass without model or network use.
Completed native gates are checkpointed atomically; interrupted runs stay marked
running and existing output is refused. Failed direct-tool checks now retain
the response envelope (including streamed reasoning), usage and finish reason.
Their budget is 1024 tokens including reasoning, matching the plain probe. The
original 256-token attempt ended both direct-tool generations at the cap with
no call and remains failed evidence; no validation requirement is relaxed.

The `groot-19b-a3b-text` pull/profile registry entry pins the published prepared
artifact, with the measured 256-token prefill step, anchors off and an 8192-token
default generation cap. Explicit user flags still win; sampling stays separate.

Qualification evidence is [published in local-model-bench](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-rc2-native-groot-tools1024-20261101):

- Offline Swift suite: 449 tests, two explicit skips, no failures. Python:
  26 exporter/probe tests pass, with Ruff and ty checks.
- Live source `e3e4bb04fbc038926ff890fb854aeb9b5c27f977`: all 13 native probes
  passed, including streaming/nonstreaming tool calls, tool-result history,
  cache reuse, exact 65,536-token prompt admission and over-cap rejection.
  The release source adds documentation only after that tested revision.
- Guarded attempt: 228.389 seconds, 45 normal-pressure samples, zero swap
  growth; all 16 model-file hashes match afterward. Owned server stopped and
  KV files removed after preserving logs.
- The first attempt failed both direct-tool checks at its 256-token limit.
  Both completed correctly in 285 tokens under the declared 1024-token probe
  budget. The original failed evidence remains; no task grade was rewritten.
- Packaged executable SHA256:
  `11b08e245c5ee19be603d215f3c572b8a02284f059a41646f3b6f7e9d821bc27`.
  Metal SHA256:
  `e87ac8aa940ace7dd1f9945c4b6a3238c3d94604391e0f36a0ac51259bd76f32`.
  Packaging must match these tested artifacts exactly.

The vmlx pin and Metal artifact are unchanged from the preceding preview. This
release does not change decode/cache arithmetic. Older benchmark scores remain
attributed to their measured source and binary, rather than to this release.

Stable promotion still requires the recorded cold determinism, one-variable
correctness and per-contender short/loaded throughput gates. A separate
five-second status GET timed out during the first run's long-context prefill;
responsiveness under load remains under investigation. Idle status and all
required native probes passed. No broad runtime, model-quality or speed claim
follows from these checks. Peixinho is a design; no actual app acceptance is
claimed. Flint remains retired from the benchmark roster.
