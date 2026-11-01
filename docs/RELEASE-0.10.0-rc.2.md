# Mei 0.10.0-rc.2 preview

Prepared release; native acceptance, packaging and publication are pending.
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

Full offline and live native evidence, exact binary/Metal hashes and remaining
qualification limits will be recorded before publication. Benchmark scores on
older pinned runtime builds are not acceptance evidence for this new binary.
