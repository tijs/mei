# Mei

Mei (梅, after Mei Long the sleeping dragon dinosaur) is a native Swift/MLX inference server for Apple Silicon, inspired by Osaurus's MLX stack and Dwarf Star.

## Goal

Build a narrow, correct, OpenAI-compatible serving engine around `osaurus-ai/vmlx-swift` and its MLX backend. Use the existing `local-model-bench` harness to measure correctness, latency, memory, useful agent throughput, and decode speed. The first target is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`; the performance goal is at least 30 decode tokens/second, comparable to the current llama.cpp setup, without the long-context serving collapse seen in Python wrappers.

## Source plan

The implementation plan is maintained in Kiem under the `proj/local_model_bench` project:

- `924ccddc-b4cb-438b-bccc-1cddd225e2b4` — native Swift MLX server on `vmlx-swift` (primary plan)
- `ac214007-76b2-473f-8fb0-89041c6260f0` — Qwen/llama.cpp speed and agentic-efficiency investigation (supporting benchmark methodology)

Retrieve the current plan with:

```bash
$HOME/bin/kiem show 924ccddc-b4cb-438b-bccc-1cddd225e2b4
```

## Working principles

- Use test-first red/green development for the server and acceptance oracle.
- Pin SwiftPM dependencies to exact revisions.
- Keep Mei's runtime, ports, logs, model staging, and shutdown isolated from other local backends.
- Preserve raw benchmark logs and historical `local-model-bench` results; never overwrite them.
- Do not claim a speed win without repeated, comparable benchmark evidence.
- Iterate through implementation, tests, review, fixes, and benchmark runs until the 30+ tok/s target is reached or a measured hardware/engine limit is documented.
