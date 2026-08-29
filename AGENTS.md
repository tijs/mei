# Mei project instructions

Mei is a new native Swift/MLX inference server project under `/Users/tijs/projects/mei`.

- The primary plan is Kiem note `924ccddc-b4cb-438b-bccc-1cddd225e2b4`; read it before implementation with `$HOME/bin/kiem show 924ccddc-b4cb-438b-bccc-1cddd225e2b4`.
- Supporting benchmark methodology is in Kiem note `ac214007-76b2-473f-8fb0-89041c6260f0` and `/Users/tijs/projects/local-model-bench/AGENTS.md`.
- The primary MVP model is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`.
- The performance goal is at least 30 decode tokens/second on the Sulaco Apple Silicon machine, while retaining correctness and agentic tool-use reliability.
- Use SwiftPM and pin `vmlx-swift` to an exact revision. Do not pull in the full Osaurus application.
- Build the acceptance oracle before the server, keep it red until the corresponding server behavior exists, and test streaming/non-streaming parity and tool calls.
- Integrate with `/Users/tijs/projects/local-model-bench` through an isolated engine backend: dedicated launch/stop scripts, port, logs, model staging, and benchmark configs.
- Never modify or overwrite historical benchmark results. Keep each experiment's configuration, logs, and measured results reproducible.
- Do not add secrets to the repository or publish anything without explicit user permission.
- Before modifying existing repositories, inspect their status and instructions. Run relevant tests and benchmarks before reporting completion.

<!-- kiem -->
This repo is Kiem project `proj/mei`. Run `kiem todos` / `kiem notes` for project state, and record progress with `kiem note add` / `kiem todo check`.
