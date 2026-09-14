# Mei project instructions

<!-- kiem -->
This repo is Kiem project `proj/mei`. Use `$HOME/bin/kiem todos` and
`$HOME/bin/kiem notes` for project state and decisions; do not turn this file
into a project log.

## Project

Mei is a native Swift/MLX inference server under
`/Users/tijs/projects/mei`. It loads local model bundles and exposes an
OpenAI-compatible `/v1` HTTP API for chat completions, streaming, and native
tool calls. It is integrated with the benchmark harness in
`/Users/tijs/projects/local-model-bench` through isolated launch/stop scripts,
ports, logs, model staging, and YAML configs.

## Source and tools

- `Sources/` — Mei server and inference implementation.
- `Tests/` — Swift unit and acceptance tests.
- `scripts/` — build/runtime helpers, including Metal resource preparation.
- `tools/` — model inspection, staging, and safetensors utilities.
- `docs/RELEASE-RUNBOOK.md` — release procedure; read it before a release.
- `Package.swift` / `Package.resolved` — SwiftPM dependencies; pin
  `vmlx-swift` to an exact revision and do not pull in the Osaurus app.

Use SwiftPM with the Xcode toolchain explicitly selected when needed:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --parallel
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release
```

Before changing the repo, inspect Git status and project instructions. Keep
model provenance, launch arguments, logs, and measured results reproducible;
never overwrite historical benchmark evidence or add secrets. Do not publish
or push without explicit permission.

During the benchmark experiments, CoCore remains off. Mei work must not
restart or restore it automatically.
