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

## Scheduler verification before PRs

For a PR touching `Sources/MeiCore/Scheduling.swift` or its scheduler checks,
run `bash scripts/check_scheduler.sh`, then ask `Run a Quint review of this
branch against origin/main` (substitute an explicitly specified base). Follow
the installed `quint-studio` skill's review workflow against that merge-base. Scope the assessment to changed components and the scheduler
contract in `verification/scheduler/README.md`; do not expand it to cache/GPU
behavior. Studio must be running, signed in, and connected. Record failed checks,
a missing/stale model, an unavailable connection, and uncovered behavior in the
PR description; an agent policy is not a GitHub-enforced required check.

This policy authorizes the local review workflow and updates to the existing
scheduler model. First-time modeling of any other component still requires an
explicit request. It does not authorize commits, pushes, posting, or CI setup.
