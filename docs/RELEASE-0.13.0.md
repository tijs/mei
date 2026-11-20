# Mei 0.13.0 — release preparation

This release is being qualified. The stable release remains 0.12.0 until the
combined runtime, production package and Homebrew checks are complete.

## Scheduler verification

Add a repository-owned bounded Quint lifecycle model and generated replay tests
against the actual Swift scheduler. Cover one/two slots, bounded queues,
interactive-before-background/FIFO grants, timeout, queued and active cancellation,
cancellation between grant and operation start, failure and cleanup ownership.
The public scheduler initializer retains its clock and admission defaults.

The model-free check command is `bash scripts/check_scheduler.sh`. See
`verification/scheduler/README.md` for bounds, trace regeneration and exclusions.
The 19 existing scheduler tests and generated replay of 33 Quint traces passed
both in the model-free package and inside the complete Mei package using the
published vMLX pin `3c2823b7`. The complete package check selects only those 20
scheduler tests; it does not run the full offline or real-model suite.

Quint Studio is connected. Its component map groups the scheduler with HTTP,
inference, and cache code. The repository model and replay keep verification
scoped to the scheduler. Combined release qualification remains pending.

## Release gates still pending

Finish non-finite-logit recovery and K8V4 disposition in that order. Retain only
qualified useful changes. Release tests must use a different scratch directory
from the production build. Qualify the production executable and verify binary
and Metal hashes against the extracted published artifact, then verify Homebrew.
