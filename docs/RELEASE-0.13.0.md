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
Studio account/connection assessment and combined release qualification are still
pending; local model/replay evidence must not be described as full runtime proof.

## Release gates still pending

Finish non-finite-logit recovery and K8V4 disposition in that order. Retain only
qualified useful changes. Release tests must use a different scratch directory
from the production build. Qualify the production executable and verify binary
and Metal hashes against the extracted published artifact, then verify Homebrew.
