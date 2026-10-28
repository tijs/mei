# Mei 0.10.0-rc.1

This is an experimental preview, distributed separately from stable Mei 0.9.0.
It ships retained-prefix memory bounds, rotating-cache restore fixes, explicit
cache/weight policies, an early VLM eligibility check, and default-off Flint
standard-affine/JANGH2 support. VMLX is pinned to
`f6510fc384bcddb5db9efc96e1dd5ab3afc49346` on `mei/0.10.0-rc.1`.

## Install

```bash
brew install tijs/tap/mei-preview
mei-preview --version  # mei 0.10.0-rc.1
```

The preview formula exposes `mei-preview`; the stable `mei` formula remains at
0.9.0. Manual downloads contain `bin/mei`, its Metal libraries and provenance,
licenses/notices and runtime documentation. Model weights are not bundled.

See [runtime policies](RUNTIME-POLICIES.md), [Flint](FLINT.md) and
[JANGH2](FLINT-JANGH2.md) for explicit opt-ins. Defaults keep unsafe/compiled
experiments disabled. The launcher omits unset allocator/token/cache settings;
explicit zero remains an explicit choice.

## Evidence and limits

Offline Swift tests, policy-transition regressions, launcher dry-run checks and
binary/source packaging checks are required before publishing this preview.
Real-model proof is recorded separately with its exact source/config identity.
Earlier fixed-token Flint forensics found matching native/warm replay and
independent Python/Swift logits for matching prefill partitions; shared MLX and
quant kernels do not establish base-model versus quant attribution.

Qwen3.6-35B-A3B remains resource-blocked on the 32 GiB test host. The last
low-buffer attempt hit sustained macOS warning pressure during admission, with
no scored task. No full Qwen baseline, speed improvement, stable contender
qualification, or real Peixinho app acceptance is claimed.

Stable promotion still requires the release runbook's cold deterministic
one-variable correctness comparisons and short/loaded throughput samples
(n >= 10 for each contender), plus a resource-safe Qwen baseline. See Kiem
project `proj/mei` and the linked local-model-bench release plan for open work.
