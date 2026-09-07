# D3 IMPLEMENTED: one-arg Mei change makes the MLXPress cold tier engage (10.5-14.2 GiB advised); first memory measurement used the wrong instrument (2026-09-07)

# D3 IMPLEMENTED: MLXPress cold tier now engages in Mei — but the first measurement used the wrong instrument (2026-09-07)

## The change

`Sources/MeiCore/Engine.swift:90`, branch `research/d3-mlxpress` (worktree
`~/projects/mei-d3`, built to `mei-build-d3`, metallib provisioned):

```swift
loadConfiguration: LoadConfiguration(
    jangPress: .default,                      // was: init default .disabled
    useMmapSafetensors: config.useMmapSafetensors)
```

One argument, as the spec predicted. Build clean (349.7 s, zero errors).

## It works — the tier is live for the first time

Server logs from the sweep, which never appeared on any prior Mei build:

| MLXPRESS | evidence |
|---|---|
| `0` | no `[MLXPress] advised` line — correctly disabled |
| `70` | `[MLXPress] advised 11261706240 canonical mmap routed bytes cold (pct=70)` — **10.5 GiB** |
| `95` | `[MLXPress] advised 15288238080 … (pct=95)` — **14.2 GiB** |

Plus, at 70 and 95:
`[MLXPressMmapTier] index: descriptors=1757 parsed=120 stacked=120 whole=0
experts=10240 layers=40 routedBytes=16106127360`

It correctly identifies **10,240 experts across 40 layers, 15 GiB of routed
bytes** — matching the checkpoint arithmetic exactly (256 experts x 40 layers =
10,240; routed bank 16.9 GiB total). So the mechanism sees the right weights and
issues advice proportional to `compressPct`.

`MLXPRESS=N` is now live from the launcher, confirming the policy is
`.auto(envFallback: true)` as intended.

## First measurement showed nothing — and the instrument is why

Across `MLXPRESS=0 / 70 / 95`, every figure was **identical**:

- short decode 64.0–67.4 tok/s (no regression, which is the right no-harm result)
- 30k decode 40.4–40.6 tok/s
- `mei_memory_active_bytes` 18.793 GB, peak 18.853 GB (short)
- 30k active 21.770 GB, peak 23.268 GB
- `memory after load: active 19551131108` — byte-identical in all three

**Do not read that as "the lever does nothing."** Those numbers come from MLX's
own allocator (`Memory.snapshot()`), which tracks *allocations*, not *residency*.
The cold tier works by `madvise`-ing file-backed mmap pages so the **kernel** can
evict them; the MLX allocator still counts the mapping as allocated either way.
The instrument structurally cannot observe the effect being tested.

Second reason to expect a null result on that run: `forceMode` defaults to
`.soft` = `madvise(MADV_DONTNEED)`, which the vmlx source explicitly calls
"kernel HINTS, ignored when free RAM is plentiful". The host had several GB free,
so the kernel had no reason to act on them.

Also visible: `[MLXPressRouter] … symbol=missing` — the `advise_experts` symbol
is not present in this build, so router-aware advice could not engage regardless.
That matches the decision to leave `enableRouterAdvice` off, and means nothing in
this sweep depended on it.

## Re-measuring properly (running)

`/tmp/mlxpress_rss.sh` re-runs `MLXPRESS=0` vs `95`, each with
`MLXPRESS_FORCE_MODE=soft` and `force`, measuring **process RSS** at three
points: after load, after a 30k request, and after a 20 s quiesce window. RSS is
the metric that can actually move when the kernel drops file-backed pages.

Expected outcomes and how to read them:
- **RSS falls at 95/force but not at 0** → the lever works; then the question is
  what it costs in cold-fault latency on the next request.
- **RSS unchanged everywhere** → either the pages are not actually file-backed in
  this configuration, or `msync(MS_INVALIDATE)` is not reaching them. Next step
  would be `vmmap` on the routed regions rather than more sweeps.

## Honest status

**Implemented and verified engaged; effect on memory not yet demonstrated.** The
speed no-regression result (64–67 tok/s short, 40.5 at 30k, matching the 61.70
baseline and the prior 47.5–50.3 at 30k) is solid and worth having on its own —
enabling the tier costs nothing measurable in throughput. Whether it *buys*
anything is still open, and the first sweep does not answer it either way.

#proj/mei
