# D1/D3 UNBLOCKED (spec): MLXPress axis-E expert residency is fully built in vmlx; Mei never passes the option; try env vars first (2026-09-06)

> **TITLE IS STALE — read the CORRECTION section at the end first.** The
> "try env vars first" advice in this note's title and in its "Concrete
> change required" section is WRONG: `LoadConfiguration.init` defaults
> `jangPress` to `.disabled`, which never consults the environment, so
> `MLXPRESS=70` does nothing on the current Mei binary. The fix is a
> one-argument Mei change (`jangPress: .default`), verified in source.


# D1/D3 UNBLOCKED (spec): the expert-residency mechanism is fully built in vmlx, Mei just never passes the option (2026-09-06)

The audit note recorded D1/D3 as blocked because `Engine.swift:90` constructs
`LoadConfiguration` with JangPress disabled. That is correct, but understates
how close it is: **the entire mechanism exists and is production-documented in
vmlx.** Mei is passing one field where it could pass two. Source reading only,
no GPU.

## What Mei currently does

`Sources/MeiCore/Engine.swift:87`:

```swift
let container = try await loadModelContainer(
    from: directory,
    using: #huggingFaceTokenizerLoader(),
    loadConfiguration: LoadConfiguration(useMmapSafetensors: config.useMmapSafetensors)
)
```

`LoadConfiguration` (vmlx `Libraries/MLXLMCommon/Cache/LoadConfiguration.swift`)
also carries **`jangPress: JangPressPolicy`** (:110) and
**`maxResidentBytes: ResidentCap`** (:117). Neither is set, so both default off.

## What the mechanism actually offers

`JangPressLoadOptions` (`Libraries/MLXLMCommon/Cache/JangPressLoadOptions.swift`)
— MLXPress axis E, the "cold-weight tier". This is precisely the expert-residency
lever hypothesised as F6/P5, already implemented:

| field | meaning | note |
|---|---|---|
| `enabled` | master switch | default **false**, opt-in |
| `compressPct` | 0–100, % of **routed-MoE weight mass** open to compaction during quiesce | `0` arms the failsafe controller without compacting; **`70` is documented as "the production-recommended value for tight hosts"**; `100` keeps only the top-k hot expert set pinned |
| `backend` | `.mmap` = file-backed, **"page-cache shared with MLX, zero RAM doubling"** | `.none` disables the routed-expert tier but still arms the embed/lm_head Zipfian tier |
| `forceMode` | `.soft` = `madvise(MADV_DONTNEED)` (kernel *hint*, ignored when RAM is plentiful) vs `.force` = `msync(MS_INVALIDATE)`, pages dropped immediately | `.force` documented for "memory-constrained hosts where eager reclaim is required" — that is exactly this 32 GB machine |
| `enablePrefetch` | pre-fault the hot tiles at arm time | default true; disabling causes "within-process drift at temperature 0" |
| `enableRouterAdvice` | router-aware `MADV_WILLNEED`/`MADV_DONTNEED` per expert id | default **false** because "the first CPU-readback implementation is correct but **too slow for production decode**" |

Note the `compressPct = 70` recommendation matches the "MLXPRESS=70 A/B" listed
as an unrun experiment back in the 2026-09-02 Ornith compiled-decode artifact.
That thread and D3 are the same thread.

## How this interacts with what we measured

`enableRouterAdvice` is the part D1 (routing-skew measurement) was meant to feed.
**My clean paired locality measurement says router-aware advice cannot help
speed**: addressing one expert eight times costs the same as eight distinct
experts (5.566 vs 5.531 ms) despite reading an eighth of the routed bytes, so the
gather kernel is insensitive to how much expert weight it touches. Combined with
vmlx's own note that the readback path is too slow for production decode,
**`enableRouterAdvice` should stay off.**

That leaves the genuinely promising configuration: **`enabled: true`,
`backend: .mmap`, `compressPct: 70`, `enableRouterAdvice: false`**, with
`forceMode` as the variable to sweep (`.soft` first, `.force` only if soft shows
no reclaim). This is a **memory** lever — it should be judged on resident
footprint and long-context headroom, not tok/s, and the speed gate is only there
to confirm it does not regress.

Why it matters: 93% of the model (16.9 of 18.17 GiB) is the routed expert bank,
and peak is 25.73 GB @ 65k / 28.19 GB @ 100k. Anything that keeps cold experts
out of resident memory buys context headroom directly — which, per the STATE OF
PLAY reframing, is the constraint that actually binds on this machine.

## Concrete change required in Mei

Two things, both small:

1. Extend `ServerConfig` with the MLXPress knobs (or read `MLXPRESS_*` env vars,
   which vmlx already honours — the cheaper first experiment, since it needs
   **no Mei source change at all**: `MLXPRESS_ROUTER_ADVICE=1` is named
   explicitly in the source as the experiment toggle, so sibling env vars very
   likely gate the rest. **Check that before writing any Swift** — it may make
   D3 a launcher-flag experiment rather than a code change.
2. If env vars turn out not to cover it: pass a populated `jangPress` policy in
   the `LoadConfiguration` at `Engine.swift:90`, and **hold the returned
   `JangPressRuntime` alive for the model session** — the API contract at
   `ModelFactory.swift:427-445` is explicit that the runtime owns the tiers and
   its deinit cancels the memory-pressure listener and quiesce timer. Dropping it
   silently disarms the feature. `loadModelContainer` as currently called does
   not return one, so this is a call-shape change, not just an argument.

## Revised D-phase ordering

- **D1 (routing-skew measurement): demote.** It was a prerequisite for
  router-aware advice, which the locality result and vmlx's own performance note
  both argue against. Not worth GPU time now.
- **D3: promote, and try the env-var route first.** It is the highest-ceiling
  memory lever and may not need a Mei change at all.
- **D2 (mixed 4/8-bit) unchanged** — still a checkpoint-side quality experiment
  that must go through the real bench suite.

## CORRECTION (same session) — the env-var route does NOT work; a one-line Mei change does

I suggested above trying `MLXPRESS_*` env vars first "since it needs no Mei
source change at all". **That is wrong and I verified it.**

`LoadConfiguration.init` (`Cache/LoadConfiguration.swift:219-220`) defaults
`jangPress: JangPressPolicy = .disabled`. Only the `.auto(envFallback: true)`
case ever consults the environment:

```
/// - ``auto(envFallback:)`` — pick at load time:
///   1. If `envFallback == true`, honor `MLXPRESS` env var …
///      integer `N` in `[0, 95]` → `.enabled(coldFraction: N/100.0)`
public static let `default`: JangPressPolicy = .auto(envFallback: true)
```

Because Mei calls `LoadConfiguration(useMmapSafetensors:)` and never passes
`jangPress`, the policy is `.disabled`, which short-circuits before any env
lookup. **Setting `MLXPRESS=70` on the current binary does nothing.** The audit
note's original "blocked on a Mei source change" framing was right; my
env-var shortcut was not.

The good news is how small the change is. `.default` is *already* the
production-recommended policy (auto, env-aware, enables at `coldFraction 0.70`
for routed MoE bundles whose raw bytes exceed 50% of physical memory — which
describes Ornith/Qwen3.6 exactly on a 32 GB host). So:

```swift
// Sources/MeiCore/Engine.swift:90
loadConfiguration: LoadConfiguration(
    jangPress: .default,                                  // <- add this
    useMmapSafetensors: config.useMmapSafetensors)
```

That single argument both enables the tier under its own auto-heuristic and
makes `MLXPRESS=0|70|N` live from the launcher, so the A/B becomes an env sweep
afterwards.

**The runtime-lifetime concern I raised is also resolved:** `ModelContext`
carries `jangPressRuntime` as a stored property (`ModelFactory.swift:84`), so
the container owns it and the "hold it alive" contract is satisfied by holding
the context Mei already holds. No call-shape change is needed after all — it
really is one argument.

Recommended first sweep once that lands: `MLXPRESS=0` (armed, no compaction) as
control, then `70`, then `95`, measuring **resident footprint and peak at 30k/65k**
first and tok/s only as a no-regression check.

#proj/mei