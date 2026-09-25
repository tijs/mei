# Mei vMLX fork

Mei consumes the public fork [`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift)
through SwiftPM. The fork keeps [`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift)
as its `upstream` parent; the local checkout lives at
`~/projects/vmlx-swift` and has:

```text
origin   https://github.com/tijs/vmlx-swift.git
upstream https://github.com/osaurus-ai/vmlx-swift.git
```

Mei's `Package.swift` and `Package.resolved` pin the fork's integrated `main`
revision. A fresh Mei build retrieves the fork through SwiftPM. There is no
normal build-time patch application and no dependency on a local SwiftPM
repository cache being pre-populated.

## Current pin

Mei 0.6.0 pins **`fef563a5`** (`fef563a55b4f22d4530b3439c1edb233cfc44a8f`),
the pushed `main` of `tijs/vmlx-swift`. The fork `main` integrates upstream
history through `e07bd67b` (upstream `main` at the last sync) plus the
fork-side commits below.

Since the previous Mei pin (`44461ffd`, the 0.4.2-era revision), the fork
`main` advanced 151 commits: **133 upstream commits** (integrated by two sync
merges, `d38d3c50` and the pin tip `fef563a5`) and **18 fork-side commits**.
The pinned MLX C++ submodule (`Source/Cmlx/mlx`) moved from the 0.31.1-era
revision to **0.32.2** (`c0a51a08`).

## The 18 fork-side commits since the 0.4.2-era pin

All pushed, each with a focused message, each independently cherry-pickable.
They are Mei-maintained changes; none is represented as an upstream-accepted
change.

| Commit | Scope |
|---|---|
| `a6694f8e` | `BaseConfiguration` accepts `quantization_config` as an alias for `quantization` |
| `78c0b454` | gate `prism_hadamard_qwen35` behind a default-off portability flag |
| `fb0b80eb` | Prism-Hadamard transform modules + `Load.swift` `bonsaiTransform:` seam |
| `47f838a4` | harden Prism-Hadamard manifest/sign/shape validation + install-path tests |
| `04b8959f` | validate both packed dims against the leaf; consume keys transactionally |
| `82f420f4` | later-entry transactional-consumption test fails in-loop, not during resolve |
| `379d529b` | pinned-runtime FWHT parity gate for the block-1024 Prism-Hadamard seam |
| `54f017ea` | wire the validated Prism-Hadamard plan into the LLM factory/load path |
| `f29587fb` | decode the pinned pack's nested `text_config` in the factory/load handoff |
| `6a81ca32` | flatten `language_model` wrapper; drop vision sidecar for the text-only Bonsai load |
| `12121099` | break chained cache/store-boundary trace interpolation into lets |
| `46933d60` | fail closed on integer packed scales/biases dtype |
| `559b6ee6` | pin mlx submodule to the default-off head_dim-256 full-SDPA admission experiment |
| `746d1c47` | exclude SDPA admission doctest from the Cmlx target; pin the bd=256 full kernel source |
| `8c7df5bd` | pin mlx default-off bd=256 remedy tile; mirror steel kernels |
| `d38d3c50` | merge: upstream sync into `integrate/main-upstream-sync` |
| `f1c428d1` | integrate: merge the validated Bonsai 2 Prism-Hadamard work into fork `main` |
| `fef563a5` | merge: upstream sync into `integrate/main-upstream-sync` (fork `main` tip) |

The `bonsai2`-prefixed commits (14 of the 18) are the Bonsai 2 Prism-Hadamard
work: a default-off `prism_hadamard_qwen35` portability gate plus the
transform modules, validation, load-path wiring and mlx-submodule pins that
make it reproducible. They ship in Mei 0.6.0 gated off by default; nothing
changes for the models Mei serves unless the flag is explicitly enabled.

## Earlier Mei-maintained commits (also in the fork)

The 13 fork-side commits predating the 0.4.2-era pin remain in the fork's
history:

- **The six former patch-queue commits:** `ae1783be` (quantized rotating KV
  cache), `1326d803` (quantized rotating disk store), `ab09d363` (long-prompt
  compiled-decode guard), `9b8e93b1` (bounded KV-window probe),
  `91fed8be` (explicit SSM anchor boundaries), `318a4e68`
  (gen-suffix-stripped cross-turn boundary for rotating/companion topologies,
  formerly local-only — **now pushed**; a pure source build needs no local
  SwiftPM edit).
- **The three Laguna model fixes:** `1cd53409` (unwrap `language_model.`
  prefix in sanitize), `b9dd75f4` (normalize routed gate layout before load
  dequantization), `f2a7dc0f` (compile routed SwitchGLU separated decode for
  the affine S-2.1 XS topology).
- **The three cache boundary-capture fixes:** `654eb455` (stop re-deriving
  durable boundaries on MambaCache hybrids), `4a1069a4` (capture prefix
  boundaries correctly, including after a restore), `44461ffd` (warn when a
  restore matches a stable boundary instead of its seed).
- **The 2026-09-07 upstream sync merge** `e37d1d59`.

## Upstream history

The 133 upstream commits integrated since the 0.4.2-era pin are the
upstream-accepted history of `osaurus-ai/vmlx-swift` — notably Spark 2.5
prefill fusion, rotating-cache boundary snapshot reuse, MiMo V2.6,
ModernBERT and Linux CI builds. They are **not** Mei-validated: no model Mei
serves has a profile for those additions, and Mei's acceptance matrix is
unchanged. A future re-pin must re-run Mei's acceptance probes against the
new pin before release.

## Normal workflow

For Mei work:

```bash
cd ~/projects/mei
git pull --ff-only origin main
swift package resolve
swift test
```

For vMLX work:

```bash
cd ~/projects/vmlx-swift
git fetch origin upstream --prune
git switch main
git pull --ff-only origin main
# Make and test a focused change.
git push origin main
```

Keep `upstream` available for comparison and future synchronization. Do not
rewrite fork history or force-push `main` as a shortcut.

## Preparing upstream PRs

Before proposing a change upstream, compare the fork with the latest parent:

```bash
cd ~/projects/vmlx-swift
git fetch upstream --prune
git log --oneline upstream/main..main
git diff --check upstream/main...main
```

The commits can be offered separately, in dependency order. The former
patch-queue commits `ae1783be`/`1326d803` form the rotating-KV/storage pair;
`ab09d363` is independent; `9b8e93b1` depends on the generation parameter
additions from `ab09d363`; `91fed8be` adds the SSM boundary path
independently of the bounded-window probe. The `bonsai2` Prism-Hadamard
series is a self-contained, default-off feature set. Any upstream PR must
include focused regression tests and re-run Mei's acceptance matrix before
Mei advances its pin.

## Updating Mei's pin

After a verified fork change:

1. Push the fork commit to `tijs/vmlx-swift`.
2. Update the revision in `Package.swift`.
3. Run `swift package resolve` so `Package.resolved` records the fork URL and revision.
4. Run the focused tests, release build, and relevant long-context/agentic acceptance probes.
5. Commit the Mei pin update separately from unrelated source changes.

The old `patches/` directory is intentionally not part of the normal workflow;
the fork commit history is now the source of truth for these engine changes.