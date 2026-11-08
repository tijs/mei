# Mei vMLX fork

Mei consumes the public fork [`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift)
through SwiftPM. The fork keeps [`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift)
as its `upstream` parent; the local checkout lives at
`~/projects/vmlx-swift` and has:

```text
origin   https://github.com/tijs/vmlx-swift.git
upstream https://github.com/osaurus-ai/vmlx-swift.git
```

Mei's `Package.swift` and `Package.resolved` pin an exact published fork
revision. A fresh Mei build retrieves the fork through SwiftPM. There is no
normal build-time patch application and no dependency on a local SwiftPM
repository cache being pre-populated.

## Published dependency pin: 0.11.1

Mei 0.11.1 pins vMLX to `e280bd2b87dff614991cb6f9213adaf797eb2db4`.
It includes Apple MLX core main through `fc344c98` via the pinned fork
revision `b06cc4dc5af50e281b1b6309b2f7b79719fa9427`, with the C bridge
at `1169bf6796e05e5857f8d40af5a66b361df3ae5b`. These immutable revisions
keep the qualified M1 Max attention selector and disable graph debug checks
in both C++ targets in Release mode. See [the release notes](RELEASE-0.11.1.md)
for model coverage.

## Previous dependency pin: 0.11.0

[Mei 0.11.0](https://github.com/tijs/mei/releases/tag/v0.11.0) pins vMLX
`3cb7240a39d92475345c6c1d0c9057da31a2f289`. That revision includes Osaurus
vMLX through `1bcd74ad` and Apple MLX Swift through `5f117a2e` (0.33.0
plus three commits). Its MLX core pin is
`49631e976f01ba148abf4b041ca61ff2f1d5bfdd`, based on Osaurus core
`3b39b5e7` and Apple MLX 0.32.4. The core pin also carries the
qualified M1 Max attention selector. These are release pins; they do not
track the moving upstream branches at build time.

The release preserves Mei's bounded prefix storage, raw-text generation and
request-local constrained decoding while adapting to upstream materialization,
host-read and eval-lock ownership.

Upstream's owned weight storage replaces the earlier mmap path. Mei exposes a
bounded residency request with explicit CLI/library/status behavior; see
[runtime policies](RUNTIME-POLICIES.md) for migration and limits. Existing
performance claims for mmap/aligned loading describe earlier releases.

Qwen tool-prefix fixes apply to Qwen/Ornith/Groot. Gemma template, quoted-XML
tool parsing and activation-dtype updates are inherited for future qualification
of the queued text-only Gemma pack. Source support does not establish admission
of a new checkpoint or image-conditioned behavior.

On macOS, backend `RuntimeDiagnostics` uses the private unified log by default.
For terminal or owned benchmark capture, set `VMLX_DIAGNOSTICS_STDERR=1` alongside
the selected trace flag, for example `MLXPRESS_GENERATION_PROFILE=1`. The first
flag selects the output destination; the second enables phase counters. Earlier
versions wrote those counters directly to stderr. The default preserves
best-effort logging for hosts with disconnected stderr; explicit capture uses
the existing supported opt-in. Profiled timings remain diagnostic observations.

Several upstream features need a separate integration gate:

- Mei retains its own tested strict schema compiler and token mask. Upstream's
  native schema path has different tokenizer, reasoning/tool and completion
  contracts; switching implementations requires contract and real-model parity.
- Concurrent checkpoint loading and byte-level load progress do not yet preserve
  Mei's complete key-filtering, owned-storage and residency accounting contracts.
  They are not advertised as active Mei features.
- MTP/DFlash and newly registered model families require exact checkpoint/head
  provenance, greedy correctness and measured short/long-context benefit before
  enabling a serving profile.

The [qualification and contribution index](../artifacts/upstream-refresh-20261106/README.md)
records exact artifacts, negative/positive regressions and fourteen small issue-first
upstream PRs. The overlap audit excluded snapshot guards and quantization aliases
already upstream. Fork-specific ownership policy remains separate where an
independent upstream benefit has not been established. PR publication does not
mean upstream acceptance.

## Published dependency pin: 0.10

Mei 0.10 pins vmlx **`9db62ce10416db4a33ff0108dca6fc00bdadfff7`**,
which retains the phase-flag lookup optimization and pins MLX core
**`bbdcd4d1431e64a4ae44f353cea6ec0be72f4bee`** from
[`tijs/mlx`](https://github.com/tijs/mlx), based on `osaurus-ai/mlx` at
`c0a51a085b9252dc6469afe1aed772b1f538c9b0`.

The core change admits recent buffer shapes within a bounded allocator pool
and releases oversized returns. Four real-Metal regressions reproduce the old
behavior and pass through the new published dependency chain. They do not
establish a model speed improvement; native/context/numerical and unprofiled
throughput qualification remain required before release. Exact application,
dependency, binary and Metal identities belong in the release evidence.

The shared optional cache-store guard also vetoes nonempty stores when the
kernel reports warning or critical host pressure. It preserves normal-pressure
byte budgeting and reclaimed-mmap footprint accounting, adds no operator
setting, and logs why a checkpoint was skipped. A pressure sample is not a RAM
reservation; this guard does not establish the cause of a prefill peak or
replace real-model qualification.

The earlier pins and fork history below are preserved as historical evidence.

## Historical pins: 0.6.1 and 0.7.0

The released Mei 0.6.1 build pinned **`fef563a5`**
(`fef563a55b4f22d4530b3439c1edb233cfc44a8f`). Mei 0.7.0 pins
**`633fe166`** (`633fe166630ef04310aea7d5a1795555ab32970d`), the pushed
`main` of `tijs/vmlx-swift`. The new commit adds the public request
logit-processor seam used by Mei's token-level structured-output decoding. The
fork `main` integrates upstream history through `e07bd67b` (upstream `main` at
the last sync) plus the fork-side commits below.

Since the previous Mei pin (`44461ffd`, the 0.4.2-era revision), the fork
`main` advanced 152 commits: **133 upstream commits** (integrated by two sync
merges, `d38d3c50` and `fef563a5`) and **19 fork-side commits**.
The pinned MLX C++ submodule (`Source/Cmlx/mlx`) moved from the 0.31.1-era
revision to **0.32.2** (`c0a51a08`).

## The 19 fork-side commits since the 0.4.2-era pin

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
| `fef563a5` | merge: upstream sync into `integrate/main-upstream-sync` (fork `main` tip at the 0.6.1 release) |
| `633fe166` | add request-scoped logit-processor composition to `TokenIterator`, with focused ordering/copy/default-path tests |

The `bonsai2`-prefixed commits (14 of the 19) are the Bonsai 2 Prism-Hadamard
work: a default-off `prism_hadamard_qwen35` portability gate plus the
transform modules, validation, load-path wiring and mlx-submodule pins that
make it reproducible. They ship gated off by default, unchanged from Mei
0.6.0; nothing changes for the models Mei serves unless the flag is
explicitly enabled.

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
