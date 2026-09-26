# Changelog

All notable changes to Mei are documented here.

## [0.6.1] - 2026-09-26

A hardening patch on top of 0.6.0, with no dependency change — the vmlx pin
stays `fef563a5`. Metallib provisioning and release packaging stop reusing a
Metal library built for a different MLX version and prefer the binary's own
source-built Cmlx kernels; streaming answers leave the post-token
finalization tail off the client's critical path; the vmlx decode-policy
switches are visible in the startup log; and the reference launcher script
starts a server again. No inference, admission, or quality behaviour is
claimed to change.

### Fixed

- **Streaming answers no longer wait for post-token finalization.** vmlx
  yields `.info` as soon as generation completes and then keeps working — the
  GPU drain, the cache store and the advisor drain run before the stream
  ends. Mei answered only on stream termination, so every streaming turn paid
  that tail before the client saw its finish frame and usage. The finish
  frame is now emitted at `.info` while the single-flight lock is still held
  through producer end — the next request still cannot race the cache store —
  and the no-`.info` fallback keeps the previous behaviour. The run log gains
  `finalize_ms` (from `.info` to producer end); on a truncated 12-layer proxy
  (self-comparison only — not a product benchmark) the client-visible
  completion point moved 27-35 ms earlier, with byte-identical generated
  text, `finish_reason` and usage. The completion blocks in the streaming,
  non-streaming chat and `/v1/completions` paths are consolidated into one
  helper (commit `8529b37`).
- **The reference launcher could not start a server at all.**
  `scripts/start_mei_server.sh` passed the removed
  `--optimization-profile auto|generic|ornith` flag, which both shipped
  binaries reject outright, so every launch through the script exited 2
  before loading a model. `MEI_MODEL_PROFILE=<name>` now feeds
  `--model-profile` (omitted when unset, so the server runs on architecture
  defaults and says so at startup), and the obsolete
  `MEI_OPTIMIZATION_PROFILE` fails fast naming its replacement. The benchmark
  path never noticed — local-model-bench keeps its own (correct) copy of this
  launch logic. New `scripts/test_start_mei_server_flags.sh` checks the
  removed flag is gone, the guard fires, and every flag the script hands the
  binary is accepted by that binary (commit `0abbc5a`).
- **Stale `mlx.metallib` reuse across MLX version changes.** Build
  directories are reused across vmlx pins, and a colocated library was
  re-accepted on a structural check alone — even when its provenance named a
  different vendored MLX (e.g. a 0.31.1-era library under the 0.32.2
  runtime). Provisioning now reads the `vendored_mlx` recorded in the
  provenance sidecar and re-provisions when it disagrees with the derived
  runtime version (commit `25478fe`).

### Changed

- **Metallib provisioning and packaging prefer the source-built Cmlx
  kernels.** With Xcode's Metal Toolchain installed, SwiftPM's build emits the
  pinned checkout's own `mlx-swift_Cmlx.bundle`; provisioning copies that
  build product ahead of any cached or wheel library. An explicit
  `MEI_METALLIB_BUILD_DIR` makes a missing bundle fatal instead of silently
  selecting a wheel, and packaging sets it automatically to the selected
  binary's build directory — release packaging is source-first and
  fail-closed, never an unrelated wheel discovered on the packaging machine.
  The provenance sidecar records the source path, SHA-256, selected checkout
  revision (`vmlx_checkout_revision`) and vendored MLX version (commit
  `83087c3`).
- **Decode-policy switch values are logged at startup.** A vmlx re-pin can
  turn an upstream default on under a model family while every local setting
  stays silent — exactly how the 0.6.0 regression hid (upstream #455 made
  compiled routed-MoE decode the default for text-only `qwen3_5_moe`, an
  experiment this repo had already measured at -9.7% short decode and
  rejected). Startup now reports every decode-policy switch with its
  effective value and source (`operator`, `mei-profile`, or
  `upstream-default`); the operator's environment is snapshotted before any
  `setenv`, so the two sources stay distinguishable. The release runbook
  gains the matching re-pin step: a throughput gate, not only a correctness
  gate (commit `0772ba3`).

### Docs

- The source-first provisioning path is documented in `docs/DEVELOPMENT.md`
  and the release runbook (Metal Toolchain prerequisite, strict packaging)
  (commit `83087c3`), and the active docs that still used the removed
  `--optimization-profile` spelling are fixed: `docs/ARCHITECTURE.md`'s
  "Optimization profiles" section and the Qwen3.6/Nemotron launch examples in
  `docs/MODELS.md`. Historical release notes keep their original text
  (commit `0abbc5a`).

## [0.6.0] - 2026-09-25

The CoCore attached-engine integration is fixed, every model gets the
alignment warning, the release tooling stopped packaging stale binaries, and
the vmlx engine advances to the pushed fork `main`.

### Fixed

- **Nested OpenAI `tool_choice` function names are honored.** The OpenAI
  forced form `{"type": "function", "function": {"name": ...}}` puts the name
  inside `function`; Mei only read a top-level `name`, so a client forcing a
  tool that way — CoCore's attached-engine canary included — never pinned the
  call (`tool_choice` became `required` with no name to bind to), and the
  model was not advertised for tool-calling jobs. The nested `function.name`
  is now read when no top-level `name` is present, so the forced canary is
  actually pinned to `report_status` and passes (commit `2be7975`, covered by
  `OpenAITypesTests.testCoCoreForcedToolCanaryPinsReportStatus`).
  `response_format` remains **unsupported**: the structured-output canary
  still fails by design, and CoCore does not advertise schema jobs for Mei.
- **Alignment warnings for every model, not only declared profiles.** The
  misaligned-weights warning was gated behind a per-profile
  `requiresAlignedWeights` flag that not every profile set, so a community
  quant crashed with no explanation when realignment would not fit — measured
  up to a hard Metal OOM on a 32 GB machine (18.2 GB of realignment on top of
  20 GB of weights). Startup now checks alignment for every model and, for
  profiles that declare it, still says which repack the settings were
  measured on (commit `50fd9ab`).

### Changed

- **vmlx-swift re-pinned** `44461ffd` -> `fef563a5`, the pushed `main` of
  [`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift). The fork `main`
  integrated two upstream syncs plus the Bonsai 2 Prism-Hadamard work
  (default-off gated, integrated by `f1c428d1`) and the `quantization_config`
  alias — 151 commits past the old pin (133 upstream, 18 fork-side), with the
  MLX C++ submodule at 0.32.2 (was 0.31.1 at the 0.4.2-era pin). The upstream
  side brings Spark 2.5 prefill fusion, rotating-cache boundary snapshot
  reuse, MiMo V2.6, ModernBERT and Linux CI builds. That is
  **upstream-integrated, not Mei-validated**: no model Mei serves has a
  profile for those additions, and Mei's acceptance matrix is unchanged. This
  is a different line than the sync `23551729` prepared and held back during
  0.5.0 (that commit is not an ancestor of the new pin). Every
  Mei-maintained fork commit — including the former local-only `318a4e68` —
  is now in the pushed fork `main`, so a pure source build of this tag needs
  no local SwiftPM edit.

### Release tooling

- `scripts/package_release.sh` refuses to package a binary older than the
  sources it ships (commit `a860630`): during 0.5.0, `--skip-build` packaged
  a previous-day binary that contained none of the release and printed
  "packaging OK".
- The release runbook now asserts instead of observing (commit `68decd5`):
  `git merge-base --is-ancestor v<tag> main` finishes the merge-back step,
  because `git checkout main` can fail inside a release worktree and the
  subsequent merge reports "Already up to date" while pushing an untouched
  ref — three false successes 0.5.0 hit.

### Docs

- README rewritten around the models actually served: it names the model to
  take (Qwen3.6 35B-A3B text-only), stops calling Ornith the default, and
  cuts stale history (commits `5d5ab55`, `b766dad`, `fd39794`).
- The OpenAI-compatible Chat Completions contract is frozen in
  `docs/OPENAI-COMPATIBILITY.md` and the CoCore attached-engine integration
  documented in `docs/COCORE.md` (commits `e795592`, `2be7975`, `bdf99d8`).
- `AGENTS.md` is trimmed to a project guide (commit `950e8c2`).

## [0.5.0] - 2026-09-12

Pick your model by name and get the settings we measured for it.

### `--model-profile` — one flag selects everything for a supported model

```
mei --model-dir DIR --model-profile qwen3.6-35b-a3b-text
```

The profile carries the chunked-prefill step, cross-conversation anchor
boundaries, the generation cap and the architecture handling together, because
those have different optima per model. Every value is traceable to a
measurement recorded in the profile's own `provenance` string.

Profiles also pin the HuggingFace repo and an exact revision — not `main` —
because the settings are calibrated to one artifact, and an upstream re-export
that silently invalidates them would be miserable to debug later.

### Anchors ship on for one model of three

Cross-conversation prefix anchors cut the cold prefill of the shared
system+tools preamble. Measured on every supported model, same protocol:

| model | prefill/turn | quality cost | anchors |
|---|---|---|---|
| Qwen3.6 text-only | 5.20 → 3.04 s (−42%) | none, across all 25 tasks | **on** |
| Ornith 1.5 | 4.64 → 2.82 s (−39%) | 2 stable tasks | off |
| Qwen3.6 vision | 4.36 → 3.40 s (−22%) | 1 stable task, both pairs | off |

Every figure above is measured cold, with each arm on its own cleared KV cache,
and scored against a noise floor derived from repeats of the *same* config.

An earlier version of this table said anchors were free on both Qwen3.6
variants. That came from A/Bs whose two arms shared one never-cleared cache —
which is precisely the defect that produces "no difference between the arms".
Re-measured, the vision variant costs `hermes_ops-targeted-edit` in two
independent pairs, deterministically: 272 completion tokens and a successful
`patch` call without anchors, 2401 tokens and an empty tool-call list with them.
Its prefill benefit is also less than half what was claimed.

Where anchors hurt, they do not produce a worse answer — they stop the model
calling tools at all. Ornith and Qwen3.6 vision both fail that way. The
text-only build of the same checkpoint, with the vision tower removed, shows no
cost on any task, and has the largest benefit of the three.

All three change what the model writes — chunked prefill is not bit-reproducible
on a GPU, so splitting it to capture a boundary shifts the last bits. Only
Ornith pays a task for it, which is why this is a per-model setting and not a
global default.

### Prefill step, for real this time

0.4.1's notes described choosing the prefill step from the device's recommended
working set. That never shipped (see the correction under 0.4.1). It ships now,
in a different shape.

A profile states the step it was **measured** at, so naming a model gives the
same answers on every machine that can afford it. Where a device cannot, the
step is reduced to 512 **and the reduction is printed**, naming what was given
up and how to override it:

```
mei: WARNING prefill step reduced 1024 -> 512. This device reports a
recommended working set below 26000000000 bytes, so the 1024-token step this
model profile was measured at does not fit.
```

Announcing it matters more than it might look. Chunked prefill is not
answer-invariant on this architecture, so the two steps are two different
configurations, not a fast one and a slow one — an operator comparing output
against our published numbers needs to know which they are running. 0.4.1's
design chose silently.

`ornith-1.5-35b-a3b` also now pins its step explicitly at 1024. It had been left
unset, which meant it inherited the architecture default of 512 — while every
Ornith measurement this project publishes, including the ones quoted in that
profile's own description, was taken at 1024.

### Why you name the model instead of Mei detecting it

The supported models are not distinguishable from their metadata. Ornith 1.5
and Qwen3.6 text-only report the same `model_type`, the same architecture and
the same layer topology. Telling them apart would mean keying behaviour off
incidental fields like `transformers_version`, which breaks the moment an
upstream re-export changes them.

Mei still detects the *architecture*, which is what keeps an unnamed or unknown
model safe — it simply will not guess which specific model you have, and says
so at startup when it is serving an architecture it has profiles for.

### Ornith now points at a published aligned repack

`ornith-1.5-35b-a3b` pins
[`Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned`](https://huggingface.co/Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned):
the official weights, bit-identical, repacked so every tensor starts at a
naturally aligned offset. The published checkpoint leaves 1,421 of 1,757
tensors unable to be mmap'd, so MLX copies them into anonymous RAM at load —
**24.28 GB resident instead of 19.55, and 117.6 s instead of 36.7 s for a fresh
80k-context prefill**. The entire difference is a few bytes of padding per
shard's JSON header, and none of it is visible from the outside.

### The upstream vmlx sync is NOT in this release

0.5.0 stays on `44461ffd`, the same engine revision 0.4.2 shipped.

The sync (`23551729` — that same capture work plus 30 upstream commits) was
prepared for this release and pulled back out of it. On the coding and tool
suites it costs `hermes_ops-multi-step-chain`, a task that does not otherwise
vary: two independent runs on that pin, from different configs on different
days, produced **byte-identical** failing trajectories — the same 15 tool calls
in the same order. Scored against this model's measured noise floor that is
18/19 stable tasks against 17/19, and 8 of the 10 tasks whose output is recorded
generate differently under it.

Nothing was given up by holding it back. `44461ffd` is the tip of the
prefix-capture work and a strict ancestor of the sync, so every cache fix
`--ssm-anchor-boundaries` depends on is present. The 30 upstream commits are
MTP, Flash-attention, Spark2.5 and MiniCPM5 work that no model Mei has a profile
for uses.

It will land when the regression is understood, not before.

### Removed

- **`--optimization-profile`.** Naming the model selects its architecture too,
  and two flags to keep in sync was the confusion this replaces.
  Migrate: `--optimization-profile ornith` → `--model-profile ornith-1.5-35b-a3b`.
  Omitting it entirely is also fine — you get architecture defaults.
- **Gemma 4 special-casing.** That model was discarded as a candidate, so its
  prefill-256 rule and disk-KV entries went with it. No compatibility shim.

## [0.4.2] - 2026-09-12

Out-of-the-box prefix reuse, and instrumentation to prove where a turn's time
actually goes.

### `mei --model-dir DIR` now reuses its cache without being told to

The qwen3_5_moe hybrid cannot restore from the paged in-memory tier, so a bare
server re-read the whole conversation on every turn while reporting cache
reuse as enabled. It now gets a disposable disk KV tier by default, like the
other topologies that need one. Measured bare, four turns of a growing
conversation: **246 s -> 66 s**. The acceptance probe reaches 13/13 on a bare
server where it previously needed an explicit `--kv-cache-dir`.

### `--request-log` — one JSON line per generation run

Records prompt and cached token counts, prefill and generate milliseconds,
peak memory and the finish reason. `runner/analyze_request_log.py` in the
benchmark repo turns it into a per-turn decomposition of prefill, generation,
server gap and the interval between turns.

This is what made the per-turn overhead question answerable rather than
arguable. On the shipped configuration it showed prefill at 4.64 s/turn against
a server gap of 0.26 s — the gap was never the problem — and that eight cold
prefills of the ~20k system+tools preamble accounted for 46% of all prefill
time, 18.6% of the suite's wall clock, in eight requests.

### Fixed

- A restored cache no longer freezes its stored boundary at the SSM anchor.
- The advancing boundary costs one template render per turn instead of the
  full canonical set (~4.4 s/turn on a long transcript).
- `testOrnithMoeModelKeepsKVCacheDirEmptyDefault` asserted the exact contract
  this release reverses; it was never updated and `swift test` was red on the
  branch. No gate ran unit tests, so it would have shipped.

## [0.4.1] - 2026-09-11

A correctness fix to prefix reuse, and an end to needing flags to get good
behaviour.

### Prefix reuse restores faithfully now

0.4.0 shipped cross-conversation prefix reuse, but the prefill boundary capture
almost never produced a usable snapshot, so the post-answer store fell back to
`cacheSnapshotForBoundary` — which replays the prefix through the model for any
topology it cannot trim. Restoring that re-derived state **changes greedy
output**. Three linked defects, all fixed in vmlx `4a1069a4`:

- the capture asked `boundarySplit` for a boundary that helper rebases itself,
  so every capture landed `promptCount - headCount` tokens short (5 on Ornith,
  this template's generation-prompt suffix) and the store was correctly refused
  for an offset mismatch;
- captured snapshots were keyed by the head's *local* length, which equals the
  absolute boundary only when the cache started empty;
- the inner capture filtered stable boundaries against that same local length,
  so after a restore every boundary beyond the remaining slice was skipped.

**Measured.** A restored prefix now reproduces cold output byte-for-byte — 5 of
5 prompts identical in text, tool calls and completion tokens, with the cold leg
asserted at `cached_tokens` 0 and the restored leg at 20,375, fresh server and
KV directory per leg. It stays exact with 422 tokens of tail.

A restoring request also no longer pays a post-answer re-derive: on a 20k shared
prefix the first restoring conversation goes **53.4 s → 1.7 s**, and eight
conversations sharing a system prompt go 117.7 s → 65.4 s. That cost is paid
once per stored boundary, so it is a one-time saving per cache rather than per
turn.

### `mei --model-dir DIR` is now a complete command

The operator should not have to know a flag exists to get good behaviour.

- `--served-model-id` is optional, defaulting to the bundle's directory name and
  reporting it at startup.
- The `ornith` profile applies the settings this project measured as best:
  `VMLX_ENABLE_UNSAFE_COMPILE=1` (+9.0% short decode on Ornith, +9.8% on Qwen3.6
  text-only, gated on token equality because that flag fails silently rather
  than loudly) and a `--max-tokens` of 8192 (the 32768 default let one
  degenerate request generate 32,768 tokens over 1,156 s, burning 19.3 of a
  run's 40.1 minutes).
- `--ssm-anchor-boundaries` implies a durable KV tier. It previously did nothing
  at all on `qwen3_5_moe` bundles unless `--kv-cache-dir` was also passed —
  silently, with the startup line still reporting the cache as enabled.
- Startup names any setting that will not apply to the loaded topology:
  `--kv-bits` on recurrent topologies, anchors without a durable tier.

Explicit settings always win; every automatic choice is printed.

### Prefill step is chosen from available memory

> **Correction (2026-09-12): this did not ship in 0.4.1, and did not ship in
> 0.4.2 either.** The implementing commit reached a release-candidate branch and
> no tag — `git show v0.4.1:Sources/MeiCore/ModelOptimizationProfile.swift`
> contains no working-set logic at all, while this section is present in
> `v0.4.1:CHANGELOG.md`. In both releases the `ornith` profile took a flat 512.
> The device check is real as of 0.5.0, in the clamped form described in that
> release's notes. The section is corrected rather than deleted, because the
> published notes claimed it and readers of those tags deserve to find out why
> their prefill step never changed.

The `ornith` profile took 512 while 1024 was measured faster. Now it picks from
the device's recommended working set. Measured at 54,016 tokens: 512 gives
311 tok/s prefill at 22.72 GB peak, 1024 gives 337 tok/s (+8.4%) at 23.77 GB. At
the full 65,536-token cap that lands near 26.2 GB against a 26.8 GB recommended
set — it fits a 32 GB machine with about 0.6 GB to spare and nothing smaller,
so it is a device check rather than an unconditional default.

## [0.4.0] - 2026-09-08

Prefix reuse across conversations, Laguna XS 2.1 support, and the end of a
per-turn re-derive that was costing ~60 s on every warm request.

### Cross-conversation prefix reuse (opt-in, `--ssm-anchor-boundaries K`)

An agent harness sends the same large system prompt and tool schemas on every
turn of every conversation. That shared prefix was re-prefilled from cold every
single time: measured 60.7 s for a 20,394-token Hermes prompt, on every request,
in every session. Three defects, all fixed here:

- **The anchor offsets were always empty.** `SSMAnchorBoundaries.compute`
  requires a prefix-additive chat template — rendering `messages[0..<i]` must
  produce a token prefix of the full render. Qwen 3.5/3.6 violates this at the
  system boundary: the system message rendered alone *with tools* is 21,834
  tokens against 20,394 for the full system+user render. The additivity
  self-check therefore returned `[]` (correctly, given its contract) and the
  feature was inert on exactly the prompt shape it was built for. Added
  `SSMAnchorBoundaries.computeByDivergence`, which renders the full template with
  one message substituted and takes the longest common token prefix. It assumes
  nothing about the template. `compute` remains the fallback.
- **Mei never declared the boundary reusable.** vmlx's post-answer store loop
  iterates `LMInput.cachePrefixTokenCounts` and, for a hybrid cache, persists
  only boundaries also listed in `cacheStablePrefixTokenCounts` — the field
  documented as "deliberately persisted for reuse by unrelated new chat
  sessions". Mei passed neither, so the only stored boundary was the
  generation-stripped one, which already contains the current turn's user
  tokens; its content key never matched another conversation.
- **Every warm turn re-derived a boundary already on disk.** See below.

Measured on both production targets, real 20k-token Hermes prompt:

| | Ornith 1.5 35B-A3B | Qwen 3.6 35B-A3B text-only |
|---|---|---|
| cold, first ever | 122.9 s | 120.3 s |
| second conversation | **1.75 s** | **1.78 s** |
| new process, same KV dir | **1.79 s** | **1.82 s** |

The cold request is a one-time cost per unique prefix and is amortised across
every later conversation, including across process restarts.

**Correctness gated.** Restored output is byte-identical to cold on content,
`reasoning_content` and `tool_calls`, on every restoring row — after a
determinism baseline proved two identical cold servers agree byte-for-byte.

Default-off. `--ssm-anchor-boundaries 0` (the default) is unchanged behaviour.

### `hasDurableDiskEntry` no longer demands a companion that is never written

For disk-only MambaCache hybrids (Ornith, Qwen 3.5/3.6 GDN MoE, Qwen 3.8-27B)
the recurrent state round-trips inside the v2 payload as `mamba_{i}_state0/1`.
`storeAfterGeneration` deliberately writes no separate SSM sidecar for them, and
`hasRequiredHybridSSM` accepts a fetched entry without one for the same reason —
but `hasDurableDiskEntry` demanded a *validated* sidecar regardless. The check
was therefore permanently false on those models, and the post-answer store loop
re-derived every boundary on every warm turn, replaying the whole prefix through
the model after the answer had already streamed. The disk write that followed
was then correctly skipped as already-validated, so the work produced nothing.
Measured ~60 s per turn on a 20k prefix. Now gated on the same flag the store
and fetch paths use. (vmlx `654eb455`.)

### Laguna XS 2.1 MLX support

- Unwrap a leading `language_model.` prefix in sanitize.
- Normalize the routed gate layout before load dequantization, fixing
  `Unhandled keys [e_score_correction_bias, proj]`.
- Compile the routed SwitchGLU separated decode for the exact affine S-2.1 XS
  MoE topology only; other model families are unaffected. Focused regression
  coverage added.

### Diagnostics

- `MEI_ANCHOR_TRACE=1` prints the anchor computation: message roles, per-prefix
  token counts, and the resulting offsets or the reason they were rejected.

### vmlx

Re-pinned to `654eb455` (`mei/0.4.0` = the 0.3.0 pin plus the three Laguna fixes
and the durability fix above). The C3 checkpoint-fused `gate_up_proj` experiment
is deliberately **not** included: it loads at zero memory cost and is
token-identical, but measured -2.9% short decode and neutral prefill, so one
wider gather loses to two narrower ones on this hardware.


## [0.3.0] - 2026-09-07

Re-pins vmlx-swift onto the Mei fork synced with 37 upstream commits, and makes
the MLXPress cold-weight tier reachable. Measured on the two active
`qwen3_5_moe` targets (Ornith 1.5 35B-A3B, Qwen 3.6 35B-A3B).

### Changed

- **vmlx-swift re-pinned** `91fed8be` -> `e37d1d59`: the fork's `main` merged
  with 37 upstream commits (clean, zero conflicts, all six fork commits
  preserved). Measured against the old pin on Ornith: short decode neutral
  (66.59 -> 66.28 tok/s), **30k decode +3.0%** (40.40 -> 41.63), peak +0.30 GB;
  four 700-token greedy temp-0 generations token-for-token identical.
  Taken primarily for the correctness fixes it carries, not the speed:
  - `a6252bc1` cache restore fails closed when attention and recurrent offsets
    disagree with the matched boundary — the upstream fix for **Ornith loops**
  - `2c036567` / `45bcb1de` DiskCache publishes rows atomically, fails closed on
    a short file, and never stores or restores NaN/Inf — our KV disk tier
  - `36b7396f` the safetensors healer is opt-in and no longer rewrites a user's
    original model files by default (it had been mutating staged checkpoints)
  - `5a63b9be` / `97676e19` wire a loaded model's weights so big bundles stop
    swapping, but only when the model fits a safe physical-RAM reserve

### Added

- `Engine` now passes `jangPress: .default` to `LoadConfiguration`, making
  MLXPress axis E (the routed cold-weight tier) reachable. `LoadConfiguration`
  defaults `jangPress` to `.disabled`, which short-circuits before any
  environment lookup, so **`MLXPRESS=N` was silently inert on every previous
  Mei build**. Verified engaged: `MLXPRESS=70` advises 10.5 GiB cold, `95`
  advises 14.2 GiB, and the mmap tier indexes 10,240 experts across 40 layers.
  No throughput cost (64.0-67.4 tok/s short, 40.4-40.6 at 30k across settings).

  It reclaims **nothing** on this configuration, and that is worth stating: a
  19.5 GB mmap'd model shows only ~150 MB `phys_footprint` and ~1.1 GB RSS, so
  the weights are clean file-backed pages macOS already evicts without any
  `madvise` help. RSS is flat across `MLXPRESS=0` vs `95` in both soft and force
  modes. Shipped because it makes the tier reachable at all — for a future model
  or host where weights are not mmap-clean — not because it helps here.

### Notes for operators

- `VMLX_ENABLE_UNSAFE_COMPILE=1` measures **+9%** short decode on both targets
  (Ornith 66.36 -> 72.32, Qwen 3.6 66.65 -> 73.21) with token-for-token
  identical greedy output, and is re-verified against this pin. It is **not**
  enabled by Mei; it is set per-deployment. Mei's one-model-per-server-process
  design makes the Osaurus #1173 model-switch corruption that gates it off
  structurally unreachable, but the macOS Tahoe Metal JIT bug
  (MLX #3329/#3201/#3256) is a separate live risk — **re-verify token equality
  after any vmlx re-pin before relying on it**.
- Two optimizations were implemented, measured, and **rejected**: enabling the
  compiled routed-MoE region on the LLM path (`compileSeparatedDecode: true`)
  costs **-9.7%** short decode and -4.9% at 30k, and graph-traced compiled
  decode fails a greedy token-equality gate outright. Neither is in this
  release.

## [0.2.0] - 2026-09-06

First **stable** Apple Silicon release: the prebuilt CLI/runtime bundle + a
Homebrew formula, promoting the `0.2.0-alpha.1` runtime work to a downloadable,
easy-to-install release. Tagged `v0.2.0` at `6a53cb0` (2026-09-06), on top of
the streaming tool-call fix `67e897e`.

### Added

- Reproducible packaging (`scripts/package_release.sh`): builds the release
  binary, provisions the version-matched `mlx.metallib` (mlx 0.31.1) via
  `scripts/prepare_metallib.sh`, assembles a `dist/mei-<version>-macos-arm64/`
  CLI/runtime bundle (`bin/mei` + `bin/mlx.metallib` + provenance + docs), tars
  it with a stable member order, and emits the SHA-256 checksum. Verifies the
  binary is arm64 and reports exactly `mei <version>` before packaging.
- Packaging smoke checks (`scripts/test_package_release.sh`): validates the
  tarball + checksum round-trip, extracted structure, `bin/mei --version ==
  "mei 0.2.0"` (the offline API/version identity proxy), Metal-library
  presence, and that no weight blobs (`*.safetensors`/`*.gguf`/`*.bin`) are
  bundled.
- Homebrew formula `tijs/tap/mei` (`brew install tijs/tap/mei`): installs the
  packaged `mei` binary + colocated Metal libraries from the anonymous GitHub
  release-download URL, with a `test` block asserting version + Metal-library
  presence.
- Stable release notes: `docs/RELEASE-0.2.0.md` documents install paths
  (Homebrew + manual download), the weight-separation boundary, and the
  reproducible build path.

### Changed

- `ServerConfig.version`: `0.2.0-alpha.1` → `0.2.0` (`mei --version` now prints
  `mei 0.2.0`).
- `README.md`: install section now leads with the Homebrew formula and the
  downloadable binary bundle.
- This stable release is the `0.2.0-alpha.1` runtime plus the streaming
  tool-call fix (commit `67e897e`, distinct SSE indexes for multiple tool
  calls) — see `CHANGELOG.md` `[0.2.0-alpha.1]` for the full inherited runtime
  work.

### Removed

- `ornith-ai/Ornith-1.5-9B-MLX-4bit` removed from the active model lineup
  (`configs/model-lineup.json`): the upstream repository became unavailable
  (2026-09-04), so the fallback option could never stage. Historical 9B
  artifacts and benchmark rows are preserved in `artifacts/` and remain
  untouched. The `mei-ornith9` worker option was dropped from
  `docs/WORKER-MODEL-OPTIONS.md` (port 8028 is no longer a Mei backend port);
  stale profile entries, if any, fail closed by connection refusal.

### Docs

- `README.md` Models section reconciled to the measured four-model state:
  Ornith-35B 30k decode 47.5-50.3 t/s (3 repeats) with the fitted env-gated
  config, Qwen3.8 30 t/s hardware-ceiling record, Gemma4 fuse-gate lever and
  GGUF A/B result, Heretic 30k 3-repeat and GGUF A/B results — each citing
  its committed artifact under `artifacts/` (see also the consolidated
  `artifacts/four-model-gate-matrix-20260904.md`).

### Release status

Published. Tagged `v0.2.0` at `6a53cb0` (2026-09-06) and released as a public,
non-preview GitHub release. The Homebrew formula `tijs/tap/mei` is live
(`brew install tijs/tap/mei`) and the prebuilt Apple Silicon bundle is
downloadable from the release assets. Model weights are never bundled (see the
weight-separation boundary in `docs/RELEASE-0.2.0.md`).

## [0.2.0-alpha.1] - 2026-09-03

Source-first preview release candidate, tagged `v0.2.0-alpha.1` at `23811db`
and published as a GitHub prerelease on 2026-09-03. Ships the verified Qwen3.8
/ Gemma 4 / Qwen3.8-Heretic runtime work on top of the 0.1.0 Ornith release
and bumps the runtime version metadata. Its runtime work was subsequently
promoted into the `0.2.0` stable release.

### Added

- Qwen3.8-27B, Gemma 4 26B-A4B and Qwen3.8-Uncensored (Heretic) MLX
  loadability/acceptance evidence on the generic profile; the staged
  `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`
  and `orcarouter/Qwen3.8-27B-Uncensored-MLX` 4-bit checkpoints are loadable
  and acceptance-gated (see `configs/model-lineup.json` for per-model status,
  exact revisions, digests, quant settings and measured results).
- Model-aware disposable on-disk KV default extended to Gemma 4 bundles
  (`model_type` `gemma4`/`gemma4_text`): exact-repeat prefixes never restore on
  the in-memory-only paged tier for this family, so with `--cache-reuse` on and
  no explicit `--kv-cache-dir` they now default to a disposable on-disk cache
  under the OS temp directory, matching the existing dense
  qwen3_5/qwen3_8 default. Explicit `--kv-cache-dir` always wins;
  `--cache-reuse false` keeps caching fully disabled; the MoE/Ornith
  `qwen3_5_moe` family is untouched. `needsDiskKVTier` /
  `diskKVRequiredModelTypes` replace the former
  `denseQwen35NeedsDiskKVTier` / `denseQwen35KVUnsafeModelTypes` (deprecated
  aliases kept).
- Reader-facing release metadata: README logo (`assets/mei-logo.png`),
  `mei --version` now reports `0.2.0-alpha.1`, an explicit staging allowlist
  and script (`configs/release-allowlist.json`,
  `scripts/stage_release_candidate.sh`) and this release-notes file
  (`docs/RELEASE-0.2.0-alpha.1.md`).
- Gemma 4 chunked-prefill default 64 → 256, arch-scoped
  (`ModelOptimizationProfile.prefill256ModelTypes` = `gemma4`/`gemma4_text`;
  Ornith stays 512, all other models stay 64, explicit `--prefill-step-size`
  always wins, malformed metadata stays 64). Measured 2026-09-03 on
  `mlx-community/gemma-4-26b-a4b-it-4bit`: 30k fresh fill 266.5/265.3/266.2
  pps (3 cold repeats) vs ~139 baseline (+91%), peak 27.23 GB unchanged,
  30k loaded decode unchanged (~7.4 t/s), acceptance pass-set identical to
  the 64 baseline (at that time the only gate not yet passed was the
  pre-existing user-gated Gemma string-args tool schema, since cleared by the
  `ba1e9df`/`00418a5` argument-typing work). The 30k-decode row closes the
  (GGUF 37.08 t/s, MLX 5.0x slower — dense/rotating-attention cost, recorded
  as a measured constraint). Note: this change postdates the staged
  v0.2.0-alpha.1 snapshot (2026-09-03T14:27Z); any future staging refresh
  must re-run `scripts/stage_release_candidate.sh`. Evidence:
  `artifacts/gemma4-prefill-step-sweep-20260903.md`.
- Safe user-local CLI installer (commit `9593126`): `scripts/install_mei.sh`
  copies an already-built `mei` executable (and any colocated `mlx.metallib` /
  `default.metallib` / `*.provenance`) into a single user-local destination
  dir, defaulting to `$HOME/.local/bin`. Source resolution order: `--binary`,
  `$MEI_BINARY`, repo `.build/release/mei`, or `bin/mei` beside the installer.
  Refuses system / package-manager prefixes (`/usr`, `/usr/local`, `/opt`,
  `/opt/homebrew`, `/bin`, `/sbin`, `/Library`) unless `--force`; preserves an
  existing differing file unless `--force`; idempotent rerun is a no-op;
  supports `--prefix`, `--dry-run`, `--help`, `--version`; never builds or
  downloads. `scripts/test_install_mei.sh` is a deterministic, weight-free,
  server-free suite (31 checks) covering help, unknown-option exit-2,
  missing-binary failure, dry-run, install + executable bit, idempotent rerun,
  `--force` overwrite, colocated Metal companions, and the system-prefix
  guard. Documented in `docs/INSTALL.md`; README Build section points at it.
  Verified against the final release binary at `00418a5`: installed to
  `/Users/tijs/.local/bin/mei` reporting `mei 0.2.0-alpha.1`, SHA-256 matching
  the final release binary, with `mlx.metallib` / `default.metallib` /
  provenance installed alongside.

### Fixed

- Raw `/v1/completions` crash on the VLM-routed Qwen3.8-27B-4bit checkpoint
  (deterministic `SmallVector out of range`, vmlx `mlx/c/array.cpp:335`, at ANY
  length): `MLXVLM.Qwen35.prepare` requires batch-first tokens; the raw
  completions path delivered a 1-D token array. Raw and chat `LMInput` sites
  now emit `[1,T]` via `expandedDimensions(axis: 0)` — safe for both the VLM
  prepare and the rank-safe LLM default prepare. The 4-bit raw path re-gated
  PASS on the full matrix (acceptance/streaming/tool/coding/KV/long-context).
- Gemma 4 fresh chat-prefill SIGTRAP (rank-3 embedding slice in
  `Gemma4.prepare`, any prompt length): chat `LMInput` built 1-D `[T]` tokens
  and the VLM prepare embedded them directly; chat sites now emit batch-first
  `[1,T]` like the raw path and the vmlx cache-restore rebuild.
- Gemma 4 growing-transcript prefix-extension reuse (cached_tokens 0 →
  786/824 tokens restored on the disk tier; cache-ON == cache-OFF
  byte-identical outputs): vmlx fork commit `318a4e68` admits standalone
  rotating-window cache topologies to the gen-suffix-stripped boundary store.
  **This fork commit is NOT yet pushed** — it is consumed via a local SwiftPM
  edit; `Package.swift`/`Package.resolved` stay pinned to the remote revision
  `91fed8be`. Pure-source builds therefore lack this fix until the fork
  `main` advances (blocker, user action).
- Gemma 4 tool-call argument typing + explicit /v1 usage contract
  (commit `ba1e9df`): a schema-aware `ToolArgumentNormalizer` coerces fields
  whose request tool JSON-Schema type is number/integer to JSON numbers
  (string→number only when the emitted value is fully numeric), so Gemma-4's
  literal `{"a":"15","b":"27"}` spelling now reaches the API as integer
  `{"a":15,"b":27}` — matching the GGUF path — while string/boolean/nested
  fields pass through untouched. One shared `Router.usage(run:)` defines
  prompt/completion/total tokens across non-streaming chat, raw
  `/v1/completions`, and the streaming finish; streaming honors
  `stream_options.include_usage` (final usage chunk emitted only when true).
  Verified live against the final release binary: Gemma tool args integer
  non-streaming and streaming; `include_usage` true→emitted / false→omitted;
  raw `/v1/completions` integer usage with `total = prompt + completion`
  (final usage tuples (prompt, completion, total, cached_tokens), all
  integer, total arithmetic valid: `(168,22,190,0)` chat non-streaming,
  `(168,22,190,167)` chat streaming `include_usage=true` — prefix reused —
  and `(5,16,21,4)` raw `/v1/completions`). Canonical `MEIAcceptanceTests`
  vs the Ornith release binary
  (`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`) 5/5 PASS. Focused suites against
  the release binary: `ToolArgumentNormalizerTests` 16/16, `OpenAITypesTests`
  12/12, `ServerConfigParsingTests` 28/28, `SSMAnchorBoundariesTests` 9/9,
  `CacheRestoreTrackerTests` 6/6, `QuantizedRotatingKVCacheTests` 6/6.
  Evidence: `artifacts/gemma4-tool-api-contract-20260903.md`.

### Hardening

- Tool-argument normalization hardening (commit `00418a5`, the final head for
  this candidate): the schema-aware `ToolArgumentNormalizer` now recurses
  **every** array element against its `items` schema, so primitive numeric
  arrays (`items: {type: integer|number}`) coerce numeric strings exactly like
  object numeric properties, while string-typed items and properties pass
  through untouched; moreover numeric parsing now **rejects non-finite
  conversions** (NaN / ±infinity / overflow-to-infinity stay strings) so JSON
  serialization can never throw and silently fall back to `{}` — which would
  drop every argument. `ToolArgumentNormalizerTests` grew 9/9 → 16/16
  (strict TDD). Verified against the final release binary: exact integer
  tool args `{"a":15,"b":27}` non-streaming and streaming; all count fields
  integer with total arithmetic valid. Final binary SHA-256
  `e998782c9f2019449a73ccbeb348e91a8cb568a73a392a8ab792be7bb90aa60a`.
  Evidence: `artifacts/gemma4-tool-api-contract-20260903.md`.

### Known blockers

Full list with measured evidence in `docs/RELEASE-0.2.0-alpha.1.md`; summary:

- Fork commit `318a4e68` (Gemma4 reuse fix + cache-fetch diagnostics) is
  un-pushed; external/source-only builds resolve `91fed8be` without it.
- Qwen3.8-27B decode is below the 30 t/s primary target: 4-bit 15.66 t/s
  (sd 0.060, 3 cold repeats, peak 18.9 GB), 5-bit parity artifact 13.11 t/s —
  hardware ceiling accepted and recorded in the plan (2026-09-02); 4-bit raw
  path is fixed and re-gated.
- Ornith-1.5-35B reaches >= 30 t/s only with the env-gated fused gate/up cache
  disabled (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, user-gated); MTP /
  speculative decode stays out of scope.

### Release status

Source-first prerelease, tagged `v0.2.0-alpha.1` at `23811db` and published on
GitHub as a prerelease on 2026-09-03. The validated runtime target is macOS 15+
on Apple Silicon with local MLX/Metal. Model weights are never bundled (see
`docs/RELEASE-0.2.0-alpha.1.md` for the weight-separation boundary).

## [0.1.0] - 2026-09-02

Initial public release for Apple Silicon.

### Added

- Native Swift/MLX OpenAI-compatible server for one local model per process.
- Automatic `auto|generic|ornith` runtime profile selection from model metadata.
- Validated Ornith profile: aligned checkpoint support, prefill step 512, and
  Ornith-only fused gate/up cache disablement.
- Conservative generic profile with experimental compiled decode, rotating KV
  quantization, bounded windows, and SSM anchors disabled by default.
- In-process and optional disk-tier KV/prefix reuse.
- Scoped disposable-cache cleanup and a 20 GiB free-space launch guard.
- Local vMLX patch queue pinned to an immutable upstream revision.
- Focused unit tests and release provenance.

### Changed

- Generic-profile safety default: dense Qwen3.5/Qwen3.8-lineage checkpoints
  (`model_type` `qwen3_5`/`qwen3_5_text`) crash the in-memory-only paged KV
  cache tier (vmlx `array.cpp:335`, crash trigger isolated by a bounded 2x2
  on 2026-09-02), so with `--cache-reuse` on and no explicit `--kv-cache-dir`
  they now default to a disposable on-disk cache under the OS temp
  directory. Explicit `--kv-cache-dir` always wins; `--cache-reuse false`
  keeps caching fully disabled; the MoE/Ornith `qwen3_5_moe` family is
  untouched.

### Release status

This is an initial public, source-first release. The validated runtime target
is macOS 15+ on Apple Silicon with the aligned Ornith 1.5 35B checkpoint. The
model is not bundled. Full model/GPU acceptance requires the local MLX/Metal
runtime and is not reproduced by every CI runner.

[0.2.0]: https://github.com/tijs/mei/releases/tag/v0.2.0
[0.2.0-alpha.1]: https://github.com/tijs/mei/releases/tag/v0.2.0-alpha.1
[0.1.0]: https://github.com/tijs/mei/releases/tag/v0.1.0