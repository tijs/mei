# Mei 0.2.0-alpha.1

Source-first preview release candidate — 2026-09-03.

This candidate packages the verified Qwen3.8 / Gemma 4 / Qwen3.8-Heretic
runtime work on top of the 0.1.0 Ornith release into an inspectable,
source-first release candidate. It is **not** tagged, pushed, or published:
public release requires explicit user authorization, and pushing the vMLX fork
commit below is a prerequisite for a fully source-reproducible tag.

Status metadata lives in three places and must stay in sync:

- `Sources/MeiCore/ServerConfig.swift:6` — `ServerConfig.version` (runtime
  `mei --version` output).
- `CHANGELOG.md` — human changelog.
- `configs/model-lineup.json` — machine-readable per-model lineage, quant
  settings, digests, staged paths, status, and measured results (source of
  truth for every model claim).

## What changed since 0.1.0

Runtime (all verified on Sulaco, 32 GB Apple M1 Max, port 8024, generic
profile, cap 65536, prefill 64, kv-bits none):

- **Qwen3.8-27B-4bit** (`mlx-community/Qwen3.8-27B-4bit` @ `3e6447f`, staged):
  loadable; `probe_load` 3x PASS (hello 15.21/15.60/15.23 t/s, short
  15.48/15.61/15.71 t/s, active 18.82–18.83 GB); `probe_mei` 10/12 (tool
  stream+non-stream, parity, KV reuse 6207/6212, growing-transcript all PASS);
  `probe_coding` 4/4 (~15.6 t/s); chat-threshold fills PASS to 30k. Raw
  `/v1/completions` crash (vmlx `array.cpp:335`, any length) root-caused and
  fixed in Mei (`Engine.swift`), then re-gated PASS on the full matrix —
  see `artifacts/qwen38-4bit-raw-fix-20260903.md`.
- **Qwen3.8-27B-5bit-affine-g64** (Mei-produced, published as
  `Tostibrown/Qwen3.8-27B-5bit-affine-g64` @ `f592c6f`): full parity archive —
  `probe_mei` 11/11, coding 4/4, 30k long-context, KV reuse; 3-repeat decode
  13.108 t/s (sd 0.012), peak 21.86 GB. Remains the documented parity artifact.
- **Gemma 4 26B-A4B** (`mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464`,
  staged): loadable; VLM chat-preflight crash fixed (batch-first `[1,T]`);
  growing-transcript reuse fixed in vmlx fork commit `318a4e68` (un-pushed);
  common matrix PASS except the tool strict-schema gate (string-typed args —
  see blockers). Peak ~25.8 GB 4-bit row.
- **Qwen3.8-Heretic** (`orcarouter/Qwen3.8-27B-Uncensored-MLX` @ `14963e70`
  4-bit/): lineage gate PASS (distinct tensors from base Qwen, gated source
  `404ea47a`); `probe_load` hello 15.07 / short 15.72 t/s, peak 18.98 GB;
  `probe_mei` 10/10 incl. tool stream+non-stream parity and KV reuse.

## Model status (source of truth: `configs/model-lineup.json`)

| Model | Status | Decode (3 cold repeats) | Peak working set |
|---|---|---|---|
| Ornith-1.5-35B-A3B-MLX-4bit (primary) | loadable-accept-passed-fits-32gb-env-gated | 55.0 t/s clean; 35.2–35.3 t/s @80k; 31.7 t/s @100k | 25.7–28.2 GB (fused-cache-off env) |
| Ornith-1.5-9B-MLX-4bit (fallback) | validated | benchmark rows in artifacts | — |
| Qwen3.8-27B-4bit | loadable-accept-passed-perf-memory-constrained | 15.66 t/s (sd 0.060) | 18.9 GB |
| Qwen3.8-27B-5bit-affine-g64 (Mei-produced) | loadable-parity-passed-3repeat-perf-memory-constrained-at-cap | 13.11 t/s (sd 0.012) | 21.9 GB |
| Gemma 4 26B-A4B 4-bit | loadable-accept-passed-tool-string-args | matrix rows in artifacts | ~25.8 GB |
| Qwen3.8-27B-Uncensored-MLX-4bit (Heretic) | loadable-accept-passed | 15.1–15.8 t/s | 20.6 GB |

GGUF/llama.cpp reference matching per model is a later unit (plan todo 8);
nothing here claims GGUF-UD/APEX bit-equivalence for the MLX rows.

## Model-weight separation

Mei ships **source and tooling only**. No checkpoint, shard, GGUF blob, or
converted weight file is part of the source tree or the release staging
boundary.

- Weights live outside the repo under
  `~/.local/share/local-model-bench/mei-models/` (staged via
  `scripts/stage_model.sh`, pinned by immutable revision with sha-verified
  manifests recorded in `configs/model-lineup.json`).
- The staging allowlist (`configs/release-allowlist.json`) explicitly excludes
  model directories, the HF cache, `artifacts/` (historical benchmark
  evidence), `.git/`, build products, and the previous `dist/` bundle.
- `scripts/stage_release_candidate.sh` enforces the boundary mechanically
  (see Staging below) and writes a per-file sha256 manifest.

## Fork pin and the `318a4e68` caveat

`Package.swift` / `Package.resolved` pin `tijs/vmlx-swift` at
`91fed8be21319f92ce5220622c6dcde0b851bdae` (remote `main`). The verified Gemma 4
growing-transcript reuse fix is fork commit `318a4e68` **on top of** `91fed8be`
and is **not pushed**; the release build for this candidate consumes it via a
local SwiftPM edit:

```bash
swift package --scratch-path <scratch> edit vmlx-swift \
  --path /Users/tijs/projects/vmlx-swift   # HEAD = 318a4e68
```

Consequences, stated plainly:

- A build from the remote pin alone (fresh CI / end user) resolves `91fed8be`
  and does **not** contain the Gemma4 growing-reuse fix (correct, strictly
  slower growing-transcript reuse for that family) nor the cache-fetch
  diagnostics env vars.
- The candidate is therefore not fully source-reproducible until the fork
  `main` is advanced to `318a4e68` (or the fix is cherry-picked into a pushed
  revision) by the user. This is the single most important readiness blocker.

## Known blockers

| # | Blocker | Evidence | Owner |
|---|---|---|---|
| B1 | vMLX fork `318a4e68` un-pushed (Gemma4 rotating-boundary reuse fix) | `artifacts/gemma4-growing-reuse-fix-20260903.md` (cached 0 → 786/824, byte-identical cache-ON/OFF) | user (push) |
| B2 | Gemma 4 tool calls emit string-typed JSON args; strict-schema tool gate FAILS | `artifacts/gemma4-26b-common-matrix-20260903.md` | user go/no-go |
| B3 | Qwen3.8-27B decode 15.7 t/s (4-bit) / 13.1 t/s (5-bit) < 30 t/s primary target — hardware ceiling accepted | plan 0b87b76a speed-gate exhaustion note, `artifacts/qwen38-4bit-5bit-ab-20260902.md` | recorded |
| B4 | Ornith 35B >= 30 t/s only with env-gated fused gate/up cache off (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`) | `artifacts/ornith-35B-fuse-gateup-eliminated-20260902.md` | user go/no-go for env default |
| B5 | Qwen3.8-27B-4bit full-cap 65536 fill memory-constrained (active 30.2 GB / peak 34.6 GB exceeds 32 GB physical) | lineup note, parity matrix | recorded |
| B6 | Heretic long-context 30k + trohrbaugh Q5_K_M GGUF behavioral comparison pending (todo 8) | lineup note | worker |
| B7 | MTP/Next-N speculative decode out of scope (no win on this hardware in bench data) | plan | scope |

## Staging boundary

The release staging allowlist is `configs/release-allowlist.json`
(schema_version 1). It enumerates exactly which files enter the candidate:

- root files: `LICENSE`, `NOTICE.md`, `SECURITY.md`, `CONTRIBUTING.md`,
  `AGENTS.md`, `README.md`, `CHANGELOG.md`, `Package.swift`,
  `Package.resolved`, `.gitignore`, `.kiem`
- directories: `Sources/`, `Tests/`, `docs/`, `scripts/`, `tools/` (minus
  `__pycache__`/`*.pyc`), `configs/`, `assets/`
- excluded outright: `artifacts/` (historical benchmark evidence — never
  shipped), `.git/`, `.build/`, `.swiftpm/`, `dist/` (prior bundles), model
  weights and caches (not in the repo by design)

`scripts/stage_release_candidate.sh <version>` copies the allowlist into
`dist/mei-<version>-src/`, then fails loudly unless: every allowlisted path is
present; no staged path is outside the allowlist; no excluded directory leaks
in; and `.git` metadata is absent. It writes `STAGING-MANIFEST.sha256` plus a
JSON manifest. The 0.1.0 binary bundle under `dist/` is left untouched.

## Build and test record (this candidate, 2026-09-03)

Commands run on Sulaco:

```bash
# 1. package resolution. NOTE: SwiftPM 6.3.3's `resolve` regenerates
#    Package.resolved WITHOUT revision-pinned entries (vmlx-swift is pinned by
#    revision in Package.swift and is therefore dropped from the resolved
#    file). The committed 0.1.0-era Package.resolved keeps the fork pin; after
#    any resolve, restore it with: git show HEAD:Package.resolved > Package.resolved
swift package resolve

# 2. release build with the verified runtime (local fork edit, 318a4e68)
swift package --scratch-path ~/.local/share/local-model-bench/mei-build \
  edit vmlx-swift --path /Users/tijs/projects/vmlx-swift
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build

# 3. version + unit tests (acceptance tests need a live server)
~/.local/share/local-model-bench/mei-build/release/mei --version   # 0.2.0-alpha.1
swift test --scratch-path ~/.local/share/local-model-bench/mei-build \
  --skip MeiAcceptanceTests

# 4. staging + boundary diff checks
bash scripts/stage_release_candidate.sh 0.2.0-alpha.1

# 5. working-tree diff review vs HEAD
git diff --stat
```

Expected results (measured, see evidence note for exact numbers):

- `mei --version` prints `mei 0.2.0-alpha.1`.
- Non-acceptance unit suite passes (52/52 in prior runs; 5 live-server
  `MeiAcceptanceTests` failures are the known server-required set, unchanged
  across 0.1.0 and this candidate).
- Release build completes with exit 0; the staged tree passes all boundary
  checks and the manifest matches the allowlist one-for-one.

## Rollback

Source: revert metadata/README/CHANGELOG edits and keep the 0.1.0 pin —
previous verified revision is `91fed8be` in `Package.swift`/`Package.resolved`
(either way, do not reset the repository or delete evidence). Runtime: use the
same rollback env set as 0.1.0 (`MEI_OPTIMIZATION_PROFILE=generic`,
`MEI_PREFILL_STEP_SIZE=64`, compiled/window/anchor features off).

## Publication status

NOT published. No tag, no push, no GitHub release, no upload. Remote fork
push (B1) and release authorization are user decisions outside the autonomous
worker's authority.