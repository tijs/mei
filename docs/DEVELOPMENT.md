# Development

Build, test, and package Mei, understand the repo layout, the `mlx.metallib`
dependency, and the vMLX fork pin / workflow.

## Repository layout

```text
Sources/
  MeiCore/            engine glue, router, HTTP server, OpenAI DTOs
  Mei/main.swift      CLI entry point
Tests/MeiTests/       unit tests + black-box acceptance oracle
  MeiAcceptanceTests  run RED against a missing server, go green once the
                      server behaves; enabled via MEI_ACCEPTANCE_BASE_URL
                      (default http://127.0.0.1:8024/v1)
tools/                probe/bench drivers and standalone tools (authoritative
                      copies; see BENCHMARKING.md)
scripts/              stage_model.sh, start_mei_server.sh, stop_mei_server.sh,
                      prepare_metallib.sh, install_mei.sh, package_release.sh
                      and their test_* companions
configs/              model-lineup.json, release-allowlist.json
artifacts/            historical benchmark / evidence notes (never rewritten)
Casks/  tap/          unrelated untracked packaging scratch (do not commit here)
```

`configs/model-lineup.json` is the machine-readable source of truth for
pinned revisions, GGUF blob SHA-256 digests, quant settings, local staged
paths, status, test phases, and published model URLs. `tools/` mirrors some
drivers to `local-model-bench/runner/` in a read-only boundary.

## Build

```bash
# One-time setup if `xcrun metal --version` reports a missing toolchain:
xcodebuild -downloadComponent MetalToolchain
swift package resolve
swift build            # debug
swift test             # unit tests (acceptance tests need a live server)
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
# The release launcher resolves the fork-pinned dependency automatically:
scripts/start_mei_server.sh
```

`Package.resolved` is committed. Re-pinning or changing the fork revision is a
deliberate decision that must re-run the whole acceptance suite.

## Put the built CLI on your PATH (user-local)

```bash
MEI_VMLX_CHECKOUT="$PWD/.build/checkouts/vmlx-swift" \
  scripts/prepare_metallib.sh .build/release   # prefer the built Cmlx resource bundle
scripts/install_mei.sh                       # -> $HOME/.local/bin/mei
```

`scripts/install_mei.sh` puts a built `mei` on your PATH without touching
Homebrew, system directories, or anything managed by a package manager.
Options: `--prefix DIR` (default `$HOME/.local/bin`), `--binary PATH`,
`--dry-run`, `--force`, `-h`, `--version`. Source resolution order:
`--binary` → `$MEI_BINARY` → `.build/release/mei` → `bin/mei` beside the
installer. Safety contract: writes only to `$PREFIX`; refuses to overwrite a
differing installed file unless `--force`; idempotent on identical rerun;
never handles weights. The authoritative install docs —
including the **stable v0.2.0 Homebrew / release-asset paths** — are
[`docs/INSTALL.md`](INSTALL.md). Deterministic tests:
`scripts/test_install_mei.sh`.

## Packaging a release bundle

```bash
# Build a downloadable Apple Silicon release bundle (dist/.../mei-<v>-macos-arm64.tar.gz + .sha256):
scripts/package_release.sh 0.2.0 --skip-build
scripts/test_package_release.sh         # packaging/install/version smoke checks
```

`package_release.sh` builds the release binary, provisions the Metal library
from the Cmlx resource bundle beside that binary, assembles the bundle, tars it with
a stable member order, and emits the SHA-256 checksum. It verifies the binary
is arm64 and reports exactly `mei <version>` before packaging.
`test_package_release.sh` validates the tarball/checksum round-trip, extracted
structure, `bin/mei --version`, Metal-library presence, and that no weight
blobs (`*.safetensors` / `*.gguf` / `*.bin`) are bundled.

## Metal kernel library (`mlx.metallib`)

With Xcode's Metal Toolchain installed, SwiftPM's Xcode build system emits
`mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` beside the binary
(the flat bundle layout is also supported). `scripts/prepare_metallib.sh`
copies this build product to the colocated `mlx.metallib` that MLX loads first,
overwriting an older colocated library even if its MLX version is unchanged.
Rebuild after changing the dependency pin; provisioning does not compile or
validate the freshness of an arbitrary build directory.

- `MEI_METALLIB_SOURCE` remains the highest-priority explicit override.
- `MEI_METALLIB_BUILD_DIR` selects the binary's build-product directory when
  provisioning elsewhere. If explicitly set, a missing bundle is fatal rather
  than silently selecting a wheel. Packaging sets this to the selected binary's
  directory, so release packaging is source-first and fail-closed.
- Set `MEI_VMLX_CHECKOUT` to the checkout used for that build (especially with
  custom scratch paths or `--binary`). The sidecar records the selected checkout
  revision, derived MLX version, source path and installed library's SHA-256;
  the checkout revision is context, not proof that an arbitrary artifact was
  built from it.
- Every candidate is structurally checked (MTLB magic, size, `file`
  classification). An invalid build product fails rather than falling back.
- Legacy standalone provisioning without an explicit build directory retains
  existing-library/wheel fallbacks. Wheel use warns even on an MLX version
  match: a version number does not establish fork kernel identity. Some wheel
  installations have no bundled library; do not depend on them for releases.

The precompiled library is only part of MLX's Metal code: quantized kernels
also use runtime JIT compilation from embedded source. A stock wheel library
does not imply that all fork kernel changes are absent. Runtime model probes
remain required; a structural check or `mei --version` is not a kernel test.

## vMLX fork pin and workflow

Mei consumes the public fork [`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift)
through SwiftPM, keeping [`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift)
as `upstream`. The fork's `main` contains five separate Mei-maintained commits
ported from the former local patch queue, plus one commit (`318a4e68`) that
exists only in the local fork checkout (not yet pushed). See
[`docs/VMLX-FORK.md`](VMLX-FORK.md) — the authoritative reference — for the
commit mapping, normal workflow, upstream-PR preparation, and how to update
Mei's pin.

Quick reference:

```bash
# Mei work
cd ~/projects/mei
git pull --ff-only origin main
swift package resolve
swift test

# vMLX work
cd ~/projects/vmlx-swift
git fetch origin upstream --prune
git switch main
git pull --ff-only origin main
git push origin main      # after a focused, tested change

# Prepare an upstream PR (compare against the parent)
git fetch upstream --prune
git log --oneline upstream/main..main
git diff --check upstream/main...main
```

Never rewrite fork history or force-push `main` as a shortcut. Each fork commit
is independently cherry-pickable; any upstream PR must include focused
regression tests and re-run Mei's acceptance matrix before Mei advances its
pin.

**Updating Mei's pin:** push the fork commit, update the revision in
`Package.swift`, `swift package resolve` to record the fork URL/revision in
`Package.resolved`, run the focused tests + release build + long-context /
agentic acceptance probes, then commit the pin update separately from
unrelated source changes.

## Process and contributing

Keep model status/test phases in the lineup `configs/model-lineup.json` in
sync with the model docs ([`docs/MODELS.md`](MODELS.md)), and keep benchmark
methodology in [`docs/BENCHMARKING.md`](BENCHMARKING.md). See
[`CONTRIBUTING.md`](../CONTRIBUTING.md).