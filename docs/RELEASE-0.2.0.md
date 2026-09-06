# Mei 0.2.0

First stable, downloadable Apple Silicon release — 2026-09-06.

Mei 0.2.0 is the stable promotion of the `0.2.0-alpha.1` runtime work, plus the
streaming tool-call fix, shipped as a **prebuilt Apple Silicon CLI/runtime
bundle** — either via
Homebrew (`brew install tijs/tap/mei`) or by downloading the release asset. The
server binary is a native Swift/MLX OpenAI-compatible local inference server;
it never bundles model weights (see *Weight-separation boundary*).

Implementation for this release:

- Released publicly as `v0.2.0` on 2026-09-06; the tag `v0.2.0` points at
  commit `6a53cb0` (the `release(0.2.0)` commit). Everything below is an
  ancestor of that tag and is part of this release.
- `67e897e` — `fix(tool-calls): stream multiple tool calls under distinct SSE
  indexes`.
- `00418a5` lineage (tool-argument typing + hardening), `9593126` (installer),
  and the full `0.2.0-alpha.1` runtime work.

The packaged binary is built from the commit tagged `v0.2.0` via
[`scripts/package_release.sh`](../scripts/package_release.sh) (reproducible
build/package path; see *Building the release artifact*).

## What's in this release

Runtime (verified on Sulaco, 32 GB Apple M1 Max, port 8024; see
`configs/model-lineup.json` for the source of truth on every model):

- **Ornith-1.5-35B-A3B-MLX-4bit** (primary) — 30k decode 47.5–50.3 t/s (3
  repeats) with the env-gated fused gate/up-cache config; ≥30 t/s target met.
- **Qwen3.8-27B-4bit** — loadable, acceptance-gated; tool stream/non-stream,
  parity, KV reuse and 30k long-context all PASS.
- **Gemma 4 26B-A4B** — tool strict-schema gate cleared (integer args), chunked
  prefill 256 (+91% fresh fill), disk-tier reuse.
- **Qwen3.8-Heretic (Uncensored)** — lineage gate PASS, tool parity PASS.
- **Tool-call streaming fix** (`67e897e`): multiple tool calls are now streamed
  under distinct SSE indexes (correctly delimited, not coalesced).

Tooling / packaging (new in this release):

- `scripts/install_mei.sh` — safe user-local installer for an already-built
  `mei` binary (+ colocated Metal library). Source-resolution order
  `--binary` → `$MEI_BINARY` → `.build/release/mei` → `bin/mei`.
- `scripts/package_release.sh` — reproducible builder/packager producing the
  downloadable `mei-<version>-macos-arm64.tar.gz` bundle (+ `.sha256`).
- `scripts/test_package_release.sh` — deterministic smoke checks for the
  packaging/install/version/API-identity surface of the artifact.
- Homebrew formula `tijs/tap/mei` — `brew install tijs/tap/mei`.

## Install on Apple Silicon

Prerequisites: macOS 15+, Apple Silicon (arm64). Model weights are not included
(see below).

**Homebrew (recommended):**

```bash
brew install tijs/tap/mei
mei --version        # -> mei 0.2.0
```

**Manual — download the release asset:**

```bash
# from the GitHub release page, or:
curl -fLO \
  https://github.com/tijs/mei/releases/download/v0.2.0/mei-0.2.0-macos-arm64.tar.gz
shasum -a 256 -c mei-0.2.0-macos-arm64.tar.gz.sha256   # optional integrity check
tar -xzf mei-0.2.0-macos-arm64.tar.gz
./mei-0.2.0-macos-arm64/bin/mei --version
```

Point the server at locally staged MLX checkpoints:

```bash
mei --model-dir <dir> --served-model-id <id>
```

The release bundle carries `mlx.metallib` next to the executable (vmlx loads it
from the executable's directory first); the installer copies it wherever `mei`
lands. Full local-install docs: [`docs/INSTALL.md`](INSTALL.md).

## Weight-separation boundary

Mei ships **source and binary only** — no model weights. Each model's `.safetensors`
checkpoint is a separate, explicitly staged artifact (see
`configs/model-lineup.json` for pinned revisions, digests, quant settings and
status). The runtime release never bundles `*.safetensors` / `*.gguf` / `*.bin`
blobs; the packager asserts this (`scripts/test_package_release.sh`).

## Building the release artifact (reproducible)

```bash
# from a clean-ish main checkout:
swift build -c release                    # -> .build/release/mei
scripts/prepare_metallib.sh .build/release # -> mlx.metallib + provenance
scripts/package_release.sh 0.2.0 --skip-build
# -> dist/mei-0.2.0-macos-arm64.tar.gz + .sha256
scripts/test_package_release.sh 0.2.0     # smoke checks
```

`package_release.sh` builds the release binary, provisions the version-matched
Metal library (mlx 0.31.1, matching the vendored vmlx), assembles the bundle,
tars it with a stable member order, and emits the SHA-256 checksum. It verifies
the binary is arm64 and reports exactly `mei 0.2.0` before packaging.

## Known caveats

- The vmlx fork commit `318a4e68` (Gemma 4 growing-transcript reuse fix +
  cache-fetch diagnostics) is **not yet pushed** to the fork `main`. The
  **packaged binary includes it** (built from a local edit), but an external
  pure-source build of tag `v0.2.0` resolves `Package.swift` → fork revision
  `91fed8be`, which lacks that one fix. Pushing the fork commit is a
  prerequisite for full source-reproducibility of the Gemma 4 reuse behavior.
- Qwen3.8-27B decode (4-bit 15.66 t/s, 5-bit parity 13.11 t/s) is below the 30
  t/s primary target — a hardware ceiling accepted and recorded in the plan
  (2026-09-02); Ornith-1.5-35B reaches ≥30 t/s only with the env-gated fused
  gate/up-cache disabled (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, user-gated).
- MTP / speculative decode stays out of scope.

Full acceptance evidence for every model row: see `docs/RELEASE-0.2.0-alpha.1.md`
and the cited `artifacts/*` notes.