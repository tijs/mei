# Installing the Mei CLI

This document covers the **source-built / user-local path** — installing a
`mei` binary you have *already built* into a single destination directory via
[`scripts/install_mei.sh`](../scripts/install_mei.sh).

Most end users should instead install the **stable v0.2.0 prebuilt release:**

- **Homebrew (recommended):**
  ```bash
  brew install tijs/tap/mei
  mei --version     # -> mei 0.2.0
  ```
- **Manual — GitHub release asset:**
  ```bash
  curl -fLO https://github.com/tijs/mei/releases/download/v0.2.0/mei-0.2.0-macos-arm64.tar.gz
  shasum -a 256 -c mei-0.2.0-macos-arm64.tar.gz.sha256
  tar -xzf mei-0.2.0-macos-arm64.tar.gz
  ./mei-0.2.0-macos-arm64/bin/mei --version
  ```

Both already carry the prebuilt `mei` binary and its `mlx.metallib` companion.
You do **not** need to build anything for those paths.

## About binaries and weights

The **source tree** contains no binary and no model weights. That is unchanged
whether you build from source or pull the release. The two differ in what the
release bundle actually ships:

- The **stable v0.2.0 release bundle includes the prebuilt binary** (and its
  `mlx.metallib`) — download and run, no build step.
- This **installer copies an already-built binary** into a single, user-local
  destination directory. It never builds, downloads, or handles weights, and
  it never writes outside that directory.

> The installer is the recommended way to put a source-built `mei` on your
> `PATH` without touching Homebrew, system directories, or anything managed by
> a package manager. (It also supports copying a prebuilt binary already
> present next to the installer — see the resolution order below.)

## Prerequisites

1. Build the release binary (see [`README.md`](../README.md#build)):

   ```bash
   swift package resolve
   swift build -c release
   # -> .build/release/mei
   ```

2. Provision the Metal kernel library the binary needs at runtime:

   ```bash
   scripts/prepare_metallib.sh .build/release
   # -> .build/release/mlx.metallib (+ default.metallib, provenance sidecar)
   ```

The installer finds these automatically next to the built binary and copies
them alongside the installed executable (vmlx loads `mlx.metallib` from the
executable's directory first).

## Install

```bash
scripts/install_mei.sh
```

With no arguments the installer:

- resolves the source binary as `.../.build/release/mei` (repo release build),
  an explicit `--binary` path, or `$MEI_BINARY`
- installs it to `$HOME/.local/bin/mei` and sets the executable bit
- copies any colocated `mlx.metallib`, `default.metallib`, and `*.provenance`
  alongside it
- refuses to touch system / package-manager prefixes (`/usr`, `/usr/local`,
  `/opt`, `/opt/homebrew`, `/bin`, `/sbin`, `/Library`) unless `--force`

## Options

| Option | Effect |
|---|---|
| `--prefix DIR` | destination directory; the executable is written to `DIR/mei`. Default `$HOME/.local/bin` |
| `--binary PATH` | explicit path to the built `mei` executable |
| `--dry-run` | print the install plan and exit; write nothing |
| `--force` | overwrite an existing, differing installed file (and companions) |
| `-h`, `--help` | usage and exit 0 |
| `-v`, `--version` | installer version and exit 0 |

Source resolution order: `--binary`, then `$MEI_BINARY`, then
`.build/release/mei` next to the repo, then `bin/mei` beside the installer
(for a bundle that ships a prebuilt binary).

## Safety contract

- **Single directory.** All writes go to `$PREFIX` (default `$HOME/.local/bin`).
  Nothing else is created or modified.
- **Preserve unless forced.** If an installed `mei` already exists and differs,
  the installer refuses (exit 1) unless `--force` is passed. A rerun with an
  identical file is a no-op (idempotent). Model weights are never involved.
- **Loud failures.** A missing source binary or a missing `--prefix` argument
  exits nonzero with a `FATAL` message; nothing is written on a failed or
  `--dry-run` invocation.

## Tests

```bash
scripts/test_install_mei.sh
```

Deterministic, weight-free, server-free: exercises `--help`, unknown-option
usage errors, missing-binary failure, `--dry-run`, install + executable bit,
idempotent rerun, `--force` overwrite, colocated Metal companions, and the
system-prefix guard.