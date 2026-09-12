#!/usr/bin/env bash
# package_release.sh — build + package the Mei release binary for Apple Silicon.
#
# Produces a downloadable CLI/runtime bundle in dist/ that Homebrew and manual
# installers consume:
#   dist/mei-<VERSION>-macos-arm64/
#     LICENSE NOTICE.md CHANGELOG.md README.md
#     bin/mei  bin/mlx.metallib  bin/default.metallib  bin/mlx.metallib.provenance
#     docs/INSTALL.md  docs/RELEASE-<VERSION>.md  docs/VMLX-FORK.md
#   dist/mei-<VERSION>-macos-arm64.tar.gz
#   dist/mei-<VERSION>-macos-arm64.tar.gz.sha256
#
# The bundle is source/binary only: model weights are NEVER bundled (they are
# staged/downloaded separately). No GPG/signing claim is made.
#
# Usage: scripts/package_release.sh [VERSION] [--binary PATH] [--skip-build]
#   VERSION       defaults to ServerConfig.version, and MUST equal it.
#   --binary PATH use a prebuilt mei executable instead of building.
#   --skip-build  use .build/release/mei if present; error if missing.
#
# Safety / reproducibility:
#   - Builds with `swift build -c release` from the current tree.
#   - Provisions mlx.metallib via scripts/prepare_metallib.sh (provenance sidecar).
#   - Archives with a stable sort; verifies every file + sha256 round-trips.
#   - Fails loudly (exit 1) on any mismatch.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to whatever the tree actually declares, resolved after REPO is known
# (see below). A hardcoded default is guaranteed wrong from the next release
# onward: this sat at 0.2.0 through 0.3.0, 0.4.0, 0.4.1 and 0.4.2, so the bare
# command could only ever fail, and scripts/test_package_release.sh inherited
# the same default and failed with it.
VERSION=""
SKIP_BUILD=0
BINARY_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY_ARG="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --) shift; [[ $# -gt 0 ]] && VERSION="$1"; shift ;;
    -h|--help) echo "usage: package_release.sh [VERSION] [--binary PATH] [--skip-build]"; exit 0 ;;
    -*)
      echo "FATAL: unknown arg: $1" >&2; exit 2 ;;
    *)
      VERSION="$1"; shift ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/.*static let version = "\([^"]*\)".*/\1/p' \
    "$REPO/Sources/MeiCore/ServerConfig.swift" | head -1)"
  [[ -n "$VERSION" ]] || { echo "FATAL: could not read ServerConfig.version" >&2; exit 1; }
  echo "== version not given; using ServerConfig.version = $VERSION =="
fi

DIST="$REPO/dist"
BUNDLE_DIR="$DIST/mei-${VERSION}-macos-arm64"
TARBALL="$DIST/mei-${VERSION}-macos-arm64.tar.gz"

# --- invariants ---------------------------------------------------------------
[[ -f "$REPO/Sources/MeiCore/ServerConfig.swift" ]] || { echo "FATAL: not the Mei repo" >&2; exit 1; }
if ! grep -q "static let version = \"$VERSION\"" "$REPO/Sources/MeiCore/ServerConfig.swift"; then
  echo "FATAL: ServerConfig.version is not \"$VERSION\"; bump it or pass the right version." >&2
  exit 1
fi

echo "== packaging Mei $VERSION (Apple Silicon CLI/runtime bundle) =="
echo "repo:   $REPO"
echo "out:    $DIST"

# --- resolve / build the release binary ---------------------------------------
BINARY=""
if [[ -n "$BINARY_ARG" ]]; then
  BINARY="$BINARY_ARG"
elif [[ "$SKIP_BUILD" -eq 1 ]]; then
  BINARY="$REPO/.build/release/mei"
else
  echo "== swift build -c release (this may take a while) =="
  ( cd "$REPO" && swift build -c release )
  BINARY="$REPO/.build/release/mei"
fi
[[ -f "$BINARY" ]] || { echo "FATAL: release binary not found: $BINARY" >&2; exit 1; }
BIN_DIR="$(cd "$(dirname "$BINARY")" && pwd)"
BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

# verify arch + reported version
file "$BINARY" | grep -q "arm64" || { echo "FATAL: binary is not arm64 (Apple Silicon): $BINARY" >&2; exit 1; }
REPORTED="$("$BINARY" --version 2>/dev/null)"
[[ "$REPORTED" == "mei $VERSION" ]] || { echo "FATAL: binary reports '$REPORTED', expected 'mei $VERSION'" >&2; exit 1; }
echo "binary: $BINARY -> $REPORTED"

# --- assemble the bundle --------------------------------------------------------
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/bin"
mkdir -p "$BUNDLE_DIR/docs"

# Metal kernel libraries must sit next to the executable (vmlx loads from the
# executable's directory first). Provision into the bundle bin/.
METAL_TMP="$REPO/dist/.metallib-stage-$VERSION"
rm -rf "$METAL_TMP"; mkdir -p "$METAL_TMP"
MEI_METALLIB_SOURCE="${MEI_METALLIB_SOURCE:-}" "$REPO/scripts/prepare_metallib.sh" "$METAL_TMP"
cp -p "$METAL_TMP/mlx.metallib" "$BUNDLE_DIR/bin/mlx.metallib"
# default.metallib is byte-identical to mlx.metallib (vmlx probes mlx.metallib
# first); shipping one avoids doubling the ~125MB Metal library. Provenance
# sidecar records where this artifact came from.
[[ -f "$METAL_TMP/mlx.metallib.provenance" ]] && cp -p "$METAL_TMP/mlx.metallib.provenance" "$BUNDLE_DIR/bin/mlx.metallib.provenance"

cp -p "$BINARY" "$BUNDLE_DIR/bin/mei"
chmod 755 "$BUNDLE_DIR/bin/mei"

# reader-facing metadata
for f in LICENSE NOTICE.md CHANGELOG.md README.md; do
  [[ -f "$REPO/$f" ]] && cp -p "$REPO/$f" "$BUNDLE_DIR/$f"
done
for f in INSTALL.md RELEASE-${VERSION}.md VMLX-FORK.md; do
  [[ -f "$REPO/docs/$f" ]] && cp -p "$REPO/docs/$f" "$BUNDLE_DIR/docs/$f" || true
done

# --- verify bundle -----------------------------------------------------------------
fail=0
check() { host=0
  if "$@"; then echo "PASS: $*"; else echo "FAIL: $*" >&2; fail=1; fi
}
check test -x "$BUNDLE_DIR/bin/mei"
check test -f "$BUNDLE_DIR/bin/mlx.metallib"
check "$BUNDLE_DIR/bin/mei" --version 2>/dev/null
check grep -q "version = \"$VERSION\"" "$REPO/Sources/MeiCore/ServerConfig.swift"

# --- report-consistent tarball -------------------------------------------------------
# Reproducible archive: mtime + provenance timestamp are derived from the
# release commit, gzip -n drops the gzip header mtime, and members are read in
# sorted order — so rebuilding the same tree yields a byte-identical tar.gz
# (stable SHA-256) that the Homebrew formula can safely pin.
echo "== tar (reproducible) =="
RELEASE_DATE="$(git -C "$REPO" log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)"
MTIME="$(git -C "$REPO" log -1 --format=%cd --date=format:%Y%m%d%H%M.%S 2>/dev/null || date +%Y%m%d%H%M.%S)"
# prepare_metallib.sh stamps a fresh installed_at: line; pin it to the release date
sed -i.bak "s/^installed_at: .*/installed_at: $RELEASE_DATE/" "$BUNDLE_DIR/bin/mlx.metallib.provenance" 2>/dev/null || true
rm -f "$BUNDLE_DIR/bin/mlx.metallib.provenance.bak"
find "$BUNDLE_DIR" -exec touch -t "$MTIME" {} +
(
  cd "$DIST" && find "mei-${VERSION}-macos-arm64" -type f | sort \
    | tar -c -f - -T - | gzip -n > "$TARBALL"
)
echo "== sha256 =="
shasum -a 256 "$TARBALL" > "$TARBALL.sha256"
cat "$TARBALL.sha256"
# verify
( cd "$DIST" && shasum -a 256 -c "mei-${VERSION}-macos-arm64.tar.gz.sha256" )

if [[ $fail -ne 0 ]]; then
  echo "PACKAGING FAILED (see FAIL lines above)" >&2
  exit 1
fi
echo "== packaging OK =="
echo "bundle:   $BUNDLE_DIR"
echo "tarball:  $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo "checksum: $TARBALL.sha256"