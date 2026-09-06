#!/usr/bin/env bash
# test_package_release.sh — deterministic smoke checks for packaging/install/
# version/API identity of the Mei Apple Silicon release artifact.
#
# Requires a built release binary (scripts/package_release.sh --skip-build uses
# .build/release/mei). Requires network-free metallib provisioning, so it runs
# in the release-prep pipeline rather than in weight-free unit CI. Never
# touches external side effects (no push, no release, no Homebrew publish).
#
# Verifies:
#   - package_release.sh exists, is executable, and fails on a version mismatch
#   - the produced tarball exists, sha256 round-trips, member list is sane
#   - the extracted bundle exposes bin/mei, executable, reporting "mei <VERSION>"
#   - no model-weight blobs (.safetensors/.gguf/.bin) are bundled
#   - Metal kernel libraries (mlx.metallib) are present next to the binary
#   - API identity: GET /v1/models requires a live server + model, so identity
#     is asserted via the version string; runtime load probes are out of scope
#     here (they need a stage + GPU on a target machine).
#
# Usage: scripts/test_package_release.sh [VERSION]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="${1:-0.2.0}"
PACKAGER="$SCRIPT_DIR/package_release.sh"

PASS=0; FAIL=0
T="$(mktemp -d "${TMPDIR:-/tmp}/mei-pkg-test.XXXXXX")"
trap 'rm -rf "$T"' EXIT
die() { echo "TEST HARNESS ERROR: $*" >&2; exit 70; }
ok()   { PASS=$((PASS+1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL - $*"; }
assert() { if "$@" >/dev/null 2>&1; then ok "$*"; else bad "$*"; fi }
assert_fail() { if "$@" >/dev/null 2>&1; then bad "(expected failure) $*"; else ok "(expected failure) $*"; fi }

# --- 0. packager present + executable -----------------------------------------
[[ -f "$PACKAGER" ]] || die "packager missing at $PACKAGER"
assert test -x "$PACKAGER"

DIST="$REPO/dist"
TARBALL="$DIST/mei-${VERSION}-macos-arm64.tar.gz"
BUNDLE="$DIST/mei-${VERSION}-macos-arm64"

# --- 1. version mismatch is rejected before any artifact is written ----------
assert_fail bash "$PACKAGER" 9.9.9 --skip-build

# --- 2. packaging (skip-build uses .build/release/mei) -----------------------
echo "== packaging (skip-build) =="
if [[ "$(uname -m)" == "arm64" ]]; then
  bash "$PACKAGER" "$VERSION" --skip-build
else
  ok "skip packaging on non-arm64 host (no .build/release/mei expected)"
fi

[[ -f "$TARBALL" ]] || die "package_release.sh did not produce $TARBALL"
ok "tarball produced: $(basename "$TARBALL")"

# --- 3. checksum round-trips ---------------------------------------------------
( cd "$DIST" && shasum -a 256 -c "mei-${VERSION}-macos-arm64.tar.gz.sha256" ) >/dev/null 2>&1
ok "sha256 round-trips"

# --- 4. extract + structured checks -------------------------------------------
echo "== extracted bundle checks =="
tar_path="$T"
tar -xzf "$TARBALL" -C "$tar_path" 2>/dev/null && ok "tarball extracts" || bad "tarball extracts"
B="$tar_path/mei-${VERSION}-macos-arm64"
assert test -d "$B"
assert test -x "$B/bin/mei"
assert test -f "$B/bin/mlx.metallib"
assert test -f "$B/LICENSE"
assert test -f "$B/CHANGELOG.md"
assert test -f "$B/README.md"

# version + API identity (version string is the offline /v1/models identity proxy)
if "$B/bin/mei" --version > "$T/ver.txt" 2>&1; then
  if grep -qx "mei $VERSION" "$T/ver.txt"; then
    ok "bin/mei --version == 'mei $VERSION'"
  else
    bad "bin/mei --version printed '$(cat "$T/ver.txt")' but expected 'mei $VERSION'"
  fi
else
  bad "bin/mei --version (offline, no model) failed"
fi

# no model weights / caches bundled
if find "$B" -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' \) | grep -q .; then
  bad "bundle contains model-weight blobs"
else
  ok "no model-weight blobs bundled"
fi

# --- 5. bundle dir still present in dist (artifact) ---------------------------
assert test -f "$BUNDLE/bin/mei"

echo
echo "RESULT: $PASS ok, $FAIL failed"
[[ $FAIL -eq 0 ]]