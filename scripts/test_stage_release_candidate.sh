#!/usr/bin/env bash
# test_stage_release_candidate.sh — deterministic, weight-free checks for the
# source-first release staging path: scripts/stage_release_candidate.sh +
# configs/release-allowlist.json.
#
# Verifies:
#   - the stager's version default derives from ServerConfig.version (no
#     hardcoded release number), so a version bump cannot leave the source
#     staging path behind
#   - the allowlist release metadata matches the repo's ServerConfig.version
#   - a default (no-argument) staging run succeeds and stages the allowlist
#     closure for that version: version constant, changelog and release notes
#     agree, excluded dirs (artifacts/.git/.build/dist) and weight blobs are
#     absent, and the sha256 manifest round-trips
#   - a mismatched version fails loudly instead of staging a candidate
# No network, no server, no model. Leaves the staged candidate
# dist/mei-<version>-src in place on success (gitignored); the failed
# mismatch probe tree is removed.
#
# Usage: scripts/test_stage_release_candidate.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGER="$SCRIPT_DIR/stage_release_candidate.sh"

PASS=0; FAIL=0
die()  { echo "TEST HARNESS ERROR: $*" >&2; exit 70; }
ok()   { PASS=$((PASS+1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL - $*"; }
assert()      { if "$@" >/dev/null 2>&1; then ok "$@"; else bad "$@"; fi }
assert_fail() { if "$@" >/dev/null 2>&1; then bad "(expected failure) $*"; else ok "(expected failure) $*"; fi }

[[ -f "$STAGER" ]] || die "stager missing at $STAGER"

# --- expected version: derived exactly like the stager does -----------------
VERSION="$(sed -n 's/.*static let version = "\([^"]*\)".*/\1/p' \
  "$REPO/Sources/MeiCore/ServerConfig.swift" | head -1)"
[[ -n "$VERSION" ]] || die "cannot read ServerConfig.version"
echo "== source staging checks (version $VERSION) =="

# --- 0. metadata: allowlist release matches ServerConfig.version ------------
assert python3 -c "
import json,sys
d=json.load(open('$REPO/configs/release-allowlist.json'))
sys.exit(0 if d['release'] == '$VERSION' else 1)
"

# --- 1. default derivation: a no-arg run stages dist/mei-<VERSION>-src ------
# (Remove the target first: after the run it must exist because the no-arg
# default derived this version, not a hardcoded one.)
rm -rf "$REPO/dist/mei-$VERSION-src"
if bash "$STAGER" >/dev/null 2>&1; then
  ok "stage_release_candidate.sh (no arg) exits 0"
else
  bad "stage_release_candidate.sh (no arg) exits 0"
fi
STAGE="$REPO/dist/mei-$VERSION-src"
assert test -d "$STAGE"

# --- 2. version metadata consistent inside the staged tree ------------------
assert grep -q "static let version = \"$VERSION\"" "$STAGE/Sources/MeiCore/ServerConfig.swift"
assert grep -q "## \[$VERSION\]" "$STAGE/CHANGELOG.md"
assert test -f "$STAGE/docs/RELEASE-$VERSION.md"
assert grep -q "Mei $VERSION" "$STAGE/docs/RELEASE-$VERSION.md"

# --- 3. release content present (0.9.0 concurrency slice) -------------------
assert test -f "$STAGE/scripts/test_concurrent_generations_integration.py"
assert test -f "$STAGE/scripts/integration/peix_like.py"
assert test -f "$STAGE/docs/INFERENCE-SCHEDULING.md"

# --- 4. exclusions: no history/build/bundle/weight paths --------------------
assert bash -c "cd '$STAGE' && ! test -e artifacts && ! test -e .git && ! test -e .build && ! test -e dist && ! test -e __pycache__"
assert bash -c "cd '$STAGE' && ! find . -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' \) | grep -q ."

# --- 5. manifest: summary fields + sha256 round-trip ------------------------
assert python3 -c "
import json,sys
m=json.load(open('$STAGE/staging-manifest.json'))
sys.exit(0 if m['release']=='$VERSION' and m['kind']=='source-first' and m['file_count']>0 else 1)
"
assert bash -c "cd '$STAGE' && shasum -a 256 -c STAGING-MANIFEST.sha256 >/dev/null 2>&1"

# --- 6. a mismatched version fails loudly -----------------------------------
assert_fail bash "$STAGER" 9.9.9
rm -rf "$REPO/dist/mei-9.9.9-src"

echo
echo "RESULT: $PASS ok, $FAIL failed"
[[ $FAIL -eq 0 ]]
