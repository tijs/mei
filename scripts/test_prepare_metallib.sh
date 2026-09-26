#!/usr/bin/env bash
# test_prepare_metallib.sh — hermetic checks for scripts/prepare_metallib.sh.
#
# Everything runs in disposable tmp dirs with a fresh HOME per test: no
# network, no writes outside $T, no real 125 MiB artifacts required. The MTLB
# fixture is synthetic: "MTLB" magic + 2 MiB of zeros. `file` classifies that
# as "MetalLib, version 0.0.0", which satisfies prepare_metallib.sh's
# structural gate (magic, size > 1 MiB, file classification) — the exact gate
# under test; a fully compiled library would add nothing here.
#
# Verifies:
#   - the vendored MLX version is DERIVED from MEI_VMLX_CHECKOUT's mlx-version.h
#   - fallback to 0.32.2 when no header is readable
#   - wheel scoring runs against the DERIVED version: a 0.31.1 wheel must NOT
#     score as exact against a 0.32.2 pin (the 0.6.0 release bug) and must
#     provision only with an explicit mismatch warning; all wheels warn that
#     version matching does not establish fork kernel identity
#   - a version-matched wheel that bundles no mlx/lib/mlx.metallib is reported
#     truthfully instead of silently skipping
#   - MEI_METALLIB_SOURCE override still wins over every wheel candidate
#   - an already-provisioned, structurally valid dest is a no-op
#   - ...unless its provenance names a different vendored MLX version
#
# Usage: scripts/test_prepare_metallib.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PREPARE="$SCRIPT_DIR/prepare_metallib.sh"
[[ -f "$PREPARE" ]] || { echo "TEST HARNESS ERROR: prepare_metallib.sh missing" >&2; exit 70; }
command -v file >/dev/null 2>&1 || { echo "TEST HARNESS ERROR: 'file' required" >&2; exit 70; }

PASS=0; FAIL=0
T="$(mktemp -d "${TMPDIR:-/tmp}/mei-metallib-test.XXXXXX")"
trap 'rm -rf "$T"' EXIT
ok()   { PASS=$((PASS+1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL - $*"; }

# --- helpers ----------------------------------------------------------------
make_fixture() { # path  -> synthetic MTLB-backed metallib (passes the gate)
  { printf 'MTLB'; dd if=/dev/zero bs=1048576 count=2 2>/dev/null; } > "$1"
}

make_checkout() { # dir version(MAJOR.MINOR.PATCH | "none")
  local dir="$1" ver="$2"
  local header="$dir/Source/Cmlx/include-framework/mlx-version.h"
  mkdir -p "$(dirname "$header")"
  if [[ "$ver" != "none" ]]; then
    local ma mi pa
    IFS=. read -r ma mi pa <<< "$ver"
    printf '#pragma once\n#define MLX_VERSION_MAJOR %s\n#define MLX_VERSION_MINOR %s\n#define MLX_VERSION_PATCH %s\n' \
      "$ma" "$mi" "$pa" > "$header"
  fi
}

make_wheel_env() { # home version fixture  -> home/.local/.../mei-build/metallib-src/.venv-<tag>
  local home="$1" ver="$2" fx="$3"
  local root="$home/.local/share/local-model-bench/mei-build/metallib-src/.venv-${ver//./}"
  local sp="$root/lib/python3.12/site-packages"
  mkdir -p "$sp/mlx-${ver}.dist-info" "$sp/mlx/lib"
  cp "$fx" "$sp/mlx/lib/mlx.metallib"
}

source_fixture="$T/source.metallib"
make_fixture "$source_fixture"

run_prepare() { # dest checkout home [extra env...]
  local dest="$1" checkout="$2" home="$3"
  shift 3
  mkdir -p "$T/run-$(basename "$dest")"
  HOME="$home" MEI_VMLX_CHECKOUT="$checkout" "$@" bash "$PREPARE" "$dest" \
    2> "$T/run-$(basename "$dest")/stderr" > "$T/run-$(basename "$dest")/stdout"
  echo "$?"
}

# --- 1. vendored version DERIVED from mlx-version.h (0.32.2 pin) ------------
make_checkout "$T/co-0322" "0.32.2"
make_wheel_env "$T/h1" "0.32.2" "$source_fixture"
rc=$(run_prepare "$T/dest-exact" "$T/co-0322" "$T/h1")
if [[ "$rc" -eq 0 ]] && [[ -f "$T/dest-exact/mlx.metallib" ]] \
   && grep -q '^vendored_mlx: 0\.32\.2$' "$T/dest-exact/mlx.metallib.provenance" \
   && grep -q 'mlx-version\.h' "$T/dest-exact/mlx.metallib.provenance" \
   && grep -q 'version 0\.32\.2' "$T/dest-exact/mlx.metallib.provenance" \
   && grep -q '^warning: Using a stock Python wheel' "$T/dest-exact/mlx.metallib.provenance"; then
  ok "exact-match wheel scored against derived 0.32.2, with stock-wheel warning"
else
  bad "exact-match 0.32.2 wheel (rc=$rc): $(cat "$T/run-dest-exact/stderr")"
fi

# --- 2. 0.31.1 wheel against the 0.32.2 pin must NOT be exact (the 0.6.0 bug) -
make_checkout "$T/co-0322b" "0.32.2"
make_wheel_env "$T/h2" "0.31.1" "$source_fixture"
rc=$(run_prepare "$T/dest-mismatch" "$T/co-0322b" "$T/h2")
if [[ "$rc" -eq 0 ]] && [[ -f "$T/dest-mismatch/mlx.metallib" ]] \
   && grep -q '^vendored_mlx: 0\.32\.2$' "$T/dest-mismatch/mlx.metallib.provenance" \
   && grep -q '^warning: mlx wheel version 0\.31\.1 does not match vendored mlx 0\.32\.2' \
        "$T/dest-mismatch/mlx.metallib.provenance"; then
  ok "0.31.1 wheel vs 0.32.2 pin: provisioned with explicit version-mismatch warning"
else
  bad "0.31.1 wheel scored wrong vs 0.32.2 pin (rc=$rc): $(cat "$T/run-dest-mismatch/stderr")"
fi

# --- 3. same-minor (0.32.1) provisions with stock-wheel warning --------------
make_checkout "$T/co-0322c" "0.32.2"
make_wheel_env "$T/h3" "0.32.1" "$source_fixture"
rc=$(run_prepare "$T/dest-sameminor" "$T/co-0322c" "$T/h3")
if [[ "$rc" -eq 0 ]] && [[ -f "$T/dest-sameminor/mlx.metallib" ]] \
   && grep -q 'version 0\.32\.1' "$T/dest-sameminor/mlx.metallib.provenance" \
   && grep -q '^warning: Using a stock Python wheel' "$T/dest-sameminor/mlx.metallib.provenance"; then
  ok "same-minor 0.32.1 wheel scored against derived 0.32.2, with stock-wheel warning"
else
  bad "same-minor 0.32.1 wheel (rc=$rc): $(cat "$T/run-dest-sameminor/stderr")"
fi

# --- 4. fallback: no readable header -> 0.32.2, truthful FATAL ---------------
make_checkout "$T/co-none" "none"
rc=$(run_prepare "$T/dest-fallback" "$T/co-none" "$T/h4")
if [[ "$rc" -ne 0 ]] && grep -q '0\.32\.2' "$T/run-dest-fallback/stderr" \
   && ! grep -q '0\.31\.1' "$T/run-dest-fallback/stderr"; then
  ok "no header: falls back to current-release 0.32.2 in FATAL, never claims 0.31.1"
else
  bad "fallback version wrong (rc=$rc): $(cat "$T/run-dest-fallback/stderr")"
fi

# --- 5. header beats fallback (0.33.0 test pin) ------------------------------
make_checkout "$T/co-0330" "0.33.0"
rc=$(run_prepare "$T/dest-0330" "$T/co-0330" "$T/h5")
if [[ "$rc" -ne 0 ]] && grep -q '0\.33\.0' "$T/run-dest-0330/stderr" \
   && ! grep -q '0\.32\.2' "$T/run-dest-0330/stderr"; then
  ok "header-derived version (0.33.0) drives FATAL, not the 0.32.2 fallback"
else
  bad "header derivation ignored (rc=$rc): $(cat "$T/run-dest-0330/stderr")"
fi

# --- 6. version-matched wheel with no bundled metallib is reported -----------
make_checkout "$T/co-0322d" "0.32.2"
mkdir -p "$T/h6/.local/share/local-model-bench/mei-build/metallib-src/.venv-0322/lib/python3.12/site-packages/mlx-0.32.2.dist-info"
rc=$(run_prepare "$T/dest-wheelnometal" "$T/co-0322d" "$T/h6")
if [[ "$rc" -ne 0 ]] && grep -q 'does not bundle mlx/lib/mlx\.metallib' \
     "$T/run-dest-wheelnometal/stderr"; then
  ok "matched wheel without bundled metallib reported truthfully in FATAL"
else
  bad "matched-no-metallib wheel not reported (rc=$rc): $(cat "$T/run-dest-wheelnometal/stderr")"
fi

# --- 7. MEI_METALLIB_SOURCE override wins over wheels ------------------------
make_checkout "$T/co-0322e" "0.32.2"
make_wheel_env "$T/h7" "0.32.2" "$source_fixture"
rc=$(HOME="$T/h7" MEI_VMLX_CHECKOUT="$T/co-0322e" MEI_METALLIB_SOURCE="$source_fixture" \
  bash "$PREPARE" "$T/dest-override" 2>/dev/null >/dev/null; echo "$?")
if [[ "$rc" -eq 0 ]] && grep -q "^source: $source_fixture\$" "$T/dest-override/mlx.metallib.provenance" \
   && grep -q '^vendored_mlx: 0\.32\.2$' "$T/dest-override/mlx.metallib.provenance"; then
  ok "MEI_METALLIB_SOURCE wins over wheels; provenance records derived 0.32.2"
else
  bad "MEI_METALLIB_SOURCE override (rc=$rc)"
fi

# --- 8. already-provisioned valid dest is a no-op ----------------------------
mkdir -p "$T/dest-noop"
cp "$source_fixture" "$T/dest-noop/mlx.metallib"
rc=$(run_prepare "$T/dest-noop" "$T/co-0322e" "$T/h8")
if [[ "$rc" -eq 0 ]] && grep -q 'already present and verified' "$T/run-dest-noop/stdout"; then
  ok "already-provisioned valid metallib is a no-op"
else
  bad "no-op path (rc=$rc): $(cat "$T/run-dest-noop/stdout" "$T/run-dest-noop/stderr")"
fi

# --- 9. a dest provisioned for another MLX version is re-provisioned ---------
mkdir -p "$T/dest-stale"
cp "$source_fixture" "$T/dest-stale/mlx.metallib"
printf 'source: old\nvendored_mlx: 0.31.1\n' > "$T/dest-stale/mlx.metallib.provenance"
make_wheel_env "$T/h9" "0.32.2" "$source_fixture"
rc=$(run_prepare "$T/dest-stale" "$T/co-0322e" "$T/h9")
if [[ "$rc" -eq 0 ]] && grep -q 'provisioned for mlx 0\.31\.1' "$T/run-dest-stale/stderr" \
   && grep -q '^vendored_mlx: 0\.32\.2$' "$T/dest-stale/mlx.metallib.provenance" \
   && grep -q 'version 0\.32\.2' "$T/dest-stale/mlx.metallib.provenance"; then
  ok "metallib provisioned for 0.31.1 is replaced under a 0.32.2 pin"
else
  bad "stale-version dest kept (rc=$rc): $(cat "$T/run-dest-stale/stdout" "$T/run-dest-stale/stderr")"
fi

# --- Build-product selection: overwrite a valid old library; package elsewhere.
for layout in Contents/Resources .; do
  build="$T/built-${layout//\//-}"
  mkdir -p "$build/mlx-swift_Cmlx.bundle/$layout"
  built="$build/mlx-swift_Cmlx.bundle/$layout/default.metallib"
  cp "$source_fixture" "$built"
  printf 'source-build' >> "$built"
  cp "$source_fixture" "$build/mlx.metallib"
  rc=$(run_prepare "$build" "$T/co-0322" "$T/h1")
  if [[ "$rc" -eq 0 ]] && cmp -s "$built" "$build/mlx.metallib" \
     && grep -q '^label: SwiftPM Cmlx build product$' "$build/mlx.metallib.provenance" \
     && grep -q "^sha256: $(shasum -a 256 "$built" | awk '{print $1}')$" "$build/mlx.metallib.provenance"; then
    ok "Cmlx bundle ($layout) wins over existing library and exact wheel"
  else
    bad "Cmlx bundle selection ($layout, rc=$rc)"
  fi
  rc=$(run_prepare "$T/package-${layout//\//-}" "$T/co-0322" "$T/h1" env MEI_METALLIB_BUILD_DIR="$build")
  if [[ "$rc" -eq 0 ]] && cmp -s "$built" "$T/package-${layout//\//-}/mlx.metallib"; then
    ok "explicit build-product dir supports separate packaging destination ($layout)"
  else
    bad "separate packaging destination ($layout, rc=$rc)"
  fi
  rc=$(run_prepare "$T/override-${layout//\//-}" "$T/co-0322" "$T/h1" env \
    MEI_METALLIB_BUILD_DIR="$build" MEI_METALLIB_SOURCE="$source_fixture")
  if [[ "$rc" -eq 0 ]] && cmp -s "$source_fixture" "$T/override-${layout//\//-}/mlx.metallib"; then
    ok "explicit override still wins over Cmlx bundle ($layout)"
  else
    bad "explicit override lost to build product ($layout)"
  fi
  printf 'invalid' > "$built"
  rc=$(run_prepare "$build" "$T/co-0322" "$T/h1")
  if [[ "$rc" -ne 0 ]]; then
    ok "invalid Cmlx bundle fails instead of reusing old library or wheel ($layout)"
  else
    bad "invalid Cmlx bundle silently bypassed ($layout)"
  fi
done
rc=$(run_prepare "$T/missing-built" "$T/co-0322" "$T/h1" env MEI_METALLIB_BUILD_DIR="$T/absent")
if [[ "$rc" -ne 0 ]]; then
  ok "explicit build-product directory without library fails even with a wheel available"
else
  bad "missing requested build product silently fell back"
fi

echo
echo "RESULT: $PASS ok, $FAIL failed"
[[ $FAIL -eq 0 ]]