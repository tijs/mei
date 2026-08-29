#!/usr/bin/env bash
# Provision MLX's Metal kernel library (mlx.metallib) next to the mei binary.
#
# vmlx-swift's SwiftPM build does NOT emit a compiled Metal kernel library:
# the vendored mlx ships kernels as .metal sources that need Xcode's
# `metallib` archiver to compile, and on this machine that archiver component
# is NOT installed (xcrun -find metallib fails; Xcode 26.6 without the tool).
# The fallback is a prebuilt mlx.metallib from a Python mlx wheel.
#
# Version compatibility matters: vmlx-swift vendors mlx **0.31.1**
# (Source/Cmlx/include-framework/mlx-version.h: MLX_VERSION 0.31.1) and the
# C++ runtime looks kernels up by name at runtime (device.cpp
# load_default_library: colocated mlx.metallib first, then SwiftPM bundle,
# then compile-time METAL_PATH). A metallib from a distant mlx version can
# silently miss renamed kernels, so this script prefers the same minor series
# and records provenance so every artifact is auditable.
#
# Search order:
#   1. $MEI_METALLIB_SOURCE if set and readable (explicit user pin)
#   2. an existing colocated mlx.metallib (no-op; already provisioned)
#   3. a Python mlx wheel whose version matches the vendored 0.31.x series,
#      preferring the exact 0.31.1
#   4. any other adjacent-version wheel, with an explicit warning
#   5. compile via vmlx-swift's prepare-mlx-metal.sh (needs metal+metallib;
#      verified unavailable on this machine)
#
# Every candidate is verified structurally (MTLB magic + size + `file`
# classification) before install, and a provenance sidecar records the source
# for the benchmark record. The definitive verification is runtime: the mei
# server loads the library at startup and fails loudly (missing kernels /
# "Failed to load the default metallib") if the artifact is wrong.
set -euo pipefail

DEST_DIR="${1:?usage: prepare_metallib.sh DEST_DIR}"
CHECKOUT="${MEI_VMLX_CHECKOUT:-$HOME/.local/share/local-model-bench/mei-build/checkouts/vmlx-swift}"
VENDORED_MLX_VERSION="0.31.1"  # keep in sync with Source/Cmlx/include-framework/mlx-version.h
mkdir -p "$DEST_DIR"

DEST="$DEST_DIR/mlx.metallib"
PROVENANCE="$DEST_DIR/mlx.metallib.provenance"

# ---------------------------------------------------------------------------
# Structural verification: a real Metal library starts with the MTLB magic and
# is large enough to hold the vendored kernel set (the 0.31.1 wheel's is
# ~131MB; anything under 1MB is a stub/error payload).
# ---------------------------------------------------------------------------
verify_metallib() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local size
  size=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo 0)
  [[ "$size" -gt 1048576 ]] || return 1
  local magic
  magic=$(head -c 4 "$path" 2>/dev/null | od -An -tx1 | tr -d ' \n')
  [[ "$magic" == "4d544c42" ]] || return 1
  if command -v file >/dev/null 2>&1; then
    file "$path" | grep -q "MetalLib" || return 1
  fi
  return 0
}

provision_from() {
  local source="$1" label="$2" warn="${3:-}"
  if [[ -s "$source" ]] && verify_metallib "$source"; then
    cp "$source" "$DEST"
    cp "$source" "$DEST_DIR/default.metallib" 2>/dev/null || true
    {
      echo "source: $source"
      echo "label: $label"
      echo "verification: structural (MTLB magic, size, file classification)"
      echo "installed_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "vendored_mlx: $VENDORED_MLX_VERSION"
      if [[ -n "$warn" ]]; then
        echo "warning: $warn"
      fi
    } > "$PROVENANCE"
    echo "mlx.metallib installed from $label ($source)"
    if [[ -n "$warn" ]]; then
      echo "WARNING: $warn" >&2
    fi
    return 0
  fi
  return 1
}

# 1) Explicit pin wins.
if [[ -n "${MEI_METALLIB_SOURCE:-}" ]]; then
  if provision_from "$MEI_METALLIB_SOURCE" "MEI_METALLIB_SOURCE override"; then
    exit 0
  fi
  echo "FATAL: MEI_METALLIB_SOURCE is set but the file is not a valid metallib: $MEI_METALLIB_SOURCE" >&2
  exit 1
fi

# 2) Already provisioned and structurally intact: no-op.
if verify_metallib "$DEST"; then
  echo "mlx.metallib already present and verified at $DEST"
  [[ -f "$PROVENANCE" ]] || { echo "source: (pre-existing, no provenance recorded)" > "$PROVENANCE"; }
  exit 0
fi

# 3+4) Python mlx wheels. Score each candidate's mlx version against the
# vendored 0.31.1: exact match ranks highest, same minor series next,
# anything else last (warned).
# shellcheck disable=SC2016
score_version() {
  local version="$1"
  if [[ "$version" == "$VENDORED_MLX_VERSION" ]]; then echo 10
  elif [[ "$version" == 0.31.* ]]; then echo 5
  elif [[ "$version" =~ ^0\. ]]; then echo 1
  else echo 0
  fi
}

BEST_SCORE=0
BEST_SOURCE=""
BEST_LABEL=""
BEST_VERSION=""
declare -a WHEEL_CANDIDATES=(
  "$HOME/.local/share/local-model-bench/mei-build/metallib-src/.venv-0311"
  "$HOME/.local/share/local-model-bench/vmlx-venv"
  "$HOME/.local/share/local-model-bench/omlx-venv"
  "$HOME/.local/share/local-model-bench/.venv"
  "$HOME/projects/local-model-bench/.venv"
)
for env_root in "${WHEEL_CANDIDATES[@]}"; do
  [[ -d "$env_root" ]] || continue
  # Find the wheel's site-packages: venv lib/python*/site-packages.
  local_sp=$(find "$env_root/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -1)
  [[ -n "$local_sp" ]] || continue
  metallib="$local_sp/mlx/lib/mlx.metallib"
  [[ -f "$metallib" ]] || continue
  # mlx version from the wheel dist-info directory name.
  version=""
  dist_info=$(find "$local_sp" -maxdepth 1 -type d -name "mlx-*.dist-info" 2>/dev/null | head -1)
  if [[ -n "$dist_info" ]]; then
    version=$(basename "$dist_info" | sed -E 's/^mlx-([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
  fi
  score=$(score_version "${version:-0.0.0}")
  if [[ "$score" -gt "$BEST_SCORE" ]]; then
    BEST_SCORE=$score
    BEST_SOURCE=$metallib
    BEST_LABEL="Python mlx wheel version ${version:-unknown} ($env_root)"
    BEST_VERSION=$version
  fi
done

if [[ -n "$BEST_SOURCE" ]]; then
  warn=""
  if [[ "$BEST_SCORE" -lt 5 ]]; then
    warn="mlx wheel version ${BEST_VERSION:-unknown} does not match vendored mlx $VENDORED_MLX_VERSION; kernels are looked up by name at runtime and may be missing. Install mlx==$VENDORED_MLX_VERSION into a scratch venv for a version-matched artifact."
  fi
  if provision_from "$BEST_SOURCE" "$BEST_LABEL" "$warn"; then
    exit 0
  fi
fi

# 5) Compile path (only works where metal+metallib exist; not this machine).
if [[ -d "$CHECKOUT/scripts" ]]; then
  if bash "$CHECKOUT/scripts/prepare-mlx-metal.sh" "$DEST_DIR/" 2>/dev/null; then
    if verify_metallib "$DEST"; then
      {
        echo "source: compiled via $CHECKOUT/scripts/prepare-mlx-metal.sh"
        echo "label: vendored .metal sources"
        echo "verification: structural (MTLB magic, size, file classification)"
        echo "installed_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "vendored_mlx: $VENDORED_MLX_VERSION"
      } > "$PROVENANCE"
      echo "mlx.metallib compiled via prepare-mlx-metal.sh"
      exit 0
    fi
  fi
fi

echo "FATAL: could not provision a valid mlx.metallib for $DEST_DIR" >&2
echo "  - Install MLX's metallib archiver component or" >&2
echo "  - Install the version-matched Python wheel: uv venv && uv pip install 'mlx==$VENDORED_MLX_VERSION'" >&2
exit 1