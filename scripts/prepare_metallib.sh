#!/usr/bin/env bash
# Provision MLX's Metal kernel library (mlx.metallib) next to the mei binary.
#
# vmlx-swift's SwiftPM build does NOT emit a compiled Metal kernel library:
# the vendored mlx ships kernels as .metal sources that need Xcode's
# `metallib` archiver to compile, and on this machine that archiver component
# is NOT installed (xcrun -find metallib fails; Xcode 26.6 without the tool).
# The fallback is a prebuilt mlx.metallib from a Python mlx wheel.
#
# Version compatibility matters: vmlx-swift vendors a pinned MLX C++ version
# (Source/Cmlx/include-framework/mlx-version.h; Mei's 0.6.0 pin fef563a5
# vendors 0.32.2) and the C++ runtime looks kernels up by name at runtime
# (device.cpp load_default_library: colocated mlx.metallib first, then
# SwiftPM bundle, then compile-time METAL_PATH). A metallib from a distant mlx
# version can silently miss renamed kernels, so this script DERIVES the
# vendored version from the checkout's mlx-version.h when one is readable
# (MEI_VMLX_CHECKOUT, or the default vmlx-swift checkouts path) and falls back
# to 0.32.2 (the current release pin) when it is not, then scores every
# candidate wheel against that derived version and records provenance so every
# artifact is auditable.
#
# Search order:
#   1. $MEI_METALLIB_SOURCE if set and readable (explicit user pin)
#   2. an existing colocated mlx.metallib (no-op; already provisioned)
#   3. a Python mlx wheel whose version equals the derived vendored version,
#      preferring a venv whose name tags that version (metallib-src/.venv-0322)
#   4. a wheel in the same minor series (major.minor of the derived version)
#   5. any other adjacent-version wheel, with an explicit warning
#   6. compile via vmlx-swift's prepare-mlx-metal.sh (needs metal+metallib;
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

# Derive the vendored MLX version from the vmlx-swift checkout's
# mlx-version.h when readable (the version the C++ runtime is built against at
# the current pin), so wheel scoring and provenance track the actual pin
# instead of a hardcoded number. Fall back to 0.32.2 — the MLX version
# vendored at Mei's 0.6.0 vmlx pin (fef563a5) — when no header is readable.
MLX_VERSION_HEADER="$CHECKOUT/Source/Cmlx/include-framework/mlx-version.h"
VENDORED_MLX_VERSION=""
VENDORED_MLX_SOURCE="fallback 0.32.2 (no readable mlx-version.h at $MLX_VERSION_HEADER)"
if [[ -r "$MLX_VERSION_HEADER" ]]; then
  hdr_major=$(sed -n 's/^#define[[:space:]]*MLX_VERSION_MAJOR[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$MLX_VERSION_HEADER" | head -1)
  hdr_minor=$(sed -n 's/^#define[[:space:]]*MLX_VERSION_MINOR[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$MLX_VERSION_HEADER" | head -1)
  hdr_patch=$(sed -n 's/^#define[[:space:]]*MLX_VERSION_PATCH[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$MLX_VERSION_HEADER" | head -1)
  if [[ -n "$hdr_major" && -n "$hdr_minor" && -n "$hdr_patch" ]]; then
    VENDORED_MLX_VERSION="$hdr_major.$hdr_minor.$hdr_patch"
    VENDORED_MLX_SOURCE="$MLX_VERSION_HEADER"
  fi
fi
[[ -n "$VENDORED_MLX_VERSION" ]] || VENDORED_MLX_VERSION="0.32.2"
VMLX_VERSION_TAG="${VENDORED_MLX_VERSION//./}"   # 0.32.2 -> 0322 (wheel-venv tag)

mkdir -p "$DEST_DIR"

DEST="$DEST_DIR/mlx.metallib"
PROVENANCE="$DEST_DIR/mlx.metallib.provenance"

# ---------------------------------------------------------------------------
# Structural verification: a real Metal library starts with the MTLB magic and
# is large enough to hold the vendored kernel set (compiled libraries are
# ~125 MiB; anything under 1 MiB is a stub/error payload).
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
      echo "vendored_mlx_source: $VENDORED_MLX_SOURCE"
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

# 3+4+5) Python mlx wheels. Score each candidate's mlx version against the
# DERIVED vendored version: exact match ranks highest, same minor series next,
# anything else last (warned).
# shellcheck disable=SC2016
score_version() {
  local version="$1"
  if [[ "$version" == "$VENDORED_MLX_VERSION" ]]; then echo 10
  elif [[ "$version" == "${VENDORED_MLX_VERSION%.*}".* ]]; then echo 5
  elif [[ "$version" =~ ^0\. ]]; then echo 1
  else echo 0
  fi
}

BEST_SCORE=0
BEST_SOURCE=""
BEST_LABEL=""
BEST_VERSION=""
MATCHED_WHEEL_MISSING_METALLIB=""
declare -a WHEEL_CANDIDATES=(
  # Version-tagged scratch venv, discoverable per the derived pin (0.32.2 ->
  # .venv-0322). The legacy .venv-0311 entry stays so older wheels are still
  # found (and scored against the derived version, i.e. warned when stale).
  "$HOME/.local/share/local-model-bench/mei-build/metallib-src/.venv-$VMLX_VERSION_TAG"
  "$HOME/.local/share/local-model-bench/mei-build/metallib-src/.venv-0311"
  "$HOME/.local/share/local-model-bench/vmlx-venv"
  "$HOME/.local/share/local-model-bench/omlx-venv"
  "$HOME/.local/share/local-model-bench/.venv"
  "$HOME/projects/local-model-bench/.venv"
)
# Discover any other version-tagged scratch venv (e.g. .venv-0321) and score
# it against the derived version like every other candidate.
for tag_env in "$HOME/.local/share/local-model-bench/mei-build/metallib-src"/.venv-*; do
  [[ -d "$tag_env" ]] || continue
  case " ${WHEEL_CANDIDATES[*]} " in
    *" $tag_env "*) ;;                       # already listed (exact-tag first)
    *) WHEEL_CANDIDATES+=("$tag_env") ;;
  esac
done
for env_root in "${WHEEL_CANDIDATES[@]}"; do
  [[ -d "$env_root" ]] || continue
  # Find the wheel's site-packages: venv lib/python*/site-packages.
  local_sp=$(find "$env_root/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -1)
  [[ -n "$local_sp" ]] || continue
  # mlx version from the wheel dist-info directory name.
  version=""
  dist_info=$(find "$local_sp" -maxdepth 1 -type d -name "mlx-*.dist-info" 2>/dev/null | head -1)
  if [[ -n "$dist_info" ]]; then
    version=$(basename "$dist_info" | sed -E 's/^mlx-([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
  fi
  metallib="$local_sp/mlx/lib/mlx.metallib"
  if [[ ! -f "$metallib" ]]; then
    # A version-matched wheel with no bundled library explains the failure
    # truthfully (mlx 0.32.x wheels ship no mlx/lib/mlx.metallib).
    if [[ "$version" == "$VENDORED_MLX_VERSION" ]]; then
      MATCHED_WHEEL_MISSING_METALLIB="$env_root"
    fi
    continue
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

# 6) Compile path (only works where metal+metallib exist; not this machine).
if [[ -d "$CHECKOUT/scripts" ]]; then
  if bash "$CHECKOUT/scripts/prepare-mlx-metal.sh" "$DEST_DIR/" 2>/dev/null; then
    if verify_metallib "$DEST"; then
      {
        echo "source: compiled via $CHECKOUT/scripts/prepare-mlx-metal.sh"
        echo "label: vendored .metal sources"
        echo "verification: structural (MTLB magic, size, file classification)"
        echo "installed_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "vendored_mlx: $VENDORED_MLX_VERSION"
        echo "vendored_mlx_source: $VENDORED_MLX_SOURCE"
      } > "$PROVENANCE"
      echo "mlx.metallib compiled via prepare-mlx-metal.sh"
      exit 0
    fi
  fi
fi

echo "FATAL: could not provision a valid mlx.metallib for vendored mlx $VENDORED_MLX_VERSION at $DEST_DIR" >&2
if [[ -n "$MATCHED_WHEEL_MISSING_METALLIB" ]]; then
  echo "  - mlx==$VENDORED_MLX_VERSION is installed at $MATCHED_WHEEL_MISSING_METALLIB but its wheel does not bundle mlx/lib/mlx.metallib" >&2
fi
echo "  - Supply MEI_METALLIB_SOURCE pointing at a structurally valid, vendored-version ($VENDORED_MLX_VERSION) mlx.metallib, or" >&2
echo "  - Install Xcode's metal+metallib archiver and use the compile fallback (prepare-mlx-metal.sh), or" >&2
echo "  - Install the version-matched Python wheel and confirm its wheel carries mlx/lib/mlx.metallib: uv venv && uv pip install 'mlx==$VENDORED_MLX_VERSION'" >&2
exit 1
