#!/usr/bin/env bash
# Apply the Mei vmlx-swift patch series onto pinned checkouts.
#
# Provenance contract:
#   - Every patched checkout must be vmlx-swift at the exact pinned revision
#     aeb5e21c195d8519609488ef75a25ce7e48d8f88 (the revision Mei's
#     Package.resolved pins); the script refuses to patch anything else.
#   - patches/0001..0003 are generated from this checkout's git diff, so
#     they apply byte-exactly to the pristine pinned tree.
#   - SwiftPM keeps a dependency copy under BOTH the in-repo default
#     build dir (.build/checkouts) and any --scratch-path build dir
#     (e.g. ~/.local/share/local-model-bench/mei-build/checkouts); the
#     release binary is built from the scratch copy, so both must be
#     patched. The script patches every copy it can find.
#   - Idempotent: per checkout, if the marker exists AND the patches
#     reverse-apply cleanly, it does nothing.
#
# Usage: scripts/apply_vmlx_patches.sh [--reset]
#   --reset   revert every checkout to the pristine pinned revision first.
set -euo pipefail

MEI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="aeb5e21c195d8519609488ef75a25ce7e48d8f88"

PATCHES=(
  "$MEI_REPO/patches/0001-quantized-rotating-kv.patch"
  "$MEI_REPO/patches/0002-quantized-rotating-diskstore.patch"
  "$MEI_REPO/patches/0003-compiled-decode-threshold.patch"
)

CHECKOUT_CANDIDATES=(
  "$MEI_REPO/.build/checkouts/vmlx-swift"
  "$HOME/.local/share/local-model-bench/mei-build/checkouts/vmlx-swift"
)

RESET="${1:-}"
for CHECKOUT in "${CHECKOUT_CANDIDATES[@]}"; do
  [[ -d "$CHECKOUT/.git" ]] || { echo "skip (no checkout at $CHECKOUT)"; continue; }
  MARKER="$CHECKOUT/.mei-patches-applied"

  if [[ "$RESET" == "--reset" ]]; then
    git -C "$CHECKOUT" checkout -- . 2>/dev/null || true
    rm -f "$MARKER"
    echo "reset $CHECKOUT to pristine pinned tree"
  fi

  CURRENT="$(git -C "$CHECKOUT" rev-parse HEAD)"
  if [[ "$CURRENT" != "$PIN" ]]; then
    echo "FATAL: checkout $CHECKOUT is at $CURRENT, not the pinned revision $PIN" >&2
    exit 1
  fi

  if [[ -f "$MARKER" ]]; then
    all_applied=true
    for p in "${PATCHES[@]}"; do
      if ! git -C "$CHECKOUT" apply --reverse --check "$p" >/dev/null 2>&1; then
        all_applied=false
        break
      fi
    done
    if $all_applied; then
      echo "patches already applied at $CHECKOUT (marker $MARKER)"
      continue
    fi
  fi

  for p in "${PATCHES[@]}"; do
    [[ -f "$p" ]] || { echo "FATAL: missing patch $p" >&2; exit 1; }
    if git -C "$CHECKOUT" apply --reverse --check "$p" >/dev/null 2>&1; then
      # Already in the working tree (marker may be stale/missing) — fine.
      continue
    fi
    if ! git -C "$CHECKOUT" apply --check "$p" >/dev/null 2>&1; then
      echo "FATAL: patch $p does not apply cleanly at $CHECKOUT (checkout modified?)" >&2
      exit 1
    fi
  done

  for p in "${PATCHES[@]}"; do
    if git -C "$CHECKOUT" apply --reverse --check "$p" >/dev/null 2>&1; then
      echo "already present: $(basename "$p")"
      continue
    fi
    git -C "$CHECKOUT" apply "$p"
    echo "applied $(basename "$p") at $CHECKOUT"
  done

  find "$CHECKOUT" -name '*.swift' -exec chmod u+w {} + 2>/dev/null || true
  touch "$MARKER"
  echo "vmlx patch series applied at $CHECKOUT (marker $MARKER)"
done
echo "done"