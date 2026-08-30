#!/usr/bin/env bash
# Apply the Mei vmlx-swift patch series onto pinned checkouts.
#
# Provenance contract:
#   - Every patched checkout must be vmlx-swift at the exact pinned revision
#     aeb5e21c195d8519609488ef75a25ce7e48d8f88 (the revision Mei's
#     Package.resolved pins); the script refuses to patch anything else.
#   - patches/0001..0004 are generated from this checkout's git diff, so
#     they apply byte-exactly to the pristine pinned tree, in order.
#   - SwiftPM keeps a dependency copy under BOTH the in-repo default build
#     dir (.build/checkouts) and any --scratch-path build dir
#     (e.g. ~/.local/share/local-model-bench/mei-build/checkouts); the
#     release binary is built from the scratch copy, so both must be
#     patched. The script patches every copy it can find.
#   - Idempotence is sentinel-based (each patch has a distinctive line the
#     working tree must contain; partial stacks are repaired by reseting
#     that checkout and re-applying the whole series).
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
  "$MEI_REPO/patches/0004-max-kv-window-probe.patch"
)

# Distinctive sentinel strings per patch, checked in the working tree.
SENTINELS=(
  "Libraries/MLXLMCommon/KVCache.swift:class QuantizedRotatingKVCache"
  "Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift:serializeQuantizedRotatingLayer"
  "Libraries/MLXLMCommon/Evaluate.swift:compiledDecodeMaxPromptOffset"
  "Libraries/MLXLLM/Models/Qwen35.swift:maxKVWindowSize"
)
# The Evaluate sentinel is in two patches (0003+0004); keep an extra check
# so a half-applied stack is caught.
EXTRA_SENTINELS=(
  "Libraries/MLXLLM/Models/Qwen35.swift:Experimental bounded-window probe"
)

CHECKOUT_CANDIDATES=(
  "$MEI_REPO/.build/checkouts/vmlx-swift"
  "$HOME/.local/share/local-model-bench/mei-build/checkouts/vmlx-swift"
)

RESET="${1:-}"
for CHECKOUT in "${CHECKOUT_CANDIDATES[@]}"; do
  [[ -d "$CHECKOUT/.git" ]] || { echo "skip (no checkout at $CHECKOUT)"; continue; }

  if [[ "$RESET" == "--reset" ]]; then
    git -C "$CHECKOUT" checkout -- . 2>/dev/null || true
    echo "reset $CHECKOUT to pristine pinned tree"
  fi

  CURRENT="$(git -C "$CHECKOUT" rev-parse HEAD)"
  if [[ "$CURRENT" != "$PIN" ]]; then
    echo "FATAL: checkout $CHECKOUT is at $CURRENT, not the pinned revision $PIN" >&2
    exit 1
  fi

  applied=true
  for sentinel in "${SENTINELS[@]}" "${EXTRA_SENTINELS[@]}"; do
    file="${sentinel%%:*}"
    needle="${sentinel#*:}"
    if ! grep -qF "$needle" "$CHECKOUT/$file" 2>/dev/null; then
      applied=false
      break
    fi
  done

  if $applied; then
    echo "patch series already applied at $CHECKOUT (sentinel check)"
    continue
  fi

  # Not applied (or partial): repair from the pristine tree, then apply.
  echo "repairing $CHECKOUT (partial or missing patch state)"
  git -C "$CHECKOUT" checkout -- . 2>/dev/null || true
  # 0004 touches Qwen35.swift which needs to exist at the pinned revision.
  for p in "${PATCHES[@]}"; do
    [[ -f "$p" ]] || { echo "FATAL: missing patch $p" >&2; exit 1; }
    if ! git -C "$CHECKOUT" apply --check "$p" >/dev/null 2>&1; then
      echo "FATAL: patch $p does not apply cleanly at $CHECKOUT" >&2
      exit 1
    fi
    git -C "$CHECKOUT" apply "$p"
    echo "applied $(basename "$p") at $CHECKOUT"
  done

  find "$CHECKOUT" -name '*.swift' -exec chmod u+w {} + 2>/dev/null || true
  echo "vmlx patch series applied at $CHECKOUT"
done
echo "done"