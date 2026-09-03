#!/usr/bin/env bash
# install_mei.sh — safe, user-local installer for the already-built Mei
# executable (plus any colocated Metal companions).
#
# Scope / safety contract:
#   - Installs ONLY to a user-local destination dir (default "$HOME/.local/bin");
#     never creates files under Homebrew/new brew prefixes, /opt, /usr(/local),
#     /bin, /sbin, or /Library unless the user explicitly passes --force.
#   - Never touches package-manager paths, system directories, or writes
#     outside the single destination dir.
#   - Installs an executable the user has ALREADY built (or the staged source
#     bundle carries next to it in bin/); this script never runs `swift build`
#     and never downloads anything.
#   - Fails loudly (exit 1) if no source binary can be found.
#   - Preserves an existing installed file unless --force is given; a rerun
#     with an identical file is a no-op (idempotent).
#   - Copies any mlx.metallib / default.metallib / *.provenance colocated with
#     the source binary into the same dir as the installed binary, because
#     vmlx loads the Metal kernel library from next to the executable first.
#
# Source resolution order (first match wins):
#   1. --binary PATH            explicit
#   2. $MEI_BINARY              environment variable
#   3. $SCRIPT_ROOT/../.build/release/mei   (repo release build)
#   4. $SCRIPT_ROOT/bin/mei                 (staged bundle shipping bin/mei)
#
# Usage:
#   scripts/install_mei.sh [--prefix DIR] [--binary PATH] [--dry-run] [--force]
#
# Options:
#   --prefix DIR   destination directory; the executable is written to
#                  DIR/mei. Default: "$HOME/.local/bin".
#   --binary PATH  explicit path to the built mei executable.
#   --dry-run      print the plan and exit 0; write nothing.
#   --force        overwrite an existing, differing installed file (and
#                  companions) instead of preserving it.
#   -h, --help     this help; exit 0.
#   -v, --version  print installer version; exit 0.
set -euo pipefail

INSTALLER_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# default destination: user-local bin dir
PREFIX="${HOME}/.local/bin"
SOURCE=""
DRY_RUN=0
FORCE=0

usage() {
  cat <<EOF
Usage: install_mei.sh [options]

Safely installs the already-built Mei executable (and any colocated Metal
companion files) to a single user-local destination directory.

  --prefix DIR   destination dir; the executable is written to DIR/mei.
                 Default: \$HOME/.local/bin
  --binary PATH  explicit path to the built mei executable.
  --dry-run      print the install plan; change nothing.
  --force        overwrite an existing, differing installed file.
  -h, --help     this help.
  -v, --version  print installer version.

Source resolution (first match wins): --binary, \$MEI_BINARY,
${REPO_ROOT}/.build/release/mei, or bin/mei beside this script.

Safety: refuses system / package-manager prefixes (e.g. /usr/local, /opt,
/opt/homebrew, /bin, /sbin, /Library) unless --force is given. Never builds,
downloads, or writes outside the single destination directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -v|--version) echo "install_mei ${INSTALLER_VERSION}"; exit 0 ;;
    --prefix)
      [[ $# -ge 2 ]] || { echo "FATAL: --prefix requires a directory" >&2; exit 2; }
      PREFIX="$2"; shift 2 ;;
    --binary)
      [[ $# -ge 2 ]] || { echo "FATAL: --binary requires a path" >&2; exit 2; }
      SOURCE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "FATAL: unknown option: $1" >&2; echo "Try 'install_mei.sh --help'." >&2; exit 2 ;;
  esac
done

# --- resolve source binary ----------------------------------------------
resolve_source() {
  # 1 explicit, 2 env, 3 repo build, 4 staged bundle
  local cand="$SOURCE"
  [[ -z "$cand" && -n "${MEI_BINARY:-}" ]] && cand="$MEI_BINARY"
  [[ -z "$cand" && -x "$REPO_ROOT/.build/release/mei" ]] && cand="$REPO_ROOT/.build/release/mei"
  [[ -z "$cand" && -x "$SCRIPT_DIR/bin/mei" ]] && cand="$SCRIPT_DIR/bin/mei"
  if [[ -z "$cand" ]]; then
    echo "FATAL: no source binary found." >&2
    echo "  Pass --binary PATH, set \$MEI_BINARY, or build the release first" >&2
    echo "  (swift build -c release) so '$REPO_ROOT/.build/release/mei' exists." >&2
    exit 1
  fi
  if [[ ! -f "$cand" ]]; then
    echo "FATAL: source binary not found: $cand" >&2
    exit 1
  fi
  SOURCE="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
}

source_matches() {
  # $1 src, $2 dst -> 0 if byte-identical
  [[ -f "$2" ]] || return 1
  [[ "$(shasum -a 256 "$1" 2>/dev/null | awk '{print $1}')" == "$(shasum -a 256 "$2" 2>/dev/null | awk '{print $1}')" ]]
}

# --- system / package-manager prefix guard -------------------------------
guard_prefix() {
  if [[ "$PREFIX" != /* ]]; then
    PREFIX="$(cd "$(dirname "$PREFIX")" 2>/dev/null && pwd)/$(basename "$PREFIX")"
  fi
  local norm
  norm="$(cd "$PREFIX" 2>/dev/null && pwd || printf '%s' "$PREFIX")"
  local bad=''
  case "$norm" in
    /usr|/usr/*|/opt|/opt/*|/bin|/bin/*|/sbin|/sbin/*|/Library|/Library/*|/) bad=1 ;;
  esac
  # Homebrew new-prefix lives under /opt/homebrew (already covered by /opt).
  if [[ -n "$bad" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      echo "WARNING: --force overrides system-prefix guard for $norm" >&2
      return 0
    fi
    echo "FATAL: refusing to write to system / package-manager prefix '$norm'." >&2
    echo "  Use a user-local prefix (default \$HOME/.local/bin) or pass --force" >&2
    echo "  to override this safety guard." >&2
    exit 1
  fi
}

resolve_source
guard_prefix

# companion files colocated with the source binary (mlx needs metallib next to it)
SRC_DIR="$(dirname "$SOURCE")"
declare -a COMPANIONS=()
for f in mlx.metallib default.metallib; do
  [[ -f "$SRC_DIR/$f" ]] && COMPANIONS+=("$f")
done
for f in "$SRC_DIR"/*.provenance; do
  [[ -f "$f" ]] && COMPANIONS+=("$(basename "$f")")
done
# dedupe, keep stable order
if (( ${#COMPANIONS[@]} )); then
  COMPANIONS=($(printf '%s\n' "${COMPANIONS[@]}" | awk '!seen[$0]++'))
fi

DEST="$PREFIX/mei"

echo "== install_mei (mei binary installer) =="
echo "source:    $SOURCE"
echo "prefix:    $PREFIX"
echo "target:    $DEST"
echo "mode:      $([[ "$DRY_RUN" -eq 1 ]] && echo dry-run || echo install)$([[ "$FORCE" -eq 1 ]] && echo ", force")"

# plan a single file write with overwrite semantics
# prints the intended action; exits the whole script (status 1) on the
# fatal "existing file differs, no --force" case. As an `if` test:
#   return 0 -> install/overwrite needed;  return 1 -> up to date (no change)
plan_file() {
  # $1 src, $2 dst, $3 label
  local src="$1" dst="$2" label="$3"
  if [[ -e "$dst" ]]; then
    if source_matches "$src" "$dst"; then
      echo "  $label: already up to date (no change)"
      return 1   # nothing to do
    fi
    if [[ "$FORCE" -eq 1 ]]; then
      echo "  $label: overwrite (different existing file, --force)"
    else
      echo "  $label: FATAL — existing file differs; pass --force to overwrite" >&2
      exit 1
    fi
  else
    echo "  $label: install"
  fi
  return 0
}

would_change=0
if plan_file "$SOURCE" "$DEST" "mei"; then
  would_change=1
fi

if (( ${#COMPANIONS[@]} )); then
for f in "${COMPANIONS[@]}"; do
  if plan_file "$SRC_DIR/$f" "$PREFIX/$f" "${f}"; then
    would_change=1
  fi
done
fi

if [[ "$would_change" -eq 0 ]]; then
  echo "== nothing to do (already installed): $DEST =="
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "== dry-run: no files written =="
  exit 0
fi

# --- perform install -----------------------------------------------------
mkdir -p "$PREFIX"
if ! cp -p "$SOURCE" "$DEST" 2>/dev/null || ! chmod 755 "$DEST"; then
  echo "FATAL: failed to install $DEST (disc full / no permission?)" >&2
  exit 1
fi
if (( ${#COMPANIONS[@]} )); then
for f in "${COMPANIONS[@]}"; do
  cp -p "$SRC_DIR/$f" "$PREFIX/$f"
done
fi

echo "== installed: $DEST =="
echo "   executable: $([[ -x "$DEST" ]] && echo yes || echo NO)"
# Best-effort runtime self-check (real binary reports version; stub may not)
if "$DEST" --version >/dev/null 2>&1; then
  echo "   version:    $("$DEST" --version 2>/dev/null)"
else
  echo "   version:    (started; server runtime requires a model + Metal library)"
fi