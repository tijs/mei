#!/usr/bin/env bash
# test_start_mei_server_flags.sh — keep scripts/start_mei_server.sh honest about
# the CLI it launches.
#
# Why this exists: the launcher passed `--optimization-profile auto|generic|ornith`
# long after that flag was removed and replaced by `--model-profile <model name>`.
# No shipped binary accepted it, so every launch through this script exited 2
# before loading a model — and because the benchmark runner keeps its own copy
# of the launch logic, nothing in the benchmark path noticed. A flag typo in a
# script that cannot be smoke-tested without a 20 GB checkpoint needs a cheap
# static check instead.
#
# Verifies:
#   - the script parses (bash -n)
#   - the removed flag and its env var are gone from the script
#   - MEI_OPTIMIZATION_PROFILE fails fast with a message naming the replacement
#     (checked without a model, since the guard runs before any model lookup)
#   - MEI_MODEL_PROFILE is threaded into the launch as --model-profile
#   - every flag the script hands the binary is accepted by that binary
#     (skipped with a clear note when no built binary is available)
#
# Usage: scripts/test_start_mei_server_flags.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$SCRIPT_DIR/start_mei_server.sh"
[[ -f "$LAUNCHER" ]] || { echo "TEST HARNESS ERROR: start_mei_server.sh missing" >&2; exit 70; }

PASS=0; FAIL=0
T="$(mktemp -d "${TMPDIR:-/tmp}/mei-launch-flags.XXXXXX")"
trap 'rm -rf "$T"' EXIT
ok()  { PASS=$((PASS+1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL - $*"; }

# --- 1. parses ---------------------------------------------------------------
if bash -n "$LAUNCHER" 2>"$T/syntax"; then
  ok "start_mei_server.sh parses"
else
  bad "start_mei_server.sh does not parse: $(cat "$T/syntax")"
fi

# --- 2. the removed flag is not PASSED (a comment naming it is fine) -------
# Only the lines that build the launch arguments count: the ARGS array (which
# spans lines, so a flag can hide on a continuation line) plus any ARGS+=(...).
sed -n '/^ARGS=(/,/^$/p' "$LAUNCHER" > "$T/args.txt"
grep -oE 'ARGS\+?=\(.*' "$LAUNCHER" >> "$T/args.txt"
if grep -q -- '--optimization-profile' "$T/args.txt"; then
  bad "removed flag --optimization-profile is still passed by the launcher"
else
  ok "launcher no longer passes the removed --optimization-profile"
fi

# --- 3. the obsolete env var fails fast, naming the replacement --------------
if MEI_OPTIMIZATION_PROFILE=auto bash "$LAUNCHER" >"$T/obsolete.out" 2>"$T/obsolete.err"; then
  bad "MEI_OPTIMIZATION_PROFILE was accepted silently (exit 0)"
else
  rc=$?
  if [[ "$rc" -eq 2 ]] && grep -q 'obsolete' "$T/obsolete.err" \
     && grep -q 'MEI_MODEL_PROFILE' "$T/obsolete.err"; then
    ok "MEI_OPTIMIZATION_PROFILE exits 2 with the replacement named"
  else
    bad "obsolete-var guard: rc=$rc stderr=$(head -c 200 "$T/obsolete.err")"
  fi
fi

# --- 4. the replacement reaches the launch ----------------------------------
# shellcheck disable=SC2016  # the literal $MODEL_PROFILE is the point here
if grep -q -- 'ARGS+=(--model-profile "$MODEL_PROFILE")' "$LAUNCHER"; then
  ok "MEI_MODEL_PROFILE is threaded through as --model-profile"
else
  bad "launcher does not pass --model-profile from MEI_MODEL_PROFILE"
fi

# --- 5. every launch flag is accepted by the binary -------------------------
BIN="${MEI_BINARY:-}"
[[ -n "$BIN" ]] || BIN="$REPO/.build/release/mei"
[[ -x "$BIN" ]] || BIN="$SCRIPT_DIR/../bin/mei"
if [[ ! -x "$BIN" ]]; then
  echo "skip - no built mei binary (set MEI_BINARY or build .build/release/mei);"
  echo "       flag-contract check not run"
else
  # Only the flags handed to the binary: the ARGS array plus the ARGS+=(...)
  # conditionals (already extracted above). Flags meant for swift or the disk
  # guard live outside those.
  untested=0
  while read -r line; do
    for flag in $(printf '%s' "$line" | grep -oE '\-\-[a-z][a-z-]+'); do
      case "$flag" in
        # Flags inside the collected lines that belong to another tool.
        --runtime-root|--cache-dir|--min-free-gib|--retain|--all-disposable|--scratch-path|--package-path) continue ;;
      esac
      out="$(MEI_REPO="$REPO" "$BIN" --model-dir /nonexistent-model-dir --served-model-id x "$flag" 1 2>&1 | head -1)"
      case "$out" in
        *"unknown option $flag"*) bad "binary rejects $flag (passed by the launcher)" ;;
        *) untested=$((untested+1)) ;;
      esac
    done
  done < <(sort -u "$T/args.txt")
  if [[ $FAIL -eq 0 ]]; then
    ok "all launcher flags are accepted by $BIN ($untested checked)"
  fi
fi

echo
echo "RESULT: $PASS ok, $FAIL failed"
[[ $FAIL -eq 0 ]]
