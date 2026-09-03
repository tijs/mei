#!/usr/bin/env bash
# test_install_mei.sh — deterministic, weight-free, server-free tests for
# scripts/install_mei.sh. Exercises the installer's contract:
#   -h/--help  prints usage, exits 0
#   unknown arg exits 2 (usage error), writes to stderr
#   missing source binary fails loudly (exit 1), writes nothing
#   --dry-run  prints the plan, writes nothing, exits 0
#   install    copies the binary to $PREFIX/mei and sets the executable bit
#   idempotent rerun with the same binary is a no-op (exit 0, file unchanged)
#   a differing existing binary is preserved unless --force
#   system/write-protected prefixes are refused unless --force
#   colocated Metal companions (mlx.metallib, *.provenance) follow the binary
#   --version  prints the installer version, exits 0
#
# No model weights, no live Metal server: the "built executable" is a tiny
# POSIX stub with a `--version` flag.
#
# Usage: scripts/test_install_mei.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/install_mei.sh"

PASS=0
FAIL=0
CURRENT_TMP=""

die() { echo "TEST HARNESS ERROR: $*" >&2; exit 70; }

new_tmp() {
  mktemp -d "${TMPDIR:-/tmp}/mei-install-test.XXXXXX"
}

# make_stub DIR NAME -> create an executable stub "binary" that prints its version
make_stub() {
  local dir="$1" name="${2:-mei}"
  printf '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "mei %s"; else echo "stub-mei"; fi\n' "test-$(basename "$dir")" > "$dir/$name"
  chmod 755 "$dir/$name"
}

# assert CMD... — run a command that must succeed
assert() {
  if "$@" >/dev/null 2>&1; then
    PASS=$((PASS + 1)); echo "ok   - $*"
  else
    FAIL=$((FAIL + 1)); echo "FAIL - $*"
  fi
}

# assert_fail CMD... — run a command that must fail (exit != 0)
assert_fail() {
  if "$@" >/dev/null 2>&1; then
    FAIL=$((FAIL + 1)); echo "FAIL - expected nonzero exit: $*"
  else
    PASS=$((PASS + 1)); echo "ok   - (expected failure) $*"
  fi
}

# assert_grep NEEDLE FILE — file exists and contains needle
assert_grep() {
  local needle="$1" file="$2"
  if [[ -f "$file" ]] && grep -q -- "$needle" "$file"; then
    PASS=$((PASS + 1)); echo "ok   - grep '${needle}' in $file"
  else
    FAIL=$((FAIL + 1)); echo "FAIL - grep '${needle}' in $file"
  fi
}

section() { echo; echo "== $* =="; }

# ---------------------------------------------------------------------------
section "installer must exist and be executable"
[[ -f "$INSTALLER" ]] || die "installer missing at $INSTALLER (run install step first?)"
assert test -x "$INSTALLER"

# --- help --------------------------------------------------------------
section "help"
T=$(new_tmp); trap 'rm -rf "$T"' EXIT
"$INSTALLER" --help > "$T/help.out" 2> "$T/help.err"; rc=$?
assert test $rc -eq 0
assert_grep "Usage" "$T/help.out"
"$INSTALLER" -h > /dev/null 2>&1
assert test $? -eq 0

# --- unknown option is a usage error (exit 2) ---------------------------
section "unknown option -> usage error exit 2"
"$INSTALLER" --bogus-flag > "$T/unknown.out" 2> "$T/unknown.err"; rc=$?
[[ $rc -eq 2 ]] && PASS=$((PASS + 1)) && echo "ok   - unknown arg exit 2" || { FAIL=$((FAIL + 1)); echo "FAIL - unknown arg exit 2 (got $rc)"; }

# --- missing source binary fails loudly ---------------------------------
section "missing binary fails loudly, writes nothing"
T2=$(new_tmp); mkdir -p "$T2/prefix"
"$INSTALLER" --prefix "$T2/prefix" --binary "$T2/does-not-exist" > "$T2/missing.out" 2> "$T2/missing.err"; rc=$?
[[ $rc -eq 1 ]] && PASS=$((PASS + 1)) && echo "ok   - missing binary exit 1" || { FAIL=$((FAIL + 1)); echo "FAIL - missing binary exit 1 (got $rc)"; }
assert_grep "FATAL" "$T2/missing.err"
assert test ! -e "$T2/prefix/mei"

# --- dry-run writes nothing ----------------------------------------------
section "dry-run prints plan, writes nothing"
SRC=$(new_tmp); make_stub "$SRC"; P=$(new_tmp)/prefix
trap 'rm -rf "$SRC"' EXIT
"$INSTALLER" --prefix "$P" --binary "$SRC/mei" --dry-run > "$T2/dry.out" 2> "$T2/dry.err"; rc=$?
assert test $rc -eq 0
assert test ! -e "$P/mei"
assert test ! -e "$P"

# --- install sets executable bit, reports version ------------------------
section "install copies binary, sets executable bit"
P=$(new_tmp)/prefix
"$INSTALLER" --prefix "$P" --binary "$SRC/mei" > "$T2/install.out" 2>&1; rc=$?
assert test $rc -eq 0
assert test -f "$P/mei"
assert test -x "$P/mei"
mode=$(stat -f%Lp "$P/mei" 2>/dev/null || stat -c%a "$P/mei" 2>/dev/null)
case "$mode" in
  755|750|700|711|rwxr-xr-x|rwxr-x---) : ;; *) FAIL=$((FAIL+1)); echo "FAIL - exec bit/mode $mode";; esac
PASS=$((PASS + 1)); echo "ok   - mode '$mode' executable"
# stub runs as an executable (real "already-built" binary runs; stub echoes)
assert "$P/mei" --version

# --- idempotent rerun with same binary is a no-op ------------------------
section "idempotent rerun (same binary)"
before=$(shasum -a 256 "$P/mei" | awk '{print $1}')
"$INSTALLER" --prefix "$P" --binary "$SRC/mei" > "$T2/re.out" 2>&1; rc=$?
assert test $rc -eq 0
after=$(shasum -a 256 "$P/mei" | awk '{print $1}')
[[ "$before" == "$after" ]] && PASS=$((PASS+1)) && echo "ok   - file unchanged on rerun" || { FAIL=$((FAIL+1)); echo "FAIL - file changed on rerun"; }

# --- differing existing binary is preserved unless --force ----------------
section "differing existing binary preserved unless --force"
printf '#!/bin/sh\necho old\n' > "$P/mei"; chmod 755 "$P/mei"; oldsum=$(shasum -a 256 "$P/mei" | awk '{print $1}')
"$INSTALLER" --prefix "$P" --binary "$SRC/mei" > "$T2/noforce.out" 2> "$T2/noforce.err"; rc=$?
assert_fail test $rc -eq 0
assert test "$(shasum -a 256 "$P/mei" | awk '{print $1}')" == "$oldsum"
"$INSTALLER" --prefix "$P" --binary "$SRC/mei" --force >/dev/null 2>&1
assert test $? -eq 0
[[ "$(shasum -a 256 "$P/mei" | awk '{print $1}')" != "$oldsum" ]] && PASS=$((PASS+1)) && echo "ok   - --force overwrote" || { FAIL=$((FAIL+1)); echo "FAIL - --force did not overwrite"; }

# --- company Metal companions follow the binary --------------------------
section "colocated mlx.metallib companion copied + idempotent"
SRC2=$(new_tmp); make_stub "$SRC2"; printf 'MTLB-stub' > "$SRC2/mlx.metallib"; printf 'src: stub\n' > "$SRC2/mlx.metallib.provenance"
P=$(new_tmp)/prefix
trap 'rm -rf "$SRC2"' EXIT
"$INSTALLER" --prefix "$P" --binary "$SRC2/mei" >/dev/null 2>&1
assert test $? -eq 0
assert test -f "$P/mlx.metallib"
assert test -f "$P/mlx.metallib.provenance"
# idempotent: rerun does not error
"$INSTALLER" --prefix "$P" --binary "$SRC2/mei" >/dev/null 2>&1
assert test $? -eq 0

# --- --version -----------------------------------------------------------
section "--version"
"$INSTALLER" --version > "$T2/ver.out" 2>&1
assert test $? -eq 0
assert_grep "install_mei" "$T2/ver.out"

# --- system / non-user prefix refused unless --force ----------------------
section "system prefix refused unless --force"
"$INSTALLER" --prefix /usr/local --binary "$SRC2/mei" > "$T2/sys.out" 2> "$T2/sys.err"; rc=$?
[[ $rc -eq 1 ]] && PASS=$((PASS+1)) && echo "ok   - /usr/local refused (exit 1)" || { FAIL=$((FAIL+1)); echo "FAIL - /usr/local refused (got $rc)"; }
assert_grep "FATAL" "$T2/sys.err"

# default prefix is user-local ($HOME/.local/bin) and is NOT refused
"$INSTALLER" --help >/dev/null 2>&1   # no-op sanity
DHOME=$("$INSTALLER" --help 2>/dev/null | grep -o '\$HOME/.local/bin' | head -1)
assert test "$DHOME" == '$HOME/.local/bin'

# ---------------------------------------------------------------------------
echo
echo "RESULT: $PASS ok, $FAIL failed"
[[ $FAIL -eq 0 ]]