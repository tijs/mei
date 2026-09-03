#!/usr/bin/env bash
# stage_release_candidate.sh — build the source-first release candidate tree
# from the explicit allowlist and verify the staging boundary mechanically.
#
# Usage: scripts/stage_release_candidate.sh [VERSION]
#   VERSION defaults to 0.2.0-alpha.1.
#
# Produces dist/mei-<VERSION>-src/ (gitignored) with:
#   - exactly the allowlisted files (configs/release-allowlist.json)
#   - STAGING-MANIFEST.sha256 (per-file sha256, relative paths)
#   - staging-manifest.json (release, allowlist sha256, file count, timestamp)
# Fails loudly (exit 1) unless every check passes. Never touches artifacts/
# or the historical dist/ bundle.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-0.2.0-alpha.1}"
ALLOWLIST="$REPO/configs/release-allowlist.json"
STAGE_ROOT="$REPO/dist"
STAGE_DIR="$STAGE_ROOT/mei-${VERSION}-src"

[[ -f "$ALLOWLIST" ]] || { echo "FATAL: allowlist missing: $ALLOWLIST" >&2; exit 1; }

echo "== staging Mei $VERSION (source-first) =="
echo "repo:    $REPO"
echo "stage:   $STAGE_DIR"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# --- copy allowlisted content ---------------------------------------------
ROOT_FILES=($(python3 -c "
import json,sys
d=json.load(open('$ALLOWLIST'))
print(' '.join(d['root_files']))
"))
for f in "${ROOT_FILES[@]}"; do
  [[ -f "$REPO/$f" ]] || { echo "FATAL: allowlisted root file missing in repo: $f" >&2; exit 1; }
  cp -p "$REPO/$f" "$STAGE_DIR/$f"
done

python3 - "$ALLOWLIST" "$REPO" "$STAGE_DIR" <<'PY'
import json, os, shutil, sys
allowlist = json.load(open(sys.argv[1]))
repo, stage = sys.argv[2], sys.argv[3]
for d in allowlist["directories"]:
    src = os.path.join(repo, d["path"])
    dst = os.path.join(stage, d["path"])
    if not os.path.isdir(src):
        sys.exit(f"FATAL: allowlisted directory missing in repo: {d['path']}")
    excludes = d.get("exclude", [])
    def excluded(rel):
        import fnmatch
        return any(fnmatch.fnmatch(rel, pat) for pat in excludes)
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if not excluded(os.path.relpath(os.path.join(root, x), src))]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), src)
            if excluded(rel):
                continue
            full = os.path.join(root, f)
            if os.path.islink(full):  # never follow symlinks out of the repo
                continue
            os.makedirs(os.path.dirname(os.path.join(dst, rel)) or dst, exist_ok=True)
            shutil.copy2(full, os.path.join(dst, rel))
print("copy: directories copied from allowlist")
PY

# --- verification ----------------------------------------------------------
fail=0
check() { # $1 = description, $2 = exit-status-command...
  local desc="$1"; shift
  if "$@"; then echo "PASS: $desc"; else echo "FAIL: $desc" >&2; fail=1; fi
}

cd "$REPO"

# 1. every allowlisted root file + directory exists in the staged tree
check "allowlisted root files present" \
  python3 -c "
import json,sys,os
d=json.load(open('configs/release-allowlist.json')); stage='dist/mei-$VERSION-src'
missing=[f for f in d['root_files'] if not os.path.isfile(os.path.join(stage,f))]
missing+=[x['path'] for x in d['directories'] if not os.path.isdir(os.path.join(stage,x['path']))]
sys.exit(1 if missing else 0)
"

# 2. no staged path is outside the allowlist (compare staged files vs allowlist closure)
check "no out-of-allowlist files staged" \
  python3 - <<PY
import json, os, sys
d = json.load(open('configs/release-allowlist.json'))
stage = 'dist/mei-$VERSION-src'
allowed = set(d['root_files'])
for x in d['directories']:
    for root, dirs, files in os.walk(x['path']):
        dirs[:] = [z for z in dirs if z not in ('__pycache__',)]
        for f in files:
            allowed.add(os.path.relpath(os.path.join(root, f), '.'))
staged = set()
for root, dirs, files in os.walk(stage):
    dirs[:] = [z for z in dirs if z not in ('__pycache__', '.git')]
    for f in files:
        staged.add(os.path.relpath(os.path.join(root, f), stage))
extra = sorted(staged - allowed)
if extra:
    print('EXTRA staged files outside allowlist:')
    for e in extra: print('  ', e)
    sys.exit(1)
print(f'staged {len(staged)} files == allowlist closure {len(allowed)} files')
PY

# 3. excluded directories absent from the staged tree
check "excluded dirs absent (artifacts/.git/.build/dist)" \
  bash -c "cd '$STAGE_DIR' && ! test -e artifacts && ! test -e .git && ! test -e .build && ! test -e dist && ! test -e __pycache__"

# 4. no model weights / caches present (no .safetensors / .gguf / .bin blobs)
check "no weight blobs staged" \
  bash -c "cd '$STAGE_DIR' && ! find . -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' \) | grep -q ."

# 5. release version metadata consistent across the three locations
check "version metadata consistent (0.2.0-alpha.1)" \
  bash -c "
grep -q 'static let version = \"$VERSION\"' '$STAGE_DIR/Sources/MeiCore/ServerConfig.swift' &&
grep -q '## \[$VERSION\]' '$STAGE_DIR/CHANGELOG.md' &&
grep -q 'Mei ${VERSION}' '$STAGE_DIR/docs/RELEASE-${VERSION}.md'"

# 6. per-file sha256 manifest, then verify it round-trips
check "manifest writes + verifies" \
  bash -c "
cd '$STAGE_DIR' &&
find . -type f ! -name STAGING-MANIFEST.sha256 | sort | xargs shasum -a 256 > STAGING-MANIFEST.sha256 &&
shasum -a 256 -c STAGING-MANIFEST.sha256 >/dev/null 2>&1
"

python3 - "$VERSION" "$ALLOWLIST" "$STAGE_DIR" <<'PY'
import hashlib, json, os, sys, time
version, allowlist, stage = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(os.path.relpath(os.path.join(r, f), stage)
               for r, _, fs in os.walk(stage) for f in fs if f != 'STAGING-MANIFEST.sha256')
manifest = {
    "release": version,
    "kind": "source-first",
    "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "allowlist_sha256": hashlib.sha256(open(allowlist, 'rb').read()).hexdigest(),
    "file_count": len(files),
    "staging_dir": os.path.relpath(stage, os.getcwd()),
}
with open(os.path.join(stage, 'staging-manifest.json'), 'w') as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
print(f"manifest: {manifest['file_count']} files, allowlist sha {manifest['allowlist_sha256']}")
PY

if [[ $fail -ne 0 ]]; then
  echo "STAGING FAILED (see FAIL lines above)" >&2
  exit 1
fi
echo "== staging OK: $STAGE_DIR =="
echo "   manifest: $STAGE_DIR/STAGING-MANIFEST.sha256"
echo "   summary:  $STAGE_DIR/staging-manifest.json"