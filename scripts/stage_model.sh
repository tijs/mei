#!/usr/bin/env bash
# Stage the primary MVP model for Mei into an isolated model directory.
#
# Isolated runtime convention (matches local-model-bench's oMLX staging):
#   models live under ~/.local/share/local-model-bench/mei-models/<name>
# Never uses the benchmark ports, logs, or venvs of other engines.
set -euo pipefail

MODEL_ID="${MEI_MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit}"
MODEL_ROOT="${MEI_MODEL_ROOT:-$HOME/.local/share/local-model-bench/mei-models}"
HF_TOKEN_SOURCE="${MEI_HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
BENCH_REPO="${MEI_BENCH_REPO:-$HOME/projects/local-model-bench}"

usage() {
  cat <<EOF
Usage: stage_model.sh [--model-id ID] [--model-root DIR]

Downloads the model (default $MODEL_ID) into the isolated Mei model root
(default $MODEL_ROOT) using the local-model-bench Hugging Face credentials.
The download is idempotent: an existing complete directory is left untouched.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="${2:?}"; shift 2 ;;
    --model-root) MODEL_ROOT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

TARGET_DIR="$MODEL_ROOT/$(basename "$MODEL_ID")"
mkdir -p "$MODEL_ROOT"
[[ -f "$HF_TOKEN_SOURCE" ]] || { echo "FATAL: no HF token at $HF_TOKEN_SOURCE" >&2; exit 1; }

if [[ -f "$TARGET_DIR/config.json" ]] && [[ -f "$TARGET_DIR/tokenizer_config.json" ]]; then
  echo "Model already staged at $TARGET_DIR"
  exit 0
fi

HF_TOKEN="$(cat "$HF_TOKEN_SOURCE")"
export HF_TOKEN
if [[ -x "$BENCH_REPO/.venv/bin/python" ]]; then
  PY="$BENCH_REPO/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

echo "Staging $MODEL_ID -> $TARGET_DIR (this is a ~20GB download)"
"$PY" - "$MODEL_ID" "$TARGET_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
model_id, target = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=model_id, local_dir=target)
print(f"staged at {path}")
PYEOF

# Guard against a half-written model (truncated shard) — refuse to serve it.
python3 - "$TARGET_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
expected = config.get("safetensors", {}).get("total")
shards = sorted(root.glob("model-*.safetensors"))
actual = sum(p.stat().st_size for p in shards)
if expected is not None:
    # total is byte count in the index; compare loosely (index vs file sizes
    # can differ by header bytes).
    if actual < expected - 16 * 1024 * 1024:
        sys.exit(f"FATAL: safetensors total {actual} < expected {expected}; model is incomplete")
print(f"verified {len(shards)} shards, {actual/1e9:.1f}GB")
PYEOF
echo "Staged OK."