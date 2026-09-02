#!/usr/bin/env bash
# Start the isolated Mei server for local-model-bench.
#
# Isolation conventions (mirrors start_omlx_server.sh):
#   - dedicated port (default 8024 — never benchmark ports 8012/8015/8016/8018/8020)
#   - dedicated log dir  ~/.local/share/local-model-bench/mei-logs
#   - dedicated build dir (scratch path) ~/.local/share/local-model-bench/mei-build
#   - pid file at        ~/.local/share/local-model-bench/mei-runtime/server.pid
#   - refuses to launch if the port is already listening
set -euo pipefail

MEI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_BASE="${MEI_RUNTIME_BASE:-$HOME/.local/share/local-model-bench/mei-runtime}"
CACHE_ROOT="${MEI_CACHE_ROOT:-$HOME/.local/share/local-model-bench/mei-runtime}"
BUILD_DIR="${MEI_BUILD_DIR:-$HOME/.local/share/local-model-bench/mei-build}"
DISK_GUARD="$MEI_REPO/tools/mei_disk_guard.py"
MIN_FREE_GIB="${MEI_MIN_FREE_GIB:-20}"
RETAIN_KV_CACHE="${MEI_RETAIN_KV_CACHE:-false}"
LOG_DIR="$RUNTIME_BASE/logs"
PID_FILE="$RUNTIME_BASE/server.pid"
SWIFT_BIN="${MEI_SWIFT:-swift}"

MODEL_DIR="${MEI_MODEL_DIR:-$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit}"
if [[ -z "${MEI_MODEL_DIR:-}" && -d "$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned" ]]; then
  MODEL_DIR="$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned"
fi
SERVED_MODEL_ID="${MEI_SERVED_MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit}"
OPTIMIZATION_PROFILE="${MEI_OPTIMIZATION_PROFILE:-auto}"
PORT="${MEI_PORT:-8024}"
CONTEXT_CAP="${MEI_CONTEXT_CAP:-65536}"
MAX_TOKENS="${MEI_MAX_TOKENS:-32768}"
PREFILL_STEP_SIZE="${MEI_PREFILL_STEP_SIZE:-}"
KV_BITS="${MEI_KV_BITS:-}"
TEMPERATURE="${MEI_TEMPERATURE:-0.6}"
TOP_P="${MEI_TOP_P:-0.95}"
TOP_K="${MEI_TOP_K:-20}"
EMIT_REASONING="${MEI_EMIT_REASONING:-true}"
CACHE_REUSE="${MEI_CACHE_REUSE:-true}"
MEMORY_LIMIT_BYTES="${MEI_MEMORY_LIMIT_BYTES:-0}"
CACHE_LIMIT_BYTES="${MEI_CACHE_LIMIT_BYTES:-0}"
KV_CACHE_DIR="${MEI_KV_CACHE_DIR:-}"
LOG_REQUESTS="${MEI_LOG_REQUESTS:-false}"
SSM_REDERIVE="${MEI_SSM_REDERIVE:-true}"
SSM_ANCHOR_BOUNDARIES="${MEI_SSM_ANCHOR_BOUNDARIES:-0}"
COMPILED_DECODE="${MEI_COMPILED_DECODE:-false}"
COMPILED_DECODE_THRESHOLD="${MEI_COMPILED_DECODE_THRESHOLD:-}"
MAX_KV_WINDOW="${MEI_MAX_KV_WINDOW:-0}"
LOAD_MMAP="${MEI_LOAD_MMAP:-true}"

usage() {
  cat <<EOF
Usage: start_mei_server.sh [options]

The full launch configuration is expressed through MEI_* environment
variables (see the script); the bench config yaml sets them explicitly.

Optimization profiles:
  MEI_OPTIMIZATION_PROFILE=auto    detect qwen3_5_moe as Ornith, else generic
  MEI_OPTIMIZATION_PROFILE=generic force conservative non-Ornith defaults
  MEI_OPTIMIZATION_PROFILE=ornith  force the validated Ornith profile

Disk safety:
  MEI_MIN_FREE_GIB=20          refuse launch below this free-space floor
  MEI_RETAIN_KV_CACHE=true     retain an explicitly named disposable KV cache
                                for a reuse experiment; default is cleanup
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -d "$MODEL_DIR" ]] || { echo "FATAL: model directory missing: $MODEL_DIR (run scripts/stage_model.sh)" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "FATAL: missing config.json: $MODEL_DIR/config.json" >&2; exit 1; }

if /usr/sbin/lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "FATAL: port $PORT is already listening; refuse to attach to a stale process" >&2
  /usr/sbin/lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  exit 1
fi

mkdir -p "$RUNTIME_BASE" "$CACHE_ROOT" "$LOG_DIR" "$BUILD_DIR"

# Remove only stale, disposable experiment caches before checking the floor.
# Protected model/recent-35B caches are never selected by --all-disposable.
python3 "$DISK_GUARD" cleanup --runtime-root "$CACHE_ROOT" --all-disposable \
  >> "$LOG_DIR/start.log" 2>&1 || true
if ! python3 "$DISK_GUARD" check --runtime-root "$CACHE_ROOT" --min-free-gib "$MIN_FREE_GIB" \
  >> "$LOG_DIR/start.log" 2>&1; then
  echo "FATAL: refusing Mei launch because the disk free-space floor is not met" >&2
  exit 1
fi

echo "mei: resolving pinned vmlx dependency ..." | tee -a "$LOG_DIR/start.log"
"$SWIFT_BIN" package resolve --scratch-path "$BUILD_DIR" --package-path "$MEI_REPO" \
  > "$LOG_DIR/resolve.log" 2>&1 || {
  echo "FATAL: SwiftPM dependency resolution failed — see $LOG_DIR/resolve.log" >&2
  exit 1
}
bash "$MEI_REPO/scripts/apply_vmlx_patches.sh" --reset \
  > "$LOG_DIR/vmlx-patches.log" 2>&1 || {
  echo "FATAL: pinned vmlx patch series failed — see $LOG_DIR/vmlx-patches.log" >&2
  exit 1
}

echo "mei: building (release, scratch: $BUILD_DIR) ..." | tee -a "$LOG_DIR/start.log"
"$SWIFT_BIN" build -c release --scratch-path "$BUILD_DIR" --package-path "$MEI_REPO" \
  > "$LOG_DIR/build.log" 2>&1 || {
  echo "FATAL: swift build failed — see $LOG_DIR/build.log" >&2
  exit 1
}
BIN="$BUILD_DIR/release/mei"
[[ -x "$BIN" ]] || { echo "FATAL: built binary missing at $BIN" >&2; exit 1; }
bash "$MEI_REPO/scripts/prepare_metallib.sh" "$BUILD_DIR/release" || { echo "FATAL: missing Metal kernel library" >&2; exit 1; }

ARGS=(--model-dir "$MODEL_DIR" --served-model-id "$SERVED_MODEL_ID"
  --optimization-profile "$OPTIMIZATION_PROFILE"
  --host 127.0.0.1 --port "$PORT"
  --context-cap "$CONTEXT_CAP" --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
  --emit-reasoning "$EMIT_REASONING" --cache-reuse "$CACHE_REUSE"
  --memory-limit-bytes "$MEMORY_LIMIT_BYTES" --cache-limit-bytes "$CACHE_LIMIT_BYTES"
  --log-requests "$LOG_REQUESTS" --ssm-rederive "$SSM_REDERIVE" --compiled-decode "$COMPILED_DECODE" --load-mmap "$LOAD_MMAP")
[[ -n "$PREFILL_STEP_SIZE" ]] && ARGS+=(--prefill-step-size "$PREFILL_STEP_SIZE")
[[ -n "$COMPILED_DECODE_THRESHOLD" ]] && ARGS+=(--compiled-decode-threshold "$COMPILED_DECODE_THRESHOLD")
[[ "$MAX_KV_WINDOW" != "0" && -n "$MAX_KV_WINDOW" ]] && ARGS+=(--max-kv-window "$MAX_KV_WINDOW")
[[ "$SSM_ANCHOR_BOUNDARIES" != "0" && -n "$SSM_ANCHOR_BOUNDARIES" ]] && ARGS+=(--ssm-anchor-boundaries "$SSM_ANCHOR_BOUNDARIES")
[[ -n "$KV_BITS" ]] && ARGS+=(--kv-bits "$KV_BITS")
[[ -n "$KV_CACHE_DIR" ]] && ARGS+=(--kv-cache-dir "$KV_CACHE_DIR")

printf 'mei isolated launch: '
printf '%q ' "$BIN" "${ARGS[@]}"
printf '\n'
"$BIN" "${ARGS[@]}" >> "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"
on_exit() {
  local rc=$?
  rm -f "$PID_FILE"
  if [[ -n "$KV_CACHE_DIR" ]]; then
    local cleanup_args=(cleanup --runtime-root "$CACHE_ROOT" --cache-dir "$KV_CACHE_DIR")
    [[ "$RETAIN_KV_CACHE" == "1" || "$RETAIN_KV_CACHE" == "true" || "$RETAIN_KV_CACHE" == "yes" ]] && cleanup_args+=(--retain)
    python3 "$DISK_GUARD" "${cleanup_args[@]}" >> "$LOG_DIR/start.log" 2>&1 || \
      echo "mei: cache cleanup skipped/refused for $KV_CACHE_DIR" >> "$LOG_DIR/start.log"
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait "$SERVER_PID"