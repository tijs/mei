#!/usr/bin/env bash
# Mei measurement-cycle orchestrator (uncontended-window gated, FOREGROUND).
#
# Runs the five workstreams' measurement phases in order, each phase only
# when the machine is uncontended (no foreign inference runners + memory
# floor). Every phase writes timestamped artifacts under artifacts/. The
# script records a machine-state boundary at the START of each phase.
#
# Policy (2026-08-30 session C): NO detached/nohup waiters behind a
# contention gate. The gate is polled at most --max-wait-min minutes
# (default 5, per the shared-GPU contention cap); if the window does not
# open, the script writes a gated-boundary artifact and exits 3 so the
# caller can move to CPU-side work. Phases run in the foreground, each
# supervised by tools/run_bounded.py with a hard wall-cap.
#
# Phases:
#   A  characterization sweep (baseline fp16 matrix)            [workstream 3]
#   B  llama.cpp hardware ceiling  (+ q8_0 KV variant)          [workstream 4]
#   C  kv-bits 8 / kv-bits 4 / compiled-threshold 16K /
#      combined compile+quant, each with acceptance + 30K/80K
#      survival probes                                          [workstreams 1-2]
#
# Usage:
#   scripts/run_measurement_cycle.sh [--phase A|B|C|ALL]
#       [--max-wait-min N=5] [--phase-timeout-min N=720] [--force]
#   --force  run even with foreign processes resident (rows labeled).
# Exit codes: 0 done, 1 phase failure, 2 usage, 3 gate expired.
set -euo pipefail

MEI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MEI_REPO"
# Single-instance lock: refuse to start if another cycle is alive (gate
# polls can outlive a session; double-launches would collide on port 8024).
LOCKDIR="/tmp/mei-cycle.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "FATAL: another measurement cycle is running (lock $LOCKDIR)" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
BUILD_DIR="${MEI_BUILD_DIR:-$HOME/.local/share/local-model-bench/mei-build}"
VENV_PY="${MEI_SWEEP_VENV:-$HOME/.local/share/local-model-bench/mei-runtime/venv/bin/python}"
MODEL_DIR="${MEI_MODEL_DIR:-$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-9B-MLX-4bit}"
MODEL_ID="ornith-ai/Ornith-1.5-9B-MLX-4bit"
GGUF="$HOME/.local/share/local-model-bench/mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

FORCE=""
PHASE="ALL"
MAX_WAIT_MIN=5
PHASE_TIMEOUT_MIN=720
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE="--no-contention-gate" ;;
    --phase) PHASE="${2:-ALL}"; shift ;;
    --max-wait-min) MAX_WAIT_MIN="${2:-5}"; shift ;;
    --phase-timeout-min) PHASE_TIMEOUT_MIN="${2:-720}"; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

contended() {
  # Gate 1: another agent's SUITE RUNNER must not be active (its llama.cpp
  # backend may stay resident between tasks — that is fine as long as it is
  # idle and the machine has memory headroom; an idle-but-resident server
  # never gets a CPU/GPU cycle stolen from it).
  if ps -axo command= | grep -E "run_fixture_suite|run_bench\.py" >/dev/null; then
    return 0
  fi
  # Gate 2: enough reclaimable memory for an uncontended 9B row (idle
  # llama-server with ~24GB resident will keep this below the floor).
  local free_kb inactive_kb
  free_kb=$(vm_stat | awk '/Pages free/{print $3}' | tr -d '.')
  inactive_kb=$(vm_stat | awk '/Pages inactive/{print $3}' | tr -d '.')
  local reclaimable_gb=$(( (free_kb + inactive_kb) * 16384 / 1024 / 1024 / 1024 ))
  if (( reclaimable_gb < 10 )); then
    echo "[cycle] only ${reclaimable_gb}GB reclaimable; deferring until memory clears"
    return 0
  fi
  return 1
}

gate() {
  if [[ -n "$FORCE" ]]; then
    echo "[cycle] FORCE: running with co-resident workloads (rows labeled)"
    return 0
  fi
  local waited=0 sample_file="artifacts/cycle-gate-sample-$TS.txt"
  : > "$sample_file"   # one sample file per cycle run, appended per minute
  while (( waited < MAX_WAIT_MIN )); do
    if contended; then
      local runners reclaimable_gb
      runners=$(ps -axo pid=,command= | grep -E "run_fixture_suite|run_bench\.py|llama-server" | grep -v grep || true)
      free_kb=$(vm_stat | awk '/Pages free/{print $3}' | tr -d '.')
      inactive_kb=$(vm_stat | awk '/Pages inactive/{print $3}' | tr -d '.')
      reclaimable_gb=$(( (free_kb + inactive_kb) * 16384 / 1024 / 1024 / 1024 ))
      {
        echo "sample $(date -u +%H:%M:%SZ) reclaimable=${reclaimable_gb}GB"
        echo "$runners"
      } >> "$sample_file"
      echo "[cycle] $(date -u +%H:%M:%SZ) machine contended; waiting for an uncontended window (${waited}/${MAX_WAIT_MIN} min)..."
      sleep 60
      waited=$((waited + 1))
      continue
    fi
    # Binary must contain the fork's full patch surface (0004 window flag).
    local BIN="$BUILD_DIR/arm64-apple-macosx/release/mei"
    if ! "$BIN" --help 2>/dev/null | grep -q "max-kv-window"; then
      echo "[cycle] $(date -u +%H:%M:%SZ) release binary missing the fork flags; waiting for rebuild..."
      sleep 60
      waited=$((waited + 1))
      continue
    fi
    return 0
  done
  # Gate expired: record the boundary and exit 3 (caller moves to CPU-side
  # work; never leave a detached waiter behind a contention gate).
  {
    echo "gated-boundary $(date -u +%Y-%m-%dT%H:%M:%SZ) max-wait-min=$MAX_WAIT_MIN"
    echo "--- foreign runners/servers ---"
    ps -axo pid=,command= | grep -E "llama-server|vllm|omlx|cocore|run_fixture_suite|run_bench" | grep -v grep || true
    echo "--- memory ---"
    memory_pressure 2>/dev/null | head -4 || true
    free_kb=$(vm_stat | awk '/Pages free/{print $3}' | tr -d '.')
    inactive_kb=$(vm_stat | awk '/Pages inactive/{print $3}' | tr -d '.')
    echo "reclaimable_gb=$(( (free_kb + inactive_kb) * 16384 / 1024 / 1024 / 1024 ))"
  } > "artifacts/cycle-gated-boundary-$TS.txt"
  echo "[cycle] GATED: no uncontended window within ${MAX_WAIT_MIN}min; boundary -> artifacts/cycle-gated-boundary-$TS.txt" >&2
  exit 3
}

boundary() {
  ps -axo pid=,command= | grep -E "llama-server|vllm|omlx|cocore" | grep -v grep \
    > "artifacts/cycle-$1-boundary-$TS.txt" || true
}

# run_bounded MINUTES -- cmd args...  (foreground, hard wall-cap, exit 124 on expiry)
run_bounded() {
  local minutes="$1"; shift
  # "$1" is "--"
  shift
  echo "[cycle]   (bounded ${minutes}min) $*"
  python3 tools/run_bounded.py "$minutes" -- "$@" || local rc=$?
  local rc="${rc:-0}"
  if (( rc == 124 )); then
    echo "[cycle] PHASE BOUND EXCEEDED (${minutes}min) for: $*" >&2
  fi
  return "$rc"
}

phase_a() {
  gate
  echo "[cycle] PHASE A: baseline fp16 characterization sweep"
  boundary A
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 512,4096,16384,33175,45000 \
    --prefill-steps 512 --ssm-rederive true --kv-bits none \
    --repeats-45k 3 --chat-40k --kv-cache-dir \
    --output "artifacts/sweep-cliff-baseline-$TS.json"
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 16384,33175,45000 \
    --prefill-steps 512,2048,4096 --ssm-rederive true \
    --repeats-45k 1 --kv-cache-dir \
    --output "artifacts/sweep-cliff-prefillsteps-$TS.json"
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 16384,33175,45000 \
    --prefill-steps 512 --ssm-rederive true,false \
    --repeats-45k 1 --kv-cache-dir \
    --output "artifacts/sweep-cliff-ssm-$TS.json"
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 16384,33175,45000 \
    --prefill-steps 512 --cache-limit-gb 0,2,8 \
    --repeats-45k 1 --kv-cache-dir \
    --output "artifacts/sweep-cliff-cachelimit-$TS.json"
}

phase_b() {
  gate
  echo "[cycle] PHASE B: llama.cpp hardware ceiling"
  boundary B
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/llama_ceiling.py --gguf "$GGUF" \
    --alias "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M" \
    --repeats 3 --chat-40k \
    --output "artifacts/llama-ceiling-fp16kv-$TS.json"
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/llama_ceiling.py --gguf "$GGUF" \
    --alias "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M" \
    --repeats 3 --chat-40k --kv-cache-type-q8 \
    --output "artifacts/llama-ceiling-q8kv-$TS.json"
}

run_variant_cell() {
  # run_variant_cell TAG KV COMPILED COMPILED_THRESHOLD CONTEXT_CAP [KV_WINDOW]
  local tag="$1" kv="${2:-}" compiled="${3:-false}" threshold="${4:-}" ctxcap="${5:-65536}" window="${6:-0}"
  echo "[cycle] variant $tag: kv=$kv compiled=$compiled threshold=${threshold:-none} ctx=$ctxcap window=${window:-default}"
  local extra=()
  [[ -n "$kv" ]] && extra+=(--kv-bits "$kv")
  [[ "$compiled" == "true" ]] && extra+=(--compiled true)
  [[ -n "$threshold" ]] && extra+=(--compiled-decode-threshold "$threshold")
  [[ "$window" != "0" ]] && extra+=(--max-kv-window "$window")
  run_bounded "$PHASE_TIMEOUT_MIN" -- python3 tools/sweep_mei.py --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 512,16384,33175,45000 --repeats-45k 3 --chat-40k --kv-cache-dir \
    --context-cap "$ctxcap" "${extra[@]}" \
    --output "artifacts/sweep-variant-$tag-$TS.json"
  gate
  boundary "C-$tag"
  local cell_kv="$HOME/.local/share/local-model-bench/mei-runtime/kv-cache-cell-$tag-$TS"
  rm -rf "$cell_kv"   # cold acceptance cell: fresh disk tier per run
  local envs=(MEI_MODEL_DIR="$MODEL_DIR" MEI_SERVED_MODEL_ID="$MODEL_ID"
    MEI_KV_BITS="$kv" MEI_COMPILED_DECODE="$compiled"
    MEI_CONTEXT_CAP="$ctxcap" MEI_SSM_REDERIVE=true MEI_KV_CACHE_DIR="$cell_kv")
  if [[ -n "$threshold" ]]; then envs+=(MEI_COMPILED_DECODE_THRESHOLD="$threshold"); fi
  if [[ "$window" != "0" ]]; then envs+=(MEI_MAX_KV_WINDOW="$window"); fi
  # start_mei_server.sh runs the server in the foreground; background it and
  # wait for readiness through probe_mei's own health poll.
  env "${envs[@]}" bash scripts/start_mei_server.sh &
  local spid=$!
  # Give the (cached) build a head start; probe_mei retries /v1/models with
  # a generous timeout of its own, so a slow build is tolerated.
  sleep 20
  "${VENV_PY}" tools/probe_mei.py --base-url http://127.0.0.1:8024/v1 \
    --model "$MODEL_ID" --tokenizer "$MODEL_DIR" --context-cap "$ctxcap" \
    --output "artifacts/acceptance-variant-$tag-$TS.json" || true
  # 30K survival always; 80K survival when the cell's context cap allows.
  local lengths="30000"
  if (( ctxcap >= 131072 )); then lengths="30000 80000"; fi
  "${VENV_PY}" tools/probe_long_context.py --base-url http://127.0.0.1:8024/v1 \
    --model "$MODEL_ID" --tokenizer "$MODEL_DIR" \
    --lengths $lengths --output "artifacts/survival-variant-$tag-$TS.json" || true
  bash scripts/stop_mei_server.sh || true
  kill "$spid" 2>/dev/null || true
}

phase_c() {
  run_variant_cell kv8 "8"
  run_variant_cell kv4 "4"
  run_variant_cell compiled16 "" "true" "16384"
  run_variant_cell combined-kv8-compiled16 "8" "true" "16384"
  # Experimental bounded-window probes (attention scans at most the ring).
  # Correctness-bounded: a full-attention model drops context beyond the
  # window; decode throughput is expected to become context-independent.
  run_variant_cell window8k "" "" "" "65536" "8192"
  run_variant_cell window16k "" "" "" "65536" "16384"
  # THE gate candidate: window16k + compiled decode with NO threshold —
  # the promote+trace is O(window)=16K cheap even at a 45K offset, so the
  # compiled replay runs at 16K-width attention at 45K context.
  run_variant_cell window16-compiled "" "true" "" "65536" "16384"
  # 80K survival probe at an extended context cap (separate cell)
  run_variant_cell kv8-80k "8" "" "" "131072"
  run_variant_cell survival80k "" "" "" "131072"
}

case "$PHASE" in
  A|a) phase_a ;;
  B|b) phase_b ;;
  C|c) phase_c ;;
  ALL) phase_a; phase_b; phase_c ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
echo "[cycle] done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"