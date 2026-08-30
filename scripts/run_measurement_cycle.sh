#!/usr/bin/env bash
# Mei measurement cycle orchestrator (uncontended-window gated).
#
# Runs the five workstreams' measurement phases in order, each phase only
# when the machine is uncontended (no foreign inference processes). Every
# phase writes timestamped artifacts under artifacts/. The script records
# a machine-state boundary at the START of each phase.
#
# Phases:
#   A  characterization sweep (baseline fp16 matrix)            [workstream 3]
#   B  llama.cpp hardware ceiling  (+ q8_0 KV variant)          [workstream 4]
#   C  kv-bits 8 / kv-bits 4 / compiled-threshold 16K /
#      combined compile+quant, each with acceptance + 30K/80K
#      survival probes                                          [workstreams 1-2]
#
# Usage: scripts/run_measurement_cycle.sh [--force] [--phase A|B|C]
#   --force  run even with foreign processes resident (rows labeled).
set -euo pipefail

MEI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MEI_REPO"
BUILD_DIR="${MEI_BUILD_DIR:-$HOME/.local/share/local-model-bench/mei-build}"
VENV_PY="${MEI_SWEEP_VENV:-$HOME/.local/share/local-model-bench/mei-runtime/venv/bin/python}"
MODEL_DIR="${MEI_MODEL_DIR:-$HOME/.local/share/local-model-bench/mei-models/Ornith-1.5-9B-MLX-4bit}"
MODEL_ID="ornith-ai/Ornith-1.5-9B-MLX-4bit"
GGUF="$HOME/.local/share/local-model-bench/mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
FORCE=""
PHASE="ALL"
if [[ "${1:-}" == "--force" ]]; then FORCE="--no-contention-gate"; PHASE="${2:-ALL}"; fi

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
  for _ in $(seq 1 360); do
    if contended; then
      echo "[cycle] $(date -u +%H:%M:%SZ) machine contended; waiting for an uncontended window..."
      sleep 60
      continue
    fi
    # Binary must contain the fork's full patch surface (0004 window flag).
    local BIN="$BUILD_DIR/arm64-apple-macosx/release/mei"
    if ! "$BIN" --help 2>/dev/null | grep -q "max-kv-window"; then
      echo "[cycle] $(date -u +%H:%M:%SZ) release binary missing the fork flags; waiting for rebuild..."
      sleep 60
      continue
    fi
    return 0
  done
  echo "[cycle] FATAL: no uncontended window within 6h" >&2
  exit 1
}

boundary() {
  ps -axo pid=,command= | grep -E "llama-server|vllm|omlx|cocore" | grep -v grep \
    > "artifacts/cycle-$1-boundary-$TS.txt" || true
}

phase_a() {
  gate
  echo "[cycle] PHASE A: baseline fp16 characterization sweep"
  boundary A
  python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 512,4096,16384,33175,45000 \
    --prefill-steps 512 --ssm-rederive true --kv-bits none \
    --repeats-45k 3 --chat-40k --kv-cache-dir \
    --output "artifacts/sweep-cliff-baseline-$TS.json"
  python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 16384,33175,45000 \
    --prefill-steps 512,2048,4096 --ssm-rederive true \
    --repeats-45k 1 --kv-cache-dir \
    --output "artifacts/sweep-cliff-prefillsteps-$TS.json"
  python3 tools/sweep_mei.py \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 16384,33175,45000 \
    --prefill-steps 512 --ssm-rederive true,false \
    --repeats-45k 1 --kv-cache-dir \
    --output "artifacts/sweep-cliff-ssm-$TS.json"
  python3 tools/sweep_mei.py \
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
  python3 tools/llama_ceiling.py --gguf "$GGUF" \
    --alias "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M" \
    --repeats 3 --chat-40k \
    --output "artifacts/llama-ceiling-fp16kv-$TS.json"
  python3 tools/llama_ceiling.py --gguf "$GGUF" \
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
  python3 tools/sweep_mei.py --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --contexts 512,16384,33175,45000 --repeats-45k 3 --chat-40k --kv-cache-dir \
    --context-cap "$ctxcap" "${extra[@]}" \
    --output "artifacts/sweep-variant-$tag-$TS.json"
  gate
  boundary "C-$tag"
  local envs=(MEI_MODEL_DIR="$MODEL_DIR" MEI_SERVED_MODEL_ID="$MODEL_ID"
    MEI_KV_BITS="$kv" MEI_COMPILED_DECODE="$compiled"
    MEI_CONTEXT_CAP="$ctxcap" MEI_SSM_REDERIVE=true MEI_KV_CACHE_DIR="$HOME/.local/share/local-model-bench/mei-runtime/kv-cache-cell-$tag")
  if [[ -n "$threshold" ]]; then envs+=(MEI_COMPILED_DECODE_THRESHOLD="$threshold"); fi
  if [[ "$window" != "0" ]]; then envs+=(MEI_MAX_KV_WINDOW="$window"); fi
  # start_mei_server.sh runs the server in the foreground; background it and
  # wait for readiness through probe_mei's own health poll.
  env "${envs[@]}" bash scripts/start_mei_server.sh &
  local spid=$!
  # Build may take a while on a fresh scratch; probe_mei --skip-server-check
  # is not available, so rely on its built-in retry (it waits for /v1/models
  # with a generous timeout). Give the build a head start.
  sleep 20
  "${VENV_PY}" tools/probe_mei.py --base-url http://127.0.0.1:8024/v1 \
    --model "$MODEL_ID" --tokenizer "$MODEL_DIR" --context-cap "$ctxcap" \
    --output "artifacts/acceptance-variant-$tag-$TS.json" || true
  "${VENV_PY}" tools/probe_long_context.py --base-url http://127.0.0.1:8024/v1 \
    --model "$MODEL_ID" --tokenizer "$MODEL_DIR" \
    --lengths 30000 --output "artifacts/survival30k-variant-$tag-$TS.json" || true
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