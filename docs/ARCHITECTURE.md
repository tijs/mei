# Architecture

How Mei runs: the runtime flow, optimization profiles, memory / KV / prefill
behavior, and the design rationale. This document assembles the detail that
used to live in the README so the README can stay a front door.

## Runtime flow

`mei` is a single native Swift/MLX process. It loads one checkpoint into the
pinned `vmlx-swift` fork engine and serves the OpenAI `chat/completions`
surface. The high-level flow:

1. **Startup** — the CLI parses `--model-dir`, `--served-model-id`,
   `--optimization-profile`, and the context/memory/cache options; loads the
   model's `config.json` and `.safetensors`; provisions the Metal kernel
   library (`mlx.metallib`, vmlx loads it from the executable's directory
   first); prints the post-load footprint.
2. **Identity** — `GET /v1/models` returns exactly the `--served-model-id`
   string. The model id must be byte-identical to the served id (the
   acceptance suite enforces this).
3. **Chat** — `POST /v1/chat/completions` runs chunked prefill, then decode,
   with in-process KV/prefix reuse across turns. Streaming and non-streaming
   shapes are both supported; `tool_calls` are mapped from the engine's
   Qwen/Ornith-style `<tool_call>{json}` envelopes.
4. **Reasoning** — thinking-model thought text is exposed as
   `reasoning_content` (opt-out via `--emit-reasoning false`); the
   thinking/visible streams are kept separate.
5. **Fail-closed boundaries** — no fallback model substitution, an explicit
   context-cap rejection beyond `--context-cap`, and a 20 GiB free-disk floor
   enforced by `tools/mei_disk_guard.py` before a server/measurement cycle
   starts.

The launch scripts (`scripts/start_mei_server.sh`, `stop_mei_server.sh`) run an
isolated runtime: dedicated port 8024, own logs/build/model staging under
`~/.local/share/local-model-bench/mei-*`.

## Optimization profiles

`--optimization-profile auto|generic|ornith`:

| Profile | `--prefill-step-size` | Fused gate/up cache | Notes |
|---|---|---|---|
| `auto` | detects validated `qwen3_5_moe` metadata → Ornith behavior; else generic | per model | Reads the local model's `config.json`; unknown/malformed metadata stays generic |
| `generic` | 64 (default) | unchanged | Default fallback |
| `ornith` | 512 | **disabled before model load** | The validated Ornith path |

Ornith (and the Ornith-profile only) uses prefill step 512 and disables the
fused gate/up cache before model loading — this is what meets the >=30 tok/s
goal. `generic` uses prefill step 64 and leaves that cache unchanged.
Compiled decode, rotating-KV quantization, bounded windows, and SSM anchors
remain default-off. Per-model recommendation is in
[`docs/MODELS.md`](MODELS.md).

## Chunked prefill

`GenerateParameters.prefillStepSize` is always on — the long-context safeguard
for hybrid (GatedDelta) architectures; the Python serving wrappers this project
replaces lacked it. The window is set with `--prefill-step-size` (default 64
for generic; the auto/Ornith profile uses 512 for validated
`qwen3_5_moe`; gemma4 bundles default to a measured 256). `--compiled-decode`
and `--compiled-decode-threshold` gate the graph-traced compiled decode;
skipping the upstream default promptOffset-sized trace avoids a multi-minute
prefill tax at 45K.

## KV / prefix reuse

- In-process (and cross-restart, via an on-disk tier) prefix reuse is owned by
  `vmlx-swift`'s `CacheCoordinator` — hash-chained KV blocks plus hybrid
  companion state.
- Hybrid families (Ornith's `qwen3_5_moe`) are **disk-backed-restore only**,
  so the disk KV tier must be enabled for reuse: `--kv-cache-dir DIR` or
  `MEI_KV_CACHE_DIR`. Without it every request full-prefills (correct, just
  slower).
- Dense `qwen3_5`/`qwen3_8` checkpoints crash the in-memory-only paged tier
  (`SmallVector out of range`, vmlx `mlx/c/array.cpp:335`), and gemma4 bundles
  never restore exact-repeat prefixes on it. With cache reuse on and no
  `--kv-cache-dir`, both families default to a disposable cache under the OS
  temp dir; an explicit `--kv-cache-dir` always wins.
- The chat template's generation-prompt suffix is stripped at store time so
  the standard agentic pattern (identical system prompt, growing transcript)
  hits. `usage.prompt_tokens_details.cached_tokens` reports the reused prefix;
  `/v1/mei/status` exposes live paged/disk/SSM counters.
- Any divergence or missing companion state falls back to a full prefill —
  always correct.
- The SSM re-derive pass after each chat turn costs ~1x prefill at turn end
  (upstream default on; can be disabled for A/B rows — name the mode in the
  row).
  `--ssm-anchor-boundaries K` (vMLX fork commit `91fed8be`, default off) stores
  extra SSM companion anchors at the first K chat role-turn boundaries so a
  mid-transcript diverging agentic edit restores from a retained boundary
  instead of full-prefilling — a TTFT/latency lever, not a decode tok/s lever
  (see `artifacts/design-anchor-ssm-0005.md`).
- `--max-kv-window N` (EXPERIMENTAL): cap the rotating-KV ring at N tokens
  (attention scans at most the ring). Correctness-bounded only; 0 = default
  ring.

## Memory behavior

Memory numbers come from MLX's allocator (`Memory.snapshot()`:
active/cache/peak bytes, plus `--memory-limit-bytes` and `--cache-limit-bytes`)
and `GPU.deviceInfo()` (architecture, physical memory, recommended working
set). The long-gone Cmlx `get_physical_memory` entry point does not exist in
this stack and is not used. `/v1/mei/status` exposes live allocator state, and
the startup log prints the post-load footprint.

**The hang failure mode:** an MLX memory limit below the model working set
makes alloc calls *wait on scheduled tasks* instead of failing. So
`--memory-limit-bytes` must be set **above** the model's working set — the MLX
default limit can otherwise sit below it and hang. Benchmark configs pin
explicit limits (see [`docs/BENCHMARKING.md`](BENCHMARKING.md)).

## Design notes

- **One model per process.** Each server serves exactly one checkpoint by its
  exact served id. Run other models as separate instances on their own ports
  with their own `--kv-cache-dir`.
- **No MTP / speculative decode** — explicitly out of scope (the benchmark's
  own data shows no MTP/speculative win on this hardware).
- **Disk safety.** `tools/mei_disk_guard.py` enforces a 20 GiB free-space floor
  before a server or measurement cycle starts. Disposable per-experiment
  caches (`kv-cache-sweep`, `kv-cache-cell-*`, `kv-cache-anchors-*`,
  `kv-cache-exp-*`) are removed on completion or interruption; pass
  `MEI_RETAIN_KV_CACHE=true` only for a named reuse experiment. Model weights,
  repository artifacts, historical evidence, and protected recent-model caches
  are never auto-cleaned. Decisions are appended to `mei-disk-guard.log` under
  the runtime root.
- **Quantization of the engine's cache** (`--kv-bits 4|8`, `--kv-group-size`,
  `--quantized-kv-start`) is available for rotating-KV quantization but stays
  default-off by convention.

## Scope

Mei targets a compact machine: an M1 Mac with 32 GB of unified memory, with
the primary goal of >=30 decode tok/s, correct tool-calling, and no
long-context collapse on the Ornith primary model. This narrow scope (one
model per process, a single API surface) is deliberate.