# Mei (梅)

Mei — after Mei Long, the sleeping dragon dinosaur — is a narrow, native
Swift/MLX OpenAI-compatible inference server for Apple Silicon, built directly
on the pinned `osaurus-ai/vmlx-swift` engine (not the full Osaurus app).

The project's scope is deliberately small: one model per server process, the
OpenAI chat/completions surface this repo's benchmark needs, chunked prefill
on for hybrid architectures, and in-process KV/prefix reuse across turns.
The primary target is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`; the goal is
>= 30 decode tokens/second with correct tool-calling and no long-context
collapse on this 32GB Apple Silicon machine.

## Layout

- `Sources/MeiCore/` — engine, router, HTTP server, OpenAI DTOs
- `Sources/Mei/main.swift` — CLI entry point
- `Tests/MeiTests/` — unit tests + black-box acceptance oracle
  (`MeiAcceptanceTests`: run RED against a missing server, go green once the
  server behaves; enabled via `MEI_ACCEPTANCE_BASE_URL`, default
  `http://127.0.0.1:8024/v1`)
- `tools/probe_mei.py` — standalone acceptance/perf probe (the authoritative
  copy; `local-model-bench/runner/probe_mei.py` mirrors it)
- `scripts/` — stage_model.sh / start_mei_server.sh / stop_mei_server.sh
  (isolated runtime: dedicated port 8024, own logs/build/model staging under
  `~/.local/share/local-model-bench/mei-*`)

## Build

```bash
swift build            # debug
swift test             # unit tests (acceptance tests need a live server)
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
```

`vmlx-swift` is pinned by revision (`aeb5e21c…`, the revision osaurus-ai's own
`Package.resolved` pins) and `Package.resolved` is committed. Re-pinning is a
deliberate decision that must re-run the whole acceptance suite.

### Metal kernel library (mlx.metallib)

`vmlx-swift`'s SwiftPM build does not emit the compiled Metal kernel library
(the vendored mlx ships kernels as `.metal` sources; compiling them needs
Xcode's `metallib` archiver, which is not installed on this machine).
`scripts/prepare_metallib.sh` provisions a prebuilt `mlx.metallib` next to the
release binary:

- prefers a wheel whose mlx version matches the vendored `0.31.1`
  (`Source/Cmlx/include-framework/mlx-version.h`) — the exact
  version-matched artifact, since kernels are looked up by name at runtime
- verifies every candidate structurally (MTLB magic, size, `file`
  classification) before installing — never a blind copy
- records provenance in `mlx.metallib.provenance` next to the artifact
- falls back to the compile path on machines that do have the archiver

The definitive verification is runtime: the server loads the library at
startup and fails loudly if kernels are missing.

## Memory measurement

Memory numbers come from MLX's allocator (`Memory.snapshot()`:
active/cache/peak bytes, plus explicit `--memory-limit-bytes` and
`--cache-limit-bytes` config) and `GPU.deviceInfo()` (architecture, physical
memory, recommended working set). The long-gone Cmlx `get_physical_memory`
entry point does not exist in this stack and is not used; `/v1/mei/status`
exposes live allocator state and the startup log prints the post-load
footprint. An MLX memory limit below the model working set makes alloc calls
wait on scheduled tasks (the hang failure mode), so benchmark configs pin
explicit limits.

## Run

```bash
scripts/stage_model.sh   # downloads the primary model once (~20GB)
scripts/start_mei_server.sh    # builds + serves on 127.0.0.1:8024
scripts/stop_mei_server.sh
```

or directly:

```bash
swift run mei --model-dir ~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit \
  --served-model-id ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --port 8024 \
  --context-cap 65536 --prefill-step-size 512 \
  --kv-cache-dir ~/.local/share/local-model-bench/mei-runtime/kv-cache
```

Hybrid families (Ornith's qwen3_5_moe) need the disk tier for prefix reuse:
`--kv-cache-dir` (or `MEI_KV_CACHE_DIR`) enables it; without it every
request full-prefills (correct, just slower). Set `--memory-limit-bytes`
above the model's working set — the MLX default limit can otherwise sit
below it and make allocation wait on scheduled tasks (the hang failure
mode).

### Models

- **Primary (MVP)**: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` — the model
  author's official plain 4-bit text-generation MLX quant (19.5 GB). The 30
  decode tok/s goal is defined against this artifact on the 32 GB Sulaco
  machine.
- **Documented fallback (same family, smaller)**: `ornith-ai/Ornith-1.5-9B-MLX-4bit`
  — the official 9B plain 4-bit quant (5.0 GB), same `qwen3_5` architecture,
  same 4-bit-affine/group-64 quant scheme, same chat template and
  qwen3_coder-style tool-call format. Used to validate server correctness and
  tool-call behavior on hardware where the 35B cannot be resident
  (e.g. while other engines hold the machine's RAM). It is a fallback for
  exercising the identical Mei path, never a silent substitution for the
  primary artifact.

## Bench integration

The `local-model-bench` harness drives Mei as a first-class
`inference_engine` (its commit `9df216e` wires the launch scripts and
config; the boundary preserves that repo read-only — Mei's own copy of the
probe/bench drivers lives in `tools/` with outputs under `artifacts/`):

- `tools/probe_mei.py` — acceptance/parity/tooling gate (authoritative copy)
- `tools/probe_diverging_chat.py` — patch-0005 evidence probe: 5-turn
  tool-calling transcript, run A = strict growth, run B diverges at turn 5
  (place_order -> cancel_order); records cached_tokens/prefill_ms/TTFT per
  request plus deterministic transcript/schema/output checks (`--self-test`
  validates everything without a server; a `tool` result whose id has no
  pending assistant tool_call, missing argument keys, or non-additive
  transcripts fail loudly)
- `tools/probe_long_context.py` — chunked-prefill survival at 30K/80K
- `tools/bench_mei.py` — full benchmark rows (short, tool, 45K-loaded
  fresh + reuse, 40K chat) with engine-reported tok/s, TTFT/prefill ms and
  allocator bytes; artifact-only output under `artifacts/`
- `tools/gguf_meta.py` — GGUF header + tensor-name reader (`--check-mtp`
  reports MTP/Next-N head presence via `nextn_predict_layers` and
  `blk.*.nextn.*` tensors; `--tensors [filter]` lists names; handles the
  GGUF v2 vs v3 layout difference — v3 dropped the tensor-info count field
  and stores dims as u64). Used for llama.cpp-ceiling comparator hygiene:
  never compare an MTP variant against Mei's no-MTP baseline.
- `tools/llama_ceiling.py` — llama.cpp hardware-ceiling driver on Mei-owned
  port 8074. `--provenance-only` verifies the llama-server binary, the GGUF
  header (arch qwen35, context >= 64K, MTP-head presence) and the sha256
  against the pinned official blob digest
  (`ornith-ai/Ornith-1.5-9B-GGUF@abdd624b` = local
  `70c11219…e8fab6`) WITHOUT launching the server — safe under contention;
  a digest mismatch is FATAL and refuses to measure. The full run records
  the provenance block in its artifact before any row.

## Design notes

- **Chunked prefill**: `GenerateParameters.prefillStepSize` defaults to 512
  and is always on — the long-context safeguard for hybrid (GatedDelta)
  architectures; the Python serving wrappers this project replaces lacked it.
- **KV/prefix reuse**: in-process (and cross-restart, via the on-disk tier)
  prefix reuse is owned by `vmlx-swift`'s `CacheCoordinator` — hash-chained
  KV blocks plus hybrid companion state (Ornith's GatedDelta/SSM layers are
  disk-backed-restore only, so the disk tier must be enabled for reuse on
  this family). The chat template's generation-prompt suffix is stripped at
  store time so the standard agentic pattern (identical system prompt,
  growing transcript) hits. `usage.prompt_tokens_details.cached_tokens`
  reports the reused prefix; `/v1/mei/status` exposes live paged/disk/SSM
  counters. The disk cache lives under `MEI_KV_CACHE_DIR`
  (default `~/.local/share/local-model-bench/mei-runtime/kv-cache`) and is
  transient runtime state like the build dir, never part of the artifacts.
  Any divergence or missing companion state falls back to a full prefill
  (always correct). The SSM re-derive pass after each chat turn costs ~1x
  prefill at turn end (upstream default on; `--ssm-rederive false` turns it
  off for A/B rows). `--ssm-anchor-boundaries K` (patch 0005, default off)
  stores additional SSM companion anchors at the first K chat role-turn
  boundaries (exact token offsets from the request's own rendering path,
  with an additivity self-check) so a mid-transcript diverging agentic
  edit restores from a retained boundary instead of full-prefilling —
  a TTFT/latency lever, not a decode tok/s lever; see
  `artifacts/design-anchor-ssm-0005.md`.
- **Tool calls**: vmlx-swift parses Qwen/Ornith-style `<tool_call>{json}`
  envelopes inside its generate loop; Mei maps `.toolCall` events to OpenAI
  `tool_calls` in both streaming and non-streaming shapes.
- **Reasoning**: Ornith is a Qwen3.5-lineage thinking model; thought text is
  exposed as `reasoning_content` (opt-out via `--emit-reasoning false`) and
  the thinking/visible streams are kept separate.
- **No MTP**: explicitly out of scope (the benchmark's own data shows no
  MTP/speculative win on this hardware).