# Mei

![Mei sleeping-dinosaur logo](assets/mei-logo.jpg)

Mei — after *Mei long*, the sleeping-dragon dinosaur — is a narrow, native
Swift/MLX OpenAI-compatible inference server for Apple Silicon, built directly
on the pinned `tijs/vmlx-swift` fork of `osaurus-ai/vmlx-swift` (not the full
Osaurus app).

The project's scope is deliberately small: one model per server process, the
OpenAI chat/completions surface this repo's benchmark needs, chunked prefill
on for hybrid architectures, and in-process KV/prefix reuse across turns.
Inspired by [DwarfStar](https://github.com/antirez/ds4)'s compact native
local-inference approach, Mei targets an even smaller machine: an M1 Mac with
32 GB of unified memory. The primary model is
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`; the goal is >= 30 decode tokens/second
with correct tool-calling and no long-context collapse on that target.

**Initial public release:** `0.1.0` (source-first, macOS/Apple Silicon).
Mei's source is MIT-licensed; model weights are not included. See
[`NOTICE.md`](NOTICE.md) and [`docs/RELEASE-0.1.0.md`](docs/RELEASE-0.1.0.md)
for dependency, patch-queue, checkpoint, and benchmark provenance.

## Optimization profiles

Mei supports `auto`, `generic`, and `ornith` profiles. `auto` reads the local
model's `config.json` and recognizes the validated `qwen3_5_moe`/
`qwen3_5_moe_text` shape; unknown or malformed metadata stays generic.
Ornith uses prefill step 512 and disables the fused gate/up cache before model
loading. Generic uses prefill step 64 and leaves that cache unchanged.
Compiled decode, rotating-KV quantization, bounded windows, and SSM anchors
remain default-off.

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
swift package resolve
swift build            # debug
swift test             # unit tests (acceptance tests need a live server)
swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build
# The release launcher resolves the fork-pinned dependency automatically:
scripts/start_mei_server.sh
```

`vmlx-swift` is pinned to the Mei-maintained fork
([`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift)) at revision
`91fed8be…`. The fork's `main` contains five separate, documented commits
ported from the former Mei patch queue; see [`docs/VMLX-FORK.md`](docs/VMLX-FORK.md)
for the mapping and upstream-PR workflow. `Package.resolved` is committed.
Re-pinning or changing the fork revision is a deliberate decision that must
re-run the whole acceptance suite.

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
swift run mei --model-dir ~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned \
  --served-model-id ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --optimization-profile auto --port 8024 \
  --context-cap 65536 --prefill-step-size 512 \
  --kv-cache-dir ~/.local/share/local-model-bench/mei-runtime/kv-cache
```

Hybrid families (Ornith's qwen3_5_moe) need the disk tier for prefix reuse:
`--kv-cache-dir` (or `MEI_KV_CACHE_DIR`) enables it; without it every
request full-prefills (correct, just slower). Set `--memory-limit-bytes`
above the model's working set — the MLX default limit can otherwise sit
below it and make allocation wait on scheduled tasks (the hang failure
mode).

### Models (four-candidate MLX lineup)

- **Primary (MVP)**: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` — the model
  author's official plain 4-bit text-generation MLX quant of the MoE
  `qwen3_5_moe` architecture (4-bit affine / group-64; mlp gates and
  shared-expert gates held at 8-bit; ~19.5 GB weights, staged complete
  under `mei-models/`). The >= 30 decode tok/s goal is defined against this
  artifact on the 32 GB Sulaco machine. Its working set keeps it
  blocked-on-memory while other engines are resident, so the 9B proxy below
  exercises the identical Mei text path in that case.
- **Documented fallback (same family, smaller)**: `ornith-ai/Ornith-1.5-9B-MLX-4bit`
  — the official 9B dense `qwen3_5` 4-bit quant (5.0 GB, staged complete),
  same 4-bit-affine/group-64 scheme, same chat template and
  qwen3_coder-style tool-call format. Used to validate server correctness,
  tool-call behavior, and the optimization matrix on hardware where the 35B
  cannot be resident (e.g. while other engines hold Metal). It is a fallback,
  never a silent substitution for the primary artifact.
- **Secondary comparator — Qwen3.8 (dense qwen35)**: MLX candidate
  `mlx-community/Qwen3.8-27B-4bit` (regular 4-bit, `qwen3_5`,
  source `Qwen/Qwen3.8-27B`; staged at `mei-models/Qwen3.8-27B-4bit`, pinned
  `3e6447f`; **loadability pending GPU**). GGUF
  reference `unsloth/Qwen3.8-27B-GGUF` `UD-Q5_K_M` is cached complete and
  carries an MTP/Next-N head (compare without `--spec-type`). The MLX 4-bit
  is **not** UD-Q5 and must not be claimed as GGUF-UD equivalence; it is a
  regular 4-bit MLX comparator.
- **Secondary comparator — Gemma 4 26B-A4B (APEX-I-Quality)**: MLX candidate
  `mlx-community/gemma-4-26b-a4b-it-4bit` (regular 4-bit, `gemma4`, source
  `google/gemma-4-26B-A4B-it`; staged at
  `mei-models/gemma-4-26b-a4b-it-4bit`, pinned `0d77464`; **loadability
  pending GPU**). GGUF
  reference `mudler/gemma-4-26B-A4B-it-APEX-GGUF` `APEX-I-Quality`
  (arch `gemma4`, no MTP) is cached complete. Separate architecture from
  qwen3_5; the VLM load/template/tool-call path is added only when required
  by evidence.
- **Secondary comparator — Qwen3.8 Uncensored/Heretic (separate provenance)**:
  cached GGUF reference `trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5`
  `Q5_K_M` (qwen35, MTP present). The first-party MLX artifact in this
  lineage is `orcarouter/Qwen3.8-27B-Uncensored-MLX` (single commit
  `14963e70`), which provides a **native 4-bit affine/group-64 quant under
  `4-bit/`** (3 shards, 16.05 GB) of the Uncensored lineage — same
  chat-template/tokenizer/vocab bytes as the gated source
  `orcarouter/Qwen3.8-27B-Uncensored`, distinct from base Qwen3.8. No source
  conversion is required; it is staged at
  `mei-models/Qwen3.8-27B-Uncensored-MLX-4bit` pinned `14963e70`
  (**loadability pending GPU**). This supersedes the earlier "2-bit only"
  claim.

The machine-readable lineup is `configs/model-lineup.json` and is the source
of truth for exactly pinned revisions, GGUF blob SHA-256 digests, quant
settings, local staged paths, status, and test phases. Keep model status and
test phases there in sync with this section. `scripts/stage_model.sh` can
stage an explicitly selected Hugging Face repository with `--model-id`; it
must not be run for the cached GGUF files, which require llama.cpp rather than
Mei's MLX loader.

### MLX quantization strategy

- Prefer **first-party or established `mlx-community` checkpoints** already
  optimized for Apple/MLX, pinned by immutable revision. Where no suitable MLX
  checkpoint exists, convert from the **original model source** (never from
  GGUF) with a reproducible, Mei-owned conversion command that records bits,
  group size, calibration/data choices, and revision.
- Choose per-model bit depth by **weight size, long-context/KV-cache
  headroom, tool reliability, and measured speed**; start memory-safe at
  4-bit (affine / group-64) for 26–35B models and test 5-bit or higher only
  when the memory gate leaves headroom. Do not force one bit depth or recipe
  across architectures.
- Secondary VLM checkpoints carry preprocessor/processor files; architecture-
  specific load/template/tool-call handling is added only when required by
  evidence.
- A model is not Mei-ready until it loads, generates, passes the
  acceptance/tool-call checks, survives the long-context checks, and has
  reproducible provenance.


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
  off for A/B rows). `--ssm-anchor-boundaries K` (vMLX fork commit `91fed8be`, default off)
  stores additional SSM companion anchors at the first K chat role-turn
  boundaries (exact token offsets from the request's own rendering path,
  with an additivity self-check) so a mid-transcript diverging agentic
  edit restores from a retained boundary instead of full-prefilling —
  a TTFT/latency lever, not a decode tok/s lever; see
  `artifacts/design-anchor-ssm-0005.md`.
- **Disk safety**: `tools/mei_disk_guard.py` enforces a 20 GiB free-space
  floor before a server or measurement cycle starts. Disposable per-experiment
  caches (`kv-cache-sweep`, `kv-cache-cell-*`, `kv-cache-anchors-*`, and
  `kv-cache-exp-*`) are removed on completion or interruption; pass
  `MEI_RETAIN_KV_CACHE=true` only for a named reuse experiment. Model weights,
  repository artifacts, historical evidence, and protected recent-model caches
  are never selected by automatic cleanup. Decisions are appended to
  `mei-disk-guard.log` under the runtime root.
- **Tool calls**: vmlx-swift parses Qwen/Ornith-style `<tool_call>{json}`
  envelopes inside its generate loop; Mei maps `.toolCall` events to OpenAI
  `tool_calls` in both streaming and non-streaming shapes.
- **Reasoning**: Ornith is a Qwen3.5-lineage thinking model; thought text is
  exposed as `reasoning_content` (opt-out via `--emit-reasoning false`) and
  the thinking/visible streams are kept separate.
- **No MTP**: explicitly out of scope (the benchmark's own data shows no
  MTP/speculative win on this hardware).

## Name origin

Mei is named after [*Mei long*](https://en.wikipedia.org/wiki/Mei_long), a small
Early Cretaceous troodontid dinosaur whose fossil was found preserved in a
sleeping posture. The name is commonly translated as “sleeping dragon.” The
dinosaur reference is also a nod to [Osaurus](https://github.com/osaurus-ai/osaurus),
whose components inspired and are reused by this project.

The idea of a *small* dragon echoes [DwarfStar](https://github.com/antirez/ds4),
another inspiration for Mei. DwarfStar is named after a small star; Mei follows
the same pattern of giving a compact local-inference project a modest cosmic
name.
