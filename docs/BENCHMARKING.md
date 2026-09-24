# Benchmarking

How Mei is measured, probed, and integrated with `local-model-bench`, plus the
methodology and evidence conventions that keep numbers honest and reproducible.

## Memory measurement

Memory numbers come from MLX's allocator (`Memory.snapshot()`:
active/cache/peak bytes, plus `--memory-limit-bytes` and `--cache-limit-bytes`)
and `GPU.deviceInfo()` (architecture, physical memory, recommended working
set). The long-gone Cmlx `get_physical_memory` entry point does not exist in
this stack and is not used. `/v1/mei/status` exposes live allocator state; the
startup log prints the post-load footprint.

**The hang failure mode:** an MLX memory limit **below the model working set**
makes alloc calls *wait on scheduled tasks* rather than fail. Benchmark
configs therefore pin explicit `--memory-limit-bytes` / `--cache-limit-bytes`
above the working set. (The MLX default limit is 1.5x the Metal recommended
working set and can sit below a model's working set otherwise.)

## Acceptance probes

The black-box acceptance oracle lives in `Tests/MeiTests/MeiAcceptanceTests`:
run RED against a missing server, go green once the server behaves; enabled
via `MEI_ACCEPTANCE_BASE_URL` (default `http://127.0.0.1:8024/v1`).

**The P0 compatibility contract** (what "OpenAI-compatible" means — the
tested subset, deferred features, error envelope, `max_completion_tokens`
policy): **[docs/OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md)**.

Standalone drivers live in `tools/` (authoritative copies; the
`local-model-bench/runner/probe_mei.py` mirrors `tools/probe_mei.py`):

| Tool | Purpose |
|---|---|
| `tools/probe_mei.py` | acceptance + **P0 contract matrix** gate (`probe_load`, `probe_mei`, `probe_coding`) — `--skip-advanced` drops the P0 probes; `--skip-context`/`--skip-cache` as before; `--self-test` validates the probe's own assertion logic without a server. Run against one live server: `python3 tools/probe_mei.py --base-url http://127.0.0.1:8024/v1 --model <served-id> --tokenizer <model-dir> --context-cap N --output artifacts/p0-matrix-<ts>.json` |
| `tools/probe_diverging_chat.py` | patch-0005 evidence probe: 5-turn tool-calling transcript, run A = strict growth, run B diverges at turn 5 (`place_order` → `cancel_order`); records `cached_tokens`/`prefill_ms`/TTFT per request + deterministic transcript/schema/output checks (`--self-test` validates without a server) |
| `tools/probe_long_context.py` | chunked-prefill survival at 30K/80K |
| `tools/bench_mei.py` | full benchmark rows (short, tool, 45K-loaded fresh + reuse, 40K chat) with engine-reported tok/s, TTFT/prefill ms, allocator bytes; artifact-only output under `artifacts/` |
| `tools/gguf_meta.py` | GGUF header + tensor-name reader (`--check-mtp` reports MTP/Next-N head via `nextn_predict_layers` and `blk.*.nextn.*` tensors; `--tensors [filter]` lists names; handles GGUF v2 vs v3 layout difference) |
| `tools/llama_ceiling.py` | llama.cpp hardware-ceiling driver on port 8074; `--provenance-only` verifies the binary, GGUF header (arch, context >= 64K, MTP-head presence) and sha256 against the pinned official blob digest WITHOUT launching the server (safe under contention); a digest mismatch is FATAL |
| `tools/convert_mlx_quant.py` | Mei-owned reproducible source→MLX quant wrapper (`mlx_lm.convert` 0.31.3), enforces a 20 GiB free-disk floor, writes `<name>.provenance.json` (schema v1) only after success, never claims UD/GGUF equivalence, `--dry-run` plans without downloading |
| `tools/validate_worker_options.py` | re-checks the served-id identity contract against the lineup, cross-checks the four `mei.yaml` backend registrations (read-only), verifies staged checkpoint completeness, probes each option's port |
| `tools/mei_disk_guard.py` | 20 GiB free-space floor before a server/measurement cycle starts; protects weights, artifacts, and recent-model caches from auto-cleanup |

The deterministic half of the P0 matrix lives in the Swift unit suites
(`OpenAIRequestValidationTests`, `GenerationControlSelectionTests`,
`OpenAIResponseShapeTests`, `RouterSSEFrameTests`, `MessageMappingTests`,
`ToolArgumentNormalizerTests`, plus the existing
`OpenAITypesTests`/`RouterSSEToolCallIndexingTests`/`CacheRestoreTrackerTests`)
and runs without any model:
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --skip MeiAcceptanceTests`.
The live half is the probe matrix above. Both halves are pinned in
docs/OPENAI-COMPATIBILITY.md §8, including the known Xcode-27 flaky
type-check caveat on the pinned vmlx dependency (report any such failure
separately from Mei failures). The published matrix evidence
(`artifacts/p0-matrix-qwen36-text-20260922T201844Z.json`) is kept as a
historical record — the two probe-side failures it shows are fixed in
`probe_mei.py` since `1e2f7a7` and covered by `--self-test`; a clean passing
full-matrix re-run against a live server is the remaining P0 gate (plan
unit 6), see OPENAI-COMPATIBILITY.md §8.

## `local-model-bench` integration

The `local-model-bench` harness drives Mei as a first-class `inference_engine`
(its commit `9df216e` wires the launch scripts and config; the boundary keeps
that repo read-only — Mei's own probe/bench drivers live in `tools/` with
outputs under `artifacts/`). Backend ports 8024–8027 are the isolated Mei
backend ports registered by `local-model-bench/configs/*/mei.yaml`; the
`umans-coder` profile routes Mei's candidates exactly as served, one custom
provider per port (see [`docs/WORKER-MODEL-OPTIONS.md`](WORKER-MODEL-OPTIONS.md)).
Ornith's ~26–28 GB working set prevents co-residency with other models on the
32 GB target, so exactly **one** Mei process — one model — runs at a time there
unless a workflow explicitly configures otherwise (e.g. separate ports and
`--kv-cache-dir`s for smaller candidates).

## Methodology

- **Never compare MTP against no-MTP.** `tools/gguf_meta.py --check-mtp`
  hygiene is applied before any llama.cpp/GGUF ceiling comparison: GGUF
  references (e.g. `unsloth/Qwen3.8-27B-GGUF` UD-Q5_K_M) often carry an
  MTP/Next-N head — compare without `--spec-type` and do not claim
  GGUF-UD equivalence for plain MLX 4-bit rows.
- **A/B in the same suite.** GGUF vs MLX and env-gate-on/off rows are run in
  the same suite (3 repeats where reported) before drawing a conclusion, e.g.
  `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` lifts 30k decode (Ornith 47.5→50.3
  t/s; Gemma 4 ~7.4→21.2 t/s). The env gate is carried by the bench launcher
  env, NOT a Mei source default.
- **Explicit memory limits.** Pinned `--memory-limit-bytes` above the working
  set; record peak allocator bytes, not only t/s.
- **SSM re-derive A/B.** The ~1x-prefill-at-turn-end re-derive pass can be
  turned off with `--ssm-rederive false` for A/B rows; name the mode in the
  row.
- **Disk floor.** A measurement cycle starts only if the 20 GiB floor is met.
- **Clean Mk. 2 evidence:** every artifact note is dated, names repeats,
  names ports, names the model/checkpoint revision, and cites exact tool
  invocation.

## Evidence conventions

Evidence artifacts live under `artifacts/` and are **never rewritten** —
historical benchmark notes and rows are preserved even after a model is
removed from the active lineup (e.g. the former `Ornith-1.5-9B` proxy). Each
benchmark run:

- records the provenance block (binary, metallib, checkpoint revision, quant
  recipe, GGUF blob digest where applicable) **before** any measurement row
- notes the hardware (Sulaco = 32 GB Apple M1 Max unless stated otherwise)
- reports engine-reported tok/s plus TTFT/prefill ms and allocator peak bytes
- labels env-gated settings explicitly (which env variables are the perf
  path vs Mei source defaults)
- cites the relevant artifact file(s) next to any speed/load claim

Cached `mei-models/` and runtime state (build dir, `kv-cache`) are transient
and never part of the evidence; only the dated artifact notes and
`configs/model-lineup.json` are authoritative for what was measured.