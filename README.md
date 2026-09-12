<p align="center">
  <img src="assets/mei-logo.png" alt="Mei logo" width="300" height="300">
</p>

# Mei

Mei ([*Mei long*](https://en.wikipedia.org/wiki/Mei_long), the "sleeping dragon"
dinosaur) is a narrow, **native Swift/MLX OpenAI-compatible inference server for
Apple Silicon**. It is built directly on the pinned
[`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift) fork of
`osaurus-ai/vmlx-swift` (the engine, not the full Osaurus app). Inspired by
[DwarfStar](https://github.com/antirez/ds4), Mei targets a compact machine — an
M1 Mac with 32 GB of unified memory — and a 30+ decode tok/s goal with correct
tool-calling and no long-context collapse.

Mei's scope is deliberately small: **one model per server process**, the
OpenAI `chat/completions` surface, **chunked prefill** on for hybrid
architectures, and **in-process KV/prefix reuse** across turns. It is a focused
runtime, not a general MLX gateway — you run one `mei` server per model.

**License:** MIT. Model weights are never bundled; see
[`NOTICE.md`](NOTICE.md). Current stable release: **0.5.0**.

**0.5.0 (2026-09-12)** adds `--model-profile`: name a supported model and Mei loads the settings measured for it — architecture handling, prefill step, anchor boundaries and the generation cap together, because those have different optima per model. Profiles pin an exact HuggingFace revision, `mei pull <profile>` fetches it and verifies the weights are the artifact the settings were measured on, and Ornith now points at a published aligned repack worth 4.7 GB of memory and a 3.2x faster 80k prefill. Replaces `--optimization-profile`.

**0.4.2 (2026-09-12)** makes `mei --model-dir DIR` reuse its cache without being told to: the hybrid topology could not restore from the in-memory tier, so a bare server silently re-read the whole conversation every turn — measured bare over four turns, 246 s becomes 66 s. Adds `--request-log`, one JSON line per generation run, which is what turned "where does a turn's time go" into a measurement rather than an argument.

**0.4.1 (2026-09-11)** fixes prefix-reuse correctness and removes the flags you needed to know about. A restored prefix now reproduces byte-for-byte what the same configuration produces cold; before, it restored from re-derived state that could change the answer even against itself. Note the comparison: this does **not** mean enabling prefix reuse leaves your answers unchanged. Capturing a reusable boundary splits the prefill, and on this hybrid architecture a split prefill does not produce bit-identical state to a continuous one, so `--ssm-anchor-boundaries` on and off can give different output. `mei --model-dir DIR` is now a complete command — the profile applies the measured-best settings for the model it detects and reports them, and any setting that will not apply to the loaded topology says so.

**0.4.0 (2026-09-08)** adds opt-in cross-conversation prefix reuse
(`--ssm-anchor-boundaries K`), which takes a warm agent turn on a shared 20k
system+tools prefix from ~62 s to ~1.8 s on Ornith and Qwen 3.6 with output
byte-identical to cold; removes a per-turn boundary re-derive that cost ~60 s
on every warm request for MambaCache hybrids; and adds Laguna XS 2.1 MLX
support. See the changelog.

**0.3.0 (2026-09-07)** re-pins vmlx-swift onto the fork synced with 37 upstream
commits — carrying the upstream fix for Ornith cache-restore loops, DiskCache
atomicity, and an opt-in safetensors healer — and makes the MLXPress
cold-weight tier reachable (`MLXPRESS=N` was silently inert before). Measured
30k decode +3.0% with token-for-token identical greedy output.

## Supported models

Pick by what you care about, then pass the name:

| you want | model | `--model-profile` | weights |
|---|---|---|---|
| the validated default — best measured coding quality | Ornith 1.5 35B-A3B | `ornith-1.5-35b-a3b` | ~19 GB |
| text and tools only, fastest prefill | Qwen3.6 35B-A3B text-only | `qwen3.6-35b-a3b-text` | ~19 GB |
| images as well as text | Qwen3.6 35B-A3B (vision) | `qwen3.6-35b-a3b` | ~19 GB |

All three are MoE models that run in about 20 GB of unified memory and were
measured on a 32 GB machine at a 65,536-token context.

```bash
mei pull qwen3.6-35b-a3b-text        # fetches the exact pinned revision, verifies it
mei --model-dir ~/.cache/mei/models/Qwen3.6-35B-A3B-4bit-textonly \
    --model-profile qwen3.6-35b-a3b-text
```

`mei pull <profile> --dry-run` shows you what it would fetch and where, without
downloading anything.

The profile sets the chunked-prefill step, cross-conversation anchor
boundaries, the generation cap and the architecture handling together — these
have different optima per model, so one flag selects the whole set. Any flag you
pass explicitly still wins.

**Why you name the model instead of Mei detecting it.** The supported models are
not distinguishable from their metadata — Ornith 1.5 and Qwen3.6 text-only
report the same `model_type`, the same architecture and the same layer topology.
Telling them apart would mean keying behaviour off incidental fields like
`transformers_version`, which breaks the moment an upstream re-export changes
them. Mei still detects the *architecture* from `config.json`, which is what
keeps an unnamed or unknown model safe; it just will not guess which specific
model you have.

Serving a model with no profile works fine — you get architecture defaults
rather than tuned ones, and Mei says so at startup.

**Pull the repo the profile names, not a lookalike.** Ornith's profile points at
an *aligned repack* rather than the published checkpoint. The two have
bit-identical weights; they differ only in a few bytes of padding in each
shard's JSON header. That padding is worth 4.7 GB of memory and a 3.2× faster
80k-context prefill, because the published layout leaves 1,421 of 1,757 tensors
unable to be mmap'd and MLX copies them into anonymous RAM at load. Nothing
about that is visible from the outside, which is why the profile pins an exact
repo and revision.


## How it works (high level)

- **One native process.** `mei` is a single Swift/MLX server binary
  (arm64). No Python glue, no model orchestrator — the fork-pinned vmlx engine
  loads one checkpoint and serves it.
- **OpenAI-compatible API.** `POST /v1/chat/completions` (streaming and
  non-streaming), `GET /v1/models` identity, `reasoning_content` for
  thinking models. Default `http://127.0.0.1:8024/v1`.
- **Chunked prefill.** Prefill runs in bounded windows (`--prefill-step-size`),
  the long-context safeguard for hybrid (GatedDelta) architectures.
- **KV/prefix reuse.** vmlx's `CacheCoordinator` reuses the in-process KV/prefix
  across turns (`--cache-reuse`, default on); an on-disk KV tier
  (`--kv-cache-dir`) extends reuse to hybrid and dense families.
- **One model per process.** Each server serves exactly one checkpoint by its
  exact served id. Run other models as separate `mei` instances on their own
  ports.

Deeper detail on the runtime, optimization profiles, memory behavior, and
design: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Install on Apple Silicon (stable 0.5.0)

Apple Silicon (arm64), macOS 15+. Model weights are never bundled.

**Homebrew (recommended):**

```bash
brew install tijs/tap/mei
mei --version     # -> mei 0.5.0
```

**Manual — release asset:** download `mei-0.5.0-macos-arm64.tar.gz` from the
GitHub release page, verify the `.sha256`, and unpack
(`.../bin/mei --version`). The bundle carries the required `mlx.metallib`
beside the executable.

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full install paths and the
source-built `scripts/install_mei.sh` installer. Release notes:
[`docs/RELEASE-0.2.0.md`](docs/RELEASE-0.2.0.md); 0.3.0's changes are in
[`CHANGELOG.md`](CHANGELOG.md).

## One-time prerequisite: `hf`

Mei downloads its own **weights**, which are never bundled. The quickstarts use
the Hugging Face `hf` CLI (`hf download REPO --local-dir DIR`). Install it once
into an isolated tool directory (bypassing macOS's PEP-668
externally-managed-Python protection via Homebrew `uv`; the `hf` binary is made
available on your PATH without touching your system Python):

```bash
brew install uv                       # only if you don't already have `uv`
uv tool install --upgrade huggingface_hub   # installs/updates the `hf` CLI
hf --version                          # confirm it is on your PATH
```

Already have `hf` on your PATH? **Skip all three lines.** The quickstarts below
assume `hf` resolves in your shell.

## Quickstart: run the primary model (Ornith)

The validated primary is **[`Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned`](https://huggingface.co/Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned)**
— the official `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` weights, bit-identical,
repacked so every tensor sits at a naturally aligned offset.

Download that one, not the official checkpoint. The published layout leaves
1,421 of 1,757 tensors unable to be mmap'd, so MLX copies them into anonymous
RAM at load: **24.28 GB resident instead of 19.55, and 117.6 s instead of 36.7 s
for a fresh 80k-context prefill**. The entire difference is a few bytes of
padding in each shard's JSON header, and nothing about it is visible from the
outside. (Earlier versions of this guide had you download the raw checkpoint and
repack it yourself; the repack is published now, so that step is gone.)

```bash
MODEL_DIR="$HOME/.cache/mei/models/Ornith-1.5-35B-A3B-MLX-4bit-aligned"

# One-time download (~19 GB)
hf download Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned --local-dir "$MODEL_DIR"

# Run it. Blocking; Ctrl-C stops it. One mei server at a time.
mei --model-dir "$MODEL_DIR" \
    --served-model-id ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
```

That is the whole command. Mei detects the architecture and applies the
measured-best settings for it — prefill step, generation cap, the environment
the MoE needs — and prints what it chose at startup. Anything you pass
explicitly still wins.

If you want to check the weights are the ones you think, the repo ships
`MEI_ALIGN_MANIFEST.json` with a per-shard payload hash; the model card has a
short script that verifies it.

In a second terminal, verify the server is up and answering (no `jq` needed):

```bash
export MODEL_ID="ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

# identity on /v1/models
curl -s http://127.0.0.1:8024/v1/models

# a chat smoke test, grepping the plain-text content reply
curl -s http://127.0.0.1:8024/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":16}" \
  | grep -o '"content":"[^"]*"' | head -1
```

When done, press **Ctrl-C** in the server terminal.

> **Why the aligned repack is the measured path:** `hf download` pulls the
> plain 4-bit MLX quant. It runs correctly, but vmlx's mmap loader copies
> ~20 GB into anonymous RAM instead of zero-copy mapping it (tensors aren't at
> naturally aligned offsets), so the raw directory is not the validated >=30
> tok/s path. The `-aligned` repack is byte-identical on the payload (verified
> in its `MEI_ALIGN_MANIFEST.json`) and is the measured one on this 32 GB machine.

## Other model quickstarts

Two further candidates are tracked alongside the validated Ornith primary.
Both serve user-local weights on port **8024** with context cap **65536** and
use a model-appropriate optimization profile. Only **one** server runs at a
time — **Ctrl-C** the current one before starting the next. Deeper status and
provenance for each: **[docs/MODELS.md](docs/MODELS.md)**.

### Qwen3.6 — exploratory candidate

```bash
export MODEL_ID="mlx-community/Qwen3.6-35B-A3B-4bit"
export MODEL_REVISION="38740b847e4cb78f352aba30aa41c76e08e6eb46"
MODEL_DIR="$HOME/.cache/mei/models/Qwen3.6-35B-A3B-4bit"

# 1) Download weights once into a user-local cache (repeat is resumable/no-op)
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"

# 2) Start the server (blocking; Ctrl-C stops it). Port 8024.
mei --model-dir "$MODEL_DIR" --served-model-id "$MODEL_ID"
```

Smoke-test it exactly like Ornith above (`curl` the same
`/v1/models` and `/v1/chat/completions` calls on port 8024).

> **Status caveat.** Exploratory
> ([`docs/MODELS.md#qwen36--exploratory-candidate`](docs/MODELS.md#qwen36--exploratory-candidate)):
> the bounded Mei gate passed for the **text/tool-only path**; admission and
> provenance reconciliation are still open, and no multimodal claim is made.

### Nemotron — experimental, pending gate

```bash
export MODEL_ID="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
export MODEL_REVISION="55ac8c89261109b36c04371cd3f479a4594208c8"
MODEL_DIR="$HOME/.cache/mei/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"

# 1) Download weights once into a user-local cache (repeat is resumable/no-op)
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"

# 2) Start the server (blocking; Ctrl-C stops it). Port 8024.
#    Conservative 32 GB gate baseline, NOT an optimized/validated preset.
#    There is no --model-profile for this model because nobody has measured
#    one; Mei falls back to architecture defaults. --memory-limit-bytes
#    30000000000 and prefill 256 are starting guardrails, not measured
#    settings.
mkdir -p "$HOME/.cache/mei/runtime/kv"
mei \
  --model-dir        "$MODEL_DIR" \
  --served-model-id  "$MODEL_ID" \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 256 \
  --memory-limit-bytes 30000000000 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv" \
  --max-tokens 32768 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --emit-reasoning true --cache-reuse true --compiled-decode false
```

Smoke-test it exactly like Ornith above (`curl` the same
`/v1/models` and `/v1/chat/completions` calls on port 8024).

> **Status caveat.** Experimental, pending gate
> ([`docs/MODELS.md#nemotron--experimental-pending-gate`](docs/MODELS.md#nemotron--experimental-pending-gate)):
> **not yet validated** — conservative `generic` profile, modest 256 prefill,
> disk KV, and an explicit allocator ceiling as a guardrail. Treat any run as
> exploratory.

Each of these standalone blocks follows the same convention as the Ornith
primary: user-local `$HOME/.cache/mei/models/...`, port 8024, context cap
65536, and model-appropriate profile/prefill/KV setup.

> **Why these M1 knobs (32 GB target).** `--memory-limit-bytes 30000000000`
> (30 GB) raises the explicit MLX allocator ceiling above the ~22.4 GB default,
> which is below the 30–35B working set and can hang at load; it fits the
> 32 GB machines these presets target. `--prefill-step-size 512` (256 for the
> conservative Nemotron block) is the chunked-prefill window for hybrid
> GatedDelta architectures. `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` disables
> the fused gate/up cache on the validated Ornith/Qwen3.6 path. A user-local
> `--kv-cache-dir` provides the on-disk KV tier those families need.
> `--compiled-decode false` keeps graph-traced decode disabled (the default),
> avoiding the multi-minute compile tax. These presets target **32 GB Apple Silicon**;
> **16 GB machines likely cannot fit** these 30–35B checkpoints. Omitted
> experimental knobs (`--max-kv-window`, `--ssm-anchor-boundaries`, KV
> quantization) remain off and unvalidated.

## Build

Mei is bundled SwiftPM ([`Package.swift`](Package.swift), fork-pinned
`Package.resolved`); most users just install the release. To build from source
(debug/release, test, package, `mlx.metallib`, installer), see
**[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — the release build is fast on
Apple Silicon. The packaged Homebrew binary and the `docs/INSTALL.md`
installer both assume an already-built (or prebuilt) binary.

## The docs

- **[docs/MODELS.md](docs/MODELS.md)** — candidate status, provenance and
  quantization; the historical four-model comparator lineup; per-model
  quickstarts.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — runtime flow, optimization
  profiles, memory/KV/prefill behavior, design notes.
- **[docs/BENCHMARKING.md](docs/BENCHMARKING.md)** — memory measurement,
  acceptance probes, `local-model-bench` integration, methodology, evidence
  conventions.
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — repo layout, build/test/
  package, `mlx.metallib`, vMLX fork pin and workflow.
- **[docs/INSTALL.md](docs/INSTALL.md)** — installer paths and safety contract
  (authoritative).
- **[docs/VMLX-FORK.md](docs/VMLX-FORK.md)** — the vMLX fork commits and
  upstream-PR workflow (authoritative).