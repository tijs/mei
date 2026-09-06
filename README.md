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
[`NOTICE.md`](NOTICE.md). Current stable release: **0.2.0**.

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

## Install on Apple Silicon (stable 0.2.0)

Apple Silicon (arm64), macOS 15+. Model weights are never bundled.

**Homebrew (recommended):**

```bash
brew install tijs/tap/mei
mei --version     # -> mei 0.2.0
```

**Manual — release asset:** download `mei-0.2.0-macos-arm64.tar.gz` from the
GitHub release page, verify the `.sha256`, and unpack
(`.../bin/mei --version`). The bundle carries the required `mlx.metallib`
beside the executable.

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full install paths and the
source-built `scripts/install_mei.sh` installer. Release notes:
[`docs/RELEASE-0.2.0.md`](docs/RELEASE-0.2.0.md).

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

The validated primary is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`. The steps
below download ~19 GB of weights into a user-local directory (no system paths),
start one server on port **8024**, and smoke-test it. Only **one** `mei` server
should run at a time; press **Ctrl-C** in the server terminal to stop it.

```bash
export MODEL_ID="ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

# 1) Download weights once into a user-local cache (repeat steps are resumable/no-op)
hf download "$MODEL_ID" --local-dir "$HOME/.cache/mei/models/Ornith-1.5-35B-A3B-MLX-4bit"

# 2) Start the server (blocking; Ctrl-C stops it). Port 8024.
#    * env gate VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 disables the fused
#      gate/up cache required for the validated >=30 tok/s perf path.
mkdir -p "$HOME/.cache/mei/runtime/kv"
VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 mei \
  --model-dir        "$HOME/.cache/mei/models/Ornith-1.5-35B-A3B-MLX-4bit" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile ornith \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 512 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv"
```

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

> **Note on Ornith performance:** the >=30 tok/s path is validated with the
> *aligned repack* of this checkpoint (byte-identical weights, aligned for
> loading) plus the env gate above, 512 prefill, and the disk KV tier. The
> official checkpoint that `hf download` pulls is the plain 4-bit MLX quant;
> it runs correctly (correct output, tool-calling) but the aligned repack is
> the measured performance path for the 32 GB machine.

## Other model quickstarts

Two further candidates are tracked alongside the validated Ornith primary.
Both serve user-local weights on port **8024** with context cap **65536** and
use a model-appropriate optimization profile. Only **one** server runs at a
time — **Ctrl-C** the current one before starting the next. Deeper status and
provenance for each: **[docs/MODELS.md](docs/MODELS.md)**.

### Qwen3.6 — exploratory candidate

```bash
export MODEL_ID="mlx-community/Qwen3.6-35B-A3B-4bit"
MODEL_DIR="$HOME/.cache/mei/models/Qwen3.6-35B-A3B-4bit"

# 1) Download weights once into a user-local cache (repeat is resumable/no-op)
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

# 2) Start the server (blocking; Ctrl-C stops it). Port 8024.
mkdir -p "$HOME/.cache/mei/runtime/kv"
VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 mei \
  --model-dir        "$MODEL_DIR" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile auto \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 512 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv"
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
MODEL_DIR="$HOME/.cache/mei/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"

# 1) Download weights once into a user-local cache (repeat is resumable/no-op)
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

# 2) Start the server (blocking; Ctrl-C stops it). Port 8024.
mkdir -p "$HOME/.cache/mei/runtime/kv"
mei \
  --model-dir        "$MODEL_DIR" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile generic \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 256 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv"
```

Smoke-test it exactly like Ornith above (`curl` the same
`/v1/models` and `/v1/chat/completions` calls on port 8024).

> **Status caveat.** Experimental, pending gate
> ([`docs/MODELS.md#nemotron--experimental-pending-gate`](docs/MODELS.md#nemotron--experimental-pending-gate)):
> **not yet validated** — conservative `generic` profile, modest 256 prefill,
> disk KV. Treat any run as exploratory.

Each of these standalone blocks follows the same convention as the Ornith
primary: user-local `$HOME/.cache/mei/models/...`, port 8024, context cap
65536, model-appropriate profile/prefill/KV setup.

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