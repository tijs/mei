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

**0.5.0** — `--model-profile <name>` loads the settings measured for a supported
model (architecture handling, prefill step, anchor boundaries, generation cap),
and `mei pull <name>` fetches its pinned revision and verifies the weights are
the artifact those settings were measured on. Earlier releases fixed prefix
reuse, added cross-conversation prefix anchors and per-request instrumentation:
see **[CHANGELOG.md](CHANGELOG.md)**.

## Supported models

Pick by what you care about, then pass the name:

**Take `qwen3.6-35b-a3b-text` unless you need image input.** It is equal or
better than the alternatives on every measurement we have.

| model | `--model-profile` | images | decode | reply latency | turns per task | coding |
|---|---|---|---|---|---|---|
| **Qwen3.6 35B-A3B text-only** | `qwen3.6-35b-a3b-text` | no | **58 tok/s** | **1.05 s** | **8.9** | **100%** |
| Qwen3.6 35B-A3B vision | `qwen3.6-35b-a3b` | **yes** | 48 tok/s | 1.07 s | 9.5 | 87% |
| Ornith 1.5 35B-A3B | `ornith-1.5-35b-a3b` | no | 57 tok/s | 1.28 s | 11.1 | 93% |

"Reply latency" is time-to-first-token on an ongoing conversation, which is what
you feel while using an agent. "Turns per task" is how many agent round-trips a
coding task took — fewer is faster end to end, and it compounds with latency.

**Why not Ornith?** It was this project's default for a long time and is the most
heavily exercised model here, but it no longer wins on anything a user
experiences: same decode, higher latency, ~25% more turns per task, lower coding
pass rate, and it is the one model where cross-conversation prefix reuse costs
quality, so it ships with that turned off. Keep it if you specifically want a
second model family rather than two builds of one, or if you are reproducing
older results. Otherwise it is strictly the worse choice.

**The vision build costs about 18% of decode speed even on pure text** — 48
against 58 tok/s, same architecture and quantisation. Take it only if you
actually feed it images.

**Memory does not distinguish them.** All three are 4-bit MoE checkpoints, ~19 GB
on disk, peaking near **24 GB** in use. All three want a 32 GB machine and none
fits a 16 GB one.

**All three take about 50 s on the very first turn**, reading the ~20k-token
system+tools prompt for the first time; after that a turn starts in about a
second. Only the text-only build also keeps that prefix across *separate*
conversations, because it is the only model where cross-conversation anchors
cost nothing on our suite.

Figures from two cold runs of each shipped configuration on a 32 GB M1 at a
65,536-token context, plus the coding pass rate from the benchmark's composite
leaderboard.

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

Every number in a profile is the number it was **measured** at, so naming a
model gives the same answers on any machine that can afford them. One setting
can be walked back: a 1024-token prefill step needs about 26 GB of recommended
working set, and on a smaller device Mei drops to 512 and says so at startup.
It says so because the two are not simply a fast and a slow setting — chunked
prefill is not answer-invariant on this architecture, so a clamped server can
generate different text than the numbers here were measured on. Pass
`--prefill-step-size` yourself to override the clamp in either direction.

**Why you name the model rather than Mei detecting it:** the supported models
are not distinguishable from their metadata — Ornith and Qwen3.6 text-only
report the same `model_type`, architecture and layer topology. Mei detects the
*architecture*, which keeps an unknown model safe, but will not guess which
specific model you have. Serving without a profile works; you get architecture
defaults and Mei says so at startup. Rationale in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Use the repo the profile names.** Profiles pin an exact repo and revision, and
in one case that is a repack rather than the published checkpoint — same weights,
different padding, worth 4.7 GB of memory and a 3.2× faster 80k prefill. `mei
pull` handles this; details in [docs/MODELS.md](docs/MODELS.md).

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

## Quickstart

```bash
mei pull qwen3.6-35b-a3b-text          # ~19 GB, pinned revision, verified
mei --model-dir ~/.cache/mei/models/Qwen3.6-35B-A3B-4bit-textonly \
    --model-profile qwen3.6-35b-a3b-text
```

That is the whole setup. `mei pull` writes to `~/.cache/mei/models/<name>` and
prints the directory; `--dry-run` shows what it would fetch without downloading.
Swap the profile name for any row in the table above — the vision build if you
need image input.

The server listens on **127.0.0.1:8024** with an OpenAI-compatible
`/v1/chat/completions`. One model per process: **Ctrl-C** before starting
another. Point any OpenAI client at `http://127.0.0.1:8024/v1`.

```bash
curl -s http://127.0.0.1:8024/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-4bit-textonly",
       "messages":[{"role":"user","content":"hello"}]}'
```

Weights are never bundled and never uploaded — `mei pull` fetches from
HuggingFace into your own cache. Provenance, quantization and per-model status:
**[docs/MODELS.md](docs/MODELS.md)**.

## CoCore attached engine

[CoCore](https://github.com/graze-social/cocore) can serve a model through
Mei instead of spawning its own backend: point its engine map at this
server, and it probes `GET /v1/models`, runs a forced `report_status`
tool-calling canary plus a `response_format` structured-output canary, and
advertises exactly what passed.

```bash
mei --model-dir ~/.cache/mei/models/Qwen3.6-35B-A3B-4bit-textonly \
    --model-profile qwen3.6-35b-a3b-text \
    --served-model-id mlx-community/Qwen3.6-35B-A3B-4bit
```

```text
# ~/.cocore/engine-map — key must equal the --served-model-id above;
# value is the server root, no /v1.
mlx-community/Qwen3.6-35B-A3B-4bit = http://127.0.0.1:8024
```

Mei 0.5.0 passes the tool canary (the forced nested `tool_choice` name is
pinned to `report_status`) and **fails the structured-output canary by
design** — `response_format` is not implemented, so CoCore simply does not
advertise schema jobs for it. Full detail (endpoints, transport, canary
shapes, streaming usage, security, troubleshooting):
**[docs/COCORE.md](docs/COCORE.md)**; the wire contract it relies on is
**[docs/OPENAI-COMPATIBILITY.md](docs/OPENAI-COMPATIBILITY.md)**.

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
- **[docs/COCORE.md](docs/COCORE.md)** — CoCore attached-engine integration:
  engine map, canaries, streaming, Mei's structured-output limitation.
- **[docs/VMLX-FORK.md](docs/VMLX-FORK.md)** — the vMLX fork commits and
  upstream-PR workflow (authoritative).