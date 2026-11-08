<p align="center">
  <img src="assets/mei-logo.png" alt="Mei logo" width="300" height="300">
</p>

# Mei

[Website](https://mei.computer/) · [Get started](https://mei.computer/#get-started)

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
[`NOTICE.md`](NOTICE.md). Install the
[latest stable release](https://github.com/tijs/mei/releases/latest) with
Homebrew. Published release notes and qualification evidence identify the exact
runtime, Metal artifact and tested checkpoints; source work branches can contain
unreleased changes.

The separately installed preview formula is for explicit prerelease evaluation:
`brew install tijs/tap/mei-preview`. Check `mei-preview --version` before
comparing it with stable; a preview formula can trail the current stable release.
Earlier [0.10.0-rc.2 notes](https://github.com/tijs/mei/blob/main/docs/RELEASE-0.10.0-rc.2.md) retain text-only bundle
preparation, the Groot profile admission and their original qualification scope.

**0.7.0** introduced token-level constrained structured output (`json_object`
and a strict recursive `json_schema` subset), plus exact-decimal numeric
constraints and fail-closed schema validation. **0.7.1** fixed escape-prefix
dead ends and whitespace-only generation stalls. **0.8.0** adds bounded
admission scheduling in front of the strict single-flight execution slot — a
bounded local queue (`--max-queue-depth`), a queue-wait budget
(`--queue-wait-ms`), `mei_priority` interactive/background classes,
end-to-end cancellation, and a `/v1/mei/status` scheduler block; overload
refusals are HTTP 429 `queue_full` / 503 `queue_timeout`, and execution stays
one generation at a time. The scheduling slice is verified by model-free
tests and a 10-check live HTTP integration run against the packaged 0.8.0
binary with `mlx-community/Qwen3-4B-4bit` (queue-full HTTP status, priority
field behavior, disconnect cancellation, and queue-order readback). This is
scheduler/HTTP lifecycle evidence, not a model-quality claim or evidence for
overlapping generation.

**0.9.0** adds *opt-in* bounded-overlap concurrency: `--max-active-generations 2`
lets at most two generations run at once — never more, no batching, no
preemption — while a process started without the flag still runs exactly as
0.8.0 did. The mode is **experimental and not enabled by default**; values
outside 1...2 are rejected at parse time. On `mlx-community/Qwen3-4B-4bit` the
Peix-like multi-agent stand-in passed 10/10 paired runs (interactive TTFT p95
1.486 s vs 3.552 s serial; ≥12.25 GiB reclaimable memory; no pressure warnings
or swap growth; burst makespan median 1.42× faster than serial).
Qwen3.6-35B-A3B text-only failed the 4-GiB memory reserve in every tested
configuration (3.32–3.43 GiB) and is **not accepted** for this workload on the
tested device; Peix end-to-end orchestration remains unverified (its harness
does not exist yet), and the result does not generalize beyond the tested
model/device. See [release notes](https://github.com/tijs/mei/blob/main/docs/RELEASE-0.9.0.md).

The latest live structured-output evidence remains
the 0.7.1 runtime source (`ea5a67a`): the exact buffered and streaming CoCore
canaries passed on Qwen3.6 text-only, Qwen3.6 vision (text-only requests), and
aligned Ornith, as well as the existing Qwen3-4B checkpoint; image-conditioned
structured output on Qwen3.6 vision remains unverified. See
[inference scheduling](docs/INFERENCE-SCHEDULING.md),
[CoCore compatibility](https://github.com/tijs/mei/blob/main/docs/COCORE.md), and the
[compatibility matrix](https://github.com/tijs/mei/blob/main/docs/OPENAI-COMPATIBILITY.md).

Earlier releases fixed prefix reuse, added cross-conversation prefix anchors,
per-request instrumentation, and source-built Metal kernel packaging; see
**[CHANGELOG.md](CHANGELOG.md)**.

## Supported models

Mei's current text-model comparison uses these named presets:

| Model | `--model-profile` | Evaluated scope |
|---|---|---|
| Qwen3.6 35B-A3B text-only | `qwen3.6-35b-a3b-text` | Text, tools and household/Python workflows |
| Ornith 1.5 35B-A3B | `ornith-1.5-35b-a3b` | Text, tools and household/Python workflows |
| Groot 19B-A3B text-only | `groot-19b-a3b-text` | Text, tools and household/Python workflows |

Use the [current benchmark summary](https://github.com/tijs/local-model-bench/blob/main/results/AGENT_SUMMARY.md)
for task success, time, memory and exact runtime/model pins. The contenders have
tradeoffs; older coding scores do not establish a universal winner. Logics-SWE
is a separately pinned **llama.cpp** control in that comparison, not a qualified
Mei/MLX pack. The stand-in workflows are synthetic; they do not establish real
Peixinho application acceptance. Qualification of an unreleased runtime remains
separate from published task results.

Named presets and download revisions live in
[`ModelTuningRegistry`](https://github.com/tijs/mei/blob/main/Sources/MeiCore/ModelTuning.swift).
The vision preset `qwen3.6-35b-a3b` remains available; current text-only benchmark
results do not qualify image-conditioned behavior. Earlier profile figures and
the 0.6.1 coding comparison are preserved in
[historical model results](https://github.com/tijs/mei/blob/main/docs/HISTORICAL-MODEL-RESULTS.md).

```bash
mei pull qwen3.6-35b-a3b-text        # fetches the exact pinned revision, verifies it
mei --model-dir ~/.cache/mei/models/Qwen3.6-35B-A3B-4bit-textonly \
    --model-profile qwen3.6-35b-a3b-text
```

`mei pull <profile> --dry-run` shows you what it would fetch and where, without
downloading anything.


The 0.10.0-rc.2 preview adds `groot-19b-a3b-text`, pinned to the
[prepared text-only artifact](https://huggingface.co/Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly).
Its full native tool/cache/context admission passed on the pinned artifact:

```sh
mei-preview pull groot-19b-a3b-text
mei-preview --model-dir ~/.cache/mei/models/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly \
    --model-profile groot-19b-a3b-text --served-model-id groot
```

The profile keeps the measured 256-token prefill step, disables anchors and
sets an 8192-token default generation cap. Sampling remains independently
configurable. See the [current synthetic benchmark comparison](https://github.com/tijs/local-model-bench/blob/main/results/AGENT_SUMMARY.md)
for exact sampler/cache profiles and trial limits. Presets and explicit benchmark overrides are distinct.

The profile sets the chunked-prefill step, cross-conversation anchor
boundaries, the generation cap and the architecture handling together — these
have different optima per model, so one flag selects the whole set. Any flag you
pass explicitly still wins.

Profiles record their measured settings; they do not guarantee identical
answers across hardware or sampling trajectories. One setting
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
[docs/ARCHITECTURE.md](https://github.com/tijs/mei/blob/main/docs/ARCHITECTURE.md).

**Use the artifact named by the profile.** Profiles pin an exact repository and
revision; some use an aligned derivative. `mei pull --dry-run` shows the precise
plan before any download. Details and earlier alignment measurements are in
[docs/MODELS.md](docs/MODELS.md).

For explicit text-only preparation of compatible Qwen3.5-family bundles, see
[the payload-preserving exporter](https://github.com/tijs/mei/blob/main/tools/export_qwen_text.md). Architecture defaults
are reported as `qwen35-moe`; this identifies the family, not a named model or
validated tuning for every derivative. Existing `--model-profile` names are unchanged.

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
design: **[docs/ARCHITECTURE.md](https://github.com/tijs/mei/blob/main/docs/ARCHITECTURE.md)**.

## Install on Apple Silicon (stable)

Apple Silicon (arm64), macOS 15+. Model weights are never bundled.

**Homebrew (recommended):**

```bash
brew install tijs/tap/mei
mei --version     # -> mei 0.11.0
```

**Manual — release asset:** download `mei-0.11.0-macos-arm64.tar.gz` and its
`.sha256` file from the GitHub release page, verify the checksum, and unpack
(`.../bin/mei --version`). The bundle carries the required `mlx.metallib`
beside the executable.

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full install paths and the
source-built `scripts/install_mei.sh` installer. Release notes:
[`docs/RELEASE-0.11.0.md`](https://github.com/tijs/mei/blob/main/docs/RELEASE-0.11.0.md).

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
mei --model-dir ~/.cache/mei/models/Qwen3-4B-4bit \
    --served-model-id mlx-community/Qwen3-4B-4bit
```

```text
# ~/.cocore/engine-map — key must equal the --served-model-id above;
# value is the server root, no /v1.
mlx-community/Qwen3-4B-4bit = http://127.0.0.1:8024
```

Mei **0.9.0** carries the 0.7.1 structured-output fixes; the latest live
verification is the 0.7.1 runtime source (`ea5a67a`), which passed strict
structured-output canaries on Qwen3.6 text-only, Qwen3.6 vision (text-only
request), and aligned Ornith; 0.7.0's Qwen3-4B pass also remains valid.
CoCore's attached engine reads structured-output support for all three new
checkpoints, while forced-tool support passed only for Ornith. The installed
CoCore LaunchAgent/provider advertisement is not updated or verified by this
release. Qwen3-8B has not been retested after the fix, and image-conditioned
structured output on Qwen3.6 vision remains unverified. 0.9.0 carries the
bounded local admission queue in front of the execution slot
(429 `queue_full` / 503 `queue_timeout` refusals) — a backstop behind CoCore's
gate — and adds opt-in bounded-overlap concurrency
(`--max-active-generations 2`, experimental); the default remains strict
single-flight. See
[docs/OPENAI-COMPATIBILITY.md](https://github.com/tijs/mei/blob/main/docs/OPENAI-COMPATIBILITY.md) §8,
[docs/COCORE.md](https://github.com/tijs/mei/blob/main/docs/COCORE.md), and
[docs/INFERENCE-SCHEDULING.md](docs/INFERENCE-SCHEDULING.md) for exact
evidence.

## Build

Mei is bundled SwiftPM ([`Package.swift`](https://github.com/tijs/mei/blob/main/Package.swift), fork-pinned
`Package.resolved`); most users just install the release. To build from source
(debug/release, test, package, `mlx.metallib`, installer), see
**[docs/DEVELOPMENT.md](https://github.com/tijs/mei/blob/main/docs/DEVELOPMENT.md)** — the release build is fast on
Apple Silicon. The packaged Homebrew binary and the `docs/INSTALL.md`
installer both assume an already-built (or prebuilt) binary.

## The docs

- **[docs/MODELS.md](docs/MODELS.md)** — candidate status, provenance and
  quantization; the historical four-model comparator lineup; per-model
  quickstarts.
- **[docs/ARCHITECTURE.md](https://github.com/tijs/mei/blob/main/docs/ARCHITECTURE.md)** — runtime flow, optimization
  profiles, memory/KV/prefill behavior, design notes.
- **[docs/BENCHMARKING.md](https://github.com/tijs/mei/blob/main/docs/BENCHMARKING.md)** — memory measurement,
  acceptance probes, `local-model-bench` integration, methodology, evidence
  conventions.
- **[docs/DEVELOPMENT.md](https://github.com/tijs/mei/blob/main/docs/DEVELOPMENT.md)** — repo layout, build/test/
  package, `mlx.metallib`, vMLX fork pin and workflow.
- **[docs/INSTALL.md](docs/INSTALL.md)** — installer paths and safety contract
  (authoritative).
- **[docs/COCORE.md](https://github.com/tijs/mei/blob/main/docs/COCORE.md)** — CoCore attached-engine integration:
  engine map, canaries, streaming, and model-specific structured-output support.
- **[docs/VMLX-FORK.md](docs/VMLX-FORK.md)** — the vMLX fork commits and
  upstream-PR workflow (authoritative).
- **[docs/INFERENCE-SCHEDULING.md](docs/INFERENCE-SCHEDULING.md)** — admission
  and scheduling: the bounded queue, `mei_priority` classes, overload refusals,
  cancellation semantics, the opt-in bounded-overlap mode, and the
  `/v1/mei/status` scheduler counters.
Cache and memory controls, precedence, and requested/effective status fields are
explained in [Runtime policies](docs/RUNTIME-POLICIES.md).
