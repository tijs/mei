# Models

How Mei's model candidates are chosen, staged, quantized, and run. The
machine-readable source of truth for pinned revisions, digests, quant
settings, staged paths, and status is
[`configs/model-lineup.json`](../configs/model-lineup.json) — keep this page
in sync with it and never let it drift into claiming a model is in the lineup
when it is not.

## Candidate status at a glance

| Model | Repo (download) | Served id | Status |
|---|---|---|---|
| Ornith 1.5 35B A3B | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | **Primary — validated** (loadable, acceptance-passed, fits 32 GB env-gated) |
| Qwen3.6 35B A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `mlx-community/Qwen3.6-35B-A3B-4bit` | **Exploratory** — bounded Mei gate passed (text/tool-only path); admission/provenance reconciliation still a separate work item |
| Nemotron 3.5 Lightning 30B A3B | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` | **Experimental — pending gate**, not staged or validated |

None of Qwen3.6 or Nemotron are (yet) rows in `configs/model-lineup.json`;
they are separate evaluation candidates documented here for tracking. Do not
claim they are in the lineup.

## Provenance and quantization policy

- Prefer **first-party or established `mlx-community` checkpoints** optimized
  for Apple/MLX, pinned by immutable revision. Where no suitable MLX
  checkpoint exists, convert from the **original model source** (never GGUF)
  with a reproducible, Mei-owned command recording bits, group size,
  calibration choices, and revision.
- Choose per-model bit depth by **weight size, long-context/KV headroom, tool
  reliability, and measured speed**; start memory-safe at 4-bit
  (affine / group-64) for 26–35B models and test 5-bit+ only when the memory
  gate leaves headroom. Do not force one recipe across architectures.
- A model is not Mei-ready until it loads, generates, passes the
  acceptance/tool-call checks, survives the long-context checks, and has
  reproducible provenance.
- **Weight separation:** Mei ships source + binary only. Each checkpoint's
  `.safetensors` are a separately staged, user-downloaded artifact (see
  [`NOTICE.md`](../NOTICE.md)).

## Quickstarts (Homebrew-installed `mei`)

General rules for all three: weights go into a user-local
`$HOME/.cache/mei/models/...` directory (never a system path), the server
serves on **port 8024** with **context cap 65536**, and only **one** server
should run at a time — press **Ctrl-C** in the server terminal to stop it.
Each block is standalone copy-paste. **These 30–35B presets target 32 GB Apple
Silicon; 16 GB machines likely cannot fit them.**

### Ornith — validated primary

The validated primary is `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, pinned at
revision `19504d9`. The measured path runs the **aligned repack** (byte-identical
weights, naturally aligned for vmlx's zero-copy mmap loader); `hf download`
alone gives the **raw** checkpoint, which is correctness-capable but **not**
the measured >=30 tok/s path.

**M1 / 32 GB launch settings (validated):** env
`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, `--optimization-profile ornith`,
`--context-cap 65536`, `--prefill-step-size 512`,
`--memory-limit-bytes 30000000000` (explicitly above the ~22.4 GB MLX default,
which can hang the 35B working set), user-local `--kv-cache-dir`,
`--max-tokens 32768`, sampling `0.6/0.95/20`,
`--emit-reasoning true --cache-reuse true --compiled-decode false`. Aligned
repack 30k decode measured 47.5–50.3 t/s; long-context 80k/90k passed with peak
<30 GB on the same memory/prefill/env/disk-KV settings.

```bash
export MODEL_ID="ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
export MODEL_REVISION="19504d912fa8fc7622bf6b1de3db5d5d890b1f02"
RAW_DIR="$HOME/.cache/mei/models/Ornith-1.5-35B-A3B-MLX-4bit-raw"
ALIGNED_DIR="$HOME/.cache/mei/models/Ornith-1.5-35B-A3B-MLX-4bit-aligned"

# 1) One-time: download the official pinned checkpoint to a -raw directory (~19 GB)
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$RAW_DIR"

# 2) One-time: fetch the pure-stdlib alignment helper and build the -aligned repack
mkdir -p "$HOME/.cache/mei/tools"
curl -fsSL -o "$HOME/.cache/mei/tools/align_safetensors.py" \
  https://raw.githubusercontent.com/tijs/mei/v0.2.0/tools/align_safetensors.py
python3 "$HOME/.cache/mei/tools/align_safetensors.py" "$RAW_DIR" "$ALIGNED_DIR"

# 3) Start the server from the ALIGNED directory (blocking; Ctrl-C stops it). Port 8024.
mkdir -p "$HOME/.cache/mei/runtime/kv"
VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 mei \
  --model-dir        "$ALIGNED_DIR" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile ornith \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 512 \
  --memory-limit-bytes 30000000000 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv" \
  --max-tokens 32768 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --emit-reasoning true --cache-reuse true --compiled-decode false
```

Smoke test (second terminal, no `jq`):

```bash
export MODEL_ID="ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
curl -s http://127.0.0.1:8024/v1/models
curl -s http://127.0.0.1:8024/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":16}" \
  | grep -o '"content":"[^"]*"' | head -1
```

**Raw vs aligned:** the raw checkpoint that `hf download` pulls is the plain
4-bit MLX quant (4-bit affine, group-64; gate + shared-expert gate at 8-bit,
~19.5 GB). It runs **correctly** (right output, tool-calling) but vmlx's mmap
loader force-copies ~20 GB of the payload into anonymous RAM because tensor
offsets aren't naturally aligned — so raw is **not** the measured performance
path. The `-aligned` repack (a one-time step, needs extra disk) is
byte-identical on the payload and is the validated >=30 tok/s path.
Omitted experimental knobs (`--max-kv-window`, `--ssm-anchor-boundaries`, KV
quantization) are **not** part of this preset and remain off/unvalidated.

### Qwen3.6 — exploratory candidate

**M1 / 32 GB launch settings (measured candidate path):** env
`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`, `--optimization-profile auto`,
`--prefill-step-size 512`, explicit `--memory-limit-bytes 30000000000` for the
32 GB target, user-local `--kv-cache-dir`, `--max-tokens 32768`, sampling
`0.6/0.95/20`, `--emit-reasoning true --cache-reuse true
--compiled-decode false`. Pinned revision `38740b8`.

```bash
export MODEL_ID="mlx-community/Qwen3.6-35B-A3B-4bit"
export MODEL_REVISION="38740b847e4cb78f352aba30aa41c76e08e6eb46"
MODEL_DIR="$HOME/.cache/mei/models/Qwen3.6-35B-A3B-4bit"

hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"

mkdir -p "$HOME/.cache/mei/runtime/kv"
VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 mei \
  --model-dir        "$MODEL_DIR" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile auto \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 512 \
  --memory-limit-bytes 30000000000 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv" \
  --max-tokens 32768 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --emit-reasoning true --cache-reuse true --compiled-decode false
```

> **Status caveat.** This is an **exploratory** candidate. The bounded Mei gate
> passed for the **text/tool-only path**: native streaming/non-streaming tool
> calls, KV reuse, and the exact 65536 context boundary. A full
> `local-model-bench` run is recorded. But Mei lineup admission and provenance
> reconciliation are still a **separate work item**, and only the text/tool
> path is covered — **no multimodal or stable-lineup admission claim is made**.
> Omitted experimental knobs remain off/unvalidated.

### Nemotron — experimental, pending gate

**M1 / 32 GB launch settings (conservative gate baseline — NOT optimized or
validated):** `--optimization-profile generic`, `--prefill-step-size 256`,
user-local `--kv-cache-dir`, an explicit
`--memory-limit-bytes 30000000000` allocator ceiling as a **guardrail**,
`--max-tokens 32768`, sampling `0.6/0.95/20`, and
`--emit-reasoning true --cache-reuse true --compiled-decode false`. No
validated Mei run or optimal profile exists yet; these are starting guardrails
pending the gate. Pinned revision `55ac8c8`.

```bash
export MODEL_ID="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
export MODEL_REVISION="55ac8c89261109b36c04371cd3f479a4594208c8"
MODEL_DIR="$HOME/.cache/mei/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"

hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"

mkdir -p "$HOME/.cache/mei/runtime/kv"
mei \
  --model-dir        "$MODEL_DIR" \
  --served-model-id  "$MODEL_ID" \
  --optimization-profile generic \
  --port 8024 \
  --context-cap 65536 \
  --prefill-step-size 256 \
  --memory-limit-bytes 30000000000 \
  --kv-cache-dir "$HOME/.cache/mei/runtime/kv" \
  --max-tokens 32768 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --emit-reasoning true --cache-reuse true --compiled-decode false
```

> **Status caveat.** **Experimental and pending its gate**: this is the
> *next* experimental evaluation, **not yet staged or validated under Mei's
> acceptance/long-context checks**. It uses the conservative `generic` profile
> with a modest prefill step of 256, disk KV, and an explicit 30 GB allocator
> ceiling as a guardrail. **No validation claim is made** for this model —
> treat any run as exploratory. Omitted experimental knobs remain
> off/unvalidated.

## Historical: the four-model comparator lineup

The four established comparator rows below are the validated lineup that
shipped in 0.2.0 (see `docs/RELEASE-0.2.0.md` / `docs/RELEASE-0.2.0-alpha.1.md`
and `artifacts/*` for evidence). Their pinned revisions, digests, quant
settings, staged paths, and status are the source of truth in
`configs/model-lineup.json`. This is **historical** status as of the 0.2.0
release — do not update these rows here; update the lineup.

- **Primary — Ornith 1.5 35B A3B** (`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`,
  `qwen3_5_moe`). Plain 4-bit aff/g64 MLX quant (gate + shared-expert gate at
  8-bit), ~19.5 GB, staged at
  `mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned`. 30k loaded decode
  47.5–50.3 t/s (3 repeats), short decode 55.0–55.9 t/s, peak 25.73 GB @ 65k
  cap — with the aligned repack + fuse-gate-up env + disk KV + 512 prefill.
  The former `Ornith-1.5-9B-MLX-4bit` proxy fallback is **NOT** in the lineup:
  its upstream repo became unavailable (2026-09-04) and the entry was removed
  so staging can never silently substitute it.
- **Secondary comparator — Qwen3.8 27B** (`mlx-community/Qwen3.8-27B-4bit`,
  dense `qwen3_5`). Regular 4-bit, pinned `3e6447f`. Loadable: `probe_load`
  3x PASS (15.66 t/s short-decode mean, peak 18.87 GB), `probe_mei` 12/12,
  `probe_coding` 4/4. The 30 t/s decode target for this dense model is a
  documented HARDWARE CEILING (~19–20 t/s pure-stream floor; 4-bit is the
  fastest safe recipe). GGUF reference `unsloth/Qwen3.8-27B-GGUF` UD-Q5_K_M is
  cached but carries an MTP/Next-N head — compare without `--spec-type`; the
  MLX 4-bit is **not** UD-Q5/GGUF-equivalent.
- **Secondary comparator — Gemma 4 26B-A4B** (`mlx-community/gemma-4-26b-a4b-it-4bit`,
  `gemma4`, pinned `0d77464`). Tool strict-schema gate cleared (integer args,
  `{"a":15,"b":27}`). `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` env gate lifts
  30k decode ~7.4 → 21.2 t/s mean (18.76 GB peak). Same-suite GGUF A/B
  (`APEX-I-Quality`) reaches 37.08 t/s — the residual ~1.6x gap is a vmlx
  long-KV SDPA kernel-efficiency blocker. Separate architecture from qwen3_5.
- **Secondary comparator — Qwen3.8 Heretic/Uncensored** (`orcarouter/Qwen3.8-27B-Uncensored-MLX`,
  pinned `14963e70`). Native 4-bit aff/g64 under `4-bit/`, 3 shards ~16.05 GB.
  `probe_load` 15.07/15.72 t/s, `probe_mei` 10/10 (incl. tool stream+non-stream
  parity and KV reuse), 30k long-context fresh 11.896 / reuse 11.842 t/s, peak
  24.71 GB. Same-suite GGUF Q5_K_M A/B: MLX +1.55x @ 30k.
- **Deferred — Ling 3.0 tiny** (`rapid-mlx/Ling-3.0-tiny-MLX-4bit`,
  `experimental-post-four-model`, status
  `post-four-model-experimental-loadability-pending-not-staged`). Recorded in
  the lineup as a deferred setup; loadability still pending.

Mei-produced public artifacts (e.g. the Qwen3.8-27B-5bit-affine-g64
comparator at `Tostibrown/Qwen3.8-27B-5bit-affine-g64`) live in their own
Hugging Face model repos with model cards + `conversion-provenance.json`;
they are published comparators, not replacements for the primary Ornith
model. `scripts/stage_model.sh` stages an explicitly selected HF repository
with `--model-id` (never run it for cached GGUF files, which need llama.cpp
rather than Mei's MLX loader).

## Serving one model at a time

Mei is a one-model-per-process server. If you switch models, stop the current
server (**Ctrl-C**) before starting the next on port 8024. To run more than
one model concurrently, give each instance its own `--port` and its own
`--kv-cache-dir` (in-process prefix reuse happens per process).