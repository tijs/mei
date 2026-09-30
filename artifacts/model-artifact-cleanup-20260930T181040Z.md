# Model artifact cleanup — 2026-09-30

## Authorization and scope

At 2026-09-30T18:10:40Z, Tijs authorized deletion of every listed non-active misaligned/experimental model root. The allowlist was literal and limited to the four directories below. Active Qwen roots, evidence, configs, provenance metadata, and the separate Qwen3.6 misaligned provenance sidecar were retained.

## Deleted paths

- `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit-misaligned-20260930T171531Z`
  - safetensor payload bytes: 20,402,204,271
  - `du` before deletion: 20,429,299,712 bytes
- `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit-misaligned-20260930T171531Z`
  - safetensor payload bytes: 16,054,541,349
  - `du` before deletion: 16,081,608,704 bytes
- `/Users/tijs/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-restage-19504d9`
  - safetensor payload bytes: 19,509,024,201
  - `du` before deletion: 19,596,406,784 bytes
- `/Users/tijs/.local/share/local-model-bench/mei-models/Ternary-Bonsai-2-27B-mlx-2bit`
  - safetensor payload bytes: 8,595,477,990
  - `du` before deletion: 8,608,948,224 bytes

The four explicit roots accounted for 64,716,263,424 bytes (60.27 GiB) by `du` before deletion.

## External recoverability checks

The local shard names and byte sizes matched the public upstream tree at the recorded revisions before deletion. All four repositories reported `private: false`.

- `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` at `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`: four shard sizes matched.
- `mlx-community/Qwen3.6-35B-A3B-4bit` at `38740b847e4cb78f352aba30aa41c76e08e6eb46`: four shard sizes matched.
- `mlx-community/Qwen3.8-27B-4bit` at `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`: three shard sizes matched. The repository's current head had advanced, but the requested pinned revision remained readable and was the one checked.
- `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` at `3f926b415992eaa2ae9dd7b573706494d6bbf787`: the single shard size matched.

## Pre-delete guards

- No `mei`, `probe_mei.py`, benchmark launcher, or cleanup-related model process was running.
- No listener existed on ports 8024, 8025, or 18271–18276.
- The two recurring model/research jobs found in `~/.hermes/cron/jobs.json` were already paused; no scheduler state was changed.
- The active Qwen roots were separate from the deletion allowlist and contained their config, tokenizer, alignment manifest, and safetensor shards.
- Mei `main` was clean apart from the pre-existing untracked `Casks/` and `tap/` paths, and was six commits ahead of `origin/main`. `local-model-bench` was not modified.

## Post-delete state

- All four allowlisted directories are absent.
- Active Qwen3.6 remains at `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit` with four shards and 20,402,204,288 shard bytes.
- Active Qwen3.8 remains at `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit` with three shards and 16,054,541,360 shard bytes.
- Both active manifests and adjacent provenance now record `source_dir`/`source_root: null`, retain the original path in an `*_original` field, and mark the source as not retained.
- The compact Qwen3.6 provenance sidecar remains at `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit-misaligned-20260930T171531Z.provenance.json` as historical metadata; the model payload is gone.
- Post-delete active-path dry runs for Qwen3.6 and Qwen3.8 both exited 0.
- No model process or known benchmark listener was present after deletion.
- Four unrelated `.incomplete` Hugging Face download files under the retained aligned Ornith directory were observed and deliberately left untouched; they were outside the allowlist.

## Disk accounting

The model root filesystem changed from:

- Used: 432,585,715,712 bytes; available: 18,390,618,112 bytes; capacity: 96%

to:

- Used: 387,376,189,440 bytes; available: 63,600,144,384 bytes; capacity: 86%

The filesystem reported 45,209,526,272 bytes (42.10 GiB) recovered. This is lower than the 60.27 GiB `du` total, consistent with APFS compression/sparse-file accounting; both measurements are recorded separately.

## Evidence retained

The alignment and acceptance evidence remains committed under `artifacts/safetensors-qwen-alignment-gate-20260930T173003Z/`. This cleanup note records the deleted allowlist and does not remove benchmark results or change `local-model-bench`.
