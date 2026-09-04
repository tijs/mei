# Mei Model / Cache Artifact Cleanup — 2026-09-04

Authorized aggressive cleanup of Mei build artifacts, model copies, Hugging Face
cache, and experiment runtime caches. Recorded in the Mei repository. Source
implementation, benchmark results, local-model-bench tracked files, vmlx-swift,
and release assets were **not** modified.

## UTC

- Recorded: `2026-09-04T11:28:37Z`

## Pre-condition checks (before any deletion)

- No Mei / benchmark / model-server processes were active (`ps`), and no
  Mei-relevant model-serving ports were listening (`lsof -iTCP -sTCP:LISTEN`).
  Only unrelated listeners were present (macOS ControlCenter :5000, tailnet
  router :8766, Hermes dashboard :9119).
- Mei git branch `main` was 10 commits ahead of `origin/main`; working tree was
  clean. There was **no** working-tree modification to `Package.resolved`; it was
  left untouched (not edited, staged, or discarded).

## Pre-delete disk overview

- `df -h /`: `460 GiB` total, `410 GiB` used, `13 GiB` avail, `97%` capacity.

## Retained (KEEP) — verified present after cleanup

Four MLX model roots (serving repacks / tested checkpoints):

- `~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned` (18G)
- `~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit` (15G)
- `~/.local/share/local-model-bench/mei-models/gemma-4-26b-a4b-it-4bit` (14G)
- `~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-Uncensored-MLX-4bit` (15G)

Four Hugging Face GGUF cache repos (kept exactly as-is, shared/provenance-backed
blobs):

- `models--ornith-ai--Ornith-1.5-35B-A3B-GGUF` (20G)
- `models--unsloth--Qwen3.8-27B-GGUF` (18G)
- `models--mudler--gemma-4-26B-A4B-it-APEX-GGUF` (19G)
- `models--trohrbaugh--Qwen3.8-27B-heretic-ara-gguf-Q5` (18G)

Also retained: `Ornith-1.5-35B-A3B-MLX-4bit.provenance.json` sidecar (small
provenance for the retained aligned repack); all tracked benchmark
configs/results/docs and historical Mei evidence; `/Users/tijs/projects/mei/.build/arm64-apple-macosx/release`
(1.7G); `/Users/tijs/projects/mei/dist` in full incl. v0.1.0 and v0.2.0-alpha.1
archives/checksums (271M); `~/.local/share/local-model-bench/mei-build` (5.7G);
`mei-runtime/venv`, `mei-runtime/logs`, and `mei-runtime/mei-disk-guard.log`.

## Public HF verification (custom Qwen artifact, no credentials)

`https://huggingface.co/api/models/Tostibrown/Qwen3.8-27B-5bit-affine-g64`:

- `private`: `false`
- `sha` (commit revision): `f592c6fb6ed2e962b9f31cf0fbe30222d9dc727f`
- Tree (`/tree/main`): exactly **13 files**:
  - `.gitattributes` 1570
  - `README.md` 4684
  - `chat_template.jinja` 8952
  - `config.json` 4089
  - `conversion-provenance.json` 2067
  - `generation_config.json` 202
  - `model-00001-of-00004.safetensors` 5360418016
  - `model-00002-of-00004.safetensors` 5338181968
  - `model-00003-of-00004.safetensors` 5355925528
  - `model-00004-of-00004.safetensors` 2440187752
  - `model.safetensors.index.json` 189789
  - `tokenizer.json` 19989325
  - `tokenizer_config.json` 1161

Four shard sizes match the local Mei-produced artifact. Because the artifact is
publicly published and immutable, the local copy was removed. No new public HF
repo was created for the retained aligned Ornith repack (upstream-derived,
retained locally).

## Deleted paths (explicit allowlist only)

Local model leaves / custom artifact:

- `mei-models/Ornith-1.5-35B-A3B-MLX-4bit` (unaligned duplicate of retained
  aligned root) — 18G
- `mei-models/Ornith-1.5-9B-MLX-4bit` — 4.7G
- `mei-models/Ornith-1.5-9B-MLX-4bit.provenance.json` — 4K
- `mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf` — 5.4G
- `mei-models/Qwen3.8-27B-5bit-affine-g64` (verified published) — 17G
- `mei-models/Qwen3.8-27B-5bit-affine-g64.provenance.json` — 4K

Hugging Face partial/orphan caches:

- `~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-4bit`
  (known partial cache with four `.incomplete` shards; retained aligned local
  root is complete) — 4.3G
- `~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B-MLX-4bit`
  (empty/orphan cache) — 4K

Build trees:

- `/Users/tijs/projects/mei/.build/arm64-apple-macosx/debug` (stale debug
  products; release retained) — 3.1G
- `mei-fork-build` — 4.2G
- `mei-public-release-build` — 2.3G

`mei-runtime` disposable KV caches (venv/logs/mei-disk-guard.log retained):

- `kv-cache` 9.6G, `kv-cache-35b` 60K, `kv-cache-35b-fuseoff` 3.0G,
  `kv-cache-35b-lc80k` 8.0G, `kv-cache-gemma4` 113M, `kv-cache-heretic` 1.3G,
  `kv-cache-qwen38` 1.3G, `kv-cache-qwen38-20260902` 301M, `kv-ornith-35B` 1.8G

Experiment-specific runtime roots (committed evidence remains in results/artifacts):

- `mei-runtime-c-compiled16-smoketest`, `mei-runtime-gemma-ceiling`,
  `mei-runtime-gemma-defaultcheck`, `mei-runtime-gemma-inmem`,
  `mei-runtime-gemma-pref256`, `mei-runtime-gemma4-30k-bl`,
  `mei-runtime-gemma4-30k-cdec`, `mei-runtime-gemma4-30k-fuseoff` (6.9G),
  `mei-runtime-gemma4-30k-kv8`, `mei-runtime-gemma4-fuseoff-regate` (5.5G),
  `mei-runtime-heretic-30k-r1`/`r2`/`r3` (3.9G each), `mei-runtime-heretic-ceiling`,
  `mei-runtime-kvfix-smoke-20260902T112238Z`, `mei-runtime-ornith35-ceiling`,
  `mei-runtime-ornith35-step128`, `mei-runtime-ornith35-step64`,
  `mei-runtime-ornith9-ceiling`, `mei-runtime-q38-4bit-r1`/`r2`/`r3`,
  `mei-runtime-q38-4bit-unsafe-r1`/`r2`/`r3`, `mei-runtime-q38-5bit-r1`/`r2`/`r3`,
  `mei-runtime-q38-cdec-20260902T171635Z` (301M), `mei-runtime-q38-ceiling`,
  `mei-runtime-q38-ctrl-20260902T171528Z` (301M), `mei-runtime-q38-ref`,
  `mei-runtime-q38-unsafe-20260902T171806Z` (5.8G)

After the GGUF file was removed, `mei-models/gguf/` was empty and was removed.

Scheduled-for-deletion total measured before deletion: **121,232,540 KB ≈ 115.6 GiB**.

## Post-delete verification

- All to-be-deleted paths confirmed absent (fail-fast check).
- All four retained MLX roots exist; all four retained HF GGUF roots exist.
- No `.incomplete` files remain (the deleted partial cache path is fully gone).
- Retained aligned Ornith root contains expected key files
  (`MEI_ALIGN_MANIFEST.json`, config/tokenizer/chat_template, 4 safetensors
  shards, model index). Retained Qwen-4bit / gemma-4 / Uncensored roots each
  contain their config, tokenizer and 3 safetensors shards.
- `Ornith-1.5-35B-A3B-MLX-4bit.provenance.json` retained.
- `mei-runtime/venv`, `mei-runtime/logs`, `mei-runtime/mei-disk-guard.log` intact.
- Release build, `dist`, and `mei-build` retained.

## Post-delete disk overview

- `df -h /`: `460 GiB` total, `295 GiB` used, **128 GiB avail, 70% capacity**
  (previously 13 GiB avail / 97%). Used dropped 410 → 295 GiB
  (**~115 GiB reclaimed**).
- Post-delete retained MLX: aligned 18G, Qwen-4bit 15G, gemma 14G, Uncensored 15G.
- Post-delete HF hub cache total: `~/.cache/huggingface/hub` = 76G.

## Repo change

- `configs/model-lineup.json`: active Ornith `local_staged_path` now points to
  the retained `-aligned` directory; `artifact_audit.note` clarifies the
  original unaligned root was removed after the aligned repack was verified.
  Long evidence notes were not rewritten.
- `artifacts/model-artifact-cleanup-20260904.md`: this file.

Rationale: historical configs/results/evidence were preserved (they carry the
measured claims and are tracked in the repo), while disposable model copies,
partial/orphan HF caches, stale debug builds, and experiment runtime/KV caches
(whose committed evidence already lives in results/artifacts) were removed to
reclaim ~115 GiB. `Package.resolved` was not touched.