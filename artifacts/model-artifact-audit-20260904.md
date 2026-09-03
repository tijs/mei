# Model artifact audit — staged model directories (todo 0b87b76a#10, first leg) — 2026-09-04

Audit root: `~/.local/share/local-model-bench/mei-models` (the isolated Mei model
staging root used by `scripts/stage_model.sh` and the local-model-bench Mei MLX
backend). Method: per-directory provenance sidecar + file inventory + byte-level
comparison against upstream where a Mei-produced/repacked artifact exists. No
server was started and no benchmark ran this tick (CPU-side only).

## Classification

### Upstream artifacts (untouched, NOT Mei-produced) — 6 entries

| Staged dir | Upstream repo | Pin | Provenance sidecar |
|---|---|---|---|
| `Ornith-1.5-35B-A3B-MLX-4bit` | ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit | main | `.provenance.json` ok=true, 4 shards 19,509,024,201 B, arch qwen3_5_moe |
| `Ornith-1.5-9B-MLX-4bit` | ornith-ai/Ornith-1.5-9B-MLX-4bit | main | `.provenance.json` ok=true, single shard 5,038,161,163 B, arch qwen3_5 |
| `Qwen3.8-27B-4bit` | mlx-community/Qwen3.8-27B-4bit | main (pinned 3e6447f0 in lineup) | `.provenance.json` ok=true, 3 shards 16,054,541,349 B, arch qwen3_5 |
| `gemma-4-26b-a4b-it-4bit` | mlx-community/gemma-4-26b-a4b-it-4bit | main (pinned 0d77464e in lineup) | `.provenance.json` ok=true, 3 shards 15,341,205,776 B, arch gemma4 |
| `Qwen3.8-27B-Uncensored-MLX-4bit` | orcarouter/Qwen3.8-27B-Uncensored-MLX | 14963e70f886455cf93090ac95bdbf4c8730cbe1 (4-bit/ subdir relocated to root) | `.provenance.json` ok=true, 3 shards 16,054,541,599 B, arch qwen3_5 |
| `gguf/Ornith-1.5-9B-Q4_K_M.gguf` | ornith-ai/Ornith-1.5-9B-GGUF (reference blob) | digest abdd624b | blob sha256 70c11219…e8fab6 verified |

All upstream sidecars record repo + resolved revision + shard byte totals with
`verify.ok=true`. These are referenced, never relabeled, never claimed as
Mei-produced.

### Mei-produced artifacts — 2 entries

1. `Qwen3.8-27B-5bit-affine-g64` — **Mei-produced conversion** (mlx_lm.convert
   0.31.3 / mlx 0.32.0, `-q --q-bits 5 --q-group-size 64 --q-mode affine
   --dtype bfloat16`, source Qwen/Qwen3.8-27B @ 1d4bf0f2…, tree digest
   77d181b3…, source bytes 55,586,114,863). Carries the full sanitized
   `conversion-provenance.json` (schema mei.convert-mlx-quant/provenance-v1,
   NOT-UD / NOT-GGUF-derived claims, absolute local paths redacted) and a
   4,684 B README model card **inside the model dir**. 4 shards = 18,514,906,426 B.
   - **Published and verified this tick (read-only):**
     `Tostibrown/Qwen3.8-27B-5bit-affine-g64` — public, sha
     `f592c6fb6ed2e962b9f31cf0fbe30222d9dc727f`, created 2026-09-02T14:28:07Z,
     13 files incl. README.md (4,684 B) + conversion-provenance.json (2,067 B) +
     4 safetensors shards whose reported sizes equal the local files exactly.
     Matches lineup `publication` block. Publication requirement SATISFIED.
2. `Ornith-1.5-35B-A3B-MLX-4bit-aligned` — **Mei-produced repack**
   (tools/align_safetensors.py, manifest generated 2026-09-01T22:33:54Z,
   target alignment 8). **Byte-level verification this tick:** all **1757/1757
   tensor byte-slices are IDENTICAL** to the upstream ornith-ai 4-bit shards;
   only the safetensors header JSON (key order/serialization, header length
   delta 64–104 B/shard) and ≤6 B/shard inter-tensor padding differ. The
   manifest's per-shard payload sha256 values match the aligned files exactly
   (`manifest-ok`). So "0 realigned tensors" is accurate: this is a
   header-normalized, alignment-verified copy of upstream weights — NOT a
   weight-modifying conversion.
   - Carries `MEI_ALIGN_MANIFEST.json` but **no** conversion-provenance.json and
     **no** dedicated model card; **not published**. It is an internal runtime
     artifact (fused-gate-up cache workaround notes reference the aligned path
     for 35B serving), and because its weights are byte-identical to the
     upstream repo, publishing it would duplicate an upstream artifact — the
     plan forbids duplicating/relabeling upstream repositories. Recommended
     classification: internal serving artifact; do NOT publish. Final
     go/no-go is a user decision (autonomous publication is out of scope).

## Handle verification

Only one published handle exists across the lineup (`hf_url` / `hf_revision` /
`publication` blocks are otherwise absent): `Tostibrown/Qwen3.8-27B-5bit-affine-g64`,
verified live above (public, revision matches, sibling sizes match local).
No other Mei-produced artifact is published or queued for publication.

## Remaining under todo #10

- User go/no-go on the `-aligned` repack classification (recommend: internal,
  not published).
- Re-run this audit (same commands) after any future staging change; the
  commands used are recorded in the Kiem note for this tick.

## Artifacts

- This note: `artifacts/model-artifact-audit-20260904.md`.
- Existing evidence preserved: all `.provenance.json` sidecars, lineup
  `configs/model-lineup.json`, previous audit artifacts.