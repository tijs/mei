# BUILT: Qwen3.6-35B-A3B-4bit-textonly — vision tower removed (2026-09-06)

A text-only derivative of the staged Qwen 3.6 checkpoint, produced and verified
on 2026-09-06. **Not yet loaded by Mei** — this is a checkpoint-integrity gate
only, no runtime evidence yet.

- Source: `~/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit`
  (mlx-community/Qwen3.6-35B-A3B-4bit @ 38740b847e4cb78f352aba30aa41c76e08e6eb46,
  post-SafetensorsStorageHealer — see note ee50359f)
- Output: `~/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit-textonly`
- Tool: `tools/strip_vision_tower.py` in worktree
  `/Users/tijs/projects/mei-opt-research` (branch `research/hybrid-superopt`)
- `conversion-provenance.json` written into the output directory

## What was removed

- **333 vision tensors, 851.8 MiB** — every `vision_tower.*` key. All were
  confined to shard 1, so shards 2-4 were APFS-cloned (`cp -c`) untouched.
- **1 config key: `vision_config`.** This is the important one: it is the ONLY
  key by which the source config differed from text-only Ornith's, and it is
  what routes the bundle to vmlx's VLM factory rather than the LLM factory.
- 3 vision sidecars: `preprocessor_config.json`, `processor_config.json`,
  `video_preprocessor_config.json`. Tokenizer, chat template, and
  `generation_config.json` are preserved unchanged.

Whole job took 6.5 s.

## Verification (all independent of the writing tool)

1. **Reference implementation opens it**: `safetensors.safe_open` reads all four
   shards, **1757 tensors**, zero errors — exactly Ornith's tensor count.
2. **Tensor key patterns now identical to Ornith's**, both directions empty.
3. **Bytes preserved**: the tool sha256-verifies every retained tensor src vs
   dst as it writes. Independently re-checked afterwards on a 50-tensor random
   sample by raw file-offset digest: **50/50 bit-identical**.
4. **Alignment**: 0 unaligned tensors of 383 in the rewritten shard; data
   segment starts at 51344 (8-aligned), matching the convention in
   `mei/tools/align_safetensors.py` so vmlx's mmap loader zero-copy maps instead
   of realigning into anonymous RAM.
5. **Config**: now a strict subset of Ornith's key set — the only remaining
   differences (`bos_token_id`, `dtype`, `hidden_size`, `pad_token_id`) are keys
   Ornith has and Qwen 3.6 never had, i.e. pre-existing and unrelated to vision.
6. **Size**: 19.00 GiB -> **18.17 GiB, identical to Ornith's 18.17 GiB.**
   Bandwidth budget recomputed: 1592 MiB/token active, same component split as
   Ornith to the decimal.

## Why this is an experiment, not just a cleanup

It changes two things at once, and that is deliberate — it is the cheapest way
to settle the open load-path question from the "Qwen3.6 may load via the VLM
path" note:

- **If Qwen 3.6 was on the VLM path**: this frees ~0.83 GB resident AND flips it
  to the LLM path, which means it **loses** `compileSeparatedDecode: true`
  (passed only at `MLXVLM/Models/Qwen35.swift:2184`). Net effect could go either
  way, and the direction is itself the measurement.
- **If it was already on the LLM path**: the vision weights were being dropped
  by `Qwen35Model.sanitize` anyway, so post-load memory should be unchanged and
  the only win is 0.83 GiB of disk. That outcome falsifies the VLM hypothesis.

Either result is informative. Post-load memory is the discriminator: stock
Qwen 3.6 measured **20.44 GB** (note 9e140ea7) vs Ornith's 19.55 GB, a 0.89 GB
gap that matches the vision tower almost exactly.

**Run `VMLX_MODEL_FACTORY_TRACE=1` on the stock bundle first** — it names the
winning factory outright and costs one server start, which is cheaper and less
ambiguous than inferring it from a memory delta.

## Status and limits

- Text-only derivative: vision/VLM capability is **removed, not disabled**.
- Not an upstream artifact. Must not be relabelled as
  `mlx-community/Qwen3.6-35B-A3B-4bit`. If it is ever published it needs its own
  HF repository and model card per plan `0b87b76a`'s quantization policy; the
  `conversion-provenance.json` is already written and records source repo,
  pinned revision, removals, and claim limits.
- No Mei config or lineup entry added — `configs/model-lineup.json` is untouched
  so nothing can silently stage or serve this yet.

#proj/mei
