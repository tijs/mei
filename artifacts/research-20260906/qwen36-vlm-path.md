# Qwen3.6 is a VLM checkpoint and probably loads via a DIFFERENT path than Ornith (2026-09-06)

Found while diffing the two staged checkpoints. This is the one finding that
breaks the otherwise-clean "Ornith and Qwen 3.6 are identical" story, and it
has direct memory and optimization consequences. **Not yet confirmed live —
see the one-line check at the bottom.**

## The tensor diff

`Qwen3.6-35B-A3B-4bit` has **333 tensors Ornith does not**, totalling
**0.83 GiB**. Every one is a vision tower:

    vision_tower.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2,norm1,norm2}
    vision_tower.{merger.linear_fc1,merger.linear_fc2,merger.norm,patch_embed.proj,pos_embed}

Totals: Ornith 18.17 GiB / 1757 tensors; Qwen3.6 **19.00 GiB / 2090 tensors**.
The *language-model* half is identical in shape (both compute exactly
1592 MiB/token of active weight traffic with the same component split — checked
with `tools/bandwidth_budget.py`), so the earlier "same architecture" finding
holds for the text tower. Qwen3.6's config additionally carries `vision_config`,
`vision_start_token_id`, `vision_end_token_id`, which Ornith's does not.

Hermes is a text-only coding agent. **The vision tower is dead weight for
every use this project has.**

## Why the load path matters

Both factories register the same model_type:

- `Libraries/MLXLLM/LLMModelFactory.swift:53` — `"qwen3_5_moe"` -> `Qwen35MoEModel`
- `Libraries/MLXVLM/VLMModelFactory.swift:132` — `"qwen3_5_moe"` -> `Qwen35MoE.init(_:requesting:)`

`ModelFactory.swift:782` tries registered factories in order and takes the
first that does not throw `unsupportedModelType`. So which one wins is decided
at runtime, not by the config alone. The `vision_config` gate at
`LLMModelFactory.swift:695-714` that would force VLM routing is scoped to
**mistral3/ministral3 only** — it is not a general rule, so it does not settle
this.

Two consequences that pull in opposite directions:

1. **If the LLM factory wins:** `Qwen35Model.sanitize` (Qwen35.swift:1636-1642)
   explicitly `continue`s on any `vision_tower` / `model.visual` key, so the
   vision weights are **dropped and never become resident**. Cost is only
   0.83 GiB of wasted disk (which mattered before — this project hit a
   14.3 GB-free disk blocker on 2026-09-04, note dbdd632a).
2. **If the VLM factory wins:** the vision tower IS resident (~0.83 GiB of a
   32 GB budget), **and** `compileSeparatedDecode: true` is passed
   (`MLXVLM/Models/Qwen35.swift:2184`) — meaning Qwen 3.6 would **already be
   getting the compiled routed-MoE region that finding F4 proposes adding to
   the LLM path**, and Ornith would not.

## Arithmetic that favours option 2

Recorded post-load memory:

| model | post-load | source |
|---|---|---|
| Ornith-1.5-35B-A3B-MLX-4bit-aligned | 19.55 GB | artifacts/ornith-35B-fuse-gateup-eliminated-20260902.md |
| Qwen3.6-35B-A3B-4bit | **20.44 GB** | Kiem note 9e140ea7 (staging gate) |

Difference: **0.89 GB — almost exactly the vision tower's 0.83 GiB.** The text
towers are the same shape, so if the vision weights were being dropped the two
post-load figures should be within noise of each other. They are not.

**Working hypothesis: the VLM factory wins for Qwen 3.6, the vision tower is
resident, and Qwen 3.6 is already running the compiled routed MoE.**

## Why this is worth acting on

If the hypothesis holds it hands us a **free natural A/B that is already in the
data**: two checkpoints with an identical text architecture, one on the LLM path
without the compiled routed-MoE region (Ornith, 55.0 tok/s short decode) and one
on the VLM path with it (Qwen 3.6, 46-50 tok/s per the staging gate).

Note the direction: the model that (hypothetically) HAS the compiled region is
the **slower** one. That does not falsify F4 — Qwen3.6 also carries ~0.83 GB
more resident memory and may differ in other VLM-path details — but it means
**F4's projected +3% must not be treated as banked**. The microbenchmark
measured the region in isolation; this is the first hint that the whole-model
picture may not match, and it should temper the F4 estimate until measured
in-server.

Actions this implies, in order:

1. **Confirm the path.** One line, no A/B, next time a server starts:
   `VMLX_MODEL_FACTORY_TRACE=1` prints `[ModelFactory] <type> failed: ...` for
   each factory that declines, so the winner is unambiguous. Do this for BOTH
   models before running any of the P0-P6 experiments — several of them assume
   the LLM path.
2. **If VLM wins for Qwen 3.6: strip the vision tower** into a text-only
   repack. ~0.83 GB resident and disk back, no capability lost for Hermes.
   Fits the existing aligned-repack tooling and provenance discipline (this
   would be a Mei-produced artifact and needs its own
   `conversion-provenance.json` per the plan's quantization policy).
3. **Re-check the F4 recommendation** (add `compileSeparatedDecode: true` to
   `MLXLLM/Models/Qwen35.swift:870`) against whichever path each model actually
   takes. If Qwen 3.6 is already on the compiled VLM path, F4 is an
   **Ornith-only** change, not a shared one.

## Caveat

Everything above is static analysis plus the post-load memory arithmetic. The
0.89 GB match is strong but circumstantial — a routing difference, a different
KV pre-allocation, or a load-time healing artifact could also explain it
(note ee50359f records that Qwen3.6's first load did invoke
`SafetensorsStorageHealer` on all four shards). **Run the trace before acting.**

#proj/mei
