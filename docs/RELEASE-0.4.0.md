# Mei 0.4.0

Opt-in cross-conversation prefix reuse, Laguna XS 2.1 support, and the removal
of a per-turn re-derive that was costing about 60 seconds on every warm request.

Apple Silicon (arm64), macOS 15+. Model weights are never bundled.

## The headline: a warm agent turn goes from ~62 s to ~1.8 s

An agent harness sends the same large system prompt and tool schemas on every
turn of every conversation. That shared prefix was re-prefilled from cold every
single time — measured at 60.7 s for a 20,394-token prompt, on every request,
in every session, for the entire life of the project.

Enabling `--ssm-anchor-boundaries 2` now restores it instead:

| request | Ornith 1.5 35B-A3B | Qwen 3.6 35B-A3B text-only |
|---|---|---|
| cold, first ever | 122.9 s | 120.3 s |
| second conversation | **1.75 s** | **1.78 s** |
| new process, same KV dir | **1.79 s** | **1.82 s** |

The cold request is a one-time cost per unique prefix, amortised across every
later conversation and surviving process restarts.

**This is off by default.** `--ssm-anchor-boundaries 0` is unchanged behaviour.

## Three defects behind it

1. **The anchor offsets were always empty.** `SSMAnchorBoundaries.compute`
   requires a prefix-additive chat template: rendering `messages[0..<i]` must
   produce a token prefix of the full render. Qwen 3.5/3.6 violates this at the
   system boundary, where the system message rendered alone *with tools* is
   21,834 tokens against 20,394 for the full system+user render. The additivity
   self-check therefore returned no offsets, correctly given its contract, and
   the feature was inert on exactly the prompt shape it was built for. Added
   `computeByDivergence`, which renders the full template with one message
   substituted and takes the longest common token prefix, assuming nothing
   about the template.
2. **Mei never declared the boundary reusable.** vmlx's post-answer store loop
   persists, for a hybrid cache, only boundaries listed in
   `LMInput.cacheStablePrefixTokenCounts` — the field documented as
   "deliberately persisted for reuse by unrelated new chat sessions". Mei passed
   neither that field nor its parent, so the only stored boundary was the
   generation-stripped one, which already contains the current turn's own user
   tokens; its content key never matched another conversation.
3. **Every warm turn re-derived a boundary already on disk.**
   `hasDurableDiskEntry` demanded a validated separate SSM companion for all
   hybrid models, but disk-only MambaCache topologies never write one by design
   (the state round-trips in the payload as `mamba_{i}_state0/1`), and the fetch
   path accepts an entry without one for the same reason. The check was
   permanently false on those models, so the store loop replayed the whole
   prefix through the model after the answer had already streamed. The disk
   write that followed was then correctly skipped as already-validated, so the
   work produced nothing at all.

## Verified

- Restored output is byte-identical to cold on content, `reasoning_content` and
  `tool_calls`, on every restoring row, after a determinism baseline proved two
  identical cold servers agree byte-for-byte.
- Acceptance probe 12/12 on the release binary: streaming and non-streaming tool
  calls, cache reuse, exact context-cap boundary.

## Laguna XS 2.1 MLX

- Unwrap a leading `language_model.` prefix in sanitize.
- Normalize the routed gate layout before load dequantization, fixing
  `Unhandled keys [e_score_correction_bias, proj]`.
- Compile the routed SwitchGLU separated decode for the exact affine S-2.1 XS
  MoE topology only. Other model families are unaffected.

## vmlx

Re-pinned to `654eb455`. Deliberately excluded: the checkpoint-fused
`gate_up_proj` experiment, which loads at zero memory cost and is
token-identical but measured −2.9% short decode. Together with an earlier
compiled routed-MoE region at −9.7%, fusing gather calls does not pay on this
hardware.

## Also

- `MEI_ANCHOR_TRACE=1` prints the anchor computation: message roles, per-prefix
  token counts, and the resulting offsets or the reason they were rejected.
- `docs/RELEASE-RUNBOOK.md` documents the full release sequence, including the
  Homebrew tap step that 0.3.0 missed.
