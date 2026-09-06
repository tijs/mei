# Sequencing: isolate vs stack the hybrid-MoE levers (2026-09-06)

Answers "should each idea be an isolated experiment first, then stacked?" for
the P0-P6 plan. Short answer: **isolate where the levers interact, pre-stack
where they are structurally orthogonal** — and run two free diagnostics before
committing GPU time to any of it.

## Interaction map (this is what decides isolate-vs-stack)

| lever | mechanism | interacts with |
|---|---|---|
| P1 `VMLX_ENABLE_UNSAFE_COMPILE` | fuses MoE **elementwise** ops (SwiGLU, sigmoid gate) + Qwen35 micro-fusions | **P2** (same class of work) |
| P2 `compileSeparatedDecode: true` | wraps the 3 `gather_qmm` + silu + mul in ONE compiled region | **P1** |
| P3 pre-fused gate+up repack | **removes a dispatch** by changing data layout | largely orthogonal (measured to stack with compile: +10.4% alone, +16.5% with) |
| vision strip | removes unused weights | **changes the load path**, which decides whether P2 is even reachable |
| P4 bit depth / mixed 4-8 bit | memory + quality | orthogonal to all compute levers |

P1 and P2 both attack MoE elementwise fusion, so **expect sub-additive** —
measuring only the pair would leave you unable to say which one earned it, and
P2 costs a fork re-pin (and a full acceptance re-run) that is not worth paying
if P1 already captured the win.

## Stage 0 — two free diagnostics, before any A/B

1. `VMLX_MODEL_FACTORY_TRACE=1` — names the winning factory outright. Settles
   the LLM-vs-VLM path for both models in one server start. **P2's entire
   premise depends on this**, and so does the interpretation of the
   vision-strip result.
2. `MLXPRESS_GENERATION_PROFILE=1` on a short-decode leg — per-component decode
   budget.

**Stage 0 gates everything else.** The microbenchmarks characterised the MoE
routed blocks at 5.64 ms of the 18.18 ms step: **31%**. The other 69% (30 GDN
layers, 10 attention, lm_head, sampling) is uncharacterised. If GDN dominates
it, the whole F4/F5 line is worth ~4.6% and the effort belongs on the GDN path
instead. Roughly 10 minutes of GPU to avoid spending days optimising the wrong
third.

## Stage 1 — isolated, one variable per leg, in value order

1. **Vision-stripped Qwen 3.6 vs stock Qwen 3.6.** Already built and verified
   (note b936b20e). Settles the load path empirically, measures the ~0.83 GB
   memory win, and yields the Ornith-vs-Qwen3.6 comparison at matched
   architecture. Cheapest informative leg.
2. **P1 unsafe-compile alone** (`--compiled-decode false`). Biggest claimed win
   (+45-70% upstream), zero code change. The 2026-09-02 observation of +4-5%
   was confounded by compiled-decode being ON; this is the clean run.
3. **P2 `compileSeparatedDecode` alone.** One-line fork patch, then re-pin and
   re-run acceptance per `docs/VMLX-FORK.md`.

## Stage 2 — stack the survivors

P1 + P2 together, to measure the interaction rather than assume it. Only worth
running if both survived Stage 1 on correctness.

## Stage 3 — checkpoint work, only if the target is still unmet

P3 pre-fused repack, then P4 mixed 4/8-bit. Both are checkpoint changes needing
provenance; P4 is a **quality** experiment and must go through
local-model-bench's real suite, not speed probes.

## The gate that matters more than tok/s

For P1 and P2 the failure mode is **silent numerical corruption**, not a crash
(that is the whole reason `isCompiledDecodeSupported` is default-off). So every
leg needs **greedy, temp-0, token-for-token equality against the eager
baseline** on a fixed prompt set — not just probe pass/fail. A leg that is 20%
faster and diverges by one token is a FAIL. The 2026-09-02 run passed 12/12
probes three times with the env on, which is encouraging but was not looking
for divergence.

## Honest expectation setting

The measured F4+F5 stack is **+4.6% end-to-end** (55.0 -> ~57.5 tok/s). Real,
cheap, worth banking — but not transformative. The genuinely large unknowns are
(a) P1's upstream +45-70% claim, unverified on this model, and (b) whatever
Stage 0's profile reveals about the uncharacterised 69%. Do not let the
well-measured small levers crowd out the two big unknowns.

Also note: **none of P1-P4 apply unchanged to Nemotron.** Its routed experts
have no gate projection (fc1/fc2 only), so F4/F5 have nothing to fuse; its
linear layers are Mamba2, not GDN. See the Nemotron analysis note. Sequence it
as its own track, and use its built-in `NemotronHLayerProfiler` to get the
per-component budget that Stage 0 has to reconstruct by hand for qwen3_5_moe.

#proj/mei
