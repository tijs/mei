# Nemotron-3.5-Lightning-30B-A3B: architecture analysis before staging (2026-09-06)

Fills the gap in the 2026-09-06 hybrid-MoE research, which covered Ornith 1.5
and Qwen 3.6 but not Nemotron. Done **without downloading weights**: config.json
plus safetensors headers fetched by HTTP range request against
`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`. Nothing is staged
yet; plan `0b87b76a`'s last open todo (stage + gate Nemotron) is untouched.

## Checkpoint selection

`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` (18,291 downloads) is
the right candidate under Mei's quantization policy — an established
mlx-community 4-bit affine/group-64, matching the recipe already validated for
the other two. Others seen but NOT recommended for the first gate:
`Vontra/...-MLX-6bit` (6-bit is the slowest tier measured — see the bit-depth
sweep), `Sawfwair/...-NVFP4-MLX` (NVFP4, unvalidated in this stack),
`mlx-community/...-8bit` (31.9 GiB expert bank alone, will not fit).

Also on HF and worth remembering for later, not now:
`thoughtworks/...-Eagle3` (speculative-decoding draft head) and NVIDIA's own
`-DFlash` / `-DSpark` variants. vmlx declares a
`DFlash2StagedVerifyRollbackModel` protocol, so that machinery exists in the
stack. **Speculative decoding is unusually attractive here precisely because
decode is dispatch-bound** — verifying K tokens in one forward costs little
more than one token when you are overhead-bound rather than bandwidth-bound.
Flagged as a future thread, not part of the current plan.

## It is NOT the same shape as Ornith/Qwen3.6 — do not transfer constants

| | Ornith 1.5 / Qwen 3.6 | Nemotron 3.5 Lightning |
|---|---|---|
| model_type | `qwen3_5_moe` | `nemotron_h` |
| linear mechanism | Gated DeltaNet (GDN) | **Mamba2** (`ssm_state_size` 128, `conv_kernel` 4, 64 heads x 64 dim, `n_groups` 8) |
| layers | 40, every layer = (GDN or attn) **+** MoE | **52, specialised: 23 mamba2 + 6 attention + 23 MoE** |
| experts / top-k | 256 / 8 | 128 / 6 |
| moe_intermediate | 512 | 1856 |
| routed expert MLP | gate_proj + up_proj + down_proj (GLU) | **fc1 + fc2 only — no gate** |
| hidden / vocab | 2048 / 248320 | 2688 / 131072 |
| total weights (4-bit) | 18.17 GiB | **16.55 GiB** |

Layer counts confirmed from the tensor headers: 23x `mixer.A_log`/`conv1d`/
`dt_bias` (mamba2), 6x `mixer.{q,k,v,o}_proj` (attention), 23x
`mixer.switch_mlp.fc{1,2}` + `mixer.gate` + `mixer.shared_experts` (MoE),
52x `layers.N.norm.weight`.

## Consequences that matter for the optimization programme

**1. F4 and F5 do not apply to Nemotron at all.** Both levers are about fusing
`gate_proj` and `up_proj`. Nemotron's routed experts are `fc1`/`fc2` with no
gate projection (`shared_experts` likewise has only `up_proj`/`down_proj`), i.e.
a plain 2-matmul MLP rather than a GLU. There is nothing to fuse — its MoE block
already issues 2 `gather_qmm` per layer where qwen3_5_moe issues 3. Any Nemotron
speed work has to target the Mamba2 path instead.

**2. It has the best memory headroom of the three.** 16.55 GiB of weights
(1.6 GiB less than Ornith) and only **6 attention layers** carrying a real KV
cache (vs Ornith's 10), so long-context KV growth should be materially cheaper.
On a 32 GB machine that is the most valuable structural difference — it is the
one candidate where a higher-precision or larger-context configuration might
actually fit.

**3. Same 93% concentration.** Routed expert bank is 15.39 of 16.55 GiB (93%),
identical in character to Ornith. So the F6/P5 memory conclusion (any real
memory win must come from the expert bank) transfers even though the compute
levers do not.

**4. vmlx already has Nemotron-specific optimization work.**
`Libraries/MLXLLM/Models/NemotronH.swift` carries a hand-written Metal kernel
`nemotron_h_mamba_depthwise_decode_conv` (with a
`nemotronHMambaConvFastPathDisabled()` env escape hatch) and a
`NemotronHLayerProfiler` that records per-stage timings like `mamba.in_proj`.
**That profiler is a gift** — it gives the per-component decode budget for free,
which is exactly what P0 has to reconstruct by hand for qwen3_5_moe.

**5. No MTP head in this checkpoint.** The config carries
`num_nextn_predict_layers` and `mtp_layers_block_type`, but the header contains
zero `nextn`/`mtp` tensors — the MLX conversion dropped them. No free
self-speculative decoding from this artifact.

## Bandwidth budget (same method as the other two)

Active weight traffic per decode token: **1744 MiB** (vs Ornith's 1592).

| component | MiB/token | share |
|---|---|---|
| moe_routed_bank (top-6 of 128) | 738.7 | 42.4% |
| mamba2 | 479.1 | 27.5% |
| shared_expert | 246.2 | 14.1% |
| lm_head | 189.0 | 10.8% |
| full_attn (6 layers only) | 75.3 | 4.3% |
| router_gate | 15.1 | 0.9% |

Bandwidth ceiling: 164 tok/s @300 GB/s, **191 tok/s @350**, 219 @400.

Note the shared expert is **14.1%** of per-token traffic here versus **4.2%** on
Ornith — Nemotron leans much harder on its always-active shared expert
(`n_shared_experts`, `moe_shared_expert_overlap`, `moe_shared_expert_intermediate_size`
are all in its config). If the "promote the shared expert to 8-bit" idea from
the bit-depth work is worth anything anywhere, it is worth more here — but it
also costs proportionally more memory here, so it is a real trade rather than
the near-free one it is on Ornith.

**Prediction to test, not a result:** if Nemotron lands in the same
dispatch-bound regime, expect roughly Ornith-like tok/s despite the different
mechanism. Its 52 specialised layers issue fewer ops each than qwen3_5_moe's 40
combined layers, so total dispatch count may be similar. This is a prediction
from static analysis only — measure it, do not assume it.

## Suggested first gate

Standard bounded loadability gate per plan `0b87b76a`'s last open todo:
stage the mlx-community 4-bit, then loadability + streaming/non-streaming tool
calls + context cap + memory + short decode. Add two Nemotron-specific steps:
turn on `NemotronHLayerProfiler` during the short-decode leg (free per-component
budget), and record whether the mamba conv fast-path kernel engages.

#proj/mei
