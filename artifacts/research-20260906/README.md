# Hybrid-MoE super-optimization research — 2026-09-06

Static analysis + CPU/GPU-microbenchmark research into `qwen3_5_moe` hybrid MoE
decode on Sulaco (M1 Max, 32 GB). No model was loaded; the shared benchmark
machine was never disturbed. Mei `main` untouched — this is worktree
`research/hybrid-superopt`.

| file | what |
|---|---|
| `findings.md` | F1-F6: architecture, bandwidth budget, the two dead/disabled compile levers, memory structure |
| `moe-microbench-results.md` | measured: F4/F5 stack to +16.5% on the MoE block (+4.6% end-to-end); bit depth is not a speed dial |
| `experiment-plan.md` | P0-P6, prioritized, for when the machine is free |
| `qwen36-vlm-path.md` | Qwen3.6 carries a 0.83 GiB vision tower and may take a different load path than Ornith |

Tools (in `../../tools/`):
- `bandwidth_budget.py <model-dir>` — per-token weight-traffic budget from
  safetensors headers alone, ~1s, no GPU.
- `moe_chain_bench.py` — decode-shaped 40-layer chained MoE; F4/F5 A/B.
- `moe_bitdepth_bench.py` — routed-expert bit-depth speed sweep.
- `moe_dispatch_bench.py` — **falsification record only**; per-layer `mx.eval`
  made it wrong. Kept so the mistake is not repeated. Use `moe_chain_bench.py`.

Everything here is mirrored into Kiem under `proj/mei` so any agent can pick it
up without this worktree.
