# Mei 0.6.0 F1/C2 opt-out A/B — same-binary comparison

- **Artifact root:** `/private/tmp/mei-c2-optout-0.6.1/artifacts/f1-c2-release-compare-20260930T095614Z`
- **Generated:** 2026-09-30T12:39:08Z by `tools/f1_c2_optout_report.py` (stdlib only)
- **Scope:** 6 legs = 3 models x 2 arms (candidate = both F1/C2 switches unset, the 0.6.0 default compiled-ON; optout = both switches `"0"`); per leg 10 repeats x (short: 13 prompt tokens; 30k: 30000 prompt tokens); 20 timing rows and 25 request-log rows per leg.
- **Binary:** `/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei` — SHA256 `3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc` on all six legs.

## Validation

**6/6 legs valid** — 120 timing rows, 150 request-log rows total; probe_returncode=0 and server_stopped=true on every leg; greedy outputs match within every pair.

| Leg | Model | Arm | Timing rows | Log rows | probe rc | stopped | compiled_routed_switch_glu |
|---|---|---|---|---|---|---|---|
| `ornith-060/candidate` | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | candidate | 20/20 | 25/25 | 0 | true | **active** |
| `ornith-060-optout/optout` | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | optout | 20/20 | 25/25 | 0 | true | absent |
| `qwen36-text-060-thinking-off/candidate` | `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | candidate | 20/20 | 25/25 | 0 | true | **active** |
| `qwen36-text-060-thinking-off-optout/optout` | `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | optout | 20/20 | 25/25 | 0 | true | absent |
| `qwen36-vision-060-thinking-off/candidate` | `mlx-community/Qwen3.6-35B-A3B-4bit` | candidate | 20/20 | 25/25 | 0 | true | absent |
| `qwen36-vision-060-thinking-off-optout/optout` | `mlx-community/Qwen3.6-35B-A3B-4bit` | optout | 20/20 | 25/25 | 0 | true | absent |

## Arm definition

- **candidate:** `operator_overrides = {}` — both F1/C2 switches unset; on the 0.6.0 pin the compiled routed-MoE decode region is active by default.
- **optout:** `operator_overrides = {"VMLX_QWEN35_COMPILE_DECODE_REGIONS": "0", "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE": "0"}` — both F1/C2 switches explicitly `"0"`.

## Policy evidence (server.log)

| Leg | compiled_routed_switch_glu | Evidence |
|---|---|---|
| `ornith-060/candidate` | **active** | L17: `[Qwen4Exp] compiled_routed_switch_glu=active stock_gather_qmm=true shared_weight_inputs=true dtype=bfloat16` |
| `ornith-060-optout/optout` | absent | no compiled_routed_switch_glu policy line and no other [Qwen4Exp] lines; [Qwen4Exp] lines: 0 |
| `qwen36-text-060-thinking-off/candidate` | **active** | L17: `[Qwen4Exp] compiled_routed_switch_glu=active stock_gather_qmm=true shared_weight_inputs=true dtype=bfloat16` |
| `qwen36-text-060-thinking-off-optout/optout` | absent | no compiled_routed_switch_glu policy line and no other [Qwen4Exp] lines; [Qwen4Exp] lines: 0 |
| `qwen36-vision-060-thinking-off/candidate` | absent | no compiled_routed_switch_glu policy line (other [Qwen4Exp] lines present: 7, tolerated); [Qwen4Exp] lines: 7 |
| `qwen36-vision-060-thinking-off-optout/optout` | absent | no compiled_routed_switch_glu policy line (other [Qwen4Exp] lines present: 5, tolerated); [Qwen4Exp] lines: 5 |

## Greedy output match (per pair)

| Model | Context | candidate text (sha256) | optout text (sha256) | per-repeat match |
|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | short | '' (`e3b0c44298fc1c14`) | '' (`e3b0c44298fc1c14`) | yes |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 30k | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | yes |
| `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | short | '' (`e3b0c44298fc1c14`) | '' (`e3b0c44298fc1c14`) | yes |
| `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | 30k | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | yes |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | short | '' (`e3b0c44298fc1c14`) | '' (`e3b0c44298fc1c14`) | yes |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | 30k | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello ' ... (191 chars) (`4e0b7037f631632d`) | yes |

## Means and opt-out deltas

Means from request-log completion rows (server-reported); 10 rows per context per leg. `tail = wall - prefill - generate` (legacy; finalize_ms absent). Delta% = (optout - candidate) / candidate x 100. Greedy outputs matched within every pair, so the timing deltas are like-for-like.

### `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`

| Metric | short candidate | short optout | short delta% | 30k candidate | 30k optout | 30k delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 700.78 | 715.07 | +2.04% | 58,014.04 | 58,388.31 | +0.65% |
| prefill (ms) | 98.68 | 101.63 | +2.99% | 57,151.36 | 57,516.69 | +0.64% |
| generate (ms) | 570.55 | 581.26 | +1.88% | 692.60 | 705.01 | +1.79% |
| tail = wall - prefill - generate (ms) | 31.56 | 32.17 | +1.93% | 170.08 | 166.60 | -2.05% |
| decode (tok/s) | 56.092 | 55.063 | -1.83% | 46.203 | 45.391 | -1.76% |
| prompt (tok/s) | 131.799 | 128.006 | -2.88% | 524.922 | 521.589 | -0.63% |

| Metric | pooled candidate | pooled optout | pooled delta% |
|---|---|---|---|
| wall (ms) | 29,357.41 | 29,551.69 | +0.66% |
| prefill (ms) | 28,625.02 | 28,809.16 | +0.64% |
| generate (ms) | 631.57 | 643.14 | +1.83% |
| tail = wall - prefill - generate (ms) | 100.82 | 99.39 | -1.42% |
| decode (tok/s) | 51.147 | 50.227 | -1.80% |
| prompt (tok/s) | 328.361 | 324.798 | -1.09% |

### `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`

| Metric | short candidate | short optout | short delta% | 30k candidate | 30k optout | 30k delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 686.75 | 696.30 | +1.39% | 58,413.22 | 58,799.73 | +0.66% |
| prefill (ms) | 99.37 | 100.50 | +1.14% | 57,534.21 | 57,912.78 | +0.66% |
| generate (ms) | 554.26 | 562.19 | +1.43% | 710.27 | 714.52 | +0.60% |
| tail = wall - prefill - generate (ms) | 33.12 | 33.60 | +1.45% | 168.75 | 172.43 | +2.18% |
| decode (tok/s) | 57.739 | 56.924 | -1.41% | 45.055 | 44.785 | -0.60% |
| prompt (tok/s) | 130.946 | 129.576 | -1.05% | 521.429 | 518.021 | -0.65% |

| Metric | pooled candidate | pooled optout | pooled delta% |
|---|---|---|---|
| wall (ms) | 29,549.99 | 29,748.02 | +0.67% |
| prefill (ms) | 28,816.79 | 29,006.64 | +0.66% |
| generate (ms) | 632.26 | 638.36 | +0.96% |
| tail = wall - prefill - generate (ms) | 100.93 | 103.02 | +2.07% |
| decode (tok/s) | 51.397 | 50.855 | -1.05% |
| prompt (tok/s) | 326.188 | 323.798 | -0.73% |

### `mlx-community/Qwen3.6-35B-A3B-4bit`

| Metric | short candidate | short optout | short delta% | 30k candidate | 30k optout | 30k delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 2,352.57 | 2,340.65 | -0.51% | 83,880.99 | 83,434.92 | -0.53% |
| prefill (ms) | 290.54 | 296.17 | +1.94% | 81,330.26 | 80,709.64 | -0.76% |
| generate (ms) | 2,054.63 | 2,037.07 | -0.85% | 2,501.90 | 2,673.89 | +6.87% |
| tail = wall - prefill - generate (ms) | 7.39 | 7.41 | +0.27% | 48.84 | 51.39 | +5.22% |
| decode (tok/s) | 15.575 | 15.709 | +0.86% | 12.790 | 11.968 | -6.43% |
| prompt (tok/s) | 44.750 | 43.901 | -1.90% | 369.771 | 371.805 | +0.55% |

| Metric | pooled candidate | pooled optout | pooled delta% |
|---|---|---|---|
| wall (ms) | 43,116.78 | 42,887.79 | -0.53% |
| prefill (ms) | 40,810.40 | 40,502.91 | -0.75% |
| generate (ms) | 2,278.26 | 2,355.48 | +3.39% |
| tail = wall - prefill - generate (ms) | 28.11 | 29.40 | +4.59% |
| decode (tok/s) | 14.183 | 13.839 | -2.43% |
| prompt (tok/s) | 207.261 | 207.853 | +0.29% |

## Caveats

1. **finalize_ms absent in historical 0.6.0 logs** — No leg records finalize_ms (checked per leg across request.jsonl, result.json, probe.json and server.log; presence recorded, no value synthesized). The residual is therefore the legacy tail = wall - prefill - generate, which may absorb finalize/overhead; tail deltas must not be read as engine effects.
2. **single sequential A/B blocks: descriptive, not causal** — Each model has exactly one candidate block and one optout block, run as separate sequential server processes on one host, with 10 repeats per context per leg and no interleaving, randomization or significance testing. All candidate blocks ran before all optout blocks (candidate start epochs 1790762985-1790767673, optout 1790769043-1790770339). Host/time-varying effects cannot be separated from the F1/C2 switch effect; small deltas (order of a few percent) should not be read as effects.
3. **vision legs carry no compiled_routed_switch_glu policy line** — Both qwen36-vision legs have no compiled_routed_switch_glu line in server.log (other [Qwen4Exp] lines are present and tolerated: 7 candidate, 5 optout). The vision A/B block therefore has no direct compiled-routed-MoE signal in either arm; its deltas are recorded for completeness only.
