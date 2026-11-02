# Historical model results

The tables below preserve the numeric evidence previously shown in the README
(source snapshot `a20f353`). They use older profile/coding fixtures and do not
rank the current household/Python contenders or qualify Mei 0.10.
The current comparison is maintained in
[local-model-bench](https://github.com/tijs/local-model-bench/blob/main/results/AGENT_SUMMARY.md).

Pick by what you care about, then pass the name:

**Take `qwen3.6-35b-a3b-text` unless you need image input.** It is equal or
better than the alternatives on every measurement we have.

| model | `--model-profile` | images | decode | reply latency | turns per task | coding |
|---|---|---|---|---|---|---|
| **Qwen3.6 35B-A3B text-only** | `qwen3.6-35b-a3b-text` | no | **58 tok/s** | **1.05 s** | **8.9** | **100%** |
| Qwen3.6 35B-A3B vision | `qwen3.6-35b-a3b` | **yes** | 48 tok/s | 1.07 s | 9.5 | 87% |
| Ornith 1.5 35B-A3B | `ornith-1.5-35b-a3b` | no | 57 tok/s | 1.28 s | 11.1 | 93% |

"Reply latency" is time-to-first-token on an ongoing conversation, which is what
you feel while using an agent. "Turns per task" is how many agent round-trips a
coding task took — fewer is faster end to end, and it compounds with latency.

**Why not Ornith?** It was this project's default for a long time and is the most
heavily exercised model here, but it no longer wins on anything a user
experiences: same decode, higher latency, ~25% more turns per task, lower coding
pass rate, and it is the one model where cross-conversation prefix reuse costs
quality, so it ships with that turned off. Keep it if you specifically want a
second model family rather than two builds of one, or if you are reproducing
older results. Otherwise it is strictly the worse choice.

**The vision build costs about 18% of decode speed even on pure text** — 48
against 58 tok/s, same architecture and quantisation. Take it only if you
actually feed it images.

**Memory does not distinguish them.** All three are 4-bit MoE checkpoints, ~19 GB
on disk, peaking near **24 GB** in use. All three want a 32 GB machine and none
fits a 16 GB one.

**All three take about 50 s on the very first turn**, reading the ~20k-token
system+tools prompt for the first time; after that a turn starts in about a
second. Only the text-only build also keeps that prefix across *separate*
conversations, because it is the only model where cross-conversation anchors
cost nothing on our suite.

Figures from two cold runs of each shipped configuration on a 32 GB M1 at a
65,536-token context, plus the coding pass rate from the benchmark's composite
leaderboard.

### Mei 0.6.1 benchmark note

A separate comparison: the full-agent `local-model-bench` suite (all 25 rows)
on the 32 GB M1 Max, not the short-decode profile figures above. Mei 0.6.1 ran
the complete suite for both retained target models with zero harness errors,
against selected Mei 0.5.0 anchors. The anchors' model revisions, engine pins,
and historical generation settings differ, so this is evidence rather than a
clean isolated release benchmark.

| model | rows passed (0.5.0 → 0.6.1) | avg TTFT | avg decode | weighted completion-over-wall |
|---|---|---|---|---|
| Ornith 1.5 35B-A3B | 22/25 (88%) → 24/25 (96%) | 53.984 → 42.052 s | 10.6813 → 10.5375 tok/s | 13.4696 → 16.6776 tok/s |
| Qwen3.6 35B-A3B text-only | 24/25 and 22/25 (mean 23/25, 92%) → 24/25 (96%) | 53.561 → 42.448 s | 9.0844 → 8.935 tok/s | 11.2836 → 9.9917 tok/s |

The Qwen3.6 0.5.0 side spans two retained anchors — 24/25 and 22/25 — with the
averaged metrics above as their mean. Both models improved in quality and
TTFT; Ornith also improved weighted completion-over-wall, while Qwen3.6 did
not — Mei 0.6.1 is not uniformly faster than 0.5.0. Raw comparison artifacts:
[Ornith](https://github.com/tijs/local-model-bench/blob/main/results/comparisons/ornith-mei061-vs-mei060-mei050.json),
[Qwen3.6 text-only](https://github.com/tijs/local-model-bench/blob/main/results/comparisons/qwen36-textonly-mei061-vs-mei060-mei050.json).

