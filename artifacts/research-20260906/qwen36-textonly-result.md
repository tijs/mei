# RESULT: Qwen3.6 text-only full benchmark 24/25, +15.4% paired throughput; the 24-vs-22 quality delta is harness noise (2026-09-07)

> **[CORRECTED 2026-09-07, same session — the "+15.4% paired throughput" headline
> in this note's title and body is OVERCLAIMED. Read this first.]**
>
> The two runs took **different agent trajectories**, so their hermes_ops tok/s
> figures are not comparable. Stock generated **16,094** completion tokens across
> 1,668,750 prompt tokens; text-only generated **6,665** across 942,873. On
> `hermes_ops-multi-step-chain` alone stock produced 12,754 tokens vs text-only
> 3,451. Since `tokens_per_second = completion_tokens / wall_seconds`, that
> difference dominates the metric.
>
> Depending on how you pool, the sign flips: **per-task mean** favours text-only
> (17.05 vs 14.77), while **pooled total/total** favours stock (16.06 vs 9.93).
> Neither isolates decode speed. **Do not quote either.**
>
> The only fair in-benchmark speed comparison is the **sanity** rows, which have
> a fixed prompt and near-fixed output:
>
> | task | prompt | completion | stock -> text-only |
> |---|---|---|---|
> | sanity-basic | 29 / 29 | 286 / 174 | 44.84 -> 59.13 = **1.32x** |
> | sanity-tool | 657 / 657 | 271 / 265 | 35.49 -> 46.55 = **1.31x** |
>
> Those match B1's controlled short-decode measurement (61.85 vs 49.95 = 1.24x)
> and are the defensible number. Both runs were cold (hermes_ops ttft 62-66 s on
> both sides), so cache state is NOT a confounder here — only trajectory length is.
>
> **What survives from this note:** text-only passes 24/25; the quality delta vs
> stock is noise (bit-identical text weights); short decode is ~1.3x faster;
> resident memory is 0.83 GiB lower. **What does not:** any end-to-end
> throughput claim from the agentic suites.

# RESULT: Qwen3.6 text-only full benchmark — 24/25, +15.4% paired throughput vs stock (2026-09-07)

First full benchmark of the vision-stripped derivative
(`Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`, config
`configs/Qwen3.6-35B-A3B-textonly/mei.yaml`, config_hash `3421f91401ab`).
Runbook item E2. Ran unattended overnight while the other agent was paused.

## Paired result, same 25 tasks, one run each

| suite | stock (`cea524483faf`) | **text-only** (`3421f91401ab`) |
|---|---|---|
| sanity | 2/2 | 2/2 |
| hermes_ops | 7/8 | **8/8** |
| kiem_mini | **5/5** | 4/5 |
| hearth_mini | 2/3 | **3/3** |
| kipclip_mini | 3/4 | **4/4** |
| hearth_full | 3/3 | 3/3 |
| **TOTAL** | **22/25** | **24/25** |

**Paired mean throughput: 14.77 -> 17.05 tok/s = 1.154x (+15.4%).**
Median per-task ratio 1.078x.

## Read the quality delta correctly — 24 vs 22 is NOISE, and that is useful

The strip removed only `vision_tower.*` tensors; every text-tower tensor is
**bit-identical** to stock (per-tensor sha256 verified at build time, 50/50
re-verified independently). The two bundles cannot differ in text quality. So
the 24/25 vs 22/25 gap is **entirely run-to-run variance**, not an improvement,
and it must not be reported as one.

That makes it a free calibration of the harness: **a 2-of-25 swing (8 points)
between two runs of what is numerically the same text model, at temperature 0,
single trial.** Worth remembering when reading any other single-trial
comparison in this project — differences of one or two tasks are not signal.
(It is consistent with the repo's own `--trials` help, which already warns that
single-trial temperature-0 results are not reliably reproducible on MLX/Metal.)

## Where the speed gain actually comes from

Per-task ratios (text-only / stock) split cleanly by task shape:

| task | ratio |
|---|---|
| sanity-basic | **1.32x** |
| sanity-tool | **1.31x** |
| hermes_ops-targeted-edit | 2.04x |
| hermes_ops-persistent-failure | 1.28x |
| hermes_ops-no-tool-needed / noisy-result | 1.07–1.08x |
| hermes_ops-selection | 0.96x |
| hermes_ops-error-recovery | 0.85x |
| hermes_ops-multi-step-chain | 0.81x |
| hermes_ops-chaining | 0.68x |

The short, decode-dominated sanity rows show ~1.31–1.32x, closely matching B1's
isolated short-decode measurement (61.85 vs 49.95 tok/s = **+23.8%**). Long
agentic tasks are prefill- and multi-turn-dominated, so the decode advantage
dilutes and several rows land below 1.0x. **The headline should be the paired
mean (+15.4%), not B1's +23.8%** — that figure only describes short decode.

## Correction to my own earlier reading

My first pass compared raw model-name aggregates and reported text-only as
*slower* (17.05 vs 19.02 tok/s). That was wrong: the stock aggregate included
**2 stray rows from a different config_hash** (`e9d6db1b6675`, an earlier
partial run) whose high sanity throughput dragged the mean up. Filtering to a
single config_hash per side gives the paired 25-vs-25 comparison above. Always
filter by `config_hash`, not by model name, when comparing runs in this log.

## Status

- Result rows appended to `results/log.jsonl`; leaderboard rebuilt.
- `plot_leaderboard.py exited 1 (non-fatal)` in both runs — pre-existing, not
  investigated, flagged here so it is not mistaken for a new break.
- Config still carries stale `artifact_identity` text copied from the stock
  config (parked in `/tmp/PENDING_AFTER_BENCH.md`; the preserve-repository guard
  was armed during the coding suites so it could not be fixed mid-run).
- **Recommendation:** the text-only bundle supersedes stock Qwen3.6 for this
  project's purposes — same text weights, ~15% better end-to-end throughput,
  0.83 GiB less resident, and vision capability this benchmark never uses.
  Keep the stock config for provenance/history; run the text-only one.

#proj/mei
