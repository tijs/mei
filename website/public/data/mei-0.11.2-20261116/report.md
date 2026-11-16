# Mei 0.11.2 automatic-profile measurements

**16 November 2026 · M1 Max · 32 GB unified memory**

Fresh measurements for the three models recommended on mei.computer. The [historical 0.11.0 chart data](../mei-0.11.0-20261107/results.json) remains available.

[All metric observations and provenance](measurements.json) · [Frozen benchmark source](https://github.com/tijs/local-model-bench/tree/7d70334d368d81ea4376823b0655fa2dd6c4d001)

## Task results

| Model | Assistant | Python | Mean task time | Mean peak wired increase |
|---|---:|---:|---:|---:|
| Groot | 24/24 | 6/6 | 39.62 s | 13.64 GiB |
| Qwen3.6 | 18/24 | 6/6 | 30.17 s | 20.47 GiB |
| Ornith1.5 | 22/24 | 6/6 | 34.26 s | 20.43 GiB |

Every model completed both declared seeds, 17 and 29: 12 synthetic household tasks and 3 isolated Python tasks per seed. All failed attempts remain in the scores and task-time averages. Average time is the mean of all 15 task durations per seed, then the mean of the two seeds. Startup and admission are excluded from task time. These are synthetic workflows, not Peixinho app acceptance.

## Generation speed

| Model | 17 input tokens | 30,000 input tokens |
|---|---:|---:|
| Groot | 59.5 tok/s | 46.9 tok/s |
| Qwen3.6 | 71.4 tok/s | 53.7 tok/s |
| Ornith1.5 | 71.4 tok/s | 53.7 tok/s |

Each endpoint averages 10 measured, full 128-token greedy responses across two fresh processes, giving each process 50% weight. Cold requests and warmups are excluded; prompt-cache hits are zero. Decode excludes prefill and is not whole-request latency or long-context answer quality.

## Serving settings

| Model | Automatic prefill | Conversation anchors | Allocator cache allowance |
|---|---:|---:|---:|
| Groot | 256 | 0 | 512 MiB |
| Qwen3.6 | 1024 | 2 | 512 MiB |
| Ornith1.5 | 1024 | 0 | 128 MiB |

The original benchmark samplers were retained: Groot uses temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1 and frequency penalty 0. Qwen and Ornith use temperature 0.6, top-p 0.95 and top-k 20. Both task seeds were passed explicitly. The decode study overrides temperature to zero.

Scored runs use fresh owned disk caches; decode uses cache off. The context cap is 65,536 tokens, the normal server output budget is 8,192, and the retained-prefix allowance is 512 MiB. Compiled decode is disabled. Model-specific prefill and conversation anchors are selected automatically, with the released architecture compilation policy. These settings differ from the earlier 0.11.0 manual profiles, so changes cannot be attributed solely to runtime code.

## Validation and limitations

All three native admissions passed 13 probes, including the exact 65,536-token capacity boundary, tools and cache checks. Independent saved-grader replay, API/tool-journal comparison and submitted-script replay passed for every scored run. Full model hashes matched after every task/decode run; the retained weight payloads match the historical bundles. The task source, graders and parent lifecycle are verified against frozen benchmark-v9.11.5. The canonical exporter supports newly committed run configurations while retaining these frozen-method checks.

Memory is peak host wired bytes during the full supervised run minus the median pre-run baseline, averaged across the two runs. This is partial system-wide accounting, affected by other processes. It is not total model RAM and must not be added to stored weight size. Two seeds do not establish statistical independence, confidence intervals or universal superiority. These tests do not establish answer quality at 30,000 tokens or maximum-window quality.

Stopped, declined and launcher-failed preparation attempts are excluded. The complete reported cohort ran after simulator shutdown, under the original pressure, swap, disk and capacity guards. No failed task was rerun until it passed.

## Failed tasks (retained)

| Model | Seed | Track | Task | Failed checks |
|---|---:|---|---|---|
| Qwen3.6 | 17 | household | capture-multipart | retry_receipts_exist |
| Qwen3.6 | 17 | household | retrieve-grounded | deleted_source_not_reused |
| Qwen3.6 | 17 | household | completion-intent | no_unsafe_attempt, no_invalid_operation |
| Ornith1.5 | 17 | household | capture-multipart | no_unsafe_attempt, no_invalid_operation |
| Ornith1.5 | 17 | household | routine-occurrence | correct_explicit_schedule |
| Qwen3.6 | 29 | household | capture-ambiguous | routing_choice_requested |
| Qwen3.6 | 29 | household | share-preview | finished_with_final_answer |
| Qwen3.6 | 29 | household | routine-occurrence | correct_explicit_schedule |

## Reproduction

Use the [portable selected configurations](configs/) with the frozen benchmark source linked above. Replace `__MEI_BINARY__`, `__MEI_SOURCE__`, `__MODEL_DIR__`, `__SESSION_DIR__`, `__BENCH_ROOT__` and `__MEASUREMENT_ROOT__` with isolated absolute paths, then commit the resolved configurations before running. These are portable templates; original configuration fingerprints are retained in the metric data.

Use released tag v0.11.2 for the Mei source checkout. Exact binary, Metal and model revisions are in `measurements.json`. The frozen launcher predates `--keep-model-in-memory`; omitting the flag uses the identical released default of true.

```sh
# Native admission
uv run --locked python runner/run_mei_acceptance.py --config CONFIG --output NEW_OUTPUT --deadline 1800

# One scored seed
uv run --locked python runner/supervise_bench.py --config CONFIG --output NEW_OUTPUT --deadline 7200 --trials 1

# One context-speed process
uv run --locked python runner/qualify_mei_release.py --config CONFIG --output NEW_OUTPUT --deadline 1800
```

Never reuse an output directory. Run one local model at a time on a quiet Mac. Keep the original normal-pressure baseline, capacity budgets, two-sample warning and immediate critical-pressure stop, 2 GiB swap-growth / 16 GiB absolute swap limits, disk guard and full workloads. Every actual task/decode run was fully hashed afterward.

Raw transcripts, original path-bearing configurations, runtime logs and raw system telemetry remain in durable experiment storage. Public task and decode metric rows and their SHA-256 bindings are included here without unrelated host/process data.

## Runtime identity

- Mei source: `66d9499163a8cfc76772a2f55db16bd8142551fb`
- Binary SHA-256: `584e4deb1149f8ef0c9fbdda74640ebc69536e5db7a3c99b6ed35c72375cabc0`
- vmlx: `3aa7b086b22c67f5e36db29128f9f251b7227c63`
- Metal SHA-256: `56289ad54445d717bce56d7b964927dc092b5face4428db7881ab8f1ad5269cc`
