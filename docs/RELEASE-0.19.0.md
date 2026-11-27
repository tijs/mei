# Mei 0.19.0

Start Mei with a model name to use automatic context sizing. For chat and raw completions, the formatted prompt and generated reply share one token budget. Cached prompt tokens count once. Mei reserves enough memory for the admitted reply before generation starts.

Use `--context-cap 32768` for a fixed ceiling, or `--context-cap auto` to return to automatic sizing. A numeric ceiling still observes the model's positional limit and machine memory. The status endpoint `/v1/mei/status` exposes the selected mode, the server ceiling and an estimate of the tokens available to a new request.

Disk prefix caches use the configured context ceiling as part of their identity. After upgrading, or after changing that ceiling, the first request builds a fresh prefix entry.

If macOS memory telemetry is unavailable, Mei limits a new request to at most 8,192 total tokens (4,096 for SenseNova), subject to its configured ceiling.

If the prompt fills the entire window, shorten it to leave room for the reply. If memory admission returns HTTP 503, retry after the active request finishes or reduce the requested output length.

## Context ceilings on the 32 GiB M1 Max

| Model and cache mode | Auto ceiling, prompt plus reply |
|---|---:|
| Groot 19B text | 65,536 |
| Inkling 2 Mini, original and vocabulary Q4, default cache mode | 32,768 |
| Qwen3.6 35B text-only | 32,768 |
| Qwen3.6 35B with vision, tested on text requests | 26,624 |
| Ornith 1.5 35B | 26,624 |
| SenseNova Loop Q6 and Q8 | 26,624 |
| Inkling vocabulary Q4 with in-memory prefix snapshots | 26,624 |

The SenseNova fixtures used six- and eight-bit weight exports of `sensenova/SenseNova-Loop-9B` at revision `329a65f701ba8c48b73c30d0cbaa6f0449733673`, with float32 activations and a 64-token prefill step. Both restored 154 cached prompt tokens on the first request after a server restart and returned the expected answer. Q8 also generated 16,384 tokens after a 10,240-token prompt in 40m45s. After reusing 1,024 prompt tokens from disk, a fresh server reproduced the full reply in 40m43s. Both runs peaked at 15.45 GiB with no swap growth.

During calibration, a 32,736-token input and a 32-token reply on the vision-capable Qwen and Ornith bundles reached 23.99 and 23.89 GiB of process memory respectively. Their automatic ceiling leaves more room for macOS and other work. A 200,000-token fixed ceiling on SenseNova Q6 exercised the memory rejection path and immediately served a subsequent two-token request.

On the 32 GiB M1 Max, the custom Inkling Q4 bundle consumed 16,384 input tokens and generated 16,384 output tokens in one raw completion. Prefill took 75.03 seconds. Decode took 717.69 seconds (22.83 tokens/s) and stopped at the requested length. The sampled process footprint peaked at 21.9 GiB during prefill and 20.3 GiB during decode, with normal pressure and no swap growth. The original Inkling bundle completed the same request with a 96.25-second prefill and 16.63 tokens/s during decode. Q4 generated tokens 37% faster on this workload.

Ornith generated 16,384 tokens from a 10,240-token prompt both cold and after restoring 1,024 prompt tokens. Its replies matched byte for byte. Qwen with vision on text input reached the same cold and restored totals on a repetitive prompt. On a natural continuation it generated 16,384 tokens cold and stopped at 1,582 after restore. Mei 0.18.0 returned the same two outputs byte for byte.

Groot generated identical 16,384-token replies from 49,152 input tokens cold and after a fresh disk restore. Both runs peaked at 16.27 GiB. Qwen text did the same at 16,384 input plus 16,384 output, peaking at 22.41 GiB cold and 22.38 GiB after restore. SenseNova Q6 matched its full 16,384-token reply at 10,240 input cold and restored, peaking at 13.37 and 13.18 GiB respectively.

With in-memory prefix snapshots enabled, Inkling Q4 completed 4,096 input plus 22,528 output tokens twice in the same process. The second request reused 1,024 prompt tokens and reproduced the full reply. Peak footprint was 21.05 GiB cold and 21.08 GiB on reuse.

The 57,344-token early-marker result from the guided Inkling Q4 experiment used different runtime settings. The stock Inkling ceilings above apply to ordinary Mei launches.

## Regression checks

Across the eight fixtures, all 24 greedy comparisons matched two independent Mei 0.18.0 controls. Each fixture received a short raw prompt, a roughly 2,048-token raw prompt, and a chat request. On the longer raw prompt with 128 output tokens, decode duration differed by at most 0.29% and peak MLX allocation matched the controls. Total wall time ranged from 3.42% shorter to 0.82% longer than the control mean.

A short Groot raw request initially took 10% longer end to end. Two alternating candidate/control pairs returned identical outputs and reduced the mean timing difference to 1.3% (2.88 versus 2.84 seconds).
