# Mei 0.19.0

Start Mei with a model name to use automatic context sizing. For chat and raw completions, the formatted prompt and generated reply share one token budget. Cached prompt tokens count once. Mei reserves enough memory for the admitted reply before generation starts.

Use `--context-cap 32768` for a fixed ceiling, or `--context-cap auto` to return to automatic sizing. A numeric ceiling still observes the model's positional limit and machine memory. The status endpoint `/v1/mei/status` exposes the selected mode, the server ceiling and an estimate of the tokens available to a new request.

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

During calibration, a 32,736-token input and a 32-token reply on the vision-capable Qwen and Ornith bundles reached 23.99 and 23.89 GiB of process memory respectively. Their automatic ceiling leaves more room for macOS and other work. A 200,000-token fixed ceiling on SenseNova Q6 exercised the memory rejection path and immediately served a subsequent two-token request.

The 57,344-token early-marker result from the guided Inkling Q4 experiment used different runtime settings. The stock Inkling ceilings above apply to ordinary Mei launches.
