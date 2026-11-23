# Mei 0.17.0

Inkling 2 Mini vocabulary Q4 reads long prompts faster on the M1 Max. With a 2,048-token prompt, median prefill time fell from **8.91 to 7.51 seconds** across ten measurements per setting. All measured greedy completions matched. With the automatic profile and the default 26,624-token context cap, the same prompt took **7.66 seconds** across five measurements on a 32 GB M1 Max.

In a separate long-context check, Mei retrieved a value from the beginning of a 26,368-token prompt in 134 seconds of prefill. It then accepted 26,623 input tokens and generated one token at the exact context limit. Peak process footprint was 23.21 GiB.

Start the model with `mei --model inkling-2-mini-mei-q4`. Raw `/v1/completions` requests at temperature zero automatically use a 256-token prefill step on an M1 Max with at least 32 GiB and a context cap up to 26,624. Chat and sampled requests use the usual 128-token default. To select 128 for every request, pass `--prefill-step-size 128` at startup.

[Inkling model guide](https://github.com/tijs/mei/blob/v0.17.0/docs/INKLING.md)
