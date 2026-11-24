# Mei 0.18.0

Start Inkling 2 Mini vocabulary Q4 with `mei --model inkling-2-mini-mei-q4`. On an M1 Max with at least 32 GiB and a context cap up to 26,624, Mei selects the fused attention mask automatically. In a 2,048-token counting test on the 32 GiB M1 Max, the mask improved median generation speed from 20.43 to 22.68 tokens/s. Time to receive all 128 output tokens fell from 13.96 to 13.21 seconds. Both pairs of runs produced the same answers.

The same profile converts the dense attention projections to eight-bit weights when it loads the model. On top of the mask this raised 2,048-token decode from 22.66 to 25.01 tokens/s in two process pairs. It reduced active model memory by 555 MiB. Quantization changes token probabilities. Set `VMLX_INKLING_ATTENTION_BITS=0` when starting Mei to keep the checkpoint precision.

For repeated chat or tool requests, use `mei --model inkling-2-mini-mei-q4 --cache-mode memory`. Mei keeps one reusable 1,024-token prefix in memory. On repeated prompts with a matching prefix, chat prefill fell from 12.02–12.04 seconds to 6.30 seconds. The response reports the number of reused tokens in `usage.prompt_tokens_details.cached_tokens`.

[Inkling model guide](https://github.com/tijs/mei/blob/v0.18.0/docs/INKLING.md)
