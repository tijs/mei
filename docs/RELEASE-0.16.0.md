# Mei 0.16.0

Mei adds a lower-memory Inkling 2 Mini profile for Apple Silicon. The original remains available as `inkling-2-mini`.

```sh
mei pull inkling-2-mini-mei-q4
mei --model inkling-2-mini-mei-q4
```

On the 32 GiB M1 Max the new profile uses **26,624 total tokens** for prompt, history and output. To choose a larger window on another machine, run `mei --model inkling-2-mini-mei-q4 --context-cap 65536`.

The custom weights use 4-bit affine input embeddings and output projections, with the publisher’s mixed NVFP4/MXFP8 routed experts. The four shards contain 1.24 GiB less tensor data than the original. The Q4 profile keeps only the trained 1,024-token window for local attention and retains the full global history.

Across ten measurements per prompt and bundle, median short decode rose from 23.85 to 24.52 tokens/s. At 2,048 input tokens it rose from 20.21 to 20.44 tokens/s. Peak process footprint ranged from 20.19 to 20.21 GiB for Q4 and 21.48 to 21.95 GiB for the original weights.

**Quality trade-off.** Q4 scored 20/25 on the fixed task suite. The original release scored 22/25. Q4 additionally missed tool-error recovery and retrying after an uncertain outage. Both passed two held-out coding exercises with real file edits and executable tests.

**Context measurements.** Uncached near-full retrieval took 173 seconds at 26,624 tokens, 189 seconds at 28,672, and 222 seconds at 32,768. The 65,536-token attempt crossed a 24 GiB process budget reserved to leave room for the rest of the Mac. A 200,000-token window has not been qualified here.

[Model weights](https://huggingface.co/Tostibrown/Inkling-2-Mini-Mei-Vocab-Q4) · [Inkling guide](https://github.com/tijs/mei/blob/v0.16.0/docs/INKLING.md)
