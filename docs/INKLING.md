# Run Inkling 2 Mini with Mei

## Mei vocabulary Q4

On a 32 GB M1 Max, Mei's custom Inkling bundle uses a 26,624-token total
context by default. Input, conversation history, tool results and generated
output share that budget. Install Mei 0.16.0 and the Hugging Face `hf` CLI,
then run:

```sh
mei pull inkling-2-mini-mei-q4
mei --model inkling-2-mini-mei-q4
```

Use `inkling-2-mini-mei-q4` as the model ID in API requests. The four shards
contain 18.19 GB of tensor data. The embedding and output projection use
4-bit affine weights in groups of 64. The routed experts retain their original
NVFP4 and MXFP8 weights and scales. Mei keeps 1,024 tokens in the local
attention caches and full history in the global attention caches.

The custom profile uses the measured default on machines with at least 32 GiB
of physical memory. Below that it uses 8,192 tokens. To set a different limit
on a machine with more headroom, pass it explicitly, for example:

```sh
mei --model inkling-2-mini-mei-q4 --context-cap 65536
```

The maximum configurable value is the model's 1,048,576-token architectural
limit. Available memory and prefill latency determine the practical limit.
Set the client's context window to the server's configured value.

### Performance and quality on the M1 Max

Across two fresh processes per bundle, peak physical footprint was 20.19–20.21 GiB with Q4 and 21.48–21.95 GiB with the original weights. Median decode across ten measurements per prompt rose from 23.85 to 24.52 tokens/s for short input and from 20.21 to 20.44 tokens/s for 2,048 input tokens.

The optional Q4 profile scored **20/25** on the task suite. The original historical release scored **22/25**. Q4 passed 7/10 model-only cases, 10/12 household cases and all three Python automation cases. The additional misses relative to that release were tool-error recovery and handling an uncertain outcome after an outage. Both bundles passed the two separate coding exercises with executable tests. A fresh run of the original weights on the experimental bounded-cache backend scored 21/25.

The 26,624-token window allows room for both the prompt and response. With 256 tokens reserved for output, retrieval from the beginning took 173 seconds of uncached prefill. At 28,672 tokens it took 189 seconds, exceeding the three-minute target. The 32,768-token boundary also worked, with near-full recall taking 222 seconds. A 65,536-token test stopped at the experiment’s 24 GiB process budget. A 200,000-token window remains unverified on this Mac.

## Original mixed-precision bundle

Install Mei 0.15.0 or later and the Hugging Face `hf` CLI on an Apple Silicon
Mac. The tested NVFP4 bundle downloads about 19.5 GB. Qualification ran on a
32 GB M1 Max with one model loaded.

```sh
mei pull inkling-2-mini
mei --model inkling-2-mini
```

Mei downloads the pinned revision, skips the unused speculative decoding
weights, and applies the Inkling text profile automatically. The served model
ID is `inkling-2-mini`. If you already have a compatible local bundle, select it
with `mei --model-dir /path/to/bundle`.

Point your OpenAI-compatible client at `http://127.0.0.1:8024/v1`, or try:

```bash
curl http://127.0.0.1:8024/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Write a Python function that groups strings by their first letter."}]}'
```

The automatic profile uses a total context limit of 8,192 tokens, a 4,096-token
default output cap, a 128-token prefill step, and one active generation. Mei
uses the bundle's sampling defaults and selects the qualified native kernels
and bounded Metal command buffers. Text chat, streaming, and native tool calls
are supported. Prefix reuse is off. Image input, speculative decoding, and
grammar-constrained response formats are outside this profile.

The [tested model bundle](https://huggingface.co/nativ-community/Inkling-2-Mini-nvfp4/tree/26f71eef259a1dd7fc476778d67020351e0dfb8c)
is a conversion of [Thinking Machines Lab's Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/).
