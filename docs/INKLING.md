# Run Inkling 2 Mini with Mei

## Mei vocabulary Q4

Mei sizes context automatically when you launch the Inkling bundle by name. Input, conversation history, tool results and generated
output share that budget. Install Mei 0.19.0 or later and the Hugging Face `hf` CLI,
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

Use the status endpoint to see the configured ceiling and current headroom. To set a different limit
on a machine with more headroom, pass it explicitly, for example:

```sh
mei --model inkling-2-mini-mei-q4 --context-cap 65536
```

The maximum configurable value is the model's 1,048,576-token architectural
limit. Available memory and prefill latency determine the practical limit.
Set the client's context window to the server's configured value.

On the 32 GiB M1 Max, a vocabulary-Q4 server with prefix reuse off crossed
24 GiB of process memory when a 26,368-token request followed two 2,048-token
requests. With `--cache-limit-bytes 134217728`, Mei 0.17.0 completed the same
requests and the 26,623-input plus one-output boundary at 23.89 GiB peak.

The original publisher bundle also uses the fused attention mask on the qualified M1 Max. At its named 26,624-token context cap, 2,048-token generation improved 8.72%. Both settings scored 22/25. Both maximum-context answers matched. Use `VMLX_INKLING_FUSED_MASK=0` to run with the original mask implementation.

### Reuse a repeated prompt prefix in memory

On an M1 Max with at least 32 GiB of memory and a context cap up to 26,624,
start the vocabulary-Q4 model with the memory cache to speed up repeated chat
turns or raw completions.

```sh
mei --model inkling-2-mini-mei-q4 --cache-mode memory
```

Mei can retain the state after the first 1,024 input tokens. A subsequent
request can reuse it when its first 1,024 tokens and effective prefill step
match. Eligible prompts have 1,025 through 4,096 tokens and use a prefill step
of 128 or 256. The common prefix includes any chat template, system message,
and tool definitions. Edits later in the prompt can still reuse that prefix.

The server keeps one entry in RAM. A different eligible prefix replaces it,
and an ineligible prompt clears it. The measured entry used about 200 MiB.
In this mode, Mei defaults to 128 MiB of reusable allocator buffers. An explicit
`--cache-limit-bytes` takes precedence.
If the entry exceeds `--prefix-cache-limit-bytes`, Mei processes the whole
prompt. Prefix reuse is off by default. To see what a request reused, read
`usage.prompt_tokens_details.cached_tokens` in the response.

### Performance and quality on the M1 Max

On an M1 Max with at least 32 GiB and a context cap up to 26,624, the
vocabulary-Q4 profile converts the query, key, value, output and gate
projection weights of each attention layer to eight-bit affine groups at
model load. The model bundle on disk retains its original weights. To use
the original attention precision, launch with
`VMLX_INKLING_ATTENTION_BITS=0 mei --model inkling-2-mini-mei-q4`.

Greedy raw Q4 completions on the qualified M1 Max profile use indexed KV writes when the prompt contains at least 7,936 tokens and the available output budget is at least 2,048 tokens. On the 7,936-input, 2,048-output test, the full response fell from 114.18 to 108.03 seconds. Both process pairs returned identical choices. Set `VMLX_INKLING_INDEXED_KV=0` to disable this selection.

For greedy raw completions at the automatic 256-token prefill step, the same Q4 profile shares each depth state’s normalization across its learned queries. With the eight-bit projections already enabled, this increased median generation from 25.00 to 27.12 tokens/s for a 2,048-token prompt. Set `VMLX_INKLING_DEPTH_MIX=0` to use the original reduction order. Both the reduction order and the weight precision can change a generated continuation.

With the Q4 bundle on an M1 Max with at least 32 GiB of memory and a context cap of 26,624 or less, raw `/v1/completions` requests at temperature zero use a 256-token prefill step. For a 2,048-token prompt, median prefill time fell from 8.91 to 7.51 seconds over ten measurements per setting. All measured greedy completions matched. Chat and sampled requests use the 128-token default. Set `--prefill-step-size 128` to use the same step for every request. The status endpoint reports the raw greedy override as `greedyCompletionPrefillStepSize` when active.

Across two fresh processes per bundle, peak physical footprint was 20.19–20.21 GiB with Q4 and 21.48–21.95 GiB with the original weights. Median decode across ten measurements per prompt rose from 23.85 to 24.52 tokens/s for short input and from 20.21 to 20.44 tokens/s for 2,048 input tokens.

The 0.17.0 Q4 control scored **20/25** on the task suite. The original historical release scored **22/25**. Q4 passed 7/10 model-only cases, 10/12 household cases and all three Python automation cases. The additional misses relative to that release were tool-error recovery and handling an uncertain outcome after an outage. Both bundles passed the two separate coding exercises with executable tests. A fresh run of the original weights on the experimental bounded-cache backend scored 21/25.

The 26,624-token window allows room for both the prompt and response. A greedy raw completion retrieved an early value from a 26,368-token prompt in 134 seconds of uncached prefill. The same process accepted 26,623 input tokens and returned one output token at the exact cap. Peak footprint across both requests and cancellation recovery was 23.21 GiB. In earlier tests with the 128-token prefill step, near-full retrieval took 173 seconds at the default cap. At 28,672 tokens it took 189 seconds, exceeding the three-minute target. The 32,768-token boundary also worked, with near-full recall taking 222 seconds. A 65,536-token test stopped at the experiment’s 24 GiB process budget. A 200,000-token window remains unverified on this Mac.

## Original mixed-precision bundle

Install Mei 0.19.0 or later and the Hugging Face `hf` CLI on an Apple Silicon
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

The automatic profile uses automatic total-context admission, a 4,096-token
default output cap, a 128-token prefill step, and one active generation. Mei
uses the bundle's sampling defaults and selects the qualified native kernels
and bounded Metal command buffers. Text chat, streaming, and native tool calls
are supported. Prefix reuse is off. Image input, speculative decoding, and
grammar-constrained response formats are outside this profile.

The [tested model bundle](https://huggingface.co/nativ-community/Inkling-2-Mini-nvfp4/tree/26f71eef259a1dd7fc476778d67020351e0dfb8c)
is a conversion of [Thinking Machines Lab's Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/).
