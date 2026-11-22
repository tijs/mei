# Run Inkling 2 Mini with Mei

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
