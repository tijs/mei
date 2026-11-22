# Run Inkling 2 Mini with Mei

Install Mei 0.14.0 or later on an Apple Silicon Mac, and install the Hugging Face
`hf` CLI. The tested NVFP4 bundle downloads about 19.5 GB. Qualification ran on a
32 GB M1 Max with only one model loaded.

```bash
hf download nativ-community/Inkling-2-Mini-nvfp4 \
  --revision 26f71eef259a1dd7fc476778d67020351e0dfb8c \
  --exclude 'mtp.safetensors' --local-dir ./inkling-2-mini
mei --model-dir ./inkling-2-mini
```

Mei identifies the bundle through `config.json`. The startup log reports the
Inkling text profile, and the served model ID defaults to the directory name.
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
