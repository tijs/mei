<p align="center">
  <img src="assets/mei-logo.png" alt="Mei logo" width="300" height="300">
</p>

# Mei

**Run language models locally on your Apple Silicon Mac and connect them to your apps.**

Mei is a native Swift/MLX inference server with an OpenAI-compatible API. Use it
as a local backend for a chat app, coding agent, or your own software. Model
inference runs on your Mac; Mei serves one model per process.

[Website](https://mei.computer/) · [Releases](https://github.com/tijs/mei/releases/latest) · [API compatibility](docs/OPENAI-COMPATIBILITY.md)

## What you need

- An **Apple Silicon Mac** running **macOS 15 or later**.
- Enough unified memory for the model, its context, and your other apps. The
  larger models below have been tested on a **32 GB Mac**; memory needs vary
  with the model and request size.
- Disk space for model weights, which are downloaded separately. Check the
  model repository for download size before fetching a bundle.
- [Homebrew](https://brew.sh/) for the installation steps below.

## Get started

### 1. Install Mei and the model downloader

```sh
brew install tijs/tap/mei
mei --version

# Skip these if uv and the Hugging Face CLI are already installed.
brew install uv
uv tool install --upgrade huggingface_hub
hf --version
```

Already have Mei? Run `brew update && brew upgrade mei` to update it. You can
also download a prebuilt bundle from the [latest release](https://github.com/tijs/mei/releases/latest):
verify its accompanying SHA-256 checksum, unpack it, and run `bin/mei`. Keep
`mlx.metallib` beside the executable. No Xcode or source build is needed for
these installations.

### 2. Download and start a model

This example uses the Groot text model. `mei pull` uses the Hugging Face CLI to
download the profile's pinned model revision into your local cache.

```sh
mei pull groot-19b-a3b-text
mei --model groot-19b-a3b-text
```

Leave this terminal running. Wait for Mei to print
`listening on http://127.0.0.1:8024` before sending a request. Model loading can
take a while. Stop the server with **Ctrl-C**.

To inspect the download location and pinned revision first, use
`mei pull groot-19b-a3b-text --dry-run`. To choose a download directory, use
`mei pull groot-19b-a3b-text --dest /path/to/model`, then pass that directory
to `--model-dir`.

### 3. Send a message

In another terminal:

```sh
curl -fsS http://127.0.0.1:8024/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "groot-19b-a3b-text",
    "messages": [{"role": "user", "content": "Explain unified memory in two sentences."}],
    "max_tokens": 256
  }'
```

The response is JSON; the reply is in `choices[0].message.content`. Add
`"stream": true` to the request and `-N` to curl for server-sent events.

## Connect an app

Choose your app's OpenAI-compatible Chat Completions provider and set:

| Setting | Value for the quickstart |
|---|---|
| Base URL | `http://127.0.0.1:8024/v1` |
| Model ID | `groot-19b-a3b-text` |
| API key | No key required by Mei; use `local` if your app requires a value |

The model ID must match the one Mei serves. `--served-model-id` gives it a
custom name. By default, Mei uses the selected model name or the bundle
directory name and prints it at startup.
Check it at any time with:

```sh
curl -fsS http://127.0.0.1:8024/v1/models
```

Mei supports chat completions, streaming, native tool calls, and structured
JSON output. Support depends on the model and chat template; the
[compatibility guide](docs/OPENAI-COMPATIBILITY.md) describes the supported
request fields and schema subset. Apps must support the Chat Completions API.

Mei listens on loopback by default and has no built-in API authentication.
Keep the default for apps on the same Mac. An app connected to Mei may still
make its own network requests or execute tools locally.

## Choose a model

These named profiles can be downloaded with `mei pull <profile>`:

| Model | Profile | Input |
|---|---|---|
| Inkling 2 Mini | `inkling-2-mini` | Text |
| Inkling 2 Mini, Mei vocabulary Q4 | `inkling-2-mini-mei-q4` | Text |
| Groot 19B-A3B | `groot-19b-a3b-text` | Text |
| Qwen3.6 35B-A3B, text-only | `qwen3.6-35b-a3b-text` | Text |
| Ornith 1.5 35B-A3B | `ornith-1.5-35b-a3b` | Text |
| Qwen3.6 35B-A3B, vision bundle | `qwen3.6-35b-a3b` | Text and images; see model-specific limits |

Use the same name with `mei pull <model>` and `mei --model <model>`. Mei
recognizes the downloaded bundle and selects its runtime settings automatically; startup logs show what was selected. Explicit
flags override those settings.

You can also point Mei at an existing compatible MLX model directory containing
its configuration, tokenizer, chat template, and safetensors weights. Support is
specific to the architecture and weight format. Check [the model guide](docs/MODELS.md)
for artifact requirements, preparation steps, and limitations before downloading
an arbitrary checkpoint. Weights are not included with Mei and have their own
licenses.

For **Inkling 2 Mini**, the Mei vocabulary Q4 bundle reduces serving memory
and selects a 26,624-token context on a 32 GB M1 Max. This opt-in scored
20/25 on the task suite versus 22/25 for the original release. See the
[model guide](docs/INKLING.md) for both downloads and the context override.
The Pi example below uses the original bundle.
For measured model comparisons and their hardware/runtime scope, see the
[current benchmark results](https://github.com/tijs/local-model-bench/blob/main/results/AGENT_SUMMARY.md).

## Choose a context limit

Mei defaults to **26,624 tokens** for the curated models on a 32 GB Mac. To use a different limit, restart Mei with `--context-cap` after the model name:

```sh
mei --model groot-19b-a3b-text --context-cap 65536
```

The same flag works with `--model-dir`. Match your client’s context window to the server setting. For Inkling, the prompt, conversation history, tool results and reply share the cap. For Groot, Qwen and Ornith the cap limits the prompt. Use `--max-tokens` to set the separate reply limit.

| Mac unified memory | Example context cap | When to try it |
|---|---:|---|
| 24 GB | 8,192 | Start here if the model fits with room for macOS. Both Inkling profiles use this fallback below 32 GB. |
| 32 GB | 26,624 | Default for the curated models. On an M1 Max, a greedy raw Inkling Q4 completion retrieved an early value from 26,368 input tokens in 134 seconds of prefill. |
| 64 GB or more | 65,536 | A starting experiment for a larger Mac. Test a representative long prompt and check Memory Pressure in Activity Monitor. |

Weight size and the model’s attention cache affect how much context fits. Lower the cap if your Mac starts swapping or long prompts take too long. SenseNova Loop has a separate 4,096-token runtime limit.

## Use Mei with Pi

[Pi](https://pi.dev/) can connect to Mei through its built-in OpenAI Chat
Completions provider. This example uses Inkling 2 Mini, a text model with
streaming tool calls and a 26,624-token total budget on a 32 GB Mac for prompts and replies.
Install Mei 0.16.1 or later and the `hf` CLI. The pinned bundle is about 19.5 GB.
See the [Inkling guide](docs/INKLING.md).

```sh
mei pull inkling-2-mini
mei --model inkling-2-mini
```

Add the `mei` provider under `providers` in `~/.pi/agent/models.json`, or
create the file with the following contents. The `apiKey` is a placeholder for
Pi's credential check. The local Mei server requires no API key.

```json
{
  "providers": {
    "mei": {
      "baseUrl": "http://127.0.0.1:8024/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": { "maxTokensField": "max_tokens" },
      "models": [
        { "id": "inkling-2-mini", "contextWindow": 26624, "maxTokens": 2048 }
      ]
    }
  }
}
```

For a different server cap, change Pi’s `contextWindow` to match. Set Pi’s compaction budgets by merging this override into
the project's `.pi/settings.json`.

```json
{
  "compaction": {
    "modelOverrides": {
      "mei/inkling-2-mini": { "reserveTokens": 2048, "keepRecentTokens": 4096 }
    }
  }
}
```

Check text generation from a second terminal, then start Pi in the project
you want to work on.

```sh
pi --model mei/inkling-2-mini --no-tools --no-session -p "Reply with OK."
pi --model mei/inkling-2-mini
```

For a different model, use the ID returned by
`curl -fsS http://127.0.0.1:8024/v1/models`. Set Pi's `contextWindow` to
fit the model and Mei's configured context cap, and `maxTokens` to the desired
reply budget. Pi runs tools locally and sends their results back to Mei
for the next turn. Tool use depends on the model and its chat template.
See [Pi's model configuration reference](https://docs.pi.dev/latest/models-configuration)
and [compaction settings](https://docs.pi.dev/latest/compaction#settings).

## Running Mei

Use `mei --help` for the full list of options. Common settings are:

| Option | Purpose |
|---|---|
| `--model NAME` | Model previously downloaded with `mei pull` |
| `--model-dir DIR` | Local model bundle to load |
| `--served-model-id ID` | Model name clients send in requests |
| `--port PORT` | Listen on another port; default `8024` |
| `--context-cap TOKENS` | Override the context limit at launch. See [Choose a context limit](#choose-a-context-limit). |
| `--max-tokens TOKENS` | Set the server's maximum reply length |
| `--kv-cache-dir DIR` | Configure an on-disk KV cache; support varies by model |

Run separate Mei processes on different ports to serve multiple models. By
default, each process executes one generation at a time with a bounded queue.
The experimental `--max-active-generations 2` option allows two overlapping
generations and needs additional memory. See [inference scheduling](docs/INFERENCE-SCHEDULING.md)
before enabling it.

To check a running server:

```sh
curl -fsS http://127.0.0.1:8024/healthz
curl -fsS http://127.0.0.1:8024/v1/mei/status
```

If a client cannot connect, check that model loading has finished and that the
host and port match. If it reports an unknown model, use the ID from
`/v1/models`. If loading or generation runs out of memory, free memory, choose
a smaller model, or lower the context and reply budgets.

## Build

Mei uses SwiftPM and a pinned [vmlx-swift](https://github.com/tijs/vmlx-swift)
runtime. Source builds require Xcode and its Metal toolchain. See the
[development guide](docs/DEVELOPMENT.md) for build, test, and Metal resource
setup, and the [installer guide](docs/INSTALL.md) for installing a built binary.

## Further reading

- [Models](docs/MODELS.md) — compatible bundles, provenance, and preparation.
- [API compatibility](docs/OPENAI-COMPATIBILITY.md) — endpoints, tool calls, and structured output.
- [Runtime policies](docs/RUNTIME-POLICIES.md) — cache and memory controls.
- [Architecture](docs/ARCHITECTURE.md) — inference, chunked prefill, and prefix reuse.
- [Benchmarking](docs/BENCHMARKING.md) — measurement and qualification methods.
- [Changelog](CHANGELOG.md) — release history.

Mei is named after [*Mei long*](https://en.wikipedia.org/wiki/Mei_long), the
“sleeping dragon” dinosaur. Licensed under [MIT](LICENSE); see [NOTICE.md](NOTICE.md)
for third-party notices.
