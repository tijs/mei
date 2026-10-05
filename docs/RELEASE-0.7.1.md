# Mei 0.7.1

**Patch release — 2026-10-05**

Mei 0.7.1 fixes strict structured-output generation stalls across Qwen3.6 and
aligned Ornith while preserving token-level schema enforcement and fail-closed
behavior.

## What changed

- Reject tokenizer-token prefixes that make a JSON string escape, Unicode
  escape, surrogate pair, schema key, or enum value impossible to complete.
- Bound consecutive whitespace-only tokens while a structured root value is
  incomplete, preventing the generation budget from being exhausted without
  beginning a value.
- No prompt-only JSON instruction, output rewriting, or post-generation
  validation is used as a substitute for constrained decoding.

## Live compatibility

| Checkpoint | Structured output | Other capability notes |
|---|---|---|
| Qwen3.6 35B A3B text-only | Passed strict buffered and SSE CoCore canaries | Tool-call canary failed; do not infer tool support. |
| Qwen3.6 35B A3B vision | Passed strict buffered and SSE canaries with text-only requests | Image-conditioned structured output and tool calls are unverified/failed respectively. |
| Aligned Ornith 1.5 35B A3B | Passed strict buffered and SSE CoCore canaries | Tool-call canary passed. |
| Qwen3-4B | Existing structured-output pass remains valid from 0.7.0 | Existing tool and schema results remain unchanged. |

Strict structured output was live-tested on runtime source commit `ea5a67a`,
which is included in 0.7.1; subsequent product-source changes are the version
constant only. Because the parent is coordinating live-model gates, the built
0.7.1 package was not loaded against large models during this release run.
CoCore's attached engine separately reported `ready=true structured_output=true`
for all three; it reported `tool_calls=false` for both Qwen3.6 profiles and
`true` for Ornith. The CoCore LaunchAgent is offline and its provider
Register/PDS advertisement of these model IDs was not verified; Mei's release
does not configure that agent.

Qwen3-8B has not been re-tested after the whitespace fix. Other model families
remain individually gated; do not infer compatibility from these results.
Image-conditioned structured output on Qwen3.6 vision was not tested.

See [the model matrix](OPENAI-COMPATIBILITY.md#structured-output-live-model-matrix)
and [CoCore integration notes](COCORE.md) for exact checkpoint revisions,
evidence, and limitations.