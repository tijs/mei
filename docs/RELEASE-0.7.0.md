# Mei 0.7.0 — constrained structured output

Mei 0.7.0 adds token-level constrained output for OpenAI-compatible chat
completions. It is a model-specific capability, not a guarantee across every
checkpoint served by Mei.

## What shipped

- `response_format: {"type":"json_object"}` constrains output to a complete
  JSON value.
- Strict `json_schema` supports recursively nested objects and arrays, required
  keys, `additionalProperties: false`, `minItems`/`maxItems`, scalar enums,
  nullable scalar unions, and numeric `minimum`, `maximum`, exclusive bounds,
  and `multipleOf` constraints.
- Numeric constraints use exact decimal semantics. Unsatisfiable schemas and
  unsupported schema keywords are rejected before generation; impossible token
  continuations are masked, and incomplete constrained output fails closed.
- Structured requests force thinking off and are not combined with non-empty
  tool lists. `response_format` is supported on `/v1/chat/completions`, not
  `/v1/completions`.

## Compatibility and limitations

The exact CoCore structured-output canary and Mei's full live acceptance suite
passed with `mlx-community/Qwen3-4B-4bit`. The same canary **failed closed** on
the tested Qwen3.6 text-only, Qwen3.6 vision, and Ornith checkpoints. CoCore
must not advertise structured-output capability for those checkpoints until a
fix is made and their live canaries pass. Normal text and tool-calling behavior
is separate and remains available.

The supported JSON Schema subset is deliberately narrower than full JSON
Schema. Unsupported keywords/combinators, malformed shapes, and unsatisfiable
numeric constraints return an error before generation; see
[OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md) for the exact contract.

## Correctness and performance fixes

The release fixes integer `multipleOf` satisfiability for fractional divisors
(for example, integer multiples of 1.5 are multiples of 3), the first-item
mask bypass for arrays with `maxItems: 0`, and slow numeric-prefix feasibility
for very large bounds. A reproducible synthetic 20k-token adversarial scan of
the `1e300`-scale prefix case improved from 17.84 seconds to 73.56 milliseconds.
That is a focused grammar benchmark, not a model-throughput claim.

The complete Swift test suite passed 337/337 tests; the live acceptance suite
passed 9/9 on Qwen3-4B. Qwen3-4B model weights are not bundled; stage/download
the checkpoint separately under its publisher's license.

## Install

```bash
brew update
brew upgrade tijs/tap/mei
mei --version  # mei 0.7.0
```

Manual Apple Silicon release asset: `mei-0.7.0-macos-arm64.tar.gz` and its
`.sha256` companion on the GitHub release page.
