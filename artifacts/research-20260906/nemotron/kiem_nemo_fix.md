# Nemotron FIX FOUND: the existing `qwen3_coder` proxy parser already handles its dialect (2026-09-06)

Follow-on to the confirmed root cause (model emits tool calls without the
`<tool_call>` wrapper; vmlx's `XMLFunctionParser` is gated on that startTag and
never triggers, so the call lands in `content` as text).

## No new parser was needed

`runner/bench_local_proxy.py` already ships `qwen3_coder`, and despite the name
it matches Nemotron's format exactly:

- `QWEN3_CODER_SPLIT_MARKER = "<function="` — triggers on the **bare** form, no
  `<tool_call>` wrapper required.
- `_parse_qwen3_coder_tool_calls` looks for `<tool_call>` blocks first but
  **falls back to the whole text** when none are present
  (`if not raw_tool_calls: raw_tool_calls = [text]`).
- It then extracts `<function=NAME>…</function>` and
  `<parameter=NAME>value</parameter>`, which is precisely the dialect the
  Nemotron `chat_template.jinja` documents.

## Verified offline against the real captured output forms

| input | result |
|---|---|
| wrapped form (what the template documents) | **1 call** — `web_search {"query": "current population of Amsterdam"}` |
| **BARE form (the actual production failure)** | **1 call** — same |
| multi-parameter `write_file` | **1 call** — `{"path": "population.txt", "content": 905500}` |
| corrupted very-long-prompt form (`…web_search\n</function> <arg_0>query=…`) | **0 calls** — correctly refuses rather than inventing one |
| plain prose, no tool call | **0 calls** |

The corrupted-form row matters as much as the successes: the parser does not
hallucinate a call out of degraded output, so the very-long-prompt regime stays
a clean failure rather than becoming a silently wrong pass.

## The layering is correct — the proxy does not clobber Mei

`bench_local_proxy.py:501` only assigns `message["tool_calls"]` when its parser
actually finds a call. When Mei's native parse succeeds, `content` holds only
the prose preamble with no `<function=` text, so the proxy parser returns `[]`
and the native call passes through untouched. Mei wins when it works; the proxy
recovers only the cases Mei's strict parser drops.

This also corrects the earlier note (`hermes_style parser live verification`)
which concluded the proxy parser is always bypassed because "Mei parses the
model format before HTTP serialization". True in the success case; **not** true
in the failure case, which is exactly where recovery is needed.

## Config change applied

`configs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B/mei.yaml`:
- `orchestration.needs_proxy: false -> true`
- `orchestration.proxy_parser: qwen3_coder` (new)
- `tool_call_path: native_mei_requires_live_probe -> native_mei_plus_proxy_recovery`

**Trap worth recording:** the runner reads `orch["proxy_parser"]` at
`run_bench.py:487` as a **hard key access**, not a `.get()` with a default. A
config with `needs_proxy: true` and no `proxy_parser` raises `KeyError` mid-run,
after the model has already loaded. My first patch put the parser name at the
top level as `tool_call_recovery` and would have crashed exactly that way; it is
now under `orchestration:` where the runner reads it, with the descriptive note
kept in the header.

## NOT yet verified end-to-end

Offline parser verification only. The GPU was running the Qwen3.6 text-only
benchmark, so no live re-run happened. **Before any viability call for Nemotron,
re-run the hermes_ops suite through the proxy path** and confirm real
`tool_calls` come back — the repo's standing policy for any parser change.

Two things to expect on that re-run:
1. hermes_ops should move well off 1/8, since the tasks fail only on the dropped
   wrapper, not on task comprehension (the model narrates the correct intent).
2. The 1.02 tok/s speed-gate figure should rise substantially without any speed
   work at all: once tool calls parse, tasks become multi-turn, KV prefix reuse
   engages, and `completion_tokens / wall_seconds` stops being dominated by a
   single cold 22k-token prefill.

Known limitation inherited from the parser: no tool schema is threaded through,
so argument values get best-effort numeric coercion — a numeric-looking string
argument (`"905500"`) arrives as a number. The parser's own docstring flags this
as acceptable for grading, which compares values not types, but it is a real
difference from Mei's schema-aware native path.

#proj/mei
