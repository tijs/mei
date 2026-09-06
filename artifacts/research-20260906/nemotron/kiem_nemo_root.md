# Nemotron ROOT CAUSE CONFIRMED: the model emits tool calls WITHOUT the `<tool_call>` wrapper, and the parser is strict (2026-09-06)

Supersedes the "fails above ~18k prompt tokens" framing in the earlier
diagnosis note — that was the right symptom but the wrong mechanism. Live
evidence, dedicated Nemotron server, pinned bench build, temperature 0.

## The actual failure

At larger prompts the model emits its tool call in the raw Nemotron dialect but
**drops the enclosing `<tool_call>` tags**, so `content` comes back containing:

```
I'll search for the current population of Amsterdam and then write that number
to a file. Let me start by searching for this information.
<function=web_search>
<parameter=query>
A…
```

`tool_calls` is empty and `finish_reason` is `stop`. The template's documented
format is:

```
<tool_call>
<function=NAME>
<parameter=NAME>
value
</parameter>
</function>
</tool_call>
```

vmlx's `XMLFunctionParser` (`Libraries/MLXLMCommon/Tool/Parsers/XMLFunctionParser.swift`)
takes a `startTag` and gates detection on it. The inner
`<function=…></function>` handling is present and correct — it simply never
runs, because the wrapper the trigger looks for is absent. So the emitted call
falls through to `content` as plain text and the agent loop ends after one turn.

At still-longer prompts the format degrades further, e.g.
`…this information.web_search\n</function>    <arg_0>query=Ams` — name outside
any tag, wrong parameter tag. So there are two regimes: **wrapper omitted**
(recoverable by a tolerant parser) and **format genuinely corrupted**
(not recoverable).

## What was ruled out, with evidence

| hypothesis | test | result |
|---|---|---|
| chat template is wrong for Hermes | real 20,116-char Hermes system prompt, 4,739 tok | **tool call OK** — template fine |
| too many tools | 22 small tools, 1,830 tok | **tool call OK** |
| thinking mode interferes | relaunched with `--enable-thinking false` | **byte-identical failures** — ruled out |
| prompt-length threshold | degenerate repeated filler 6,259 tok | **tool call OK** — so not raw length |
| stronger "you MUST call tools" instruction | added to system prompt at the failing shape | **still fails** — not a prompting fix |

The degenerate-filler contrast is the key one: 6,259 tokens of a repeated
sentence still produces a valid call, while 6,003 tokens of *varied realistic
prose* does not. Token count alone does not predict it; contextual load does.

Note `--enable-thinking false` is **not** plumbed through
`runner/start_mei_server.sh` (the wrapper rejects it); the Mei binary supports
it directly. The bench config never sets it, so it runs at the template default
(thinking ON) — which the test above shows is not the problem, but the wrapper
gap is worth fixing regardless.

## Why the throughput number collapsed

Causal chain, now fully established:

**tool call unparsed → single turn → no KV prefix reuse → every task pays a full
fresh prefill → `completion_tokens / wall_seconds` collapses → 4.0 speed gate
fires → coding suites skipped.**

The 1.02 tok/s was never a decode measurement. Decode is ~68–70 tok/s, the
fastest in the Mei lineup. `runner/run_prompt.py:784` computes
`tokens_per_second = completion_tokens / wall_seconds` with prefill included; a
representative row spent **84.86 s of 85.68 s in prefill** for 55 output tokens.

## Fix options, in order of cost

1. **Tolerant parser (recommended first).** Accept a bare
   `<function=NAME>…</function>` with no `<tool_call>` wrapper. Two places it
   could live: vmlx's `XMLFunctionParser` trigger (the correct fix, but needs a
   fork change, rebuild and re-pin, which per `docs/VMLX-FORK.md` forces a full
   acceptance re-run), or `bench_local_proxy.py`'s `BENCH_TOOL_PARSER` in
   local-model-bench (cheap, testable, no re-pin — and it *does* see this text,
   because in the failing case Mei leaves it unparsed in `content`).
2. Plumb `--enable-thinking` through `runner/start_mei_server.sh`. Not a fix for
   this bug, but a real gap in the wrapper.
3. Nothing addresses the corrupted-format regime at very long prompts; that
   needs either a smaller effective tool-schema payload or acceptance that
   Nemotron is unreliable above ~30k with heavy tool schemas.

## Re-benchmark condition

Nemotron should **not** be judged on the recorded 1/8 hermes_ops result. Once a
tolerant parser lands, the suite must be re-run before any viability call — and
the speed gate should be re-evaluated too, since a model that never completes a
turn cannot demonstrate KV reuse and is structurally penalised by an end-to-end
tok/s metric.

#proj/mei
