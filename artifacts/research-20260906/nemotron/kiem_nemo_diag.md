# Nemotron ROOT CAUSE: tool calls fail above ~18k prompt tokens; "1.02 tok/s" is a prefill-dominated metric, not slow decode (2026-09-06)

Live diagnosis on a dedicated Nemotron server (port 8024, pinned build
`mei-build-67e897e`, the exact bench config). GPU was free — the other agent's
work is paused.

## The reported symptom was misleading

The benchmark recorded **1.02 tok/s average**, below the 4.0 viability gate, so
the coding suites were correctly skipped. That figure reads as "the model is
slow". It is not.

`runner/run_prompt.py:784` computes `tokens_per_second = completion_tokens /
wall_seconds`, and `wall_seconds` **includes prefill**. A representative failing
row (`hermes_ops-selection`):

| field | value |
|---|---|
| prompt_tokens | **21,934** |
| ttft_seconds | **84.858** |
| wall_seconds | 85.677 |
| completion_tokens | 55 |
| tokens_per_second | 0.64 |

**84.86 s of the 85.68 s is prefill.** Decode itself ran the 55 tokens in ~0.8 s
≈ 68 tok/s — consistent with the 70.4 tok/s the standalone gate measured, and
the **fastest decode in the whole Mei lineup**. The metric is legitimate as an
end-to-end throughput measure, but it is not a decode-speed measure, and the
model was excluded for the wrong reason.

## The actual defect: tool calls stop being emitted at long prompts

All 8 hermes_ops tasks show the same shape — `turns=1`, no tool calls, a short
prose preamble, then stop:

- chaining: *"I'll search for the current population of Amsterdam and then write
  it to a file. Let me start by searching for this information."* — 0 tool calls
- multi-step-chain: *"I'll help you find and fix the bug... Let me start by
  exploring the project structure."* — 0 tool calls
- error-recovery: *"The file doesn't exist."* — 0 tool calls
- no-tool-needed: answered correctly — **this is the 1/8 that passed**

The model understands the task and narrates the intended tool call, then never
emits one, so the agent loop ends after one turn.

### Live isolation (this session, same server, temperature 0)

| # | condition | prompt tokens | tool call? |
|---|---|---|---|
| A | minimal system + 2 tools | 263 | **YES** |
| B | no system message + 2 tools | 257 | **YES** |
| C | explicit "use the tools" nudge | 272 | **YES** |
| D | **the REAL 20,116-char Hermes system prompt** + 2 tools | 4,746 | **YES** |
| E | small system + **22 tools** | 1,830 | **YES** |
| F | ~20k chars of neutral filler + 2 tools | **18,354** | **NO** — answered from memory |

So it is **not** the chat template, **not** the Hermes system prompt content,
and **not** tool count. Leg F reproduces the failure with generic filler, and
legs D and E rule out the two obvious config suspects. **It is prompt length.**

Note legs A/C produce *exactly* the same preamble style as the failing benchmark
rows ("I need to find... Let me search for this information") — but with the
tool call attached. The failure mode is the call being dropped, not the model
misunderstanding.

The chat template's tool-call dialect is unusual —
`<tool_call><function=NAME><parameter=NAME>value</parameter></function></tool_call>`
rather than the JSON-in-`<tool_call>` form — and generation begins inside
`<think>` when `enable_thinking` is true (the default). Both were prime
suspects; legs A–E clear them at short and medium length. Whether the long-prompt
failure is the model losing the format, or Mei/vmlx failing to parse it at
length, is **not yet determined** — a length sweep is running to find the
threshold.

## Why this cascades into the throughput number

Because no tool is ever called, every task terminates after **one** turn. That
means every task pays a **full fresh prefill** and never benefits from KV prefix
reuse. Compare aggregate hermes_ops prefill across the Mei lineup:

| model | avg prompt tokens/task | prefill pps |
|---|---|---|
| Qwen3.6-35B-A3B | 208,594 | 3,233 |
| Ornith-1.5-35B-A3B | 153,449 | 2,642 |
| Qwen3.8-27B-Uncensored | 459,229 | 1,221 |
| Gemma-4-26B | 53,686 | 893 |
| **Nemotron-3.5-Lightning** | **24,699** | **292** |

**Important caveat against over-reading this table:** the other models run
multi-turn agent loops, so most of their prompt tokens are *cache-reused* turns
that cost almost nothing — which inflates their apparent pps. Nemotron's single
turns are all fresh. Against Ornith's measured *fresh* 30k prefill (~392 pps at
step 512), Nemotron's 292 pps at step 256 is roughly 1.3x slower, not 9x. The
9x in this table is mostly a cache-reuse artifact and should not be quoted as a
prefill comparison.

The causal chain is: **no tool call -> single turn -> no KV reuse -> every task
pays full fresh prefill -> end-to-end tok/s collapses -> speed gate fires.**
Fixing the tool-call emission should fix the throughput number as a side effect.

## Next

1. Length sweep (running) to pin the threshold between 4,746 and 18,354 tokens.
2. Prefill-step sweep 256/512/1024 — the bench config inherited 256 from the
   generic profile with no measurement; Ornith's own sweep found 512 optimal for
   qwen3_5_moe and Gemma4 found 256, so nemotron_h is simply untested.
3. If the threshold is sharp and low, check whether Mei/vmlx tool-call parsing
   degrades at length versus the model genuinely dropping the format — the two
   need different fixes.

#proj/mei
