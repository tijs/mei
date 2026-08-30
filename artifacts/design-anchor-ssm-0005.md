# Design note: system-anchor SSM companion states (FreeToken-derived, patch 0005 candidate)

Status: DESIGN ONLY — no fork change yet; gated on phase-A measurement +
a new diverging-chat probe. Recorded 2026-08-30 session C.

## FreeToken mapping (exact cites)
- github.com/FlashML-org/FreeToken README.md:20-21 "semantic anchor
  checkpoints for recurrent state and KV caches, allowing agentic context
  edits (e.g., tool calls, thinking blocks) to avoid redundant context
  recomputation".
- The pinned engine ALREADY implements most of this:
  - SSMReDerive.swift `captureCleanSSMStateInline` (§440, capture-during-
    prefill — zero-cost clean-state snapshot at the gen-prompt-stripped
    boundary),
  - `reDeriveAndStoreSSMStatesAtPromptBoundaries` (one replay capturing
    exact prompt boundary + gen-suffix-stripped boundary + nearest paged
    block boundaries, LRU-capped at `ssmMaxEntries` = 50, CacheCoordinator-
    Config.swift:110),
  - SSMCompanionDiskStore write-through (SSMReDerive.swift header, #110).
- The GAP: stored block-boundary anchors are the LARGEST boundaries
  (closest to the prompt END — where cross-turn extension hits land:
  SSMReDerive.swift:493-498 "Keep the LARGEST boundaries"). An agentic
  edit that diverges MID-transcript (stable system prompt + N turns, then
  a changed tool/thinking block) matches only a block boundary near the
  transcript START, which has no companion SSM state -> the coordinator
  falls back to a full prefill (always correct, ~1x prefill cost).

## Hypothesis
Storing SSM companion anchors at TRANSFORMED chat-template structural
boundaries (each role-turn start, i.e. the stable system prefix + first K
turn boundaries) lets a mid-transcript diverging edit restore KV from the
last matched block AND re-derive only the divergent suffix — converting
agentic-edit TTFT from O(full prefill) to O(suffix).

## Evidence needed before any fork
1. Phase A ssm-rederive true/false rows (does the turn-end ~1x rederive
   tax dominate chat cells?).
2. NEW probe (tools/probe_diverging_chat.py, ~60 lines): 5-turn transcript,
   turns 1-4 identical across two runs, run A completes, run B repeats
   turns 1-4 then diverges in turn 5 (different tool arguments) -> record
   cached_tokens, prefill_ms, TTFT. If cached ~= full transcript (strict
   extension) the anchors already work for the growth pattern; if cached=0
   or ~=4-turn prefix on divergence, we have the quantified gap.
3. 40K chat repeat with a mid-transcript edit at 33K/80K.

## Fork shape (only if the probe shows the gap)
Minimal patch 0005: thread `--ssm-anchor-boundaries K` from the Mei server
CLI into `reDeriveAndStoreSSMStatesAtPromptBoundaries(additionalBoundaries:
[Int])` (the parameter already exists at SSMReDerive.swift:443). The Mei
layer computes K structural boundaries from the stored prompt (chat-
template role split points via the tokenizer chat template, cached at
store time). No upstream representation change; rotating+disk tier
unchanged; Mamba state correctness preserved (same replay path).
Rollback: default 0 = current behavior.

## Constraints (unchanged)
- Never weaken the 9B correctness/40-50K gates; acceptance + 30K/80K
  survival + tool-call schema checks per variant.
- This is a TTFT/latency lever, NOT a decode tok/s lever; the >=40 gate
  candidate stays window16-compiled (see F9A bandwidth model in the log).