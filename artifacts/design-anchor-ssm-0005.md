# Design note: system-anchor SSM companion states (FreeToken-derived, patch 0005)

Status: PLUMBING LANDED (2026-08-30 session E), default OFF, no performance
claim. Measurement of the effect remains gated on an uncontended window
(probe_diverging_chat.py + phase-A rows). Original design record below,
updated with the landed shape.

## What landed (patch 0005, plumbing only)

Engine side (patches/0005-ssm-anchor-boundaries.patch, applies over 0001-0004
on the pinned aeb5e21c tree):
- `GenerateParameters.ssmAnchorBoundaries: [Int] = []` (default [] =
  upstream behavior exactly).
- Both SSM re-derive store paths union the parameter's offsets into their
  `sharedPromptAdditionalBoundaries` set:
  - Evaluate.swift `TokenIterator.storeCacheAfterGeneration` (eager/solo
    path, via the iterator's `cacheInitParameters`),
  - BatchEngine.swift `finishSlot` (batched/compiled path, via
    `slot.parameters`).
  Engine-side validation is unchanged: offsets are filtered to
  `0 < b <= promptTokenIds.count` and Set-deduped before any replay, and
  `reDeriveAndStoreSSMStatesAtPromptBoundaries` captures the recurrent
  state at each boundary from a full prompt replay — so an anchor at ANY
  offset is the exact state for that prefix (misplaced-but-in-range
  offsets can only be suboptimal, never incorrect).

Mei side (Sources/MeiCore/SSMAnchorBoundaries.swift + ServerConfig +
Engine):
- `--ssm-anchor-boundaries K` (default 0 = off).
- `SSMAnchorBoundaries.compute` derives at most K EARLY role-turn
  boundaries as absolute token offsets: user-message start positions of
  the chat template dictionary, tokenized through the request's OWN
  rendering path (same tokenizer, same tool schema, same additional
  context). An additivity self-check requires a prefix render of the full
  message list to reproduce the request's exact token count; on any
  violation the computation returns [] with a logged warning (current
  behavior is the fallback — never a possibly-misplaced offset).
- Unit coverage (non-Metal, source-level): SSMAnchorBoundariesTests
  (offset derivation, K cap, non-additivity fallback, determinism, range
  filtering) + ServerConfigParsingTests for the new flag + default-off
  guard. No performance claim is made by the plumbing.

Why K role-turn anchors: the engine keeps only the LARGEST boundaries
(prompt end, strip end, end-nearest block boundaries; SSMReDerive.swift
"Keep the LARGEST boundaries", ssmMaxEntries=50 LRU). A mid-transcript
diverging agentic edit (identical turns 1..N, changed turn N+1) has no
retained companion near the transcript start -> full-prefill fallback.
Early anchors give the coordinator a restore point; the diverged suffix
alone re-prefills. TTFT/latency lever only — NOT a decode tok/s lever.

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

## Evidence needed before any performance claim
1. Phase A ssm-rederive true/false rows (does the turn-end ~1x rederive
   tax dominate chat cells?).
2. tools/probe_diverging_chat.py (5-turn transcript, turns 1-4 identical
   across two runs, run A completes, run B diverges in turn 5): record
   cached_tokens, prefill_ms, TTFT. If cached ~= full transcript (strict
   extension) the anchors already work for the growth pattern; if cached=0
   or ~=4-turn prefix on divergence, we have the quantified gap. A/B the
   flag: `--ssm-anchor-boundaries 4` vs default.
3. 40K chat repeat with a mid-transcript edit at 33K/80K.

## Rollback
Default 0 = upstream behavior exactly (both the config default and the
GenerateParameters default). Revert = drop the flag; no representation
change, no new disk/cache format.

## Constraints (unchanged)
- Never weaken the 9B correctness/40-50K gates; acceptance + 30K/80K
  survival + tool-call schema checks per variant.
- This is a TTFT/latency lever, NOT a decode tok/s lever; the >=40 gate
  candidate stays window16-compiled (see F9A bandwidth model in the log).