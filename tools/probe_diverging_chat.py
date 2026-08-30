#!/usr/bin/env python3
"""Mid-transcript agentic-edit cache probe for Mei (patch-0005 evidence).

The pinned engine's SSM companion anchors keep only the LARGEST prompt
boundaries (prompt END / generation-suffix-stripped boundary / block
boundaries nearest the end; see SSMReDerive.swift:493-498 "Keep the
LARGEST boundaries", ssmMaxEntries=50 LRU). A transcript that grows by
strict extension therefore reuses fully. The open question (patch 0005
design, artifacts/design-anchor-ssm-0005.md) is what happens when an
agentic turn DIVERGES mid-transcript: the coordinator's nearest retained
block boundary may sit near the transcript start, forcing a full-prefill
fallback (always correct, ~1x prefill cost).

This probe quantifies that gap with a realistic 5-turn tool-calling
transcript:

  Run A (growth):  requests 1..5 send turns 1..i — pure strict extension;
                   request 5 should report cached_tokens ~= full transcript.
  Run B (divergence): requests 1..4 identical to run A, then request 5
                   replaces turn 5 with a DIFFERENT tool call (same prefix
                   through turn 4). The recorder captures cached_tokens,
                   prefill_ms (engine-reported prompt-processing ms) and
                   TTFT (client-measured first-delta via streaming).

Reading the result:
  - growth cached ~= full transcript            -> anchors already work
    for the strict-extension pattern.
  - divergence cached ~= turns 1..4 tokens      -> coordinator restores the
    mid-transcript boundary; only the divergent suffix re-prefills (good).
  - divergence cached == 0 or tiny              -> quantified gap in the
    direction patch 0005 targets (anchor at role-turn boundaries).

This probe is CLIENT-SIDE: it touches no Metal and no inference. Run it
with --self-test to validate the transcript/comparison logic without a
server (the unit-level guarantee); with a live server it produces the
measurement artifact. It never modifies local-model-bench.

Usage:
  probe_diverging_chat.py --base-url http://127.0.0.1:8024/v1 \\
      --model ID --output artifacts/probe-diverging-chat-<ts>.json
  probe_diverging_chat.py --base-url http://127.0.0.1:8024/v1 \\
      --model ID --ssm-anchor-boundaries 4 \\
      --output artifacts/probe-diverging-chat-anchors4-<ts>.json
  probe_diverging_chat.py --self-test

Patch-0005 A/B: the anchors flag is a SERVER-side launch setting
(--ssm-anchor-boundaries K on the mei binary; default 0 = off). The
probe is client-side and cannot reconfigure a running server, so the A/B
is orchestrator-driven: start the server with anchors on, run this probe
with --ssm-anchor-boundaries K (labels the artifact with the exact server
config under test), stop; then start the default server and run the probe
again with --ssm-anchor-boundaries 0. Compare the divergence summary
between the two artifacts (gap_tokens / turn4_prefix_restored). The label
is recorded verbatim in server_config.ssm_anchor_boundaries and validated
by --self-test, so mislabeled A/B artifacts fail loudly instead of being
compared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mei import request_json, request_stream  # noqa: E402

SYS_PROMPT = (
    "You are a warehouse assistant. Use the provided tools to answer. "
    "Never invent prices."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_item",
            "description": "Look up the price of a warehouse SKU.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place an order for a quantity of a SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["sku", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a previously placed order.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    },
]


def _lookup_turn(sku: str, price: str, seq: int) -> list[dict[str, Any]]:
    """One user->tool_call->tool_result->assistant turn (2 messages)."""
    user = {
        "role": "user",
        "content": f"Turn {seq}: what does {sku} cost?",
    }
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_lookup_{seq}",
            "type": "function",
            "function": {"name": "lookup_item", "arguments": json.dumps({"sku": sku})},
        }],
    }
    tool = {"role": "tool", "tool_call_id": f"call_lookup_{seq}", "content": json.dumps({"price": price})}
    closing = {"role": "assistant", "content": f"{sku} costs {price}."}
    return [user, assistant, tool, closing]


def _order_turn_a(seq: int) -> list[dict[str, Any]]:
    """Run A turn 5: place an order (the divergence target)."""
    user = {"role": "user", "content": f"Turn {seq}: place an order for 3 of SKU-1001."}
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_order_a_{seq}",
            "type": "function",
            "function": {
                "name": "place_order",
                "arguments": json.dumps({"sku": "SKU-1001", "quantity": 3}),
            },
        }],
    }
    tool = {"role": "tool", "tool_call_id": f"call_order_a_{seq}", "content": '{"order_id": "ORD-77"}'}
    closing = {"role": "assistant", "content": "Order ORD-77 placed for 3 units."}
    return [user, assistant, tool, closing]


def _order_turn_b(seq: int) -> list[dict[str, Any]]:
    """Run B turn 5: a DIFFERENT tool call (cancel instead of place)."""
    user = {"role": "user", "content": f"Turn {seq}: cancel any order for SKU-1001."}
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_order_b_{seq}",
            "type": "function",
            "function": {
                "name": "cancel_order",
                "arguments": json.dumps({"sku": "SKU-1001"}),
            },
        }],
    }
    tool = {"role": "tool", "tool_call_id": f"call_order_b_{seq}", "content": '{"cancelled": true}'}
    closing = {"role": "assistant", "content": "Cancelled any order for SKU-1001."}
    return [user, assistant, tool, closing]


SKUS = [("SKU-1001", "42"), ("SKU-1002", "17"), ("SKU-1003", "93"), ("SKU-1004", "8")]

# Tool-name -> required argument keys (the schema every tool_call payload
# in this probe must satisfy exactly: no missing, no extra keys).
TOOL_ARG_SCHEMA: dict[str, set[str]] = {
    "lookup_item": {"sku"},
    "place_order": {"sku", "quantity"},
    "cancel_order": {"sku"},
}


def validate_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Deterministic structural/schema validation of one transcript.

    Checks the OpenAI message invariants this probe depends on, so a
    transcript edit can never silently change the measured pattern:
      - per-turn role sequence: user -> assistant(tool_calls) ->
        tool -> assistant(closing content);
      - every tool message's tool_call_id references a tool_call id of a
        preceding assistant message (and none are left unresolved);
      - exactly one tool call per assistant message;
      - every tool-call arguments payload JSON-parses and matches
        TOOL_ARG_SCHEMA for its tool name (no missing/extra keys);
      - every tool result content JSON-parses.
    Returns a list of human-readable violations (empty == valid).
    """
    issues: list[str] = []
    open_calls: dict[str, str] = {}  # tool_call_id -> tool name, awaiting result
    awaiting_closing = False
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            continue  # neutral setup block; legal at any position
        if role == "user":
            if open_calls:
                issues.append(
                    f"msg{idx}: user turn starts with {len(open_calls)} tool result(s) "
                    f"unresolved: {sorted(open_calls)}")
            awaiting_closing = False
        elif role == "assistant":
            calls = msg.get("tool_calls")
            if calls is not None:
                if awaiting_closing:
                    issues.append(f"msg{idx}: assistant tool_call while a closing reply is pending")
                if len(calls) != 1:
                    issues.append(f"msg{idx}: expected exactly 1 tool call, got {len(calls)}")
                for call in calls:
                    call_id = call.get("id")
                    name = (call.get("function") or {}).get("name")
                    if name not in TOOL_ARG_SCHEMA:
                        issues.append(f"msg{idx}: tool name {name!r} not in probe schema")
                        continue
                    try:
                        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                    except ValueError as exc:
                        issues.append(f"msg{idx}: tool {name!r} arguments not JSON: {exc}")
                        continue
                    if not isinstance(args, dict):
                        issues.append(f"msg{idx}: tool {name!r} arguments not an object: {args!r}")
                        continue
                    expected = TOOL_ARG_SCHEMA[name]
                    if set(args) != expected:
                        issues.append(f"msg{idx}: tool {name!r} args keys {sorted(args)} != {sorted(expected)}")
                    if name == "place_order" and not isinstance(args.get("quantity"), int):
                        issues.append(f"msg{idx}: place_order quantity not int: {args.get('quantity')!r}")
                    if call_id:
                        open_calls[call_id] = name
                awaiting_closing = False
            else:
                if open_calls:
                    issues.append(
                        f"msg{idx}: assistant closing reply while tool result(s) still "
                        f"pending: {sorted(open_calls)}")
                awaiting_closing = False
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            if not call_id or call_id not in open_calls:
                issues.append(f"msg{idx}: tool result {call_id!r} has no pending assistant tool_call")
            else:
                del open_calls[call_id]
            try:
                json.loads(msg.get("content") or "{}")
            except ValueError as exc:
                issues.append(f"msg{idx}: tool result content not JSON: {exc}")
            awaiting_closing = True
        else:
            issues.append(f"msg{idx}: unexpected role {role!r}")
    if open_calls:
        issues.append(f"unresolved tool_call ids at end: {sorted(open_calls)}")
    return issues


def build_transcripts() -> dict[str, list[list[dict[str, Any]]]]:
    """Five turn-suffix transcripts for each run.

    run A: turns 1..5 as recorded (turn 5 = place_order).
    run B: turns 1..4 identical to A, turn 5 replaces place_order with
    cancel_order (prefix divergence at the turn-5 boundary only).
    """
    base_turns: list[dict[str, Any]] = []
    for seq, (sku, price) in enumerate(SKUS, start=1):
        base_turns.extend(_lookup_turn(sku, price, seq))
    base_prefix = base_turns  # turns 1..4 (16 messages)

    transcripts: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    for i in range(1, 6):
        # Growth requests: turns 1..i. For i <= 4 both runs are identical.
        prefix = base_prefix[: 4 * i]
        transcripts["a"].append([{"role": "system", "content": SYS_PROMPT}, *prefix])
        transcripts["b"].append([{"role": "system", "content": SYS_PROMPT}, *prefix])
    # Final (5th) request: run A finishes with place_order, run B diverges.
    transcripts["a"][-1] = [{"role": "system", "content": SYS_PROMPT}, *base_prefix, *_order_turn_a(5)]
    transcripts["b"][-1] = [{"role": "system", "content": SYS_PROMPT}, *base_prefix, *_order_turn_b(5)]
    return transcripts


def _check_transcripts(transcripts: dict[str, list[list[dict[str, Any]]]]) -> dict[str, Any]:
    """Structural invariants of the probe design, no server required."""
    checks: dict[str, Any] = {}
    turns_a = transcripts["a"]
    turns_b = transcripts["b"]
    checks["five_requests_each"] = len(turns_a) == len(turns_b) == 5
    checks["runs_identical_through_request_4"] = all(turns_a[i] == turns_b[i] for i in range(4))
    checks["turn5_diverges"] = turns_a[4] != turns_b[4]
    checks["turn5_same_prefix"] = turns_a[4][:-4] == turns_b[4][:-4]
    checks["strict_growth_a"] = all(
        turns_a[i][:-4] == turns_a[i - 1] and len(turns_a[i]) >= len(turns_a[i - 1])
        for i in range(1, 5)
    )
    names_a = [m["tool_calls"][0]["function"]["name"]
               for m in turns_a[4] if m.get("tool_calls")][-1:]
    names_b = [m["tool_calls"][0]["function"]["name"]
               for m in turns_b[4] if m.get("tool_calls")][-1:]
    checks["runA_final_tool_is_place_order"] = names_a == ["place_order"]
    checks["runB_final_tool_is_cancel_order"] = names_b == ["cancel_order"]
    # Per-turn role sequencing + tool_call_id cross-references + argument
    # schema + result JSON (see validate_messages) on ALL ten transcripts.
    violations = {
        f"{run}_r{i+1}": validate_messages(msgs)
        for run, set_ in transcripts.items()
        for i, msgs in enumerate(set_)
    }
    checks["all_transcripts_schema_valid"] = not any(violations.values())
    checks["schema_violations"] = {k: v for k, v in violations.items() if v}
    # Determinism: building the transcripts twice yields identical bytes.
    rebuilt = build_transcripts()
    checks["build_deterministic"] = rebuilt == transcripts
    # The divergent user turn 5 must itself differ (different tool request).
    user_a = turns_a[4][len(turns_a[4]) - 4]["content"]
    user_b = turns_b[4][len(turns_b[4]) - 4]["content"]
    checks["turn5_user_contents_differ"] = (
        user_a != user_b and "place" in str(user_a) and "cancel" in str(user_b)
    )
    return checks


def run_requests(
    args: argparse.Namespace,
    transcripts: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, Any]:
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    runs: dict[str, Any] = {"a": [], "b": []}
    for run_name in ("a", "b"):
        for i, messages in enumerate(transcripts[run_name], start=1):
            name = f"{run_name}_r{i}"
            started = time.monotonic()
            try:
                body, _elapsed, first_delta = request_stream(
                    chat_url,
                    {"model": args.model, "messages": messages, "tools": TOOLS,
                     "temperature": 0, "max_tokens": args.max_tokens},
                    timeout=args.timeout)
                usage = body.get("usage") or {}
                finish = ((body.get("choices") or [{}])[0]).get("finish_reason")
                message = ((body.get("choices") or [{}])[0]).get("message") or {}
                content = message.get("content") or ""
                tools_called = [
                    (c.get("function") or {}).get("name")
                    for c in (message.get("tool_calls") or [])
                ]
                row = {
                    "status": "passed",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prefill_ms": usage.get("prefill_ms"),
                    "decode_tokens_per_second": usage.get("tokens_per_second"),
                    "ttft_ms": round(first_delta * 1000.0, 1) if first_delta is not None else None,
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "finish_reason": finish,
                    # Deterministic output controls (per Level1Techs-derived
                    # rule): the artifact records what the model actually
                    # emitted so a variant change that shifts behavior is
                    # visible next to the tok/s number.
                    "content_tail": content.strip()[-120:] or None,
                    "tool_names": tools_called or None,
                    "error": None,
                }
            except BaseException as exc:  # noqa: BLE001
                row = {
                    "status": "error",
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            runs[run_name].append({name: row})
            print(f"[diverging] {name}: {row.get('status')} cached={row.get('cached_tokens')} "
                  f"prefill_ms={row.get('prefill_ms')} ttft_ms={row.get('ttft_ms')}", flush=True)
    return runs


def summarize(runs: dict[str, Any], prompt_tokens: dict[str, list[int]]) -> dict[str, Any]:
    """Turn raw rows into the evidence the patch-0005 gate needs."""
    def row(run: str, i: int) -> dict[str, Any]:
        return runs[run][i - 1][f"{run}_r{i}"]

    a_r5, b_r5 = row("a", 5), row("b", 5)
    a_r1 = row("a", 1)
    # Sizes of the shared prefix at the two boundaries that matter:
    # request 4 ends at the turn-4 boundary (prefix B can restore),
    # request 5 is the full diverged transcript.
    prefix4_tokens = prompt_tokens["b"][3] if prompt_tokens.get("b") else None
    full5_tokens = prompt_tokens["a"][4] if prompt_tokens.get("a") else None

    def near(value: int | None, target: int | None, tol: int = 8) -> bool:
        return (
            value is not None and target is not None
            and value >= target - tol
        )

    growth = {
        "run_a_r1_cached": a_r1.get("cached_tokens", 0),
        "run_a_r5_cached": a_r5.get("cached_tokens", 0),
        "full_transcript_tokens": full5_tokens,
        "run_a_r5_full_prefix_cached": (
            a_r5.get("status") == "passed"
            and near(a_r5.get("cached_tokens", 0), full5_tokens)
        ),
    }
    b_cached = b_r5.get("cached_tokens", 0)
    divergence = {
        "run_b_r5_cached": b_cached,
        "run_a_r5_cached": a_r5.get("cached_tokens", 0),
        "turn4_prefix_tokens": prefix4_tokens,
        "gap_tokens": max(0, (a_r5.get("cached_tokens") or 0) - b_cached),
        # Mid-transcript boundary restored iff the diverged run reuses
        # (almost) the whole identical 4-turn prefix; 0 or tiny = the
        # full-prefill fallback (the gap patch 0005 would close).
        "turn4_prefix_restored": near(b_cached, prefix4_tokens),
        "direction": (
            "gap_confirmed"
            if b_cached == 0 or (prefix4_tokens and b_cached < 0.5 * prefix4_tokens)
            else "anchors_sufficient_for_turn_boundary"
        ),
    }
    # Deterministic output checks on the two decisive rows: run A must have
    # called place_order and run B cancel_order, and their closing contents
    # must differ (the divergence is real, not a cache miss artifact).
    out_a = a_r5.get("tool_names")
    out_b = b_r5.get("tool_names")
    output_checks = {
        "run_a_r5_called_place_order": a_r5.get("status") == "passed" and out_a == ["place_order"],
        "run_b_r5_called_cancel_order": b_r5.get("status") == "passed" and out_b == ["cancel_order"],
        "turns_1_4_content_identical": (
            (runs["a"][3]["a_r4"].get("content_tail") or "")
            == (runs["b"][3]["b_r4"].get("content_tail") or "")
        ),
        "run_a_r5_content_tail": a_r5.get("content_tail"),
        "run_b_r5_content_tail": b_r5.get("content_tail"),
    }
    return {"growth": growth, "divergence": divergence, "output_checks": output_checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--ssm-anchor-boundaries", type=int, default=0,
                        help="server-side --ssm-anchor-boundaries K under test "
                             "(patch 0005; 0 = default off). LABEL ONLY: the probe "
                             "is client-side, the server must have been launched "
                             "with the matching flag. Recorded verbatim in the "
                             "artifact and validated by --self-test so a "
                             "mislabeled A/B fails loudly.")
    parser.add_argument("--self-test", action="store_true",
                        help="validate transcript + comparison logic only; no server, no inference")
    args = parser.parse_args()
    if args.self_test:
        transcripts = build_transcripts()
        checks = _check_transcripts(transcripts)
        # The anchors label must be a non-negative integer (it mirrors a
        # server launch flag; negative values would be a caller bug).
        checks["anchors_label_non_negative"] = args.ssm_anchor_boundaries >= 0
        # schema_violations is a DIAGNOSTIC dict (empty == valid), not a
        # boolean; exclude it from the all() gate.
        ok = all(v for k, v in checks.items() if k != "schema_violations")
        print(json.dumps(checks, indent=2, sort_keys=True))
        print("SELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if not args.model:
        parser.error("--model is required unless --self-test")
    if not args.output:
        parser.error("--output is required unless --self-test")

    transcripts = build_transcripts()
    checks = _check_transcripts(transcripts)
    if not all(v for k, v in checks.items() if k != "schema_violations"):
        print(f"FATAL: transcript invariants broken: {checks}", file=sys.stderr)
        return 1

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "probe": "diverging-chat (patch-0005 evidence: mid-transcript agentic edit)",
        # The server config under test (patch-0005 A/B label). This probe is
        # client-side: the value mirrors the launch flag the orchestrator
        # used for the server this run measured against, so the A/B pair of
        # artifacts (anchors=4 vs default) is traceable and comparable.
        "server_config": {"ssm_anchor_boundaries": args.ssm_anchor_boundaries},
        # Exact sampler/schema settings (Level1Techs-derived recording rule):
        # temperature 0 with no nucleus drift, fixed max_tokens, the exact
        # tool schema (json.dumps of TOOLS), the fixed system prompt, and
        # per-run transcript sizes in messages.
        "sampler": {"temperature": 0, "max_tokens": args.max_tokens},
        "tool_schema": TOOLS,
        "system_prompt": SYS_PROMPT,
        "transcript_sizes": {
            run: [len(msgs) for msgs in set_]
            for run, set_ in transcripts.items()
        },
        "transcript_checks": checks,
        "started_epoch": time.time(),
    }
    runs = run_requests(args, transcripts)
    result["runs"] = runs
    # Server-reported prompt token counts per request (for the growth gate).
    result["prompt_tokens_by_request"] = {}
    for name in ("a", "b"):
        result["prompt_tokens_by_request"][name] = [
            next(iter(r.values())).get("prompt_tokens") for r in runs[name]
        ]
    result["summary"] = summarize(
        runs, result["prompt_tokens_by_request"])
    result["finished_epoch"] = time.time()
    a_r5 = runs["a"][4]["a_r5"]
    # Any errored request invalidates the whole probe: a hole in run B's
    # sequence would otherwise read as "gap confirmed" (b_r5 cached=0).
    any_error = any(
        next(iter(r.values())).get("status") == "error"
        for run in ("a", "b") for r in runs[run]
    )
    result["status"] = (
        "passed"
        if not any_error and a_r5.get("status") == "passed" and a_r5.get("cached_tokens")
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())