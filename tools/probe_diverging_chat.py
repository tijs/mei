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
  probe_diverging_chat.py --self-test
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
    return {"growth": growth, "divergence": divergence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--self-test", action="store_true",
                        help="validate transcript + comparison logic only; no server, no inference")
    args = parser.parse_args()
    if args.self_test:
        transcripts = build_transcripts()
        checks = _check_transcripts(transcripts)
        ok = all(checks.values())
        print(json.dumps(checks, indent=2, sort_keys=True))
        print("SELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if not args.model:
        parser.error("--model is required unless --self-test")
    if not args.output:
        parser.error("--output is required unless --self-test")

    transcripts = build_transcripts()
    checks = _check_transcripts(transcripts)
    if not all(checks.values()):
        print(f"FATAL: transcript invariants broken: {checks}", file=sys.stderr)
        return 1

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "probe": "diverging-chat (patch-0005 evidence: mid-transcript agentic edit)",
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
    result["status"] = (
        "passed"
        if a_r5.get("status") == "passed" and a_r5.get("cached_tokens")
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())