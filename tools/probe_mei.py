#!/usr/bin/env python3
"""Black-box acceptance and P0-contract probes for one isolated Mei model.

Mirrors local-model-bench's runner/probe_omlx.py contract so the same
validation logic gates Mei: exact /v1/models identity, plain completion,
structured add-numbers tool calls (non-streaming AND streaming), cache reuse
reporting, and the exact-token context-cap 65536 pass / 65537 reject gate.

Mei-specific additions on top of the oMLX probe:
  - `usage.prompt_tokens_details.cached_tokens` on repeated prefixes
    (the in-process KV slot)
  - engine decode tok/s surfaced from the run (tokens_per_second where the
    engine reports it via usage, plus client-side wall measurements)

P0 contract probes (--skip-advanced turns them off): the frozen subset in
docs/OPENAI-COMPATIBILITY.md — max_completion_tokens alias and conflict
rejection, loud rejection of deferred fields (response_format, n>1,
logprobs, parallel_tool_calls=false, developer role), request-over-server
sampling determinism, reasoning_effort toggles, multiple indexed tool calls
non-streaming AND streaming with parity, a real two-turn tool continuation,
tool_choice none, stream usage omission without include_usage, the legacy
text_completion shape, /v1/completions stream rejection, and chat
(non-streaming and streaming) context-cap rejection.

--self-test runs the probe's own assertion checks against deterministic
fixtures (no server, no model): stream termination and error-frame handling,
usage omission, and the sampling-override stability rules.

Output artifact is JSON; exit code 0 only when every probe passed.

Probe result schema: each `probes.<name>` entry carries a reserved
`status` verdict ("passed" | "failed" | "skipped") and `elapsed_seconds`,
plus the probe's own detail. Probe detail can never clobber the reserved
keys: an expected HTTP rejection records its numeric code as
`http_status` (e.g. 400 from the over-cap or conflict gate), distinct from
the per-probe pass/fail verdict. The top-level `status` aggregates the
per-probe verdicts.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 900) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_payload = json.loads(response.read())
            if isinstance(response_payload, dict) and response_payload.get("error"):
                raise RuntimeError(
                    f"HTTP {response.status} error payload: {json.dumps(response_payload, sort_keys=True)}"
                )
            return response_payload, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def request_raw(url: str, payload: dict[str, Any], timeout: float = 900) -> tuple[int, str, dict[str, Any] | None]:
    """POST and return (status, raw body, decoded JSON-or-None) WITHOUT raising
    on error envelopes — the probes that assert REJECTIONS use this."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, _as_json(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body, _as_json(body)


def _as_json(body: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def request_stream(url: str, payload: dict[str, Any], timeout: float = 900) -> tuple[dict[str, Any], float, float | None]:
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_delta = None
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    break
                event = json.loads(encoded)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("tool_calls")) and first_delta is None:
                    first_delta = time.monotonic() - started
                if delta.get("content"):
                    content.append(delta["content"])
                for part in delta.get("tool_calls") or []:
                    index = int(part.get("index", 0))
                    call = tool_calls.setdefault(
                        index,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if part.get("id"):
                        call["id"] = part["id"]
                    function = part.get("function") or {}
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }, time.monotonic() - started, first_delta


def request_stream_plain(url: str, payload: dict[str, Any], timeout: float = 900) -> tuple[dict[str, Any], list[str]]:
    """Stream WITHOUT injecting stream_options — the client that never asked
    for usage. Returns (assembled response, raw data: lines)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content: list[str] = []
    raw_lines: list[str] = []
    finish_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                raw_lines.append(line)
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    continue
                event = json.loads(encoded)
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    return {
        "choices": [{"message": {"role": "assistant", "content": "".join(content) or None}, "finish_reason": finish_reason}],
    }, raw_lines


def validate_add_call(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        raise AssertionError(f"expected one structured tool call, got {calls!r}")
    function = calls[0].get("function") or {}
    if function.get("name") != "add_numbers":
        raise AssertionError(f"unexpected tool name: {function.get('name')!r}")
    arguments = json.loads(function.get("arguments") or "{}")
    if arguments != {"a": 15, "b": 27}:
        raise AssertionError(f"schema/argument mismatch: {arguments!r}")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason != "tool_calls":
        raise AssertionError(f"unexpected finish_reason: {finish_reason!r}")
    return {"name": function["name"], "arguments": arguments, "finish_reason": finish_reason}


def check_sampling_override(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Assert the sampling-override contract across two identical greedy runs.

    Purpose: the request-time sampling fields (temperature 0 + fixed seed)
    must OVERRIDE the server defaults — greedy decoding is byte-stable — and
    they must not leak across requests: if any state from `first` carried into
    `second`, the second run's output would drift.

    Model-independent: output is accepted from EITHER visible channel
    (`content` or `reasoning_content`). A thinking model is free to spend a
    bounded generation budget entirely inside the reasoning channel (visible
    `content` is then legitimately empty — this is what made the old
    content-only assertion fail on a contract-correct thinking run), so
    emptiness is rejected only when NOTHING was generated at all and
    byte-stability is asserted per channel.
    """

    def channels(response: dict[str, Any]) -> tuple[str, str]:
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        return (
            (message.get("content") or "").strip(),
            (message.get("reasoning_content") or "").strip(),
        )

    first_content, first_reasoning = channels(first)
    second_content, second_reasoning = channels(second)
    if not (first_content or first_reasoning):
        raise AssertionError(
            "deterministic override run produced no output in either channel "
            "(content and reasoning_content both empty)")
    if first_content != second_content:
        raise AssertionError(
            "temperature-0 override is not byte-stable across identical requests (content)")
    if first_reasoning != second_reasoning:
        raise AssertionError(
            "temperature-0 override is not byte-stable across identical requests (reasoning_content)")
    if (first.get("usage") or {}).get("completion_tokens") != (second.get("usage") or {}).get(
        "completion_tokens"
    ):
        raise AssertionError("usage differs across identical greedy requests")
    if first_content and first_reasoning:
        channel = "content+reasoning_content"
    elif first_content:
        channel = "content"
    else:
        channel = "reasoning_content"
    return {"output_channel": channel, "output": (first_content or first_reasoning)[:200]}


def sse_data_payloads(raw_lines: list[str]) -> list[str]:
    """The `data:` payloads of a raw SSE line stream, in order.

    `[DONE]` arrives as a `data: [DONE]` EVENT — the payload after the
    `data:` prefix. The raw line is `data: [DONE]`, never a bare `[DONE]`
    line, so a terminator check must compare payloads, not whole lines.
    """
    payloads = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("data:"):
            payloads.append(stripped[len("data:"):].strip())
    return payloads


def check_stream_without_include_usage(raw_lines: list[str]) -> dict[str, Any]:
    """Assert the P0 stream contract for a client that never asked for usage.

    Success stream (no error frame): NO usage payload anywhere (the
    `stream_options.include_usage` contract — usage appears mid-stream ONLY
    when requested), a finish frame with `finish_reason` set, and a terminal
    `data: [DONE]` as the final event (emitted on every successful stream,
    with or without include_usage).

    Error stream (a `data:` frame carrying the error envelope): the error
    frame TERMINATES the stream — no finish frame, no usage chunk, and no
    `[DONE]` follow it. `[DONE]` is the SUCCESS terminator only, so its
    absence after an error frame is correct behavior and is reported as the
    mid-stream failure it is, not as a missing-terminator defect. Anything
    after the error frame (including a falsely-successful `[DONE]`) is a
    contract violation.
    """
    payloads = sse_data_payloads(raw_lines)
    frames: list[tuple[str, Any]] = []
    for payload in payloads:
        if payload == "[DONE]":
            frames.append((payload, None))
            continue
        try:
            frames.append((payload, json.loads(payload)))
        except ValueError as exc:
            raise AssertionError(f"non-JSON data frame in stream: {payload[:200]!r}") from exc

    error_positions = [
        i for i, (_, event) in enumerate(frames)
        if isinstance(event, dict) and "error" in event
    ]
    if error_positions:
        first = error_positions[0]
        if first != len(frames) - 1 or len(error_positions) > 1:
            raise AssertionError(
                "frames after the mid-stream error frame: the error frame "
                "terminates the stream (no finish/usage/[DONE] may follow)")
        error = frames[first][1]["error"]
        message = (error or {}).get("message") if isinstance(error, dict) else str(error)
        raise AssertionError(f"stream failed mid-stream: {message}")

    usage_frames = [
        payload for payload, event in frames
        if isinstance(event, dict) and "usage" in event
    ]
    if usage_frames:
        raise AssertionError(
            f"usage chunk sent without stream_options.include_usage: {usage_frames[0][:200]!r}")
    finish_reasons = [
        choice.get("finish_reason")
        for _, event in frames if isinstance(event, dict)
        for choice in (event.get("choices") or [])
        if isinstance(choice, dict)
    ]
    if not any(finish_reasons):
        raise AssertionError("stream produced no finish frame (finish_reason never set)")
    dones = [i for i, (payload, _) in enumerate(frames) if payload == "[DONE]"]
    if dones != [len(frames) - 1]:
        raise AssertionError(
            "stream must terminate with exactly one final data: [DONE] event")
    return {
        "frame_count": len(raw_lines),
        "data_frames": len(frames),
        "saw_done": True,
    }


def _expect_pass(fn) -> bool:
    try:
        fn()
        return True
    except BaseException as exc:  # noqa: BLE001 — self-test reports, never crashes
        print(f"    unexpected failure: {type(exc).__name__}: {exc}")
        return False


def _expect_fail(fn, include: str, exclude: str | None = None) -> bool:
    try:
        fn()
    except AssertionError as exc:
        message = str(exc)
        ok = include in message and (exclude is None or exclude not in message)
        if not ok:
            print(f"    wrong failure: {message!r} (want {include!r}, not {exclude!r})")
        return ok
    except BaseException as exc:  # noqa: BLE001
        print(f"    wrong exception type: {type(exc).__name__}: {exc}")
        return False
    print("    unexpected pass")
    return False


def _frame(event: dict[str, Any]) -> str:
    return "data: " + json.dumps(event, sort_keys=True)


def _success_lines() -> list[str]:
    """The exact wire shape of a successful no-usage stream: content frame,
    finish frame, `data: [DONE]` — `data: <json>\\n\\n` framing split into the
    stripped lines `request_stream_plain` records."""
    return [
        _frame({"choices": [{"delta": {"content": "no-usage", "role": "assistant"}}],
                "object": "chat.completion.chunk"}),
        "",
        _frame({"choices": [{"delta": {}, "finish_reason": "stop"}],
                "object": "chat.completion.chunk"}),
        "",
        "data: [DONE]",
        "",
    ]


def _completion(content: str | None, reasoning: str | None, completion_tokens: int) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 16, "completion_tokens": completion_tokens,
                  "total_tokens": 16 + completion_tokens},
    }


def run_self_test() -> int:
    """Deterministic self-test of the probe's own assertions — no server, no
    model. Fixture line streams and response payloads exercise exactly the
    contract checks `sampling_override` and `stream_usage_omitted` run
    against a live server."""
    checks: dict[str, bool] = {}

    # -- stream contract (stream_usage_omitted) --
    checks["success_stream_with_done_passes"] = _expect_pass(
        lambda: check_stream_without_include_usage(_success_lines()))
    checks["success_stream_missing_done_fails"] = _expect_fail(
        lambda: check_stream_without_include_usage(
            [ln for ln in _success_lines() if ln.strip() != "data: [DONE]"]),
        include="DONE")
    checks["bare_done_line_is_not_a_terminator"] = _expect_fail(
        lambda: check_stream_without_include_usage(
            [ln for ln in _success_lines() if ln.strip() != "data: [DONE]"] + ["[DONE]", ""]),
        include="DONE")
    checks["missing_finish_frame_fails"] = _expect_fail(
        lambda: check_stream_without_include_usage(
            [ln for ln in _success_lines() if "finish_reason" not in ln]),
        include="finish frame")
    checks["usage_frame_without_include_usage_fails"] = _expect_fail(
        lambda: check_stream_without_include_usage(_success_lines()[:2] + [
            _frame({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
            "",
        ] + _success_lines()[2:]),
        include="usage chunk sent without stream_options.include_usage")
    error_lines = [
        _frame({"choices": [{"delta": {"content": "partial"}}], "object": "chat.completion.chunk"}),
        "",
        _frame({"error": {"message": "generation exploded", "type": "invalid_request_error",
                          "code": "stream_error"}}),
        "",
    ]
    checks["error_stream_reports_midstream_error_not_missing_done"] = _expect_fail(
        lambda: check_stream_without_include_usage(error_lines),
        include="stream failed mid-stream: generation exploded", exclude="must terminate")
    checks["done_after_error_frame_fails"] = _expect_fail(
        lambda: check_stream_without_include_usage(error_lines + ["data: [DONE]", ""]),
        include="error frame terminates")
    checks["non_json_data_frame_fails"] = _expect_fail(
        lambda: check_stream_without_include_usage(["data: {not json", "", "data: [DONE]", ""]),
        include="non-JSON")

    # -- sampling-override contract (sampling_override) --
    checks["content_and_reasoning_stable_passes"] = _expect_pass(
        lambda: check_sampling_override(
            _completion("deterministic-override", "think think", 64),
            _completion("deterministic-override", "think think", 64)))
    checks["content_only_stable_passes"] = _expect_pass(
        lambda: check_sampling_override(
            _completion("deterministic-override", None, 8),
            _completion("deterministic-override", None, 8)))
    # Regression: a thinking model may spend the whole generation budget in
    # reasoning_content; a stable reasoning-only run honors the contract.
    checks["reasoning_only_stable_passes"] = _expect_pass(
        lambda: check_sampling_override(
            _completion(None, "thinking until the budget is spent", 64),
            _completion(None, "thinking until the budget is spent", 64)))
    checks["degenerate_empty_run_fails"] = _expect_fail(
        lambda: check_sampling_override(_completion("", None, 0), _completion("", None, 0)),
        include="no output")
    checks["content_drift_fails"] = _expect_fail(
        lambda: check_sampling_override(_completion("a", None, 8), _completion("b", None, 8)),
        include="byte-stable")
    checks["reasoning_drift_fails"] = _expect_fail(
        lambda: check_sampling_override(_completion("x", "r1", 8), _completion("x", "r2", 8)),
        include="byte-stable")
    checks["usage_drift_fails"] = _expect_fail(
        lambda: check_sampling_override(_completion("x", None, 8), _completion("x", None, 9)),
        include="usage differs")

    ok = all(checks.values())
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def exact_prompt(tokenizer_path: Path, target: int) -> str:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    unit = " hello"
    if len(tokenizer.encode(unit, add_special_tokens=False)) != 1:
        raise RuntimeError("the exact-token prompt unit is not one token for this tokenizer")

    # Count the way the SERVER counts, which is with special tokens. This used
    # to build and verify with add_special_tokens=False, so for any tokenizer
    # that prepends a BOS the server saw target+1 and the exact-cap probe failed
    # on a prompt the harness believed was exactly at the cap.
    #
    # Measured 2026-09-09: Ornith 1.5 and Qwen 3.6 text-only add nothing
    # (100 -> 100), Laguna XS 2.1 adds one (100 -> 101). So the probe passed
    # 12/12 on the first two and reported a false failure on Laguna, where the
    # server had correctly rejected 65537 tokens against a 65536 cap. The server
    # was right every time; the harness was measuring a different quantity.
    overhead = len(tokenizer.encode(unit, add_special_tokens=True)) - 1
    units = target - overhead
    if units < 1:
        raise RuntimeError(
            f"context cap {target} is too small for this tokenizer's "
            f"{overhead}-token special-token overhead")
    prompt = unit * units
    measured = len(tokenizer.encode(prompt, add_special_tokens=True))
    if measured != target:
        raise RuntimeError(
            f"exact-token prompt measured {measured} with special tokens, "
            f"expected {target} (special-token overhead {overhead})")
    return prompt


def probe(name: str, output: dict[str, Any], fn) -> None:
    started = time.monotonic()
    try:
        detail = fn()
        # Collision-safe merge: `status` and `elapsed_seconds` are reserved
        # per-probe schema keys owned by this wrapper, so probe detail can
        # never clobber the pass/fail verdict or the timing. A numeric HTTP
        # status code an expected-rejection probe carries in detail (e.g.
        # {"status": 400}) is preserved as `http_status`, distinct from the
        # per-probe `status` verdict.
        entry = dict(detail) if isinstance(detail, dict) else {"detail": detail}
        raw_status = entry.pop("status", None)
        if isinstance(raw_status, int) and "http_status" not in entry:
            entry["http_status"] = raw_status
        entry["status"] = "passed"
        entry["elapsed_seconds"] = time.monotonic() - started
        output["probes"][name] = entry
    except BaseException as exc:
        output["probes"][name] = {
            "status": "failed",
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def aggregate_status(probes: dict[str, dict[str, Any]]) -> str:
    """Aggregate verdict: 'passed' only when every recorded probe passed."""
    if not probes:
        return "failed"
    return "passed" if all(p.get("status") == "passed" for p in probes.values()) else "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-cap", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument(
        "--skip-advanced",
        action="store_true",
        help="skip the P0 contract probes (max_completion_tokens alias/conflict, "
        "deferred-field rejections, sampling/reasoning overrides, multi-index "
        "tools, multi-turn loop, tool_choice none, stream usage omission, "
        "legacy completions shape/stream rejection)",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="validate the probe's own assertion logic only; "
                        "no server, no model")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if not args.model:
        parser.error("--model is required unless --self-test")
    if not args.output:
        parser.error("--output is required unless --self-test")

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "context_cap": args.context_cap,
        "started_epoch": time.time(),
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    models_url = f"{args.base_url.rstrip('/')}/models"
    completions_url = f"{args.base_url.rstrip('/')}/completions"

    def identity() -> dict[str, Any]:
        response, elapsed = request_json(models_url, timeout=30)
        ids = [entry.get("id") for entry in response.get("data", [])]
        if args.model not in ids:
            raise AssertionError(f"exact model ID {args.model!r} absent from {ids!r}")
        return {"served_ids": ids, "request_seconds": elapsed}

    probe("models_identity", result, identity)

    def status() -> dict[str, Any]:
        response, elapsed = request_json(f"{args.base_url.rstrip('/')}/mei/status", timeout=30)
        if response.get("status") != "ok":
            raise AssertionError(f"unexpected mei status response: {response!r}")
        return {
            "memory": response.get("memory"),
            "device": (response.get("memory") or {}).get("device"),
            "context_cap": response.get("context_cap"),
            "prefill_step_size": response.get("prefill_step_size"),
            "request_seconds": elapsed,
        }

    probe("mei_status", result, status)

    def plain() -> dict[str, Any]:
        response, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
            "temperature": 0,
            # Thinking models (Ornith is Qwen3.5-lineage) spend their first
            # 100+ tokens on the thinking preamble; the omlx-era 8-token
            # budget truncated before any visible content. 1024 covers a
            # full think+answer cycle while keeping the probe cheap.
            "max_tokens": 1024,
            "stream": False,
        }, timeout=args.timeout)
        choices = response.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        if not content.strip():
            raise AssertionError("plain completion returned empty content")
        return {"content": content, "usage": response.get("usage"), "request_seconds": elapsed}

    probe("plain_completion", result, plain)

    tools = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers and returns the sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }]
    tool_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to one tool."},
            {"role": "user", "content": "What is 15 + 27? You must use add_numbers to compute it."},
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "add_numbers"}},
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }

    def nonstream_tool() -> dict[str, Any]:
        response, elapsed = request_json(chat_url, tool_payload, timeout=args.timeout)
        return {"validated_call": validate_add_call(response), "raw_response": response, "request_seconds": elapsed}

    probe("tool_nonstreaming", result, nonstream_tool)

    def stream_tool() -> dict[str, Any]:
        response, elapsed, ttft = request_stream(chat_url, tool_payload, timeout=args.timeout)
        return {"validated_call": validate_add_call(response), "assembled_response": response, "request_seconds": elapsed, "client_ttft_seconds": ttft}

    probe("tool_streaming", result, stream_tool)

    # A prior assistant tool_call REPLAYED IN HISTORY, which is what every
    # multi-turn agent turn actually sends. The two probes above only test the
    # model EMITTING a call; neither ever feeds one back, and that blind spot
    # let a model through the gate 10/10 and then score 2/8 on hermes_ops.
    #
    # Laguna XS 2.1 (2026-09-09): its chat template throws
    # "Runtime error: Cannot iterate over non-iterable value" on ANY request
    # whose history contains an assistant message carrying tool_calls — with or
    # without a following tool result. Every task needing a second turn
    # therefore returned an empty response and was graded as a model failure.
    # Ornith on the same build handles 19-message tool-call histories fine, so
    # this is a per-model template defect that only a history-replay probe can
    # see. Cheap to run, and it fails loudly instead of silently costing a
    # whole benchmark.
    history_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to one tool."},
            {"role": "user", "content": "What is 15 + 27? You must use add_numbers to compute it."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_probe_0001", "type": "function",
                "function": {"name": "add_numbers", "arguments": "{\"a\": 15, \"b\": 27}"},
            }]},
            {"role": "tool", "tool_call_id": "call_probe_0001", "name": "add_numbers", "content": "42"},
        ],
        "tools": tools,
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }

    def tool_history_replay() -> dict[str, Any]:
        response, elapsed = request_json(chat_url, history_payload, timeout=args.timeout)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise AssertionError(
                "empty response to a replayed tool_call history "
                f"(finish_reason={choice.get('finish_reason')!r}, usage={response.get('usage')!r}) — "
                "the model produced nothing when its own prior tool call was sent back")
        if "42" not in content:
            raise AssertionError(f"tool result was not used in the answer: {content[:200]!r}")
        return {"content": content[:300], "usage": response.get("usage"), "request_seconds": elapsed}

    probe("tool_call_history_replay", result, tool_history_replay)

    # Streaming/non-streaming parity on plain text: same request both ways,
    # both must produce non-empty identical content and identical usage.
    parity_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: parity-ok"}],
        "temperature": 0,
        # Same thinking-preamble rationale as plain_completion: 1024 tokens
        # so both legs produce visible content to compare.
        "max_tokens": 1024,
    }

    def parity() -> dict[str, Any]:
        nonstream_response, _ = request_json(chat_url, parity_payload, timeout=args.timeout)
        stream_response, _, _ = request_stream(chat_url, parity_payload, timeout=args.timeout)
        ns_content = (nonstream_response["choices"][0]["message"].get("content") or "").strip()
        s_content = (stream_response["choices"][0]["message"].get("content") or "").strip()
        if not ns_content or not s_content:
            raise AssertionError(f"parity content empty: ns={ns_content!r} stream={s_content!r}")
        if ns_content != s_content:
            raise AssertionError(f"streaming/non-streaming content mismatch: {ns_content!r} != {s_content!r}")
        ns_usage = nonstream_response.get("usage") or {}
        s_usage = stream_response.get("usage") or {}
        if int(ns_usage.get("prompt_tokens", -1)) != int(s_usage.get("prompt_tokens", -2)):
            raise AssertionError(f"prompt token counts differ: {ns_usage!r} vs {s_usage!r}")
        return {
            "content": ns_content,
            "nonstream_usage": ns_usage,
            "stream_usage": s_usage,
        }

    probe("parity_stream_vs_nonstream", result, parity)

    # ---- P0 contract probes (skippable with --skip-advanced) -------------
    # Every probe below is part of the frozen OpenAI-compatible subset in
    # docs/OPENAI-COMPATIBILITY.md. Server-deterministic probes (rejections,
    # aliases, shapes, stream framing) are gates; the model-behavior probes
    # (multi-index tools, multi-turn loop, reasoning toggles) assert the
    # contract with the model that is staged, and any per-model limitation is
    # recorded in the artifact, not hidden.

    def max_tokens_alias() -> dict[str, Any]:
        # max_completion_tokens alone must be honored as the generation cap.
        response, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: alias-ok"}],
            "temperature": 0,
            "max_completion_tokens": 32,
            "stream": False,
        }, timeout=args.timeout)
        usage = response.get("usage") or {}
        if int(usage.get("completion_tokens", -1)) > 32:
            raise AssertionError(f"max_completion_tokens=32 not honored: {usage!r}")
        return {"usage": usage, "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("max_completion_tokens_alias", result, max_tokens_alias)

    def max_tokens_conflict() -> dict[str, Any]:
        status, body, parsed = request_raw(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "max_completion_tokens": 16,
        }, timeout=30)
        error = (parsed or {}).get("error") or {}
        if status != 400:
            raise AssertionError(f"conflicting max-token forms: expected 400, got {status}: {body[:200]!r}")
        if "conflict" not in error.get("message", "") or "max_completion_tokens" not in error.get("message", ""):
            raise AssertionError(f"conflict error must name both forms: {body[:300]!r}")
        return {"status": status, "error": error}

    if not args.skip_advanced:
        probe("max_tokens_conflict_rejected", result, max_tokens_conflict)

    def deferred_rejected() -> dict[str, Any]:
        # Each deferred platform feature must be a loud 400 that names the
        # field — never a silently-ignored request.
        cases = [
            ("response_format", {"type": "json_object"}),
            ("n", 2),
            ("logprobs", True),
            ("parallel_tool_calls", False),
            ("developer_role", None),  # special-cased below
        ]
        checked: dict[str, Any] = {}
        for field, value in cases:
            payload: dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": "hi"}],
            }
            if field == "developer_role":
                payload["messages"] = [{"role": "developer", "content": "hi"}]
            else:
                payload[field] = value
            status, body, parsed = request_raw(chat_url, payload, timeout=30)
            error = (parsed or {}).get("error") or {}
            if status != 400:
                raise AssertionError(f"{field}: expected 400, got {status}: {body[:200]!r}")
            if error.get("type") != "invalid_request_error" and error.get("type") != "engine_error":
                raise AssertionError(f"{field}: unexpected error type {error.get('type')!r}: {body[:200]!r}")
            checked[field] = {"status": status, "error_message": error.get("message")}
        return checked

    if not args.skip_advanced:
        probe("deferred_fields_rejected", result, deferred_rejected)

    sampling_text = "Reply with exactly: deterministic-override"

    def sampling_override() -> dict[str, Any]:
        # temperature 0 + fixed seed twice: the request OVERRIDES the server
        # sampling defaults (greedy is byte-stable), and the override must not
        # leak between requests (the second run must not inherit anything —
        # any leaked sampling state would make the two runs drift).
        # check_sampling_override is channel-agnostic: a thinking model may
        # spend this bounded budget entirely in reasoning_content, so output
        # in EITHER visible channel satisfies the contract (only a run with
        # nothing at all is degenerate).
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": sampling_text}],
            "temperature": 0,
            "seed": 7,
            "max_tokens": 64,
            "stream": False,
        }
        first, _ = request_json(chat_url, payload, timeout=args.timeout)
        second, elapsed = request_json(chat_url, payload, timeout=args.timeout)
        detail = check_sampling_override(first, second)
        return {**detail, "usage": first.get("usage"), "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("sampling_override_deterministic", result, sampling_override)

    def reasoning_none() -> dict[str, Any]:
        # reasoning_effort "none" must disable the thinking channel: no
        # reasoning_content field at all, visible content within budget.
        response, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: no-think"}],
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        }, timeout=args.timeout)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise AssertionError("reasoning_effort=none returned no visible content")
        if "reasoning_content" in message:
            raise AssertionError("reasoning_effort=none still exposed reasoning_content")
        return {"content": content[:200], "usage": response.get("usage"), "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("reasoning_effort_none_suppresses_thinking", result, reasoning_none)

    def reasoning_high() -> dict[str, Any]:
        # reasoning_effort high must be accepted and produce either content or
        # reasoning (model/template dependent which; the field shape must be
        # valid either way).
        response, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: think-hard"}],
            "reasoning_effort": "high",
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }, timeout=args.timeout)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = message.get("reasoning_content")
        if not content and not reasoning:
            raise AssertionError("reasoning_effort=high produced neither content nor reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise AssertionError(f"reasoning_content must be a string when present: {reasoning!r}")
        return {"has_reasoning": reasoning is not None, "usage": response.get("usage"), "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("reasoning_effort_high_accepted", result, reasoning_high)

    two_tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Adds two numbers and returns the sum.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply_numbers",
                "description": "Multiplies two numbers and returns the product.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    multi_tool_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with two tools."},
            {"role": "user", "content": (
                "Compute 15 + 27 with add_numbers AND 15 * 27 with multiply_numbers. "
                "You MUST call BOTH tools in one response.")},
        ],
        "tools": two_tools,
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
    }

    def validate_two_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        choices = response.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        calls = message.get("tool_calls") or []
        names = [call.get("function", {}).get("name") for call in calls]
        if len(set(names)) < 2:
            raise AssertionError(f"expected calls to BOTH tools, got {names!r}")
        if choices[0].get("finish_reason") != "tool_calls":
            raise AssertionError(f"expected finish_reason tool_calls, got {choices[0].get('finish_reason')!r}")
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") not in ("add_numbers", "multiply_numbers"):
                raise AssertionError(f"unexpected tool name {function.get('name')!r}")
            arguments = json.loads(function.get("arguments") or "{}")
            if set(arguments) != {"a", "b"}:
                raise AssertionError(f"argument mismatch: {arguments!r}")
            if not isinstance(arguments["a"], (int, float)) or not isinstance(arguments["b"], (int, float)):
                raise AssertionError(f"non-numeric arguments: {arguments!r}")
            ids = [c.get("id") for c in calls]
            if any(not i for i in ids) or len(set(ids)) != len(ids):
                raise AssertionError(f"tool calls must carry distinct ids: {ids!r}")
        return calls

    def multi_tool_nonstream() -> dict[str, Any]:
        response, elapsed = request_json(chat_url, multi_tool_payload, timeout=args.timeout)
        calls = validate_two_calls(response)
        return {"call_count": len(calls), "names": [c["function"]["name"] for c in calls],
                "usage": response.get("usage"), "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("tool_multi_index_nonstream", result, multi_tool_nonstream)

    def multi_tool_stream() -> dict[str, Any]:
        response, elapsed, ttft = request_stream(chat_url, multi_tool_payload, timeout=args.timeout)
        calls = validate_two_calls(response)
        usage = response.get("usage") or {}
        if not usage.get("prompt_tokens"):
            raise AssertionError("streamed multi-call run missing usage chunk: %r" % usage)
        return {"call_count": len(calls), "names": [c["function"]["name"] for c in calls],
                "usage": usage, "client_ttft_seconds": ttft, "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("tool_multi_index_stream", result, multi_tool_stream)

    def tool_choice_none() -> dict[str, Any]:
        # tool_choice "none" with tools present must suppress tool emission.
        response, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer directly."}],
            "tools": tools,
            "tool_choice": "none",
            "temperature": 0,
            "max_tokens": 128,
            "stream": False,
        }, timeout=args.timeout)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        if message.get("tool_calls"):
            raise AssertionError(f"tool_choice=none still emitted tool calls: {message['tool_calls']!r}")
        if not (message.get("content") or "").strip():
            raise AssertionError("tool_choice=none returned empty content")
        return {"content": (message.get("content") or "")[:120], "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("tool_choice_none_suppresses_tools", result, tool_choice_none)

    def multi_turn_loop() -> dict[str, Any]:
        # A REAL two-turn agentic loop: turn 1 emits a tool call, the client
        # executes it locally, turn 2 replays history + result and must use
        # the result. (The history_replay probe above covers the replay shape
        # only; this one runs the actual continuation with usage tracking.)
        turn1, elapsed = request_json(chat_url, tool_payload, timeout=args.timeout)
        calls = validate_add_call(turn1)
        result_text = str(calls["arguments"]["a"] + calls["arguments"]["b"])
        history = tool_payload["messages"] + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_p0_loop", "type": "function",
                 "function": {"name": "add_numbers", "arguments": json.dumps(calls["arguments"])}}]},
            {"role": "tool", "tool_call_id": "call_p0_loop", "name": "add_numbers", "content": result_text},
        ]
        turn2, elapsed2 = request_json(chat_url, {
            "model": args.model,
            "messages": history,
            "tools": tools,
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        }, timeout=args.timeout)
        content = (turn2.get("choices") or [{}])[0].get("message") or {}
        content_text = (content.get("content") or "").strip()
        if not content_text:
            raise AssertionError("continuation turn returned empty content")
        if result_text not in content_text:
            raise AssertionError(f"continuation did not use the tool result {result_text!r}: {content_text[:200]!r}")
        usage2 = turn2.get("usage") or {}
        return {
            "turn1_call": calls,
            "turn2_content": content_text[:200],
            "turn2_usage": usage2,
            "cached_tokens": (usage2.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "request_seconds": elapsed + elapsed2,
        }

    if not args.skip_advanced:
        probe("tool_multi_turn_continuation", result, multi_turn_loop)

    def stream_usage_omitted() -> dict[str, Any]:
        # A stream that never asked for usage must not receive a usage chunk —
        # stream_options.include_usage is the ONLY way usage appears mid-stream
        # — and a SUCCESSFUL stream must still terminate with a finish frame
        # and `data: [DONE]` (the terminator is the data payload of the final
        # event, never a bare [DONE] line). A genuine mid-stream error instead
        # ends with an error frame and NO [DONE]; that is reported as the
        # stream failure it is, not as a missing-terminator defect. The
        # full contract lives in check_stream_without_include_usage.
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: no-usage"}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
        }
        _, raw_lines = request_stream_plain(chat_url, payload, timeout=args.timeout)
        return check_stream_without_include_usage(raw_lines)

    if not args.skip_advanced:
        probe("stream_usage_omitted_without_include_usage", result, stream_usage_omitted)

    def completions_shape() -> dict[str, Any]:
        # The minimal legacy path must answer in the OpenAI text_completion
        # shape with the same usage block the chat path reports.
        response, elapsed = request_json(completions_url, {
            "model": args.model,
            "prompt": "Say hello",
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
        }, timeout=args.timeout)
        if response.get("object") != "text_completion":
            raise AssertionError(f"expected object text_completion, got {response.get('object')!r}")
        choices = response.get("choices") or []
        text = (choices[0].get("text") or "").strip() if choices else ""
        if not text:
            raise AssertionError("legacy completion returned empty text")
        if not (response.get("usage") or {}).get("prompt_tokens"):
            raise AssertionError("legacy completion missing usage")
        return {"object": response.get("object"), "text": text[:120], "usage": response.get("usage"),
                "request_seconds": elapsed}

    if not args.skip_advanced:
        probe("completions_text_shape", result, completions_shape)

    def completions_stream_rejected() -> dict[str, Any]:
        status, body, parsed = request_raw(completions_url, {
            "model": args.model,
            "prompt": "hi",
            "stream": True,
        }, timeout=30)
        error = (parsed or {}).get("error") or {}
        if status != 400 or "stream" not in error.get("message", ""):
            raise AssertionError(f"expected 400 stream rejection, got {status}: {body[:200]!r}")
        return {"status": status, "error_message": error.get("message")}

    if not args.skip_advanced:
        probe("completions_stream_rejected", result, completions_stream_rejected)

    cache_prompt = "Summarize the final instruction only. " + ("stable prefix text " * 2048) + " Final instruction: reply cache-ready."
    if not args.skip_cache:
        for repetition in (1, 2):
            def cache_request(repetition=repetition) -> dict[str, Any]:
                response, elapsed = request_json(chat_url, {
                    "model": args.model,
                    "messages": [{"role": "user", "content": cache_prompt}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": False,
                }, timeout=args.timeout)
                return {"usage": response.get("usage"), "request_seconds": elapsed}
            probe(f"cache_repeat_{repetition}", result, cache_request)

        # The agentic pattern the slot is built for: identical system prompt,
        # growing transcript. Turn 2 must reuse the turn-1 prefix (strict
        # extension => cached_tokens ≈ turn-1 prompt tokens).
        system_cache_prompt = ("system stability marker " * 256)

        def growing_turn(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
            response, elapsed = request_json(chat_url, {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            }, timeout=args.timeout)
            usage = response.get("usage") or {}
            choices = response.get("choices") or []
            assistant_content = ""
            if choices:
                assistant_content = ((choices[0].get("message") or {}).get("content") or "")
            return {
                "usage": usage,
                "cached_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "request_seconds": elapsed,
            }, assistant_content

        turn1 = [{"role": "system", "content": system_cache_prompt},
                 {"role": "user", "content": "First instruction: answer nothing yet."}]
        turn1_replies: dict[str, str] = {}

        def cache_growing_turn1() -> dict[str, Any]:
            detail, reply = growing_turn(turn1)
            turn1_replies["assistant_content"] = reply
            return detail

        probe("cache_growing_turn1", result, cache_growing_turn1)

        def cache_growing_turn2() -> dict[str, Any]:
            # The agentic pattern: turn 2 = turn 1 + assistant reply + next
            # user turn. The coordinator strips the generation-prompt suffix
            # at store time, so the re-rendered turn 2 strictly extends the
            # stored turn-1 prefix and the whole turn-1 prefix must be
            # restored from the KV cache (cached_tokens ≈ turn-1 tokens).
            detail, _ = growing_turn(turn1 + [
                {"role": "assistant", "content": turn1_replies.get("assistant_content", "")},
                {"role": "user", "content": "Second instruction: reply cache-reuse-ok."},
            ])
            expected_cache = (detail["usage"].get("prompt_tokens") or 0) >= 260
            slot_cached = detail["cached_tokens"] >= 250
            if not (expected_cache and slot_cached):
                raise AssertionError(
                    f"growing-transcript reuse failed: cached_tokens={detail['cached_tokens']} usage={detail['usage']!r}")
            return detail

        probe("cache_growing_turn2_reuses_slot", result, cache_growing_turn2)

    if not args.skip_context:
        if args.tokenizer is None:
            result["probes"]["context_exact_cap"] = {
                "status": "skipped",
                "elapsed_seconds": 0,
                "error": "no --tokenizer path provided",
            }
            result["probes"]["context_over_cap_rejected"] = {
                "status": "skipped",
                "elapsed_seconds": 0,
                "error": "no --tokenizer path provided",
            }
        else:
            prompt_at_cap = exact_prompt(args.tokenizer, args.context_cap)

            def context_at_cap() -> dict[str, Any]:
                response, elapsed = request_json(completions_url, {
                    "model": args.model,
                    "prompt": prompt_at_cap,
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                }, timeout=args.timeout)
                usage = response.get("usage") or {}
                if int(usage.get("prompt_tokens", -1)) != args.context_cap:
                    raise AssertionError(f"usage did not confirm {args.context_cap} prompt tokens: {usage!r}")
                if int(usage.get("completion_tokens", 0)) < 1:
                    raise AssertionError(f"no completion at exact context cap: {usage!r}")
                return {"usage": usage, "request_seconds": elapsed}

            probe("context_exact_cap", result, context_at_cap)

            def context_over_cap() -> dict[str, Any]:
                over = exact_prompt(args.tokenizer, args.context_cap + 1)
                try:
                    response, elapsed = request_json(completions_url, {
                        "model": args.model,
                        "prompt": over,
                        "temperature": 0,
                        "max_tokens": 1,
                        "stream": False,
                    }, timeout=args.timeout)
                except RuntimeError as exc:
                    if "HTTP 400" not in str(exc):
                        raise
                    return {
                        "rejected_as_expected": True,
                        # Numeric HTTP code, distinct from the per-probe
                        # `status` verdict (collision-safe merge in probe()).
                        "http_status": 400,
                        "error": str(exc),
                    }
                raise AssertionError(f"{args.context_cap + 1}-token prompt unexpectedly succeeded in {elapsed:.3f}s: {response!r}")

            probe("context_over_cap_rejected", result, context_over_cap)

            def chat_over_cap() -> dict[str, Any]:
                # Chat prompts pay template overhead on top of the raw text,
                # so wrapping the exact over-cap prompt in a user message is
                # guaranteed over the cap. Non-streaming must 400 cleanly.
                over = exact_prompt(args.tokenizer, args.context_cap + 1)
                status, body, parsed = request_raw(chat_url, {
                    "model": args.model,
                    "messages": [{"role": "user", "content": over}],
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                }, timeout=args.timeout)
                if status != 400 or "context cap" not in ((parsed or {}).get("error") or {}).get("message", ""):
                    raise AssertionError(f"chat over-cap: expected 400, got {status}: {body[:200]!r}")
                return {"status": status, "error": (parsed or {}).get("error")}

            probe("context_chat_over_cap_rejected", result, chat_over_cap)

            def chat_stream_over_cap() -> dict[str, Any]:
                # The streaming preflight must reject BEFORE any SSE bytes:
                # a clean JSON 400, not an error frame on a 200 stream.
                over = exact_prompt(args.tokenizer, args.context_cap + 1)
                status, body, parsed = request_raw(chat_url, {
                    "model": args.model,
                    "messages": [{"role": "user", "content": over}],
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": True,
                }, timeout=args.timeout)
                if status != 400:
                    raise AssertionError(f"streaming over-cap: expected 400, got {status}: {body[:200]!r}")
                if body.startswith("data:"):
                    raise AssertionError("streaming over-cap answered with SSE frames instead of a JSON 400")
                return {"status": status, "error": (parsed or {}).get("error")}

            probe("context_chat_stream_over_cap_rejected", result, chat_stream_over_cap)

    result["finished_epoch"] = time.time()
    result["status"] = aggregate_status(result["probes"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())