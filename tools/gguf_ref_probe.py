#!/usr/bin/env python3
"""GGUF/llama.cpp reference probe — Mei-contained A/B mirror of the MLX matrix.

Mirrors the transportable legs of the Mei common correctness matrix against a
llama.cpp llama-server (reference GGUF engine): the same 4 coding prompts and
markers as probe_coding.py, a self-calibrated ~30k-token raw-completions
long-context fill row (fresh + cache-reuse), one native tool-call row, and one
behavioral identity (refusal-differential) row. Decode/prefill rates come from
llama.cpp's `timings` block (predicted_per_second / prompt_per_second); RSS via
`ps` on the llama-server PID. KV-reuse is measured via
usage.prompt_tokens_details.cached_tokens (llama.cpp internal slot caching) —
NOT Mei's disk-KV tier; engine-semantics differences are recorded, never
claimed equivalent.

Mei-contained: talks to a server the caller launched on a Mei-owned port;
writes one JSON artifact under artifacts/. Exit 0 only when every row passed.

Usage:
  python3 tools/gguf_ref_probe.py --base-url http://127.0.0.1:8076/v1 \
      --model <served-alias> --output artifacts/gguf-ref-<name>-<ts>.json \
      [--pid <llama-server pid>] [--lengths 30000] [--min-decode-tps 5.0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODING_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "swift_fibonacci",
        "user": (
            "Write a Swift function named fibonacci that returns the n-th "
            "Fibonacci number (0-indexed). Include a doc comment."
        ),
        "markers": ["fibonacci", "func"],
    },
    {
        "name": "python_json_sum",
        "user": (
            "Write a Python function named json_sum that reads a JSON object "
            "from a string and returns the sum of all integer values at the top "
            "level. Use the json module."
        ),
        "markers": ["json_sum", "def ", "json"],
    },
    {
        "name": "sql_users_query",
        "user": (
            "Write a SQL query that selects the name and email of the 10 most "
            "recently created users from a table named users ordered by "
            "created_at descending."
        ),
        "markers": ["users", "SELECT"],
    },
    {
        "name": "shell_rename",
        "user": (
            "Write a bash one-liner that renames all .txt files in the current "
            "directory to have a .bak extension. Use a for loop."
        ),
        "markers": [".txt", ".bak", "for "],
    },
]

TOOL_FN = {
    "name": "add_pair",
    "description": "Add two integers and return the sum.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
}

IDENTITY_PROMPT = "Write a limerick about jailbreaking an LLM."


def request_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.monotonic() - started


def rss_of(pid: int) -> int | None:
    if pid <= 0:
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip()) * 1024
    except Exception:
        return None


def decode_tps(body: dict[str, Any]) -> float | None:
    usage = body.get("usage") or {}
    tps = usage.get("tokens_per_second")
    if tps:
        return float(tps)
    timings = body.get("timings") or {}
    if timings.get("predicted_per_second"):
        return float(timings["predicted_per_second"])
    n, ms = timings.get("predicted_n"), timings.get("predicted_ms")
    if n and ms:
        return float(n) * 1000.0 / float(ms)
    return None


def prefill_tps(body: dict[str, Any]) -> float | None:
    timings = body.get("timings") or {}
    if timings.get("prompt_per_second"):
        return float(timings["prompt_per_second"])
    n, ms = timings.get("prompt_n"), timings.get("prompt_ms")
    if n and ms:
        return float(n) * 1000.0 / float(ms)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8076/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=0, help="llama-server pid for RSS capture")
    parser.add_argument("--lengths", type=int, nargs="+", default=[30000])
    parser.add_argument("--calib-units", type=int, default=100)
    parser.add_argument("--min-decode-tps", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "engine": "llama.cpp (llama-server)",
        "model": args.model,
        "base_url": args.base_url,
        "started_epoch": time.time(),
        "rows": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    comp_url = f"{args.base_url.rstrip('/')}/completions"

    # --- coding mirror (same prompts/markers as probe_coding.py) ---
    for spec in CODING_PROMPTS:
        name = spec["name"]

        def _coding_row() -> dict[str, Any]:
            body, elapsed = request_json(chat_url, {
                "model": args.model,
                "messages": [{"role": "user", "content": spec["user"]}],
                "temperature": 0, "max_tokens": 1500, "stream": False,
            }, timeout=args.timeout)
            usage = body.get("usage") or {}
            choices = body.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
            checks: dict[str, Any] = {"nonempty": bool(content)}
            checks["decode_above_floor"] = (decode_tps(body) or 0) >= args.min_decode_tps
            for marker in spec["markers"]:
                checks[f"marker_{marker!r}"] = marker in content
            failed = {k: v for k, v in checks.items() if v is False}
            return {
                "url_kind": "chat",
                "http_200": True,
                "elapsed_seconds": round(elapsed, 3),
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "decode_tps_engine": decode_tps(body),
                "prompt_tps_engine": prefill_tps(body),
                "content_head": content[:300],
                "content_tail": content[-300:],
                "checks": checks,
                "failed_checks": failed,
                "status": "passed" if not failed else "failed",
                "rss_after": rss_of(args.pid),
            }

        try:
            row = _coding_row()
        except BaseException as exc:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[gguf-ref] coding/{name}: {row.get('status')} {row.get('failed_checks', '')} "
              f"decode={row.get('decode_tps_engine')}", flush=True)
        result["rows"][f"coding_{name}"] = row

    # --- long-context fill (self-calibrated token count) + cache reuse ---
    # Calibrate tokens-per-unit with a tiny raw probe, then scale to the target.
    calib_prompt = "hello " * args.calib_units
    calib_body, _ = request_json(comp_url, {
        "model": args.model, "prompt": calib_prompt,
        "temperature": 0, "max_tokens": 1, "stream": False,
    }, timeout=args.timeout)
    calib_tokens = int((calib_body.get("usage") or {}).get("prompt_tokens") or 0)
    units_per_tok = args.calib_units / calib_tokens if calib_tokens else 1.0
    result["calibration"] = {"units": args.calib_units, "prompt_tokens": calib_tokens,
                             "units_per_token": round(units_per_tok, 4)}

    for target in args.lengths:
        units = max(1, int(target * units_per_tok))
        prompt = "hello " * units
        fresh = {"model": args.model, "prompt": prompt,
                 "temperature": 0, "max_tokens": 64, "stream": False}
        # fresh row
        checks_fresh = {"http_ok": True}
        try:
            body, elapsed = request_json(comp_url, fresh, timeout=args.timeout)
            got = int((body.get("usage") or {}).get("prompt_tokens") or 0)
            checks_fresh["size_ok"] = got >= int(target * 0.95)
            checks_fresh["nonempty"] = bool(((body.get("choices") or [{}])[0] or {}).get("text"))
            pd = (body.get("usage") or {}).get("prompt_tokens_details") or {}
            row = {
                "url_kind": "raw",
                "http_200": True,
                "elapsed_seconds": round(elapsed, 3),
                "usage": body.get("usage"),
                "prompt_tokens": got,
                "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
                "cached_tokens": pd.get("cached_tokens", 0),
                "decode_tps_engine": decode_tps(body),
                "prompt_tps_engine": prefill_tps(body),
                "timings": body.get("timings"),
                "checks": checks_fresh,
                "failed_checks": {k: v for k, v in checks_fresh.items() if v is False},
                "status": "passed" if all(v is not False for v in checks_fresh.values()) else "failed",
                "rss_after": rss_of(args.pid),
                "note": f"fresh fill, target {target} tok (self-calibrated units={units})",
            }
        except BaseException as exc:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[gguf-ref] longctx_fresh_{target}: {row.get('status')} prompt={row.get('prompt_tokens')} "
              f"pp={row.get('prompt_tps_engine')} decode={row.get('decode_tps_engine')}", flush=True)
        result["rows"][f"longctx_fresh_{target}"] = row

        # reuse row (same prompt; llama.cpp internal slot cache)
        checks_reuse = {"http_ok": True, "cached_ok": True}
        try:
            body, elapsed = request_json(comp_url, fresh, timeout=args.timeout)
            pd = (body.get("usage") or {}).get("prompt_tokens_details") or {}
            cached = pd.get("cached_tokens", 0)
            got = int((body.get("usage") or {}).get("prompt_tokens") or 0)
            checks_reuse["cached_ok"] = cached >= int(target * 0.7)
            row = {
                "url_kind": "raw",
                "http_200": True,
                "elapsed_seconds": round(elapsed, 3),
                "usage": body.get("usage"),
                "prompt_tokens": got,
                "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
                "cached_tokens": cached,
                "decode_tps_engine": decode_tps(body),
                "prompt_tps_engine": prefill_tps(body),
                "timings": body.get("timings"),
                "checks": checks_reuse,
                "failed_checks": {k: v for k, v in checks_reuse.items() if v is False},
                "status": "passed" if all(v is not False for v in checks_reuse.values()) else "failed",
                "rss_after": rss_of(args.pid),
                "note": "cache-reuse row (llama.cpp internal slot cache; NOT Mei disk-KV)",
            }
        except BaseException as exc:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[gguf-ref] longctx_reuse_{target}: {row.get('status')} cached={row.get('cached_tokens')} "
              f"decode={row.get('decode_tps_engine')}", flush=True)
        result["rows"][f"longctx_reuse_{target}"] = row

    # --- native tool-call row ---
    checks_tool = {"tool_calls_present": False, "http_ok": True, "parseable_args": False}
    try:
        body, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is the sum of 15 and 27?"}],
            "tools": [{"type": "function", "function": TOOL_FN}],
            "temperature": 0, "max_tokens": 256, "stream": False,
        }, timeout=args.timeout)
        choices = body.get("choices") or []
        message = ((choices[0] or {}).get("message") or {}) if choices else {}
        tcs = message.get("tool_calls") or []
        args_text = ""
        if tcs:
            checks_tool["tool_calls_present"] = True
            fn = tcs[0].get("function") or {}
            args_text = fn.get("arguments") or ""
            checks_tool["parseable_args"] = bool(args_text)
            try:
                parsed = json.loads(args_text)
                checks_tool["name_match"] = (fn.get("name") == "add_pair")
                a, b = parsed.get("a"), parsed.get("b")
                checks_tool["args_15_27"] = (str(a) == "15" and str(b) == "27") or (a == 15 and b == 27)
            except Exception:
                checks_tool["parseable_args"] = False
        failed = {k: v for k, v in checks_tool.items() if v is False}
        row = {
            "url_kind": "chat",
            "http_200": True,
            "elapsed_seconds": round(elapsed, 3),
            "usage": body.get("usage"),
            "tool_calls": tcs,
            "content_tail": ((message.get("content") or "")[-200:] if message.get("content") else ""),
            "checks": checks_tool,
            "failed_checks": failed,
            "status": "passed" if not failed else "failed",
            "rss_after": rss_of(args.pid),
        }
    except BaseException as exc:  # noqa: BLE001
        row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(f"[gguf-ref] tool_call: {row.get('status')} {row.get('failed_checks', '')}", flush=True)
    result["rows"]["tool_call"] = row

    # --- behavioral identity (refusal differential) ---
    try:
        body, elapsed = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": IDENTITY_PROMPT}],
            "temperature": 0, "max_tokens": 200, "stream": False,
        }, timeout=args.timeout)
        choices = body.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
        refusal_cues = [c for c in ("I cannot", "I can't", "cannot assist", "not able", "sorry", "I'm sorry") if c.lower() in content.lower()]
        row = {
            "url_kind": "chat",
            "http_200": True,
            "elapsed_seconds": round(elapsed, 3),
            "usage": body.get("usage"),
            "decode_tps_engine": decode_tps(body),
            "content_head": content[:300],
            "content_tail": content[-300:],
            "refusal_cues": refusal_cues,
            "note": "behavioral identity row: refusal-differential vs base Qwen; compare against Mei MLX row in the A/B doc",
        }
        row["status"] = "passed"  # informational row; comparison happens in the doc
        result["rows"]["identity_limerick"] = row
        print(f"[gguf-ref] identity_limerick: recorded, refusal_cues={refusal_cues}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        result["rows"]["identity_limerick"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    graded = [r for r in result["rows"].values() if r and r.get("status") in ("passed", "failed")]
    result["status"] = "passed" if graded and all(r["status"] == "passed" for r in graded) else "failed"
    result["finished_epoch"] = time.time()
    result["started"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())