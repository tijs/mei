#!/usr/bin/env python3
"""Coding-prompt probe for one isolated Mei model (Qwen3.8 parity unit).

Deterministic, artifact-only coding checks — no LLM judge. Sends a fixed
set of coding prompts through /v1/chat/completions and requires:
  - HTTP 200, non-empty completion
  - engine-reported decode tok/s above a sanity floor (--min-decode-tps)
  - a deterministic marker per prompt (substring check on a tag the model
    must emit, e.g. a function signature); markers are deliberately loose
    (name + a structural token) to avoid overfitting a specific style

Each row records usage (token counts, mei_memory_*, engine tok/s, prefill
pps) so the artifact doubles as a short-context decode/prefill datapoint.

Usage:
  probe_coding.py --base-url http://127.0.0.1:8024/v1 --model ID \\
      --output artifacts/probe-coding-<ts>.json [--min-decode-tps 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mei import request_json  # noqa: E402  (shared client helpers)

PROMPTS: list[dict[str, Any]] = [
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-decode-tps", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "min_decode_tps": args.min_decode_tps,
        "started_epoch": time.time(),
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"

    for spec in PROMPTS:
        name = spec["name"]
        started = time.monotonic()
        try:
            response, elapsed = request_json(chat_url, {
                "model": args.model,
                "messages": [{"role": "user", "content": spec["user"]}],
                "temperature": 0,
                "max_tokens": 1500,
                "stream": False,
            }, timeout=args.timeout)
            usage = response.get("usage") or {}
            choices = response.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
            decode_tps = usage.get("tokens_per_second") or 0
            checks = {
                "nonempty": bool(content),
                "decode_above_floor": decode_tps >= args.min_decode_tps,
            }
            for marker in spec["markers"]:
                checks[f"marker_{marker!r}"] = marker in content
            failed = {k: v for k, v in checks.items() if v is False}
            result["probes"][name] = {
                "status": "passed" if not failed else "failed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "content_head": content[:200],
                "content_tail": content[-200:],
                "usage": usage,
                "decode_tokens_per_second_engine": decode_tps,
                "checks": checks,
                "failed_checks": failed,
            }
            print(f"[coding] {name}: {'PASSED' if not failed else 'FAILED'} {failed}", flush=True)
        except BaseException as exc:  # noqa: BLE001
            result["probes"][name] = {
                "status": "failed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[coding] {name}: ERROR {exc}", flush=True)

    result["finished_epoch"] = time.time()
    result["status"] = "passed" if result["probes"] and all(
        p["status"] == "passed" for p in result["probes"].values()
    ) else "failed"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())