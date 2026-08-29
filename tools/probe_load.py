#!/usr/bin/env python3
"""Load + hello + short-decode measurement for one Mei model process.

Records the exact evidence for a model-load attempt: startup log lines
(model load active/peak bytes, device, limits, cache topology), a live
/v1/mei/status snapshot, then one short hello completion and one measured
decode run. Output is a JSON artifact under artifacts/; exit 0 only when
the model loaded AND produced a non-empty completion.

Usage:
  probe_load.py --base-url http://127.0.0.1:8024/v1 --model ID \
    --server-log PATH --output artifacts/load-<model>-<ts>.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 1800) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "started_epoch": time.time(),
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    status_url = f"{args.base_url.rstrip('/')}/mei/status"

    def record(name: str, fn) -> None:
        started = time.monotonic()
        try:
            detail = fn()
            result["probes"][name] = {"status": "passed", "elapsed_seconds": round(time.monotonic() - started, 3), **detail}
            print(f"[load] {name}: passed", flush=True)
        except BaseException as exc:  # noqa: BLE001
            result["probes"][name] = {"status": "failed", "elapsed_seconds": round(time.monotonic() - started, 3), "error": f"{type(exc).__name__}: {exc}"}
            print(f"[load] {name}: FAILED {exc}", flush=True)

    def status() -> dict[str, Any]:
        return {"status_body": request_json(status_url, timeout=30)}

    record("status", status)

    def hello() -> dict[str, Any]:
        response = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        }, timeout=args.timeout)
        content = (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise AssertionError("hello completion returned empty content")
        return {"content_tail": content[-120:], "usage": response.get("usage")}

    record("hello", hello)

    def decode() -> dict[str, Any]:
        response = request_json(chat_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Count from 1 to 10, one per line."}],
            "temperature": 0,
            "max_tokens": args.decode_tokens,
            "stream": False,
        }, timeout=args.timeout)
        usage = response.get("usage") or {}
        return {
            "usage": usage,
            "decode_tokens_per_second_engine": usage.get("tokens_per_second"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    record("short_decode", decode)

    # Startup log evidence (load memory, device, limits, topology).
    log_text = ""
    if args.server_log.exists():
        log_text = args.server_log.read_text(errors="replace")
    result["server_log_tail"] = log_text[-6000:]

    result["finished_epoch"] = time.time()
    result["status"] = (
        "passed"
        if result["probes"].get("hello", {}).get("status") == "passed"
        else "failed"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())