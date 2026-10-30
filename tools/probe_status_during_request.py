#!/usr/bin/env python3
"""Measure status responsiveness during one isolated completion request.

Concurrent request time is not proof of model prefill: correlate samples with
server-side prefill markers. Run only against an explicitly selected test
server; never starts/stops a server or modifies its cache or configuration.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def request(url, payload=None, timeout=5):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def observe(
    base, model, prompt, *, interval=0.5, status_timeout=5, request_timeout=300
):
    if interval <= 0 or status_timeout <= 0 or request_timeout <= 0:
        raise ValueError("timeouts and interval must be positive")
    base = base.rstrip("/")
    done = threading.Event()
    ready = threading.Event()
    result = {}
    start_utc = datetime.now(UTC).isoformat()
    start = time.monotonic()

    def completion():
        ready.set()
        try:
            reply = request(
                base + "/v1/completions",
                {"model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0},
                timeout=request_timeout,
            )
            result.update(
                ok=True, elapsed_s=time.monotonic() - start, usage=reply.get("usage")
            )
        except (OSError, ValueError, urllib.error.URLError) as exc:
            result.update(
                ok=False, elapsed_s=time.monotonic() - start, error=type(exc).__name__
            )
        finally:
            done.set()

    thread = threading.Thread(target=completion, daemon=True)
    thread.start()
    ready.wait()
    samples = []
    while not done.is_set():
        begin = time.monotonic()
        sample: dict[str, object]
        try:
            status = request(base + "/v1/mei/status", timeout=status_timeout)
            sample = {"ok": status.get("status") == "ok"}
        except (OSError, ValueError, urllib.error.URLError) as exc:
            sample = {"ok": False, "error": type(exc).__name__}
        sample.update(start_s=begin - start, elapsed_s=time.monotonic() - begin)
        samples.append(sample)
        done.wait(interval)
    thread.join()
    return {
        "start_utc": start_utc,
        "completion": result,
        "status_samples": samples,
        "note": "correlate timestamps with server prefill logs; request overlap alone is not prefill proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--status-timeout", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args()
    result = observe(
        args.base,
        args.model,
        args.prompt_file.read_text(encoding="utf-8"),
        interval=args.interval,
        status_timeout=args.status_timeout,
        request_timeout=args.request_timeout,
    )
    print(json.dumps(result, indent=2))
    if (
        not result["completion"]["ok"]
        or not result["status_samples"]
        or any(not sample["ok"] for sample in result["status_samples"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
