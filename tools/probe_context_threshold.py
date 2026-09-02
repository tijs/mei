#!/usr/bin/env python3
"""Probe at which context length a Mei server dies during chunked prefill.

Bounded, exact-token context threshold finder for the todo-7 common matrix.
Builds an exact N-token prompt with the model's own tokenizer (reusing the
probe_mei exact_prompt convention), POSTs a short non-streaming completion,
and records pass/crash plus engine memory. Stops at the first server death
(RemoteDisconnected / connection refused) and does not retry dead servers.

Usage:
  probe_context_threshold.py --base-url http://127.0.0.1:8024/v1 \
      --model mlx-community/Qwen3.8-27B-4bit \
      --tokenizer <dir> --lengths 8000 16000 24000 \
      --output artifacts/probe-ctx-threshold-<ts>.json

Exit code 0 always (the artifact is the record); the row statuses carry the
result. Server liveness is checked via /v1/models before every request.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer  # type: ignore[import-not-found]


def exact_prompt(tokenizer_path: Path, target: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    unit = " hello"
    if len(tokenizer.encode(unit, add_special_tokens=False)) != 1:
        raise RuntimeError("the exact-token prompt unit is not one token for this tokenizer")
    prompt = unit * target
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    if measured != target:
        raise RuntimeError(f"exact-token prompt measured {measured}, expected {target}")
    return prompt


def server_alive(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--lengths", nargs="+", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument(
        "--endpoint",
        choices=("chat", "completions"),
        default="chat",
        help="chat -> /v1/chat/completions (messages); completions -> /v1/completions (raw prompt)",
    )
    args = ap.parse_args()

    if args.endpoint == "chat":
        chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    else:
        chat_url = f"{args.base_url.rstrip('/')}/completions"
    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "lengths": args.lengths,
        "started_epoch": time.time(),
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for length in args.lengths:
        if not server_alive(args.base_url):
            row = {
                "status": "server_dead_before_request",
                "length_tokens": length,
                "elapsed_seconds": 0.0,
            }
            result["probes"][f"fill_{length}"] = row
            continue
        prompt = exact_prompt(args.tokenizer, length)
        if args.endpoint == "completions":
            payload = {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            }
        else:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            }
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                chat_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                body = json.loads(response.read())
            usage = body.get("usage") or {}
            row = {
                "status": "passed",
                "length_tokens": length,
                "elapsed_seconds": time.monotonic() - started,
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "decode_tokens_per_second": usage.get("tokens_per_second"),
                "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
                "mem_active_GB": round(usage.get("mei_memory_active_bytes", 0) / 1e9, 2),
                "mem_peak_GB": round(usage.get("mei_memory_peak_bytes", 0) / 1e9, 2),
                "mem_cache_GB": round(usage.get("mei_memory_cache_bytes", 0) / 1e9, 2),
            }
        except urllib.error.HTTPError as exc:
            row = {
                "status": "http_error",
                "length_tokens": length,
                "elapsed_seconds": time.monotonic() - started,
                "error": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}",
            }
        except Exception as exc:  # RemoteDisconnected, ConnectionRefused, timeout
            row = {
                "status": "server_died",
                "length_tokens": length,
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["probes"][f"fill_{length}"] = row
        result["server_alive_final"] = server_alive(args.base_url)
        print(f"length={length} -> {row.get('status')} ({row.get('elapsed_seconds', 0):.1f}s)")
        if row.get("status") in ("server_died", "server_dead_before_request"):
            break

    result["finished_epoch"] = time.time()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())