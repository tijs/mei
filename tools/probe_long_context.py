#!/usr/bin/env python3
"""Long-context regression probes for Mei (Phase 5 of the plan).

The whole point of the project: chunked prefill must keep hybrid
(GatedDelta) architectures from collapsing at long context, the way the
three Python serving wrappers did (vllm_mlx 0.18 tok/s @43K, oMLX 0.75,
jjang-ai/vmlx 0.2 @30K + hard Metal OOM @80K). This probe sends synthetic
filler prompts at configured lengths via /v1/completions and requires:

- HTTP 200 with non-empty completion (no OOM crash, no collapse)
- engine-reported decode tok/s above a sanity floor (--min-decode-tps,
  default 1.0 — the Python wrappers' collapse band was 0.18-0.75)
- prompt token count matching the requested length
- a second request that STRICTLY EXTENDS the first (length + 1 tokens) with
  cached_tokens reporting the full first prompt — Mei's KV slot reuses only
  on exact sequence extension (equal-length repeats are positionally
  unsafe for the GatedDelta recurrent state, so they never reuse)

Deliberately NOT a pass/fail gate on absolute speed — that's the benchmark's
job; this gates on "did chunked prefill survive the length".

Usage:
  probe_long_context.py --base-url http://127.0.0.1:8024/v1 --model ID \
    --tokenizer PATH --lengths 30000 80000 --output probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from probe_mei import request_json  # noqa: E402  (shared client helpers)

MISSING = object()


def exact_prompt(tokenizer_path: Path, target: int) -> str:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    unit = " hello"
    if len(tokenizer.encode(unit, add_special_tokens=False)) != 1:
        raise RuntimeError("the exact-token prompt unit is not one token for this tokenizer")
    prompt = unit * target
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    if measured != target:
        raise RuntimeError(f"exact-token prompt measured {measured}, expected {target}")
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[30000, 80000])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-decode-tps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "lengths": args.lengths,
        "min_decode_tps": args.min_decode_tps,
        "started_epoch": time.time(),
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completions_url = f"{args.base_url.rstrip('/')}/completions"

    for length in args.lengths:
        for round_trip, label, sent_length in (
            (1, "fresh", length),
            # Round 2 must strictly EXTEND round 1 (one more token) to be
            # eligible for Mei's exact-extension KV slot; the cached prefix
            # then covers round 1's full prompt.
            (2, "cache_reuse", length + 1),
        ):
            name = f"fill_{length}_{label}"
            prompt = exact_prompt(args.tokenizer, sent_length)
            started = time.monotonic()
            try:
                response, elapsed = request_json(completions_url, {
                    "model": args.model,
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                }, timeout=args.timeout)
                usage = response.get("usage") or {}
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = (message.get("content") or choice.get("text") or "").strip()
                prompt_tokens = int(usage.get("prompt_tokens", -1))
                completion_tokens = int(usage.get("completion_tokens", 0))
                decode_tps = usage.get("tokens_per_second")
                cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                checks = {
                    "http_ok": True,
                    "prompt_tokens_match": prompt_tokens == sent_length,
                    "nonempty_completion": len(content) > 0 or completion_tokens > 0,
                    "decode_above_floor": (decode_tps or 0) >= args.min_decode_tps,
                }
                if round_trip == 2:
                    checks["full_prefix_cached"] = cached >= length - 8
                failed = {k: v for k, v in checks.items() if v is False}
                result["probes"][name] = {
                    "status": "passed" if not failed else "failed",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "decode_tokens_per_second": decode_tps,
                    "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
                    "cached_tokens": cached,
                    "memory_active_bytes": usage.get("mei_memory_active_bytes"),
                    "memory_peak_bytes": usage.get("mei_memory_peak_bytes"),
                    "checks": checks,
                    "failed_checks": failed,
                }
            except BaseException as exc:  # noqa: BLE001
                result["probes"][name] = {
                    "status": "error",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }

    result["finished_epoch"] = time.time()
    result["status"] = (
        "passed"
        if result["probes"]
        and all(p["status"] == "passed" for p in result["probes"].values())
        else "failed"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())