#!/usr/bin/env python3
"""Live total-context boundary probe; run only in an owned, guarded session.

Unlike the historical prompt-only probe, reserves one output token. This
does not establish cache parity, tool admission, or model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_mei import (
    aggregate_status,
    checkpointed_probe,
    exact_prompt,
    request_json,
    write_probe_result,
)


def check_boundary(base_url, model, tokenizer, cap, timeout, output):
    if cap < 2:
        raise ValueError("total context cap must be at least two")
    result = {
        "model": model,
        "context_cap": cap,
        "contract": "prompt+output",
        "probes": {},
    }
    url = f"{base_url.rstrip('/')}/completions"

    def request(tokens, budget=1):
        return request_json(
            url,
            {
                "model": model,
                "prompt": exact_prompt(tokenizer, tokens),
                "temperature": 0,
                "max_tokens": budget,
                "stream": False,
            },
            timeout=timeout,
        )

    def fits():
        # Ask for more than the remaining capacity to prove the server clamps
        # the output budget as well as accepting the largest valid prompt.
        response, elapsed = request(cap - 1, budget=8)
        usage = response.get("usage") or {}
        expected = {
            "prompt_tokens": cap - 1,
            "completion_tokens": 1,
            "total_tokens": cap,
        }
        if any(
            type(usage.get(key)) is not int or usage[key] != value
            for key, value in expected.items()
        ):
            raise AssertionError(f"total-context boundary usage mismatch: {usage!r}")
        if not response.get("choices"):
            raise AssertionError("no completion choice at total-context boundary")
        return {"usage": usage, "requested_max_tokens": 8, "request_seconds": elapsed}

    checkpointed_probe("last_prompt_token_leaves_one_output", result, fits, output)
    for size in (cap, cap + 1):

        def rejected(size=size):
            try:
                response, elapsed = request(size)
            except RuntimeError as exc:
                # An unrelated HTTP 400 (e.g. unknown model or bad request)
                # must never count as context-boundary evidence.
                expected = f"request exceeded context cap: {size} prompt tokens > {cap - 1} allowed"
                if not str(exc).startswith("HTTP 400:") or expected not in str(exc):
                    raise
                return {
                    "rejected_as_expected": True,
                    "prompt_tokens": size,
                    "error": str(exc),
                }
            raise AssertionError(
                f"{size}-token prompt unexpectedly accepted: {response!r} in {elapsed}s"
            )

        checkpointed_probe(f"prompt_{size}_rejected", result, rejected, output)
    result["status"] = aggregate_status(result["probes"])
    write_probe_result(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--context-cap", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("choose a fresh context probe output path")
    result = check_boundary(
        args.base_url,
        args.model,
        args.tokenizer,
        args.context_cap,
        args.timeout,
        args.output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
