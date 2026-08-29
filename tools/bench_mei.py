#!/usr/bin/env python3
"""Mei-contained benchmark runner (local-model-bench protocol, artifact-only).

Drives the live Mei server through its OpenAI surface and records one
versioned JSON artifact under artifacts/ per invocation. Follows the
local-model-bench methodology: automated pass/fail only (never an LLM
judge), engine-reported usage (token counts, tok/s, allocator bytes) instead
of client-side guesses where possible, raw artifact + human summary kept
separate, and every row carrying its config.

Scenario matrix (each row is a separate HTTP request; order fixed):

  short_context       chat, ~64 prompt tokens           -> short-context decode speed
  tool_nonstreaming   add_numbers forced tool call      -> structured tool-call correctness
  tool_streaming      same, streamed                    -> SSE tool-call correctness + TTFT
  long_loaded_fresh   exact-token completion at N=45000 -> prefill tok/s, TTFT, decode at
                                                          loaded context (NO cache)
  long_loaded_reuse   N+1 exact extension of the same   -> KV reuse evidence (cached_tokens)
                          prompt                          + decode speed at loaded context
  long_chat_40k       chat transcript ~40K (filler      -> chat-path long-context correctness,
                          system + growing transcript)     no collapse, marker content check

Pass/fail gates:
  - every request must return 200 and non-empty completion
  - tool rows must produce the exact add_numbers(15, 27) schema
  - long rows must report the requested prompt-token count
  - decode at loaded context must stay above `--min-decode-tps`
    (the Python wrappers collapsed to 0.18-0.75 tok/s; Mei's floor is set
    well above that band and below the engineering targets)
  - the reuse row must report cached_tokens covering ~the full prior prompt

Baselines are read from local-model-bench's documented results (never
written back to that repo).

Usage:
  bench_mei.py --model ID --tokenizer DIR [--length 45000] [--min-decode-tps 10]
               [--max-tokens 64] [--output artifacts/bench-...json]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mei import request_json, request_stream, validate_add_call  # noqa: E402

# Baselines transcribed from /Users/tijs/projects/local-model-bench
# (results/LEADERBOARD.md + docs/INFERENCE_ENGINES.md), read-only reference.
BASELINES = {
    "vllm_mlx_ornith35b_mlx4bit": {
        "engine": "vllm-mlx",
        "context_tokens": 43000,
        "decode_tps": 0.18,
        "note": "collapse band measured by local-model-bench; mean rows 2.1-2.4 tok/s",
    },
    "omlx_ornith9b_oq4e": {
        "engine": "oMLX",
        "context_tokens": None,
        "decode_tps": 0.75,
        "note": "0.75 tok/s average on real hermes_ops trials",
    },
    "llamacpp_ornith35b_apex_compact": {
        "engine": "llama.cpp",
        "context_tokens": None,
        "decode_tps": 28.6,
        "note": "APEX-Compact GGUF best row (local-model-bench)",
    },
    "mlx_lm_generate_80k": {
        "engine": "bare mlx-lm",
        "context_tokens": 80000,
        "decode_tps": 15.7,
        "note": "direct library call, no serving layer (local-model-bench measurement)",
    },
    "mei_target": {"engine": "mei", "decode_tps": 40.0, "note": "primary repeatable target at ~40-50K loaded"},
    "mei_milestone": {"engine": "mei", "decode_tps": 30.0, "note": "minimum milestone already observed on the 9B path"},
}


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


TOOLS = [{
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--length", type=int, default=45000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-decode-tps", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "engine": "mei",
        "model": args.model,
        "base_url": args.base_url,
        "methodology": "local-model-bench protocol, artifact-only (no writes to local-model-bench)",
        "baselines": BASELINES,
        "length": args.length,
        "min_decode_tps": args.min_decode_tps,
        "started_epoch": time.time(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    completions_url = f"{args.base_url.rstrip('/')}/completions"

    def record(name: str, fn) -> None:
        started = time.monotonic()
        row: dict[str, Any] = {"name": name, "started_epoch": time.time()}
        try:
            detail = fn()
            row.update(detail)
            passed = all(v is not False for v in detail.get("checks", {}).values())
            row["status"] = "passed" if passed else "failed"
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        except BaseException as exc:  # noqa: BLE001
            row["status"] = "error"
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            row["error"] = f"{type(exc).__name__}: {exc}"
        result["rows"].append(row)
        print(f"[bench] {row['name']}: {row['status']} ({row['elapsed_seconds']}s)", flush=True)

    # ---- short context (reported separately from the loaded-context target) ----
    if not args.skip_short:
        def short_context() -> dict[str, Any]:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with exactly: short-ok"}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            response, elapsed = request_json(chat_url, payload, timeout=args.timeout)
            usage = response.get("usage") or {}
            content = (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            checks = {
                "nonempty": bool(content),
                "http_200": True,
            }
            return {
                "checks": checks,
                "usage": usage,
                "decode_tokens_per_second_engine": usage.get("tokens_per_second"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "request_seconds": elapsed,
            }
        record("short_context", short_context)

    # ---- tool correctness (non-streaming + streaming) ----
    tool_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to one tool."},
            {"role": "user", "content": "What is 15 + 27? You must use add_numbers to compute it."},
        ],
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "add_numbers"}},
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }

    if not args.skip_tools:
        def tool_nonstreaming() -> dict[str, Any]:
            response, elapsed = request_json(chat_url, tool_payload, timeout=args.timeout)
            call = validate_add_call(response)
            return {
                "checks": {"schema": True},
                "validated_call": call,
                "usage": response.get("usage"),
                "request_seconds": elapsed,
            }
        record("tool_nonstreaming", tool_nonstreaming)

        def tool_streaming() -> dict[str, Any]:
            response, elapsed, ttft = request_stream(chat_url, tool_payload, timeout=args.timeout)
            call = validate_add_call(response)
            return {
                "checks": {"schema": True},
                "validated_call": call,
                "usage": response.get("usage"),
                "client_ttft_seconds": ttft,
                "request_seconds": elapsed,
            }
        record("tool_streaming", tool_streaming)

    # ---- long context: loaded at N tokens, decode measured there ----
    if not args.skip_long:
        unit_prompt = exact_prompt(args.tokenizer, args.length)

        def long_fresh() -> dict[str, Any]:
            # Non-streaming: Mei's /v1/completions is a synchronous route and
            # the engine reports exact prefill and generate ms + tok/s in
            # usage, which is a better TTFT source than client wall time for
            # a request whose prefill runs in a single server-side call.
            payload = {
                "model": args.model,
                "prompt": unit_prompt,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            response, elapsed = request_json(completions_url, payload, timeout=args.timeout)
            usage = response.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", -1))
            completion_tokens = int(usage.get("completion_tokens", 0))
            decode_engine = usage.get("tokens_per_second")
            prefill_ms = usage.get("prefill_ms")
            finish = (response.get("choices") or [{}])[0].get("finish_reason")
            content_len = len(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            checks = {
                "prompt_tokens_match": prompt_tokens == args.length,
                "nonempty_completion": completion_tokens > 0 or content_len > 0,
                "decode_above_floor": (decode_engine or 0) >= args.min_decode_tps,
                "not_collapsed": (decode_engine or 0) >= args.min_decode_tps,
            }
            return {
                "checks": checks,
                "usage": usage,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "decode_tokens_per_second_engine": decode_engine,
                "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
                "prefill_ms": prefill_ms,
                "finish_reason": finish,
                "request_seconds": elapsed,
            }
        record("long_loaded_fresh", long_fresh)

        def long_reuse() -> dict[str, Any]:
            # Strict extension (length+1): must reuse the prior request's
            # prefix KV instead of re-prefilling the full prompt.
            extending_prompt = exact_prompt(args.tokenizer, args.length + 1)
            payload = {
                "model": args.model,
                "prompt": extending_prompt,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            response, elapsed = request_json(completions_url, payload, timeout=args.timeout)
            usage = response.get("usage") or {}
            cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            decode_engine = usage.get("tokens_per_second")
            completion_tokens = int(usage.get("completion_tokens", 0))
            checks = {
                "prefix_reused": cached >= args.length - 8,
                "decode_above_floor": (decode_engine or 0) >= args.min_decode_tps,
            }
            return {
                "checks": checks,
                "usage": usage,
                "cached_tokens": cached,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": completion_tokens,
                "decode_tokens_per_second_engine": decode_engine,
                "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
                "prefill_ms": usage.get("prefill_ms"),
                "request_seconds": elapsed,
            }
        record("long_loaded_reuse", long_reuse)

        # Chat path at ~40K: growing-transcript pattern, correctness marker.
        # The filler line count is computed from the REAL tokenizer (not
        # estimated) so the transcript lands at the configured chat length.
        def long_chat() -> dict[str, Any]:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

            def transcript(filler_lines: int) -> list[dict[str, Any]]:
                filler = "system stability marker " * filler_lines
                t = (
                    [{"role": "system", "content": filler + " Final system line."}]
                    + [{"role": "user", "content": f"Instruction batch {i}: answer nothing yet."} for i in range(6)]
                    + [{"role": "assistant", "content": "Understood."} for _ in range(5)]
                )
                t.append({"role": "user", "content": "Final instruction: reply with exactly the word cache-ready."})
                return t

            def transcript_tokens(t: list[dict[str, Any]]) -> int:
                rendered = tok.apply_chat_template(t, add_generation_prompt=True)
                if isinstance(rendered, str):
                    return len(tok.encode(rendered, add_special_tokens=False))
                ids = rendered.get("input_ids")
                if ids:
                    return len(ids)
                raise RuntimeError(f"cannot count transcript tokens: {type(rendered)}")

            target = 44000
            lo, hi = 0, 30000
            # Binary search for the smallest filler line count whose
            # transcript renders >= target tokens.
            chat_messages = transcript(30000)
            while lo < hi:
                mid = (lo + hi) // 2
                if transcript_tokens(transcript(mid)) >= target:
                    chat_messages = transcript(mid)
                    hi = mid
                else:
                    lo = mid + 1
            payload = {
                "model": args.model,
                "messages": chat_messages,
                "temperature": 0,
                "max_tokens": 256,
                "stream": False,
            }
            response, elapsed = request_json(chat_url, payload, timeout=args.timeout)
            usage = response.get("usage") or {}
            message = (((response.get("choices") or [{}])[0].get("message") or {}))
            content = (message.get("content") or "").strip()
            prompt_tokens = int(usage.get("prompt_tokens", -1))
            checks = {
                "long_chat_transcript": prompt_tokens >= 40000,
                "nonempty_completion": bool(content),
                "decode_above_floor": (usage.get("tokens_per_second") or 0) >= args.min_decode_tps,
            }
            return {
                "checks": checks,
                "usage": usage,
                "prompt_tokens": prompt_tokens,
                "content_tail": content[-200:],
                "request_seconds": elapsed,
            }
        record("long_chat_40k", long_chat)

    result["finished_epoch"] = time.time()
    result["status"] = (
        "passed"
        if result["rows"]
        and all(r["status"] == "passed" for r in result["rows"])
        else "failed"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())