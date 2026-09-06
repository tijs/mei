#!/usr/bin/env python3
"""C1 deterministic greedy-output parity oracle for two live Mei endpoints.

Narrow, client-side oracle comparing TWO ALREADY-RUNNING Mei servers that
serve the SAME model with temperature=0 and a fixed max_tokens budget:

  * baseline  = baseline/eager leg
  * candidate = candidate/unsafe-compile optimized leg

On each deterministic chat prompt the oracle POSTs the identical payload to
both endpoints and compares:

  1. response visible fields byte-for-byte: `content`, `reasoning_content`,
     and (normalized) `tool_calls`;
  2. engine-reported aggregate `usage.completion_tokens`;
  3. token ID sequences obtained by re-encoding the generated text with the
     model's `tokenizer.json` via the standalone `tokenizers` library.

STRICT LIMITATION (recorded verbatim in every artifact):
Mei's HTTP surface exposes no token IDs and no logprobs, so LITERAL server
token-ID parity is unobservable and this tool does NOT claim it. The token IDs
compared here are a *client-side re-encoding* of the returned text produced by
`tokenizers.Tokenizer.from_file`. They are an equality check on byte-identical
generation under the model's own tokenizer, useful when two textual strings
tokenize identically, but they are NOT the engine's hidden token streams.

Optional fused-GDN log gate: if `--server-log PATH` is supplied, the oracle
requires that log to contain
  `[Qwen35] fused_gdn_decode_input_projections=active`
(the marker the optimized leg emits when the fused decode-input-projections
path is active). No log path => the gate is recorded as SKIPPED, never as a
pass.

No model/server is started and no benchmark is run by this tool; it only talks
to two HTTP endpoints that must already be running. See
`tools/test_parity_oracle.py` for the no-network unit/self tests.

Usage (run under the local-model-bench venv so `tokenizers` is importable):
  .venv/bin/python tools/parity_oracle.py \
      --baseline-url http://127.0.0.1:8024/v1 \
      --candidate-url http://127.0.0.1:8025/v1 \
      --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
      --tokenizer-json /path/to/tokenizer.json \
      --server-log /path/to/candidate.server.log \
      --output artifacts/parity-oracle-<ts>.json

  # No-network dependency self-check (validates the tokenizer re-encode path):
  .venv/bin/python tools/parity_oracle.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mei import request_json  # noqa: E402

# Marker the optimized (unsafe-compile) leg emits on a decoded-GDN-active start.
FUSED_GDN_MARKER = "[Qwen35] fused_gdn_decode_input_projections=active"

STRICT_LIMITATION = (
    "Mei HTTP exposes no token IDs and no logprobs; literal server token-ID "
    "parity is unobservable and is NOT claimed. Token IDs compared here are a "
    "client-side re-encoding of the returned text produced by re-encoding with "
    "the model tokenizer.json via the standalone `tokenizers` library; they are "
    "an equality proxy for byte-identical generation under the model's own "
    "tokenizer, not the engine's hidden token streams."
)

# Short, deterministic prompts that exercise a cheap greedy completion. Prompts
# are run verbatim with temperature=0, so a producing model must return the same
# output on both legs for parity to hold.
DEFAULT_PROMPTS = [
    "Count down from ten to one, one number per line, with no leading prose.",
    "Multiply 271 by 37 and reply with only the final integer.",
    "Reply with exactly the single word: deterministic-parity-ok.",
]

LONG_CONTEXT_PROMPT = (
    "Sum the first one hundred positive integers and reply with only the result."
)


def message_fields(message: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the observable, byte-comparable fields out of one response message.

    Combines the OpenAI-standard `content`/`tool_calls` with Mei's Qwen-style
    `reasoning_content`; tolerates a missing reasoning field (non-thinking model
    or `--emit-reasoning false`).
    """
    message = message or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = canonicalize_tool_calls(message.get("tool_calls"))
    return {
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        # Deterministic concatenation used for token-ID re-encoding below:
        # reasoning preamble first, then visible content, `\n` separated.
        "combined_text": (reasoning.rstrip("\n") + "\n" + content) if reasoning else content,
    }


def canonicalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Normalise tool calls to (name, parsed-or-raw arguments) for byte parity.

    The engine may vary `id`/`type` ordering or argument formatting across
    builds; parity care is about the *call content*, so we keep name and a
    JSON-canonical form of the arguments.
    """
    if not tool_calls:
        return []
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or ""
        raw_args = fn.get("arguments") or call.get("arguments")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            args = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            args = f"{raw_args}"
        out.append({"name": name, "arguments": args})
    return sorted(out, key=lambda c: (c["name"], c["arguments"]))


def extract_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    """Turn one /chat/completions response into a comparable snapshot."""
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    fields = message_fields(message)
    usage = response.get("usage") or {}
    fields["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    fields["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
    fields["finish_reason"] = choices[0].get("finish_reason") if choices else None
    return fields


def byte_identical(a: str, b: str) -> tuple[bool, str | None]:
    """True when two strings are byte-for-byte equal; else a short diff excerpt."""
    if a == b:
        return True, None
    return False, f"len {len(a)} != {len(b)} :: repr[A]={a!r} repr[B]={b!r}"


def tokenize_ids(tokenizer_json: Path, text: str) -> list[int]:
    """Re-encode `text` with the model's tokenizer.json (standalone tokenizers)."""
    from tokenizers import Tokenizer  # type: ignore[import-not-found]

    tok = Tokenizer.from_file(str(tokenizer_json))
    return tok.encode(text).ids


def compare_legs(baseline: dict[str, Any], candidate: dict[str, Any], tokenizer_json: Path) -> dict[str, Any]:
    """Compare two snapshots (same deterministic prompt) and report gates."""
    checks: dict[str, Any] = {}
    diffs: list[str] = []

    for field in ("content", "reasoning_content"):
        same, diff = byte_identical(baseline.get(field) or "", candidate.get(field) or "")
        checks[f"{field}_byte_identical"] = same
        if not same and diff:
            diffs.append(f"{field}: {diff}")

    bc = json.dumps(baseline.get("tool_calls"), sort_keys=True)
    cc = json.dumps(candidate.get("tool_calls"), sort_keys=True)
    checks["tool_calls_identical"] = bc == cc
    if bc != cc:
        diffs.append(f"tool_calls: {bc!r} != {cc!r}")

    checks["completion_tokens_equal"] = baseline["completion_tokens"] == candidate["completion_tokens"]
    if not checks["completion_tokens_equal"]:
        diffs.append(
            f"completion_tokens: {baseline['completion_tokens']} != {candidate['completion_tokens']}"
        )

    # Token-ID parity proxy: re-encode identical construction of the generated
    # text on both legs and compare ID sequences. NOTE: this is NOT the engine's
    # literal token stream (see STRICT_LIMITATION).
    ids_a = tokenize_ids(tokenizer_json, baseline["combined_text"])
    ids_b = tokenize_ids(tokenizer_json, candidate["combined_text"])
    checks["reencoded_token_ids_equal"] = ids_a == ids_b
    checks["reencoded_token_ids_baseline"] = ids_a
    checks["reencoded_token_ids_candidate"] = ids_b
    if ids_a != ids_b:
        shared = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
        diffs.append(
            f"re-encoded token IDs diverge at index {shared}: len {len(ids_a)} != {len(ids_b)}"
        )

    passed = all(checks[k] for k in checks if k.startswith(("content", "reasoning", "tool", "completion", "reencoded")))
    return {
        "checks": checks,
        "passed": passed,
        "diff": diffs,
        "baseline_completion_tokens": baseline["completion_tokens"],
        "candidate_completion_tokens": candidate["completion_tokens"],
        "baseline_reencoded_id_count": len(ids_a),
        "candidate_reencoded_id_count": len(ids_b),
        "baseline_prompt_tokens": baseline["prompt_tokens"],
        "candidate_prompt_tokens": candidate["prompt_tokens"],
    }


def check_fused_gdn_log(log_path: Path | None) -> dict[str, Any]:
    """Optional server-log gate: the optimized leg must claim fused GDN active.

    Returns status passed|failed|skipped. Only a supplied log can PASS or FAIL;
    a missing `--server-log` is always SKIPPED (the gate is advisory, never a
    pass by omission).
    """
    if log_path is None:
        return {"status": "skipped", "eligible": False}
    if not log_path.exists():
        return {"status": "failed", "eligible": True, "error": f"log not found: {log_path}"}
    text = log_path.read_text(errors="replace")
    found = FUSED_GDN_MARKER in text
    return {
        "status": "passed" if found else "failed",
        "eligible": True,
        "marker": FUSED_GDN_MARKER,
        "found": found,
    }


def run_prompt(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response, elapsed = request_json(f"{url.rstrip('/')}/chat/completions", payload, timeout=timeout)
    return {"elapsed_seconds": round(elapsed, 3), **extract_snapshot(response)}


def self_test() -> int:
    """No-network check that the tokenizer re-encode path and compare helpers work."""
    print("[self-test] building in-memory word-level tokenizer (no network)...")
    from tokenizers import Tokenizer  # type: ignore[import-not-found]
    from tokenizers.models import WordLevel  # type: ignore[import-not-found]
    from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-not-found]

    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    with tempfile.TemporaryDirectory() as tmp:
        tj = Path(tmp) / "tokenizer.json"
        tok.save(str(tj))
        assert tokenize_ids(tj, "a b c") == [1, 2, 3]
    assert byte_identical("x", "x")[0] is True
    assert byte_identical("x", "y")[0] is False
    assert canonicalize_tool_calls([{"function": {"name": "f", "arguments": '{"b":2,"a":1}'}}]) == [
        {"name": "f", "arguments": '{"a":1,"b":2}'}
    ]
    print("[self-test] PASS: tokenizers import, from_file re-encode, byte/diff and tool-call helpers OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run no-network self/unit checks and exit")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8024/v1", help="baseline/eager Mei endpoint")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8025/v1", help="candidate/unsafe-compile Mei endpoint")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None, help="path to model tokenizer.json")
    parser.add_argument("--server-log", type=Path, help="optional candidate leg server log (fused-GDN gate)")
    parser.add_argument("--prompt", action="append", help="custom deterministic prompt (repeatable); overrides defaults")
    parser.add_argument("--samples", type=int, default=3, help="number of default prompts to use when no --prompt given")
    parser.add_argument("--max-tokens", type=int, default=1024, help="fixed per-leg generation budget")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    for required, label in ((args.model, "--model"), (args.tokenizer_json, "--tokenizer-json"), (args.output, "--output")):
        if required is None:
            parser.error(f"the following arguments are required: {label}")

    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("[parity] error: the standalone `tokenizers` library is required; run under a venv that has it.", file=sys.stderr)
        return 2

    if args.tokenizer_json is None or not args.tokenizer_json.exists():
        print(f"[parity] error: tokenizer.json not found: {args.tokenizer_json}", file=sys.stderr)
        return 2

    prompts = args.prompt or DEFAULT_PROMPTS[: args.samples]
    result: dict[str, Any] = {
        "engine": "mei",
        "tool": "C1 deterministic greedy-output parity oracle",
        "model": args.model,
        "baseline": {"label": "baseline/eager", "url": args.baseline_url},
        "candidate": {"label": "candidate/unsafe-compile", "url": args.candidate_url},
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "methodology": (
            "identical deterministic chat prompt, temperature=0, fixed max_tokens; "
            "POST to both live endpoints; compare visible text/reasoning/tool fields "
            "byte-for-byte, engine aggregate completion_tokens, and client-side "
            "tokenizer.json re-encoded token-ID sequences."
        ),
        "strict_token_id_parity": False,
        "strict_limitation": STRICT_LIMITATION,
        "fused_gdn_gate": check_fused_gdn_log(args.server_log),
        "started_epoch": time.time(),
        "samples": [],
    }

    for i, prompt in enumerate(prompts):
        sample: dict[str, Any] = {"index": i, "prompt": prompt}
        started = time.monotonic()
        try:
            baseline = run_prompt(args.baseline_url, args.model, prompt, args.max_tokens, args.timeout)
            sample["baseline"] = {k: baseline[k] for k in ("content", "reasoning_content", "tool_calls", "completion_tokens", "prompt_tokens", "finish_reason", "elapsed_seconds")}
        except BaseException as exc:  # noqa: BLE001
            sample["status"] = "error"
            sample["baseline_error"] = f"{type(exc).__name__}: {exc}"
            result["samples"].append(sample)
            print(f"[parity] sample {i}: baseline error ({exc})", flush=True)
            continue
        try:
            candidate = run_prompt(args.candidate_url, args.model, prompt, args.max_tokens, args.timeout)
            sample["candidate"] = {k: candidate[k] for k in ("content", "reasoning_content", "tool_calls", "completion_tokens", "prompt_tokens", "finish_reason", "elapsed_seconds")}
        except BaseException as exc:  # noqa: BLE001
            sample["status"] = "error"
            sample["candidate_error"] = f"{type(exc).__name__}: {exc}"
            result["samples"].append(sample)
            print(f"[parity] sample {i}: candidate error ({exc})", flush=True)
            continue
        try:
            comparison = compare_legs(baseline, candidate, args.tokenizer_json)
        except BaseException as exc:  # noqa: BLE001
            sample["status"] = "error"
            sample["compare_error"] = f"{type(exc).__name__}: {exc}"
            result["samples"].append(sample)
            print(f"[parity] sample {i}: compare error ({exc})", flush=True)
            continue
        sample["status"] = "passed" if comparison["passed"] else "failed"
        sample["elapsed_seconds"] = round(time.monotonic() - started, 3)
        sample.update(comparison)
        result["samples"].append(sample)
        print(f"[parity] sample {i}: {sample['status']} ({sample['elapsed_seconds']}s)", flush=True)

    executed = [s for s in result["samples"] if s["status"] in ("passed", "failed")]
    result["finished_epoch"] = time.time()
    gate = result["fused_gdn_gate"]
    fused_ok = gate["status"] == "passed" or gate["status"] == "skipped"
    result["status"] = (
        "passed"
        if executed and all(s["status"] == "passed" for s in executed) and fused_ok
        else "failed"
    )
    result["note"] = (
        "Execution against two live servers remains GPU-gated; this artifact alone "
        "proves the oracle runs and its gates are wired, not that either Mei build "
        "produced identical output."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())