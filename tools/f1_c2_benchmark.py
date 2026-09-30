#!/usr/bin/env python3
"""F1/C2 same-binary A/B for Mei's routed-MoE decode policy.

This is a scratch driver for the isolated Mei worktree. It uses the
local-model-bench probe_mei.py read-only for the per-arm admission/tool gate,
then measures the actual OpenAI-compatible Mei endpoint at short and 30k
context. One server process is used for one arm; each arm gets a fresh process,
its own request log, and a fresh KV/runtime path.

Arms (both F1/C2 switches are cleared from the environment first; the arm then
applies its overrides on top):

  candidate  leave both switches unset. This is the upstream/profile default;
             on the 0.6.0 pin the compiled routed-MoE decode region is active
             by default, so this arm measures it ON.
  control    force both switches to "1" -- compiled region explicitly ON.
  optout     force both switches to "0" -- compiled region explicitly OFF.

The same-binary A/B for the 0.6.0 pin is candidate (default ON) versus optout
(explicit OFF); control is kept for pins whose default already flipped off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


SWITCHES = (
    "VMLX_QWEN35_COMPILE_DECODE_REGIONS",
    "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE",
)

# Arm name -> environment overrides applied after both switches are cleared.
# "candidate" intentionally stays empty so the switches remain unset (the
# 0.6.0 default is compiled ON); "optout" pins the explicit off state.
ARM_OVERRIDES = {
    "candidate": {},
    "control": {name: "1" for name in SWITCHES},
    "optout": {name: "0" for name in SWITCHES},
}


def arm_overrides(arm: str) -> dict[str, str]:
    """Return a fresh copy of the switch overrides for ``arm``."""
    return dict(ARM_OVERRIDES[arm])


def build_env(overrides: dict[str, str], base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the server environment for one arm.

    Both F1/C2 switches are cleared first, the fixed compile flags are set,
    and the arm's overrides are applied last (so "candidate" leaves the
    switches genuinely unset, "control" pins them to "1", and "optout" pins
    them to "0"). ``base`` defaults to ``os.environ`` and is never mutated.
    """
    env = dict(os.environ) if base is None else dict(base)
    for name in SWITCHES:
        env.pop(name, None)
    env["VMLX_ENABLE_UNSAFE_COMPILE"] = "1"
    env["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"] = "0"
    env.update(overrides)
    return env


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def post_json(url: str, body: dict, timeout: float = 1800.0) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read()), time.monotonic() - started


def wait_ready(process: subprocess.Popen, base: str, model: str, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with rc={process.returncode}")
        try:
            data = get_json(url, timeout=10)
            ids = [entry.get("id") for entry in data.get("data", [])]
            if model in ids:
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server was not ready after {timeout}s")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def response_text(body: dict) -> str:
    choice = (body.get("choices") or [{}])[0]
    return choice.get("text") or ((choice.get("message") or {}).get("content") or "")


def sha256_token_text(text: str, tokenizer) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def read_request_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_arm(args: argparse.Namespace, arm: str, overrides: dict[str, str]) -> dict:
    out = args.output / arm
    out.mkdir(parents=True, exist_ok=True)
    server_log = out / "server.log"
    request_log = out / "request.jsonl"
    model = args.model_id
    base = f"http://127.0.0.1:{args.port}/v1"

    env = build_env(overrides)

    command = [
        str(args.binary),
        "--model-dir", str(args.model_dir),
        "--served-model-id", model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--context-cap", "65536",
        "--prefill-step-size", "1024",
        "--max-tokens", "64",
        "--temperature", "0",
        "--top-p", "1",
        "--top-k", "1",
        "--cache-reuse", "false",
        "--request-log", str(request_log),
        "--log-requests", "true",
    ]
    if args.enable_thinking is not None:
        command.extend(["--enable-thinking", args.enable_thinking])
    metadata = {
        "arm": arm,
        "model_id": model,
        "model_dir": str(args.model_dir),
        "binary": str(args.binary),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "command": command,
        "operator_overrides": overrides,
        "started_epoch": time.time(),
        "contexts": {"short": 13, "loaded": 30000},
        "repeats_per_context": args.repeats,
    }
    with server_log.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(process, base, model)
        status = get_json(f"{base}/mei/status")
        metadata["status"] = status

        # The benchmark's real admission/tool/parity probe is read-only code
        # from local-model-bench; its output is preserved beside this arm.
        probe = subprocess.run(
            [
                sys.executable,
                str(args.probe),
                "--base-url", base,
                "--model", model,
                "--tokenizer", str(args.model_dir),
                "--output", str(out / "probe.json"),
                "--skip-context",
                "--skip-cache",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        metadata["probe_returncode"] = probe.returncode
        metadata["probe_stdout_tail"] = probe.stdout[-4000:]
        metadata["probe_stderr_tail"] = probe.stderr[-2000:]
        if probe.returncode != 0:
            raise RuntimeError(f"probe_mei.py failed with rc={probe.returncode}")

        rows = []
        prompts = {
            "short": "Count from 1 to 10, one per line.",
            "loaded": " hello" * 30000,
        }
        for context_name, prompt in prompts.items():
            for repeat in range(1, args.repeats + 1):
                body, wall = post_json(
                    f"{base}/completions",
                    {
                        "model": model,
                        "prompt": prompt,
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "max_tokens": 32,
                        "seed": 1,
                        "stream": False,
                    },
                )
                usage = body.get("usage") or {}
                text = response_text(body)
                row = {
                    "context": context_name,
                    "repeat": repeat,
                    "wall_seconds": wall,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                rows.append(row)
                print(
                    f"{arm} {context_name} r{repeat}: "
                    f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
                    f"wall={wall:.3f}s",
                    flush=True,
                )
        metadata["rows"] = rows
        metadata["request_log_rows"] = read_request_log(request_log)
        metadata["finished_epoch"] = time.time()
        (out / "result.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        stop_server(process)
        metadata["server_stopped"] = process.poll() is not None
        metadata["finished_epoch"] = time.time()
        (out / "result.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "F1/C2 same-binary A/B driver. Arms: candidate (both switches "
            "unset, the 0.6.0 default compiled-ON behavior), control (both "
            "switches '1'), optout (both switches '0')."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--arm",
        choices=list(ARM_OVERRIDES),
        required=True,
        help="candidate=switches unset; control=both '1'; optout=both '0'",
    )
    parser.add_argument("--enable-thinking", choices=["true", "false"], default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.binary = args.binary.resolve()
    args.model_dir = args.model_dir.resolve()
    args.probe = args.probe.resolve()
    args.output = args.output.resolve()
    overrides = arm_overrides(args.arm)
    result = run_arm(args, args.arm, overrides)
    print(json.dumps({"arm": args.arm, "status": "passed", "output": str(args.output / args.arm), "rows": len(result.get("rows", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
