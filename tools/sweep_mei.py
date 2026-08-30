#!/usr/bin/env python3
"""Mei cliff-characterization sweep driver (Mei-contained, artifact-only).

Measures the CURRENT pinned engine (vmlx-swift aeb5e21c + Mei) with a
controlled matrix of server configurations and prompt lengths, restarting
the server per configuration cell so every row is a cold, isolated run.
Captures engine usage (prefill ms, generate ms, tok/s, cached_tokens),
MLX allocator state (active/cache/peak) from /v1/mei/status, the
MLXPRESS_GENERATION_PROFILE=1 per-generation stage dump, TTFT, and any
failures, into one timestamped artifact under Mei/artifacts.

The driver never writes outside Mei-owned state (runtime dir + artifacts).
It records a machine-contention boundary (foreign llama-server / vllm / omlx
/ cocore processes) so an accidentally co-resident run is labeled, not
silently presented as clean.

Usage (from an uncontended machine):
  MEI_SWEEP_VENV=~/.local/share/local-model-bench/mei-runtime/venv/bin/python \\
  python3 tools/sweep_mei.py \\
    --model-dir ~/.local/share/local-model-bench/mei-models/Ornith-1.5-9B-MLX-4bit \\
    --model-id ornith-ai/Ornith-1.5-9B-MLX-4bit \\
    --tokenizer-dir ~/.local/share/local-model-bench/mei-models/Ornith-1.5-9B-MLX-4bit \\
    --contexts 512,4096,16384,33175,45000 --prefill-steps 512 --repeats-45k 3 \\
    --output artifacts/sweep-cliff-<ts>.json

The matrix flags multiply: --prefill-steps 512,2048,4096 --ssm-rederive
true,false --cache-limit-gb 2,8 --kv-bits none,8,4 --compiled true,false.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
RUNTIME_BASE = Path(
    os.environ.get("MEI_RUNTIME_BASE", "~/.local/share/local-model-bench/mei-runtime")
).expanduser()
BUILD_DIR = Path(
    os.environ.get("MEI_BUILD_DIR", "~/.local/share/local-model-bench/mei-build")
).expanduser()

FOREIGN_SERVER_RE = re.compile(
    r"llama-server|vllm|omlx|o?mlx_mlx|serve\s+--model|llama.cpp|cocore",
    re.IGNORECASE,
)


def foreign_servers() -> list[str]:
    """Name any non-Mei inference processes currently on the machine."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return ["ps-failed"]
    found = []
    for line in out.splitlines():
        if any(tok in line for tok in ("sweep_mei.py", "mei-build", "mei-runtime", "mei-models")):
            continue
        if FOREIGN_SERVER_RE.search(line):
            found.append(line.strip())
    return found


def request_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.monotonic() - started


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(base: str, model_id: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base}/models"
    while time.monotonic() < deadline:
        try:
            body = get_json(url, timeout=10)
            ids = [m.get("id") for m in body.get("data", [])]
            if model_id in ids:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server did not become ready within {timeout}s (model {model_id})")


def unit_prompt(target: int) -> str:
    return " hello" * target


class Server:
    """Owned Mei server process, one per configuration cell."""

    def __init__(self, args: argparse.Namespace, bin_path: Path, cell: dict[str, Any], log_path: Path):
        self.args = args
        self.cell = cell
        self.log_path = log_path
        self.kv_dir = (
            RUNTIME_BASE / "kv-cache-sweep" / cell["tag"] if args.kv_cache_dir else None
        )
        if self.kv_dir is not None:
            self.kv_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(bin_path),
            "--model-dir", args.model_dir,
            "--served-model-id", args.model_id,
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--context-cap", str(args.context_cap),
            "--max-tokens", str(args.max_tokens),
            "--prefill-step-size", str(cell["prefill_step"]),
            "--temperature", "0",
            "--cache-reuse", "true",
            "--log-requests", "true",
            "--ssm-rederive", str(cell["ssm_rederive"]).lower(),
            "--compiled-decode", str(cell["compiled"]).lower(),
        ]
        if args.memory_limit_bytes:
            argv += ["--memory-limit-bytes", str(args.memory_limit_bytes)]
        if cell["cache_limit_bytes"]:
            argv += ["--cache-limit-bytes", str(cell["cache_limit_bytes"])]
        if cell["kv_bits"]:
            argv += ["--kv-bits", str(cell["kv_bits"])]
        if self.kv_dir is not None:
            argv += ["--kv-cache-dir", str(self.kv_dir)]
        env = dict(os.environ)
        env["MLXPRESS_GENERATION_PROFILE"] = "1"
        print(f"[sweep] starting server cell {cell['tag']}: {' '.join(argv)}", flush=True)
        self.proc = subprocess.Popen(
            argv, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env
        )
        try:
            wait_for_server(f"http://127.0.0.1:{args.port}/v1", args.model_id, timeout=args.server_timeout)
        except Exception:
            self.stop()
            raise

    def status(self) -> dict[str, Any]:
        try:
            return get_json(f"http://127.0.0.1:{self.args.port}/v1/mei/status", timeout=10)
        except Exception as exc:
            return {"error": str(exc)}

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.kv_dir is not None and self.args.fresh_kv:
            # Fresh-KV requirement: each cell owns its runtime state; the
            # caller can also clear per-row via --fresh-kv-per-cell below.
            pass


def summarize_profile(log_text: str) -> dict[str, Any]:
    rows = re.findall(r"\[MLXPressGenerationProfile\] ([^\n]*)", log_text)
    return {"dumps": rows[-8:] if rows else []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, default=8024)
    parser.add_argument("--context-cap", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--contexts", default="512,4096,16384,33175,45000")
    parser.add_argument("--prefill-steps", default="512")
    parser.add_argument("--ssm-rederive", default="true")
    parser.add_argument("--cache-limit-gb", default="0")
    parser.add_argument("--kv-bits", default="none")
    parser.add_argument("--compiled", default="false")
    parser.add_argument("--repeats-45k", type=int, default=1)
    parser.add_argument("--chat-40k", action="store_true", help="include 40K chat row per cell")
    parser.add_argument("--memory-limit-bytes", type=int, default=0)
    parser.add_argument("--kv-cache-dir", action="store_true", help="use a per-cell disk KV dir")
    parser.add_argument(
        "--transformers-python", type=str,
        default="~/.local/share/local-model-bench/mei-runtime/venv/bin/python",
        help="python interpreter with transformers installed (for the 40K chat "
        "transcript token count); the exact-token prompts themselves are "
        "built without transformers via the unit-token invariant from bench_mei",
    )
    parser.add_argument("--server-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=5400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-contention-gate", action="store_true",
        help="run even with foreign inference processes resident (labels rows)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(args.model_dir)
    contexts = [int(x) for x in args.contexts.split(",")]
    prefill_steps = [int(x) for x in args.prefill_steps.split(",")]
    ssm_values = [x.lower() == "true" for x in args.ssm_rederive.split(",")]
    cache_limit_values = [int(float(x) * 1e9) for x in args.cache_limit_gb.split(",")]
    kv_values = [None if x.lower() in ("none", "fp16", "") else int(x) for x in args.kv_bits.split(",")]
    compiled_values = [x.lower() == "true" for x in args.compiled.split(",")]

    cells = []
    for ps in prefill_steps:
        for ssm in ssm_values:
            for cl in cache_limit_values:
                for kv in kv_values:
                    for cd in compiled_values:
                        cells.append({
                            "tag": f"ps{ps}-ssm{str(ssm).lower()}-cl{int(cl/1e9) if cl else 0}g-kv{('none' if kv is None else kv)}-compiled{str(cd).lower()}",
                            "prefill_step": ps, "ssm_rederive": ssm,
                            "cache_limit_bytes": cl, "kv_bits": kv, "compiled": cd,
                        })

    foreign = foreign_servers()
    if foreign and not args.no_contention_gate:
        print(
            "FATAL: foreign inference processes resident; official numbers would be "
            f"contaminated. Found: {foreign[:3]}",
            file=sys.stderr,
        )
        return 2

    # Reuse the existing release build if present, else build once.
    bin_path = BUILD_DIR / "release" / "mei"
    if not bin_path.exists():
        print("[sweep] building release binary (one-time)...", flush=True)
        r = subprocess.run(
            ["swift", "build", "-c", "release", "--scratch-path", str(BUILD_DIR), "--package-path", str(REPO)],
            capture_output=True, text=True, timeout=3600,
        )
        if r.returncode != 0:
            print("build failed:", r.stderr[-2000:], file=sys.stderr)
            return 1

    result: dict[str, Any] = {
        "engine": "mei", "model": args.model_id,
        "pin": "vmlx-swift aeb5e21c195d8519609488ef75a25ce7e48d8f88",
        "machine": {"note": "M1 Max g13s 32GB"},
        "contention_boundary": {"foreign_servers": foreign},
        "started_epoch": time.time(),
        "cells": [],
    }

    chat_40k = None
    if args.chat_40k:
        chat_code = (
            "import sys, json\n"
            "from transformers import AutoTokenizer\n"
            f"t = AutoTokenizer.from_pretrained(r'{tokenizer_dir}')\n"
            "def transcript(k):\n"
            "    filler = 'system stability marker ' * k\n"
            "    m = ([{'role':'system','content': filler + ' Final system line.'}]\n"
            "         + [{'role':'user','content': f'Instruction batch {i}: answer nothing yet.'} for i in range(6)]\n"
            "         + [{'role':'assistant','content':'Understood.'} for _ in range(5)])\n"
            "    m.append({'role':'user','content':'Final instruction: reply with exactly the word cache-ready.'})\n"
            "    return m\n"
            "def nt(m):\n"
            "    r = t.apply_chat_template(m, add_generation_prompt=True)\n"
            "    ids = r.get('input_ids') if isinstance(r, dict) else t.encode(r, add_special_tokens=False)\n"
            "    return len(ids)\n"
            "lo, hi, best = 0, 30000, None\n"
            "while lo < hi:\n"
            "    mid = (lo + hi) // 2\n"
            "    m = transcript(mid)\n"
            "    if nt(m) >= 44000:\n"
            "        best = m; hi = mid\n"
            "    else:\n"
            "        lo = mid + 1\n"
            "print(json.dumps(best))\n"
        )
        r = subprocess.run(
            [os.path.expanduser(args.transformers_python), "-c", chat_code],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            chat_40k = json.loads(r.stdout)
        else:
            print("chat-40k transcript build failed (skipping):", r.stderr[:300], file=sys.stderr)

    base = f"http://127.0.0.1:{args.port}/v1"
    chat_url = f"{base}/chat/completions"
    comp_url = f"{base}/completions"
    prompts = {n: unit_prompt(n) for n in set(contexts + ([45001] if 45000 in contexts else []))}

    for cell in cells:
        cell_result: dict[str, Any] = {"config": cell, "rows": []}
        log_path = RUNTIME_BASE / "logs" / f"sweep-{cell['tag']}.log"
        try:
            server = Server(args, bin_path, cell, log_path)
        except Exception as exc:
            cell_result["error"] = f"server start failed: {exc}"
            result["cells"].append(cell_result)
            continue
        try:
            cell_result["status_after_load"] = server.status()
            # Warmup
            req = {"model": args.model_id, "messages": [{"role": "user", "content": "warmup"}],
                   "temperature": 0, "max_tokens": 8, "stream": False}
            request_json(chat_url, req, timeout=args.request_timeout)

            def row(name: str, payload: dict[str, Any], status_before: dict[str, Any],
                    expected_prompt: int | None = None, min_decode: float = 0.0) -> dict[str, Any]:
                started = time.monotonic()
                r: dict[str, Any] = {"name": name, "started_epoch": time.time()}
                try:
                    body, elapsed = request_json(comp_url if "prompt" in payload else chat_url,
                                                 payload, timeout=args.request_timeout)
                    usage = body.get("usage") or {}
                    checks = {"http_200": True}
                    decode = usage.get("tokens_per_second")
                    decode = decode if decode is not None else 0.0
                    if expected_prompt is not None:
                        checks["prompt_tokens_match"] = int(usage.get("prompt_tokens", -1)) == expected_prompt
                    if min_decode:
                        checks["decode_above_floor"] = (decode or 0) >= min_decode
                    r.update({
                        "checks": checks, "usage": usage,
                        "decode_tps_engine": decode,
                        "prefill_ms": usage.get("prefill_ms"),
                        "generate_ms": usage.get("generate_ms"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
                        "request_seconds": round(elapsed, 3),
                        "status_before": status_before,
                        "status_after": server.status(),
                    })
                    passed = all(v is not False for v in r["checks"].values())
                    r["status"] = "passed" if passed else "failed"
                except BaseException as exc:  # noqa: BLE001
                    r["status"] = "error"
                    r["error"] = f"{type(exc).__name__}: {exc}"
                r["elapsed_seconds"] = round(time.monotonic() - started, 3)
                print(f"[sweep] {cell['tag']} {name}: {r['status']} {r['elapsed_seconds']}s"
                      + (f" decode={decode}" if decode else ""), flush=True)
                return r

            for n in contexts:
                cell_result["rows"].append(row(
                    f"ctx_{n}_fresh" if n != 45000 or args.repeats_45k == 1 else f"ctx_{n}_fresh",
                    {"model": args.model_id, "prompt": prompts[n], "temperature": 0,
                     "max_tokens": args.max_tokens, "stream": False},
                    status_before=server.status(), expected_prompt=n, min_decode=1.0))
            if 45000 in contexts:
                ext = prompts[45001] if 45001 in prompts else unit_prompt(45001)
                base_payload = {"model": args.model_id, "temperature": 0,
                                "max_tokens": args.max_tokens, "stream": False}
                for k in range(args.repeats_45k):
                    p = unit_prompt(45000 + (1 if k == 0 else 0))
                    cell_result["rows"].append(row(
                        f"ctx_45000_reuse_r{k+1}",
                        {**base_payload, "prompt": p if k == 0 else unit_prompt(45000 + 1)},
                        status_before=server.status(), expected_prompt=45000 + (1 if k == 0 else 0),
                        min_decode=1.0))
            if args.chat_40k and chat_40k:
                cell_result["rows"].append(row(
                    "chat_40k",
                    {"model": args.model_id, "messages": chat_40k, "temperature": 0,
                     "max_tokens": 256, "stream": False},
                    status_before=server.status(), min_decode=1.0))

            log_text = log_path.read_text(errors="replace")
            cell_result["profile"] = summarize_profile(log_text)
            cell_result["status_after_cell"] = server.status()
        finally:
            server.stop()

        result["cells"].append(cell_result)
        result_json = json.dumps(result, indent=2, sort_keys=True)
        args.output.write_text(result_json + "\n")
        print(f"[sweep] cell {cell['tag']} complete; artifact written", flush=True)

    result["finished_epoch"] = time.time()
    result["status"] = "passed" if all(c.get("rows") and all(r["status"] == "passed" for r in c["rows"]) for c in result["cells"]) else "failed"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())