#!/usr/bin/env python3
"""Independent hardware-ceiling measurement for Ornith-1.5-9B via llama.cpp.

Mei-contained (artifact-only): all drivers/configs/logs/outputs live under
Mei; local-model-bench is never touched. Runs llama-server on an
exclusively Mei-owned port (8074) with the same exact 45K-token prompt and
40K chat pattern used by the Mei bench, one request at a time, three
repeats, context >= 64K, working-set/cache memory capture, and
context-aware decode statistics.

Gate: >=40 tok/s at 45K loaded justifies the vmlx fork strongly; 25-39 is
a likely hardware ceiling; <25 means stop chasing 40 tok/s and prioritize
reuse latency + prefill quality over raw decode.

Usage:
  python3 tools/llama_ceiling.py \
    --gguf ~/.local/share/local-model-bench/mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf \
    --alias ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M \
    --output artifacts/llama-ceiling-<ts>.json
  [--repeats 3] [--kv-cache-type-q8] [--no-contention-gate]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

PORT = 8074  # Mei-owned llama.cpp ceiling port; never collides with other agents' ports.
FOREIGN_SERVER_RE = re.compile(r"llama-server|vllm|omlx|cocore", re.IGNORECASE)


def foreign_servers(exclude_self_pid: int) -> list[str]:
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return ["ps-failed"]
    found = []
    for line in out.splitlines():
        pid = int(line.split(None, 1)[0]) if line.split(None, 1)[0].isdigit() else -1
        if pid == exclude_self_pid:
            continue
        if any(tok in line for tok in ("sweep_mei.py", "llama_ceiling.py", "mei-build", "mei-runtime")):
            continue
        if FOREIGN_SERVER_RE.search(line):
            found.append(line.strip())
    return found


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.monotonic() - started


def get_json(url: str, timeout: float = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def rss_of(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
        return int(out.strip()) * 1024
    except Exception:
        return -1


def wait_for_server(port: int, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            body = get_json(f"http://127.0.0.1:{port}/v1/models", timeout=10)
            if isinstance(body, dict) and body.get("data"):
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s")


class LlamaServer:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        argv = [
            shutil.which("llama-server") or str(args.llama_bin),
            "--model", str(args.gguf),
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--ctx-size", str(args.ctx_size),
            "--temp", "0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--alias", args.alias,
            "--parallel", "1",
            "--no-webui",
            "--metrics",
        ]
        if args.kv_cache_type_q8:
            argv += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        if args.cache_reuse_disable:
            argv += ["--no-cache-prompt"]
        print(f"[ceiling] llama-server: {' '.join(argv)}", flush=True)
        self.proc = subprocess.Popen(argv, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        wait_for_server(args.port, timeout=args.server_timeout)

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {"rss_bytes": rss_of(self.proc.pid)}
        try:
            metrics = get_json(f"http://127.0.0.1:{self.args.port}/metrics", timeout=10)
            if isinstance(metrics, str):
                for name in ("llamacpp:kv_cache_usage_ratio", "llamacpp:kv_cache_tokens"):
                    m = re.search(rf"^{re.escape(name)}{{.*?}} ([\d.e+-]+)", metrics, re.M)
                    if m:
                        info[name] = float(m.group(1))
        except Exception:
            pass
        return info

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def unit_prompt(target: int) -> str:
    return " hello" * target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--llama-bin", type=Path, default=None)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--ctx-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--server-timeout", type=float, default=1200)
    parser.add_argument("--kv-cache-type-q8", action="store_true",
                        help="run attention KV as q8_0 (llama.cpp-native quantized KV variant)")
    parser.add_argument("--cache-reuse-disable", action="store_true")
    parser.add_argument("--chat-40k", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-contention-gate", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: keyed to BRING-YOUR-OWN-CLEAN-WINDOW: the script refuses to run
    # while any foreign inference process (other agents' llama-server, vllm,
    # omlx, cocore) is resident unless explicitly forced.
    foreign = foreign_servers(exclude_self_pid=os.getpid())
    if foreign and not args.no_contention_gate:
        print(f"FATAL: foreign inference processes resident: {foreign[:3]}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {
        "engine": "llama.cpp (llama-server)",
        "gguf": str(args.gguf),
        "alias": args.alias,
        "llama_version": str(subprocess.run(
            ["llama-server", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[0]
            if shutil.which("llama-server") else "unknown"),
        "methodology": "Mei-contained ceiling probe; one request at a time; 3 repeats; ctx>=64K; same 45K exact-token prompt + 40K chat pattern as the Mei bench",
        "contention_boundary": {"foreign_servers": foreign},
        "started_epoch": time.time(),
        "rows": [],
    }

    log_path = Path(os.environ.get("MEI_RUNTIME_BASE", "~/.local/share/local-model-bench/mei-runtime")).expanduser() / "logs" / "llama-ceiling.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = LlamaServer(args, log_path)
    base = f"http://127.0.0.1:{args.port}/v1"
    comp_url = f"{base}/completions"
    chat_url = f"{base}/chat/completions"

    def row(name: str, payload: dict[str, Any], expected_prompt: int | None = None) -> dict[str, Any]:
        started = time.monotonic()
        r: dict[str, Any] = {"name": name, "started_epoch": time.time(),
                             "mem_before": server.status()}
        usage: dict[str, Any] = {}
        try:
            body, elapsed = post_json(
                chat_url if "messages" in payload else comp_url, payload, timeout=args.request_timeout)
            usage = body.get("usage") or {}
            td = (body.get("choices") or [{}])[0]
            text = (td.get("message") or td).get("content") or ""
            checks: dict[str, Any] = {"http_200": True, "nonempty": bool(text)}
            if expected_prompt is not None:
                checks["prompt_tokens_match"] = int(usage.get("prompt_tokens", -1)) == expected_prompt
            r.update({
                "checks": checks,
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                # llama.cpp reports timings separately
                "timings": body.get("timings"),
                "finish_reason": td.get("finish_reason"),
                "request_seconds": round(elapsed, 3),
                "mem_after": server.status(),
            })
            decode = None
            pp_tps = None
            if body.get("timings"):
                decode = body["timings"].get("predicted_per_second")
                pp_tps = body["timings"].get("prompt_per_second")
                if decode is None and body["timings"].get("predicted_n") and body["timings"].get("predicted_ms"):
                    decode = body["timings"]["predicted_n"] * 1000.0 / body["timings"]["predicted_ms"]
                if pp_tps is None and body["timings"].get("prompt_n") and body["timings"].get("prompt_ms"):
                    pp_tps = body["timings"]["prompt_n"] * 1000.0 / body["timings"]["prompt_ms"]
            r["decode_tps_engine"] = usage.get("tokens_per_second", decode)
            r["prompt_tokens_per_second"] = usage.get("prompt_tokens_per_second", pp_tps)
            passed = all(v is not False for v in checks.values())
            r["status"] = "passed" if passed else "failed"
        except BaseException as exc:  # noqa: BLE001
            r["status"] = "error"
            r["error"] = f"{type(exc).__name__}: {exc}"
        r["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(f"[ceiling] {name}: {r['status']} {r['elapsed_seconds']}s "
              f"decode={r.get('decode_tps_engine')} cached={r.get('cached_tokens')}", flush=True)
        return r

    # Warmup short request
    row("warmup_short", {"model": args.alias, "prompt": "hi", "temperature": 0, "max_tokens": 8, "stream": False})

    p45000 = unit_prompt(45000)
    for k in range(args.repeats):
        # fresh: identical exact 45K prompt each repeat (llama.cpp slot KV
        # makes repeats 2+ prefix-cache hits — recorded as cached_tokens)
        result["rows"].append(row(
            f"45k_fresh_r{k+1}",
            {"model": args.alias, "prompt": p45000, "temperature": 0,
             "max_tokens": args.max_tokens, "stream": False},
            expected_prompt=45000))
    # extension reuse row: prompt = 45001 tokens; slot cache from previous
    # request should cover 45000 of them.
    result["rows"].append(row(
        "45k_extension_reuse",
        {"model": args.alias, "prompt": unit_prompt(45001), "temperature": 0,
         "max_tokens": args.max_tokens, "stream": False},
        expected_prompt=45001))

    if args.chat_40k:
        # Same 40K chat transcript shape as the Mei bench (transcript token
        # count is server-reported; exactness vs the model template is not
        # required for a decode-throughput row).
        filler = "system stability marker " * 30000
        messages = (
            [{"role": "system", "content": filler + " Final system line."}]
            + [{"role": "user", "content": f"Instruction batch {i}: answer nothing yet."} for i in range(6)]
            + [{"role": "assistant", "content": "Understood."} for _ in range(5)]
            + [{"role": "user", "content": "Final instruction: reply with exactly the word cache-ready."}]
        )
        result["rows"].append(row(
            "chat_40k",
            {"model": args.alias, "messages": messages, "temperature": 0,
             "max_tokens": 256, "stream": False}))

    result["server_log"] = str(log_path)
    result["finished_epoch"] = time.time()
    result["status"] = (
        "passed"
        if result["rows"] and all(r["status"] == "passed" for r in result["rows"])
        else "failed")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    server.stop()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())