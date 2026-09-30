#!/usr/bin/env python3
"""Reproducible real-model gate for the source-built Cmlx metallib.

The gate keeps one release binary fixed and swaps only the adjacent
``mlx.metallib``.  Each invocation runs one cold server and writes a durable
JSON result.  Invoke this script once per behavior leg and once per
performance leg; use ``--validate``/``--report`` after the round-robin campaign.

Behavior legs run the repository's local-model-bench ``probe_mei.py`` with
cache and exact context-cap checks enabled.  Performance legs run a warm-up and
matched short/30k-token completion requests with cache reuse disabled, so the
artifact is the only changed variable.

No model paths or benchmark results are inferred from names.  The source and
stock libraries, provenance sidecars, binary, model directory, served ID, and
probe path are explicit command-line inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SWITCHES = (
    "VMLX_QWEN35_COMPILE_DECODE_REGIONS",
    "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"FATAL: {label} is not a file: {path}")
    return path


def json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 1800.0) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def wait_ready(process: subprocess.Popen[Any], base_url: str, model_id: str, timeout: float = 1800.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with rc={process.returncode}")
        try:
            data, _ = json_request(url, timeout=10)
            ids = [entry.get("id") for entry in data.get("data", [])]
            if model_id in ids:
                return
        except (OSError, ValueError, RuntimeError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not expose {model_id!r} after {timeout}s")


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or [{}]
    choice = choices[0]
    return choice.get("text") or ((choice.get("message") or {}).get("content") or "")


def prepare_stage(args: argparse.Namespace, arm: str) -> tuple[Path, dict[str, Any]]:
    root = args.output.resolve()
    stage = root / "stage" / arm / "bin"
    stage.mkdir(parents=True, exist_ok=True)
    binary = require_file(args.binary, "fixed Mei binary")
    if arm == "source":
        library = require_file(args.source_metallib, "source-built metallib")
        provenance = require_file(args.source_provenance, "source-built provenance")
    else:
        library = require_file(args.stock_metallib, "stock metallib")
        provenance = require_file(args.stock_provenance, "stock provenance")
    staged_binary = stage / "mei"
    staged_library = stage / "mlx.metallib"
    staged_default = stage / "default.metallib"
    staged_provenance = stage / "mlx.metallib.provenance"
    shutil.copy2(binary, staged_binary)
    shutil.copy2(library, staged_library)
    shutil.copy2(library, staged_default)
    shutil.copy2(provenance, staged_provenance)
    staged_binary.chmod(0o755)
    digest = sha256(staged_library)
    sidecar = staged_provenance.read_text(encoding="utf-8", errors="replace")
    declared = ""
    for line in sidecar.splitlines():
        if line.startswith("sha256:"):
            declared = line.split(":", 1)[1].strip()
            break
    if declared and declared != digest:
        raise RuntimeError(f"{arm} provenance sha256 {declared} != staged library {digest}")
    if sha256(staged_default) != digest:
        raise RuntimeError(f"{arm} default.metallib is not byte-identical to mlx.metallib")
    metadata = {
        "arm": arm,
        "binary": str(staged_binary),
        "binary_sha256": sha256(staged_binary),
        "metallib": str(staged_library),
        "metallib_sha256": digest,
        "metallib_bytes": staged_library.stat().st_size,
        "provenance": str(staged_provenance),
        "provenance_text": sidecar,
        "declared_metallib_sha256": declared or None,
    }
    return staged_binary, metadata


def launch_command(args: argparse.Namespace, binary: Path, request_log: Path, cache_reuse: bool, kv_dir: Path, port: int) -> list[str]:
    return [
        str(binary),
        "--model-dir", str(args.model_dir),
        "--served-model-id", args.model_id,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--context-cap", str(args.context_cap),
        "--prefill-step-size", str(args.prefill_step_size),
        "--max-tokens", str(args.max_tokens),
        "--temperature", "0",
        "--top-p", "1",
        "--top-k", "1",
        "--enable-thinking", "false",
        "--emit-reasoning", "false",
        "--cache-reuse", "true" if cache_reuse else "false",
        "--kv-cache-dir", str(kv_dir),
        "--request-log", str(request_log),
        "--log-requests", "true",
    ]


def env_for_server() -> dict[str, str]:
    env = dict(os.environ)
    for name in SWITCHES:
        env.pop(name, None)
    env["VMLX_ENABLE_UNSAFE_COMPILE"] = "1"
    env["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"] = "0"
    return env


def run_probe(args: argparse.Namespace, base_url: str, output: Path) -> dict[str, Any]:
    command = [
        str(args.probe_python),
        str(args.probe),
        "--base-url", base_url,
        "--model", args.model_id,
        "--tokenizer", str(args.model_dir),
        "--output", str(output),
        "--context-cap", str(args.context_cap),
        "--timeout", str(args.request_timeout),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.probe_timeout)
    if output.exists():
        result = json.loads(output.read_text())
    else:
        result = {"status": "failed", "probes": {}, "error": "probe produced no JSON"}
    result["probe_returncode"] = completed.returncode
    result["probe_command"] = command
    result["probe_stdout_tail"] = completed.stdout[-4000:]
    result["probe_stderr_tail"] = completed.stderr[-4000:]
    return result


def tool_projection(probe: dict[str, Any]) -> dict[str, Any]:
    entry = probe.get("probes", {}).get("tool_nonstreaming", {})
    call = entry.get("validated_call") or {}
    return {"name": call.get("name"), "arguments": call.get("arguments"), "finish_reason": call.get("finish_reason")}


def cache_projection(probe: dict[str, Any]) -> dict[str, Any]:
    probes = probe.get("probes", {})
    return {
        name: {
            "status": item.get("status"),
            "cached_tokens": (item.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
        }
        for name, item in probes.items()
        if name.startswith("cache_")
    }


def run_behavior(args: argparse.Namespace, binary: Path, identity: dict[str, Any]) -> dict[str, Any]:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    port = args.port
    base = f"http://127.0.0.1:{port}/v1"
    request_log = out / "request.jsonl"
    server_log = out / "server.log"
    kv_dir = out / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    command = launch_command(args, binary, request_log, True, kv_dir, port)
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "behavior",
        "identity": identity,
        "model_dir": str(args.model_dir),
        "model_id": args.model_id,
        "context_cap": args.context_cap,
        "prefill_step_size": args.prefill_step_size,
        "command": command,
        "environment_overrides": {"VMLX_ENABLE_UNSAFE_COMPILE": "1", "VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES": "0"},
        "started_epoch": time.time(),
    }
    process: subprocess.Popen[Any] | None = None
    try:
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env_for_server(), stdout=log, stderr=subprocess.STDOUT, text=True)
        wait_ready(process, base, args.model_id)
        status, _ = json_request(f"{base}/mei/status", timeout=30)
        record["mei_status"] = status
        probe_path = out / "probe.json"
        record["probe"] = run_probe(args, base, probe_path)
        record["probe_path"] = str(probe_path)
        record["status"] = "passed" if record["probe"].get("status") == "passed" and record["probe"].get("probe_returncode") == 0 else "failed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=60)
            record["server_returncode"] = process.returncode
            record["server_stopped"] = process.poll() is not None
        else:
            record["server_stopped"] = True
        record["finished_epoch"] = time.time()
        (out / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run_performance(args: argparse.Namespace, binary: Path, identity: dict[str, Any]) -> dict[str, Any]:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    port = args.port
    base = f"http://127.0.0.1:{port}/v1"
    request_log = out / "request.jsonl"
    server_log = out / "server.log"
    kv_dir = out / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    command = launch_command(args, binary, request_log, False, kv_dir, port)
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "performance",
        "identity": identity,
        "model_dir": str(args.model_dir),
        "model_id": args.model_id,
        "context_cap": args.context_cap,
        "prefill_step_size": args.prefill_step_size,
        "repeats": args.repeats,
        "command": command,
        "environment_overrides": {"VMLX_ENABLE_UNSAFE_COMPILE": "1", "VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES": "0"},
        "started_epoch": time.time(),
        "rows": [],
    }
    process: subprocess.Popen[Any] | None = None
    try:
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env_for_server(), stdout=log, stderr=subprocess.STDOUT, text=True)
        wait_ready(process, base, args.model_id)
        warmup, warmup_wall = json_request(f"{base}/completions", {
            "model": args.model_id, "prompt": "Reply exactly warmup.", "temperature": 0, "top_p": 1, "top_k": 1, "max_tokens": 8, "stream": False,
        }, timeout=args.request_timeout)
        record["warmup"] = {"wall_seconds": warmup_wall, "text": response_text(warmup)}
        prompts = {"short": "Count from 1 to 10, one per line.", "loaded": " hello" * 30000}
        for context, prompt in prompts.items():
            for repeat in range(1, args.repeats + 1):
                body, wall = json_request(f"{base}/completions", {
                    "model": args.model_id, "prompt": prompt + (f"\nRepeat {repeat}." if context == "loaded" else ""),
                    "temperature": 0, "top_p": 1, "top_k": 1, "max_tokens": 64, "seed": 1, "stream": False,
                }, timeout=args.request_timeout)
                usage = body.get("usage") or {}
                text = response_text(body)
                completion = int(usage.get("completion_tokens", 0))
                row = {
                    "context": context, "repeat": repeat, "wall_seconds": wall,
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                    "total_tokens": usage.get("total_tokens"), "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "decode_tps_end_to_end": completion / wall if wall and completion else None,
                    "prefill_ms": usage.get("prefill_ms"),
                    "generate_ms": usage.get("generate_ms"),
                    "decode_tps": usage.get("tokens_per_second"),
                    "prompt_tps": usage.get("prompt_tokens_per_second"),
                    "mei_memory_peak_bytes": usage.get("mei_memory_peak_bytes"),
                }
                record["rows"].append(row)
                print(f"{out.name} {context} r{repeat}: prompt={row['prompt_tokens']} completion={completion} wall={wall:.3f}s", flush=True)
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=60)
            record["server_returncode"] = process.returncode
            record["server_stopped"] = process.poll() is not None
        else:
            record["server_stopped"] = True
        record["finished_epoch"] = time.time()
        (out / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("**/result.json")):
        if "stage" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            results.append({"status": "failed", "error": f"invalid JSON {path}: {exc}", "_path": str(path)})
            continue
        # Older runs of this driver wrote client timing rows before the
        # server-side decomposition was added.  Recover the exact matching
        # request-log rows by invocation order (the first completion is the
        # warm-up) so reports remain reproducible without editing raw JSON.
        if data.get("kind") == "performance" and data.get("rows"):
            request_path = path.parent / "request.jsonl"
            try:
                request_rows = [json.loads(line) for line in request_path.read_text().splitlines() if line.strip()]
                request_rows = [row for row in request_rows if row.get("kind") == "completion"]
                result_rows = data["rows"]
                if len(request_rows) >= len(result_rows) + 1:
                    request_rows = request_rows[-len(result_rows):]
                if len(request_rows) == len(result_rows):
                    for result_row, request_row in zip(result_rows, request_rows):
                        result_row.setdefault("prefill_ms", request_row.get("prefill_ms"))
                        result_row.setdefault("generate_ms", request_row.get("generate_ms"))
                        result_row.setdefault("decode_tps", request_row.get("decode_tps"))
                        result_row.setdefault("prompt_tps", request_row.get("prompt_tps"))
                        result_row.setdefault("mei_memory_peak_bytes", request_row.get("mem_peak_bytes"))
            except (OSError, ValueError, TypeError):
                pass
        data["_path"] = str(path)
        results.append(data)
    return results


def mean_rows(result: dict[str, Any], context: str, field: str) -> float | None:
    values = [row[field] for row in result.get("rows", []) if row.get("context") == context and isinstance(row.get(field), (int, float))]
    return statistics.fmean(values) if values else None


def behavior_summary(result: dict[str, Any]) -> dict[str, Any]:
    probe = result.get("probe") or {}
    probes = probe.get("probes") or {}
    return {
        "path": result.get("_path"),
        "arm": (result.get("identity") or {}).get("arm"),
        "status": result.get("status"),
        "server_stopped": result.get("server_stopped"),
        "probe_status": probe.get("status"),
        "probe_returncode": probe.get("probe_returncode"),
        "probe_checks": {name: item.get("status") for name, item in probes.items()},
        "plain_content": (probes.get("plain_completion") or {}).get("content"),
        "parity_content": (probes.get("parity_stream_vs_nonstream") or {}).get("content"),
        "tool": tool_projection(probe),
        "cache": cache_projection(probe),
        "context_cap": {
            name: (probes.get(name) or {}).get("status")
            for name in ("context_exact_cap", "context_over_cap_rejected")
        },
    }


def report(root: Path) -> tuple[dict[str, Any], str]:
    results = load_results(root)
    behaviors = [r for r in results if r.get("kind") == "behavior"]
    performances = [r for r in results if r.get("kind") == "performance"]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_arm.setdefault((result.get("identity") or {}).get("arm", "unknown"), []).append(result)
    summary: dict[str, Any] = {
        "schema": 1,
        "root": str(root),
        "results": len(results),
        "behavior_results": len(behaviors),
        "performance_results": len(performances),
        "behavior": [behavior_summary(r) for r in behaviors],
        "performance": [],
        "checks": {},
    }
    for result in performances:
        summary["performance"].append({
            "path": result.get("_path"), "arm": (result.get("identity") or {}).get("arm"), "status": result.get("status"), "server_stopped": result.get("server_stopped"),
            "rows": len(result.get("rows", [])), "short_wall_mean": mean_rows(result, "short", "wall_seconds"), "loaded_wall_mean": mean_rows(result, "loaded", "wall_seconds"),
            "short_tps_mean": mean_rows(result, "short", "decode_tps_end_to_end"), "loaded_tps_mean": mean_rows(result, "loaded", "decode_tps_end_to_end"),
            "short_decode_tps_mean": mean_rows(result, "short", "decode_tps"), "loaded_decode_tps_mean": mean_rows(result, "loaded", "decode_tps"),
            "short_prompt_tps_mean": mean_rows(result, "short", "prompt_tps"), "loaded_prompt_tps_mean": mean_rows(result, "loaded", "prompt_tps"),
            "short_prefill_ms_mean": mean_rows(result, "short", "prefill_ms"), "loaded_prefill_ms_mean": mean_rows(result, "loaded", "prefill_ms"),
            "short_generate_ms_mean": mean_rows(result, "short", "generate_ms"), "loaded_generate_ms_mean": mean_rows(result, "loaded", "generate_ms"),
            "short_peak_bytes_max": max((row.get("mei_memory_peak_bytes", 0) or 0 for row in result.get("rows", []) if row.get("context") == "short"), default=0),
            "loaded_peak_bytes_max": max((row.get("mei_memory_peak_bytes", 0) or 0 for row in result.get("rows", []) if row.get("context") == "loaded"), default=0),
        })
    checks: dict[str, Any] = {}
    checks["all_server_runs_stopped"] = bool(results) and all(r.get("server_stopped") is True for r in results)
    checks["all_runs_passed"] = bool(results) and all(r.get("status") == "passed" for r in results)
    source_behaviors = [r for r in behaviors if (r.get("identity") or {}).get("arm") == "source"]
    stock_behaviors = [r for r in behaviors if (r.get("identity") or {}).get("arm") == "stock"]
    first_source = behavior_summary(source_behaviors[0]) if source_behaviors else None
    deterministic_fields = ("arm", "status", "server_stopped", "probe_status", "probe_returncode", "probe_checks", "plain_content", "parity_content", "tool", "cache", "context_cap")
    checks["source_behavior_determinism"] = False
    if first_source is not None and len(source_behaviors) >= 2:
        checks["source_behavior_determinism"] = all(
            all(behavior_summary(r).get(field) == first_source.get(field) for field in deterministic_fields)
            for r in source_behaviors[1:]
        )
    checks["source_behavior_count"] = len(source_behaviors)
    checks["stock_behavior_count"] = len(stock_behaviors)
    if source_behaviors and stock_behaviors:
        left, right = behavior_summary(source_behaviors[0]), behavior_summary(stock_behaviors[0])
        checks["cross_library_plain_output_equal"] = left["plain_content"] == right["plain_content"]
        checks["cross_library_parity_output_equal"] = left["parity_content"] == right["parity_content"]
        checks["cross_library_tool_equal"] = left["tool"] == right["tool"]
        checks["cross_library_cache_check_equal"] = left["cache"] == right["cache"]
        checks["cross_library_context_boundary_equal"] = left["context_cap"] == right["context_cap"]
        checks["cross_library_tool_valid"] = left["tool"].get("name") == "add_numbers" and left["tool"].get("arguments") == {"a": 15, "b": 27}
    perf_by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in performances:
        perf_by_arm.setdefault((r.get("identity") or {}).get("arm", "unknown"), []).append(r)
    for arm, rows in perf_by_arm.items():
        for context in ("short", "loaded"):
            values = [mean_rows(r, context, "decode_tps_end_to_end") for r in rows]
            values = [v for v in values if v is not None]
            checks[f"performance_{arm}_{context}_runs"] = len(values)
            if values: checks[f"performance_{arm}_{context}_mean_tps"] = statistics.fmean(values)
            decode_values = [mean_rows(r, context, "decode_tps") for r in rows]
            decode_values = [v for v in decode_values if v is not None]
            if decode_values: checks[f"performance_{arm}_{context}_mean_decode_tps"] = statistics.fmean(decode_values)
            prefill_values = [mean_rows(r, context, "prefill_ms") for r in rows]
            prefill_values = [v for v in prefill_values if v is not None]
            if prefill_values: checks[f"performance_{arm}_{context}_mean_prefill_ms"] = statistics.fmean(prefill_values)
    if perf_by_arm.get("source") and perf_by_arm.get("stock"):
        for context in ("short", "loaded"):
            for metric, label in (("decode_tps", "decode_tps"), ("prefill_ms", "prefill_ms")):
                s = [mean_rows(r, context, metric) for r in perf_by_arm["source"]]
                k = [mean_rows(r, context, metric) for r in perf_by_arm["stock"]]
                s = [v for v in s if v is not None]; k = [v for v in k if v is not None]
                if s and k:
                    s_mean, k_mean = statistics.fmean(s), statistics.fmean(k)
                    if k_mean:
                        checks[f"performance_{context}_source_vs_stock_{label}_delta_pct"] = (s_mean / k_mean - 1) * 100
    summary["checks"] = checks
    summary["gate_verdict"] = "pass" if all(v is True for k, v in checks.items() if k in {"all_server_runs_stopped", "all_runs_passed", "source_behavior_determinism", "cross_library_tool_valid", "cross_library_context_boundary_equal"}) else "pending-or-fail"
    markdown = [
        "# Cmlx real-model gate", "", f"- Root: `{root}`", f"- Results: {len(results)} ({len(behaviors)} behavior, {len(performances)} performance)", f"- Gate verdict: **{summary['gate_verdict']}**", "", "## Checks", "",
    ]
    for key, value in checks.items(): markdown.append(f"- `{key}`: `{value}`")
    markdown += ["", "## Behavior legs", ""]
    for item in summary["behavior"]:
        markdown += [f"### {item['arm']} — `{item['path']}`", f"- status: `{item['status']}`; stopped: `{item['server_stopped']}`; probe: `{item['probe_status']}` / rc `{item['probe_returncode']}`", f"- tool: `{json.dumps(item['tool'], sort_keys=True)}`", f"- context: `{json.dumps(item['context_cap'], sort_keys=True)}`", f"- cache: `{json.dumps(item['cache'], sort_keys=True)}`", ""]
    markdown += ["## Performance legs", "", "| Arm | Status | Rows | Short wall mean (s) | Loaded wall mean (s) | Short end-to-end tok/s | Loaded end-to-end tok/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["performance"]:
        markdown.append(f"| {item['arm']} | {item['status']} | {item['rows']} | {item['short_wall_mean']} | {item['loaded_wall_mean']} | {item['short_tps_mean']} | {item['loaded_tps_mean']} |")
    markdown += ["", "## Interpretation", "", "The source-built and stock libraries are intentionally compared as separate artifacts. Cross-library text identity is reported, not assumed: the plan explicitly allows numerical output differences. Tool-call structure, cache behavior, and exact context-cap behavior are separate acceptance gates. Performance values are descriptive measurements; round-robin order and repeat counts must be reviewed before attributing a difference to the metallib.", ""]
    return summary, "\n".join(markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--source-metallib", type=Path)
    parser.add_argument("--source-provenance", type=Path)
    parser.add_argument("--stock-metallib", type=Path)
    parser.add_argument("--stock-provenance", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--probe", type=Path, default=Path("/Users/tijs/projects/local-model-bench/runner/probe_mei.py"))
    parser.add_argument("--probe-python", type=Path, default=Path("/Users/tijs/projects/local-model-bench/.venv/bin/python"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("source", "stock"))
    parser.add_argument("--phase", choices=("behavior", "performance"))
    parser.add_argument("--port", type=int, default=18251)
    parser.add_argument("--context-cap", type=int, default=65536)
    parser.add_argument("--prefill-step-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--probe-timeout", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--report", action="store_true")
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    required = {"--binary": args.binary, "--source-metallib": args.source_metallib, "--source-provenance": args.source_provenance, "--stock-metallib": args.stock_metallib, "--stock-provenance": args.stock_provenance, "--model-dir": args.model_dir, "--model-id": args.model_id, "--probe": args.probe, "--probe-python": args.probe_python}
    missing = [name for name, value in required.items() if value is None]
    if missing: raise SystemExit("FATAL: missing required run inputs: " + ", ".join(missing))
    args.binary = require_file(args.binary, "binary")
    args.source_metallib = require_file(args.source_metallib, "source metallib")
    args.source_provenance = require_file(args.source_provenance, "source provenance")
    args.stock_metallib = require_file(args.stock_metallib, "stock metallib")
    args.stock_provenance = require_file(args.stock_provenance, "stock provenance")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.probe = require_file(args.probe, "probe")
    # Keep the venv launcher path rather than resolving its symlink into the
    # uv-managed base interpreter: the launcher carries local-model-bench's
    # installed tokenizer dependencies.
    args.probe_python = args.probe_python.expanduser()
    if not args.probe_python.is_file():
        raise SystemExit(f"FATAL: probe Python is not a file: {args.probe_python}")
    if not (args.model_dir / "config.json").is_file(): raise SystemExit(f"FATAL: model dir has no config.json: {args.model_dir}")
    if args.repeats < 1: raise SystemExit("FATAL: repeats must be positive")


def main() -> int:
    args = build_parser().parse_args()
    if args.validate or args.report:
        summary, markdown = report(args.output.resolve())
        report_json = args.output.resolve() / "cmlx-gate-report.json"
        report_md = args.output.resolve() / "cmlx-gate-report.md"
        report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        report_md.write_text(markdown)
        print(json.dumps({"verdict": summary["gate_verdict"], "results": summary["results"], "checks": summary["checks"], "json": str(report_json), "markdown": str(report_md)}, indent=2, sort_keys=True))
        return 0 if not args.validate or summary["gate_verdict"] == "pass" else 1
    validate_inputs(args)
    if not args.arm or not args.phase: raise SystemExit("FATAL: --arm and --phase are required for a run")
    binary, identity = prepare_stage(args, args.arm)
    if args.dry_run:
        command = launch_command(args, binary, args.output.resolve() / args.arm / "request.jsonl", args.phase == "behavior", args.output.resolve() / args.arm / "kv", args.port)
        print(json.dumps({"arm": args.arm, "phase": args.phase, "identity": identity, "command": command}, indent=2, sort_keys=True))
        return 0
    result = run_behavior(args, binary, identity) if args.phase == "behavior" else run_performance(args, binary, identity)
    print(json.dumps({"status": result.get("status"), "kind": result.get("kind"), "path": str(args.output.resolve() / "result.json"), "server_stopped": result.get("server_stopped")}, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" and result.get("server_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
