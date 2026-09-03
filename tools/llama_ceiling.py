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
  python3 tools/llama_ceiling.py --gguf <path> --provenance-only
    # verify binary + GGUF header + sha256 against the pinned official
    # blob digest WITHOUT launching llama-server (safe under contention)
"""
from __future__ import annotations

import argparse
import hashlib
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

# The official ornith-ai/Ornith-1.5-9B-GGUF artifact this ceiling is pinned
# to BY CONTENT (repo revision abdd624b12ebf020b767fff532ff44fe552b28c3,
# blob sha256 verified 2026-08-30 against huggingface.co):
PINNED_GGUF_SHA256 = "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6"
GGUF_REVISION = "abdd624b12ebf020b767fff532ff44fe552b28c3"
GGUF_REPO = "ornith-ai/Ornith-1.5-9B-GGUF"


def foreign_servers(exclude_self_pid: int, extra_exclude_pids: set[int] | None = None) -> list[str]:
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return ["ps-failed"]
    found = []
    for line in out.splitlines():
        pid = int(line.split(None, 1)[0]) if line.split(None, 1)[0].isdigit() else -1
        if pid == exclude_self_pid or (extra_exclude_pids and pid in extra_exclude_pids):
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


def sha256_file(path: Path) -> str:
    """Streaming sha256 (never loads the multi-GB GGUF into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def llama_version(bin_path: str | None) -> tuple[str, str]:
    """(version line, resolved binary path). Raises RuntimeError if the
    llama-server executable cannot be found — the ceiling cannot measure
    something that is not installed."""
    resolved = bin_path or shutil.which("llama-server")
    if not resolved:
        raise RuntimeError(
            "llama-server binary not found on PATH and no --llama-bin given; "
            "install via `brew install llama.cpp`")
    out = subprocess.run(
        [resolved, "--version"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"llama-server --version failed (rc {out.returncode}): {out.stderr[:200]}")
    text = (out.stdout or "").strip() or (out.stderr or "").strip()
    lines = [ln for ln in text.splitlines() if "version" in ln.lower() or "built" in ln.lower()]
    first = (lines[0] if lines else text.splitlines()[0]) if text.splitlines() else "unknown"
    return first, resolved


def gguf_meta_summary(path: Path) -> dict[str, Any]:
    """GGUF header facts for comparator hygiene: arch, quant, context length,
    MTP head presence. Imported from gguf_meta.py (Mei tool, no deps).
    Arch-aware since llama.cpp builds write arch-prefixed keys
    (qwen35.*, qwen35moe.*, gemma4.*); bare fallbacks cover both."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_meta import read_gguf_meta  # type: ignore[import-not-found]
    keys = (
        "general.architecture,general.file_type,general.quantization_version,"
        "context_length,llama.context_length,nextn_predict_layers,"
        "full_attention_interval,block_count")
    meta = read_gguf_meta(path, [k.strip() for k in keys.split(",")])["metadata"]
    arch = meta.get("general.architecture")
    npred = meta.get(f"{arch}.nextn_predict_layers") or meta.get("nextn_predict_layers")
    return {
        "arch": arch,
        "file_type": meta.get("general.file_type"),
        "quant_version": meta.get("general.quantization_version"),
        "context_length": meta.get(f"{arch}.context_length")
        or meta.get("llama.context_length") or meta.get("context_length"),
        "full_attention_interval": meta.get(f"{arch}.full_attention_interval")
        or meta.get("full_attention_interval"),
        "block_count": meta.get(f"{arch}.block_count") or meta.get("block_count"),
        "mtp_head_present": bool(npred),
        "mtp_nextn_predict_layers": npred,
        "mtp_note": (
            "llama.cpp runs WITHOUT --spec-type draft-mtp; single-token decode, "
            "comparable to Mei's no-MTP baseline" if npred else "no MTP head"),
    }


def verify_provenance(
    gguf: Path, expected_sha256: str | None, compute_sha256: bool,
    arch: str = "qwen35", repo: str = GGUF_REPO, revision: str = GGUF_REVISION,
) -> tuple[bool, dict[str, Any]]:
    """Independent pre-launch provenance gate for the ceiling GGUF.

    Checks (each recorded, mismatch/absence is FATAL):
      1. GGUF file exists, is non-empty, and starts with the GGUF magic.
      2. sha256 matches the pinned official blob digest (content pin).
      3. GGUF header is a same-family model with context >= 65536
         and records MTP-head presence (comparator hygiene; no --spec-type
         is ever passed so the decode path stays single-token).
      4. llama-server binary exists and reports a version.
    """
    prov: dict[str, Any] = {
        "repo": repo,
        "revision": revision,
        "expected_sha256": expected_sha256,
        "path": str(gguf),
    }
    ok = True
    if not gguf.exists() or gguf.stat().st_size == 0:
        return False, {**prov, "error": f"GGUF missing or empty: {gguf}"}
    prov["size_bytes"] = gguf.stat().st_size
    with open(gguf, "rb") as f:
        magic = f.read(4)
    prov["magic_ok"] = magic == b"GGUF"
    ok = ok and magic == b"GGUF"
    try:
        meta = gguf_meta_summary(gguf)
        prov["meta"] = meta
        # Architecture is a family-prefix gate (qwen35 matches qwen35 and
        # qwen35moe; gemma matches gemma4). The sha256 content pin above is
        # the exact-identity gate; arch only guards same-family hygiene.
        meta_arch = str(meta.get("arch") or "")
        ok = ok and bool(meta_arch) and meta_arch.startswith(arch)
        # Context: only enforceable when the header records it. Several
        # official GGUFs (ornith-ai 9B/35B, mudler gemma4) carry NO
        # context_length metadata key — the runtime context is set by the
        # explicit llama-server --ctx-size launch arg, so its absence is
        # recorded as a note, not a failure, as long as the server is
        # launched with --ctx-size >= 65536.
        ctx = meta.get("context_length")
        if ctx is not None:
            ok = ok and int(ctx) >= 65536
        else:
            prov["context_note"] = (
                "context_length absent from GGUF header; runtime context is "
                "the llama-server --ctx-size launch arg")
    except Exception as exc:  # noqa: BLE001
        prov["meta_error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    if compute_sha256:
        digest = sha256_file(gguf)
        prov["sha256"] = digest
        prov["digest_matches_pinned"] = (expected_sha256 is not None) and digest == expected_sha256
        ok = ok and prov["digest_matches_pinned"]
    else:
        prov["sha256"] = "skipped (--no-verify-gguf)"
    try:
        version, resolved = llama_version(None)
        prov["llama_version"] = version
        prov["llama_bin"] = resolved
    except RuntimeError as exc:
        prov["llama_error"] = str(exc)
        ok = False
    return ok, prov


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
    parser.add_argument("--alias", default=None,
                        help="served alias; required unless --provenance-only")
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
    parser.add_argument("--mode", choices=["45k", "short"], default="45k",
                        help="45k: Ornith 45K/40K row set (default, unchanged). "
                             "short: Mei probe_load-style short decode rows "
                             "(3x /completion + 1 chat row) with timings.")
    parser.add_argument("--gguf-sha256", default=PINNED_GGUF_SHA256,
                        help="expected blob sha256 (default: Ornith-9B pin)")
    parser.add_argument("--gguf-repo", default=GGUF_REPO,
                        help="pinned source repo (default: Ornith-9B)")
    parser.add_argument("--gguf-revision", default=GGUF_REVISION,
                        help="pinned source revision (default: Ornith-9B)")
    parser.add_argument("--arch", default="qwen35",
                        help="expected GGUF architecture (default: qwen35)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-contention-gate", action="store_true")
    parser.add_argument("--no-verify-gguf", action="store_true",
                        help="skip the sha256 content check (hashes 5.8GB; "
                        "use only when provenance was already verified this session)")
    parser.add_argument("--provenance-only", action="store_true",
                        help="verify binary + GGUF header + sha256 against the "
                        "pinned official digest WITHOUT launching llama-server")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Independent provenance gate (binary, GGUF header, content digest)
    # runs BEFORE any server launch; provenance-only mode stops here.
    verified, prov = verify_provenance(
        args.gguf, args.gguf_sha256, compute_sha256=not args.no_verify_gguf,
        arch=args.arch, repo=args.gguf_repo, revision=args.gguf_revision)
    if args.provenance_only:
        # Persist the provenance block as an artifact too (evidence
        # preservation): the caller's --output must never be empty when a
        # provenance check ran, and a digest mismatch keeps the artifact
        # (with verified=False) so the failure is auditable.
        record = {
            "mode": "provenance-only",
            "verified": verified,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        record.update(prov)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(prov, indent=2, sort_keys=True))
        print("PROVENANCE", "OK" if verified else "FAIL")
        return 0 if verified else 2
    if not verified:
        print("FATAL: GGUF provenance verification failed; refusing to measure:",
              json.dumps(prov, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    if not args.alias:
        parser.error("--alias is required unless --provenance-only")

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
        "gguf_provenance": prov,
        "llama_version": prov.get("llama_version", "unknown"),
        "methodology": (
            "Mei-contained ceiling probe; one request at a time; 3 repeats; ctx>=64K; "
            "mode=" + args.mode + " (45k = same exact 45K-token prompt + 40K chat pattern "
            "as the Mei bench; short = Mei probe_load-style short decode rows)"),
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
                             "mem_before": server.status(),
                             "contended_during_row": bool(
                                 foreign_servers(exclude_self_pid=os.getpid(),
                                                 extra_exclude_pids={server.proc.pid}))}
        usage: dict[str, Any] = {}
        try:
            body, elapsed = post_json(
                chat_url if "messages" in payload else comp_url, payload, timeout=args.request_timeout)
            usage = body.get("usage") or {}
            td = (body.get("choices") or [{}])[0]
            msg = td.get("message") or {}
            text = (
                msg.get("content") or msg.get("reasoning_content")
                or td.get("content") or td.get("text")
                or body.get("content") or body.get("text") or ""
            )
            text_source = (
                "message.content" if msg.get("content")
                else "message.reasoning_content" if msg.get("reasoning_content")
                else "choice.content" if td.get("content")
                else "choice.text" if td.get("text")
                else "body.content" if body.get("content")
                else "body.text" if body.get("text")
                else "none")
            checks: dict[str, Any] = {"http_200": True, "nonempty": bool(text)}
            r["text_source"] = text_source
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

    if args.mode == "short":
        # Mei probe_load-style short decode ceiling: same short prompt and
        # token budget as Mei's short_decode probe ("Count from 1 to 10, one
        # per line.", max_tokens 32, temp 0), raw /completion so
        # timings.predicted_per_second is the decode rate (prefill excluded),
        # three repeats. One chat-format row confirms the template path works
        # end-to-end and records wall-clock t/s (usage only, no timings).
        short_prompt = "Count from 1 to 10, one per line."
        for k in range(args.repeats):
            result["rows"].append(row(
                f"short_fresh_r{k+1}",
                {"model": args.alias, "prompt": short_prompt, "temperature": 0,
                 "max_tokens": args.max_tokens, "stream": False}))
        result["rows"].append(row(
            "short_chat",
            {"model": args.alias,
             "messages": [{"role": "user", "content": short_prompt}],
             "temperature": 0, "max_tokens": args.max_tokens, "stream": False}))
    else:
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