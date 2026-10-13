#!/usr/bin/env python3
"""Opt-in, model-backed live acceptance test for Mei bounded-overlap
concurrency (two active generations) — Kiem plan `820814e1` stage 5+.

This is a purpose-built, standard-library integration/benchmark client. It
launches ONLY the --binary and --model-dir you pass, on a dynamically chosen
free loopback port (never 8024, no fallback of any kind), with all artifacts
in an explicit directory outside the repository, and it terminates only the
exact subprocess it launched (SIGTERM, then SIGKILL; never a process group,
never a pkill). It refuses to run when a foreign `mei` process is present.

The settled acceptance gates this client checks (all required):

  G1  Isolation/correctness. Fixed greedy canary requests (text buffered,
      text streaming, native tool calls buffered+streaming, structured JSON)
      must match their single-flight reference output and finish metadata
      byte-for-byte while running two-in-flight; two logical sessions with
      distinct nonces must never see each other's content; overload must
      still refuse with 429 before any SSE event; a restart must come back
      clean. No crash, no stream_error, no cross-request contamination.
  G2  Throughput. >=30 paired, counterbalanced live trials per arm after
      warmup; mixed-load aggregate completed tokens/s (interactive
      prompt-heavy request + background decode-heavy request) must improve
      at least 20% over the equivalent serialized pair.
  G3  Interactive TTFT. For prompts <=1000 input tokens: p95 TTFT <=5 s AND
      p95 TTFT <=20% slower than the same prompt run alone.
  G4  Memory. >=4 GiB free+inactive+reclaimable at peak during the run; no
      sustained yellow/red memory pressure; no test-induced swap growth.
  G5  Cancellation. Active cancellation releases its execution slot in
      <=2 s at p95 (server-side cancel-to-release and client-observed),
      with no orphaned generation (active count returns to 0, a follow-up
      request runs, the request log records `cancelled` exactly once per
      cancelled run and never a success).
  G6  Defaults. One active generation stays the default; overlap is opt-in
      (--max-active-generations 2) and the runtime rejects anything >2.

Usage:
  scripts/test_concurrent_generations_integration.py --self-check
  scripts/test_concurrent_generations_integration.py --dry-run \
      --binary ... --model-dir ...
  scripts/test_concurrent_generations_integration.py --mode canaries|benchmark|cancellation|all \
      --binary /path/to/mei --model-dir /path/to/model-bundle \
      [--served-model-id ID] [--artifacts-dir DIR] [--trials N] [--warmup N]
      [--timeout S] [--startup-timeout S] [--extra-args STRING]
      [--memory-limit-bytes N] [--keep-temp]

Safety contract (unchanged from the 0.8.0 scheduling client):
  - --binary and --model-dir are required explicitly for every live mode.
  - The port is always allocated fresh (bind port 0 on 127.0.0.1) and can
    never be 8024.
  - Only the exact subprocess Popen returned by launch_server is ever
    terminated.
  - The client never contacts any server other than 127.0.0.1 on the port it
    allocated itself.
  - Before any expensive model run: foreign-mei check, memory preflight, and
    a refusal to start below the reclaimable-memory floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOPBACK = "127.0.0.1"
# The usual local Mei port. This test never uses it: it refuses any port
# equal to it and warns when something is listening there.
PREEXISTING_PORT = 8024

# Gate thresholds (the settled acceptance gates; do not weaken).
THROUGHPUT_RATIO_GATE = 1.20          # G2: >= +20% aggregate tokens/s
TTFT_P95_GATE_SECONDS = 5.0           # G3: p95 TTFT <= 5 s
TTFT_RATIO_GATE = 1.20                # G3: <= +20% vs run alone
MEMORY_FLOOR_BYTES = 4 * 1024**3      # G4: >= 4 GiB reclaimable at peak
MEMORY_PREFLIGHT_BYTES = 8 * 1024**3  # refuse to start below this
SWAP_GROWTH_GATE_BYTES = 256 * 1024**2  # G4: test-induced swap growth bound
CANCEL_RELEASE_P95_GATE_MS = 2000.0   # G5: <= 2 s p95
TRIALS_DEFAULT = 30                   # G2: >= 30 paired trials per arm
WARMUP_DEFAULT = 2
TTFT_PROMPT_TOKEN_CAP = 1000          # G3 scope: interactive prompts <= 1000

# Server flags this client owns; --extra-args must not override them.
OWNED_FLAGS = (
    "--host",
    "--port",
    "--model-dir",
    "--served-model-id",
    "--model-profile",
    "--request-log",
    "--cache-reuse",
    "--max-queue-depth",
    "--queue-wait-ms",
    "--max-active-generations",
    "--memory-limit-bytes",
    "--version",
    "--help",
    "-h",
)

# Deterministic canary prompts (greedy, temperature 0).
PONG_PROMPT = "Reply with exactly one word: pong"
def nonce_prompt(nonce: str) -> str:
    return f"Reply with exactly this token and nothing else: {nonce}"

COUNT_PROMPT = (
    "Count upward from 1, printing one number per line. "
    "Keep counting as long as you can; do not summarize or stop early."
)
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the morning sun "
    "rises over the quiet valley and the river carries its slow water past "
    "the old mill. "
)
def prompt_heavy_prompt(repetitions: int) -> str:
    """A deterministic interactive prompt of roughly `repetitions` * 30 words.

    The exact token count is verified live from the usage chunk
    (`prompt_tokens`); gate G3 only applies to prompts <= 1000 tokens and the
    benchmark fails loudly if this prompt ever exceeds that.
    """
    return FILLER_SENTENCE * repetitions + "\n\n" + COUNT_PROMPT

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]
TOOLS_PROMPT = "Use the add_numbers tool to add 17 and 25. Do not answer in text."

STRUCTURED_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["word", "count"],
            "additionalProperties": False,
        },
    },
}
STRUCTURED_PROMPT = 'Reply as JSON with the word "pong" and the number 2.'


class TestFailure(RuntimeError):
    """A live-test assertion or precondition failed."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def check_eq(actual: Any, expected: Any, what: str) -> None:
    check(actual == expected, f"{what}: expected {expected!r}, got {actual!r}")


def safe_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", "replace")


def tail_file(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(content[-lines:])


# ---------------------------------------------------------------------------
# HTTP/1.1 client with an incremental, timestamping response reader
# ---------------------------------------------------------------------------


def parse_head(head_bytes: bytes) -> Tuple[int, str, Dict[str, str]]:
    """Parse an HTTP/1.x response head into (status, reason, headers)."""
    lines = head_bytes.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    check(len(parts) >= 2 and parts[0].startswith("HTTP/"), f"malformed status line: {lines[0]!r}")
    status = int(parts[1])
    reason = parts[2] if len(parts) > 2 else ""
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return status, reason, headers


class ChunkedDecoder:
    """Incremental `transfer-encoding: chunked` decoder (pure, self-checked)."""

    def __init__(self) -> None:
        self._state = "size"
        self._pending = b""
        self._remaining = 0
        self.error: Optional[str] = None
        self.complete = False

    def feed(self, data: bytes) -> bytes:
        out = bytearray()
        self._pending += data
        while not self.complete and self.error is None:
            if self._state == "size":
                idx = self._pending.find(b"\r\n")
                if idx < 0:
                    break
                raw_size = self._pending[:idx].split(b";")[0].strip()
                self._pending = self._pending[idx + 2:]
                try:
                    size = int(raw_size, 16)
                except ValueError:
                    self.error = f"malformed chunk size: {raw_size!r}"
                    break
                if size == 0:
                    self.complete = True
                    break
                self._remaining = size
                self._state = "data"
            elif self._state == "data":
                take = min(len(self._pending), self._remaining)
                out += self._pending[:take]
                self._pending = self._pending[take:]
                self._remaining -= take
                if self._remaining == 0:
                    self._state = "crlf"
                else:
                    break
            elif self._state == "crlf":
                if len(self._pending) < 2:
                    break
                self._pending = self._pending[2:]
                self._state = "size"
        return bytes(out)


class SSEParser:
    """Incremental parser for the server's `data: <payload>\\n\\n` SSE frames."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, data: bytes) -> List[str]:
        self._buf += data
        payloads: List[str] = []
        while b"\n\n" in self._buf:
            frame, self._buf = self._buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if line.startswith(b"data:"):
                    payloads.append(line[5:].strip().decode("utf-8", "replace"))
        return payloads


class HttpResponse:
    """Everything the reader thread observed for one HTTP response."""

    def __init__(self) -> None:
        self.status: Optional[int] = None
        self.reason = ""
        self.headers: Dict[str, str] = {}
        self.content_type = ""
        self.body = b""
        self.events: List[str] = []
        self.frames: List[Dict[str, Any]] = []
        self.saw_done = False
        self.error: Optional[str] = None
        self.eof = False
        self.aborted = False
        self.complete = False
        self.done = False
        # Timestamps (monotonic) captured by the reader thread.
        self.sent_at: Optional[float] = None
        self.head_at: Optional[float] = None
        self.first_content_at: Optional[float] = None
        self.done_at: Optional[float] = None

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def _is_content_frame(frame: Dict[str, Any]) -> bool:
    """A frame that carries the first visible output for TTFT purposes."""
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return False
    if delta.get("content"):
        return True
    if delta.get("reasoning_content"):
        return True
    if delta.get("tool_calls"):
        return True
    return False


class Session:
    """One HTTP/1.1 connection with an incremental response reader."""

    def __init__(self, port: int, timeout: float) -> None:
        self.port = port
        self.timeout = timeout
        self.resp = HttpResponse()
        self._sock = socket.create_connection((LOOPBACK, port), timeout=min(timeout, 10.0))
        self._sock.settimeout(timeout)
        self._thread: Optional[threading.Thread] = None
        self._phase = "head"
        self._buf = b""
        self._decoder: Optional[ChunkedDecoder] = None
        self._length_remaining: Optional[int] = None
        self._sse: Optional[SSEParser] = None

    # -- request side -------------------------------------------------------

    def send(self, path: str, payload: Optional[Dict[str, Any]] = None, method: str = "GET") -> None:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        if payload is not None:
            method = "POST"
        lines = [
            f"{method} {path} HTTP/1.1",
            f"Host: {LOOPBACK}:{self.port}",
            "Accept: */*",
        ]
        if payload is not None:
            lines.append("Content-Type: application/json")
        lines.append(f"Content-Length: {len(body)}")
        lines.append("Connection: close")
        self._sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body)
        self.resp.sent_at = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_loop, name="mei-cc-reader", daemon=True)
        self._thread.start()

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def abort(self) -> None:
        """Client-side disconnect: what a cancelled streaming request looks like."""
        self.resp.aborted = True
        self.close()

    # -- waiting helpers ----------------------------------------------------

    def wait_head(self, timeout: float, what: str = "the response head") -> None:
        self._wait(lambda: self.resp.status is not None, timeout, what)

    def wait_done(self, timeout: float, what: str = "the response body") -> None:
        self._wait(lambda: self.resp.done, timeout, what)

    def _wait(self, predicate: Callable[[], bool], timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return
            if self.resp.done:
                detail = f" ({self.resp.error})" if self.resp.error else ""
                raise TestFailure(f"connection closed while waiting for {what}{detail}")
            if time.monotonic() >= deadline:
                raise TestFailure(f"timed out after {timeout:g}s waiting for {what}")
            time.sleep(0.005)

    # -- reader -------------------------------------------------------------

    def _read_loop(self) -> None:
        resp = self.resp
        try:
            while True:
                try:
                    data = self._sock.recv(65536)
                except socket.timeout:
                    resp.error = f"no data for {self.timeout:g}s (read timeout)"
                    break
                except OSError as exc:
                    if not resp.aborted:
                        resp.error = f"socket error: {exc}"
                    break
                if not data:
                    resp.eof = True
                    break
                try:
                    self._feed(data)
                except Exception as exc:  # malformed bytes from the server
                    resp.error = f"parse error: {exc!r}"
                    break
                if resp.complete:
                    break
        finally:
            resp.done_at = time.monotonic()
            resp.done = True

    def _feed(self, data: bytes) -> None:
        resp = self.resp
        self._buf += data
        if self._phase == "head":
            idx = self._buf.find(b"\r\n\r\n")
            if idx < 0:
                check(len(self._buf) <= 1_000_000, "response head exceeded 1 MB")
                return
            head_bytes = self._buf[:idx]
            self._buf = self._buf[idx + 4:]
            status, reason, headers = parse_head(head_bytes)
            resp.status, resp.reason, resp.headers = status, reason, headers
            resp.head_at = time.monotonic()
            resp.content_type = headers.get("content-type", "").split(";")[0].strip().lower()
            if resp.content_type.startswith("text/event-stream"):
                self._sse = SSEParser()
            if "chunked" in headers.get("transfer-encoding", "").lower():
                self._decoder = ChunkedDecoder()
            elif "content-length" in headers:
                self._length_remaining = int(headers["content-length"])
            else:
                self._length_remaining = None  # close-delimited
            self._phase = "body"

        if self._decoder is not None:
            body_bytes = self._decoder.feed(self._buf)
            self._buf = b""
            if self._decoder.error is not None:
                raise ValueError(self._decoder.error)
            self._append_body(body_bytes)
            if self._decoder.complete:
                resp.complete = True
        elif self._length_remaining is not None:
            take = min(len(self._buf), self._length_remaining)
            self._append_body(self._buf[:take])
            self._buf = self._buf[take:]
            self._length_remaining -= take
            if self._length_remaining <= 0:
                resp.complete = True
        else:
            self._append_body(self._buf)
            self._buf = b""

    def _append_body(self, chunk: bytes) -> None:
        if not chunk:
            return
        resp = self.resp
        resp.body += chunk
        if self._sse is not None:
            for payload in self._sse.feed(chunk):
                resp.events.append(payload)
                if payload == "[DONE]":
                    resp.saw_done = True
                    continue
                if payload.startswith("{"):
                    try:
                        frame = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(frame, dict):
                        resp.frames.append(frame)
                        if resp.first_content_at is None and _is_content_frame(frame):
                            resp.first_content_at = time.monotonic()


def fetch(port: int, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> HttpResponse:
    """One complete request/response exchange (health, status, refusals)."""
    session = Session(port, timeout)
    try:
        session.send(path, payload)
        session.start()
        session.wait_done(timeout)
        resp = session.resp
        if resp.status is None:
            raise TestFailure(f"no HTTP response from {path}: {resp.error or 'connection closed'}")
        return resp
    finally:
        session.close()


def open_stream(port: int, payload: Dict[str, Any], timeout: float) -> Session:
    """A streaming request whose SSE body is read asynchronously."""
    session = Session(port, timeout)
    session.send("/v1/chat/completions", payload)
    session.start()
    return session


def fetch_scheduler(port: int, timeout: float) -> Dict[str, Any]:
    resp = fetch(port, "/v1/mei/status", timeout=timeout)
    check_eq(resp.status, 200, "/v1/mei/status HTTP status")
    payload = resp.json()
    sched = payload.get("scheduler")
    check(isinstance(sched, dict), f"/v1/mei/status has no scheduler block: {payload}")
    return sched


def wait_scheduler(
    port: int,
    predicate: Callable[[Dict[str, Any]], bool],
    timeout: float,
    what: str,
    proc: subprocess.Popen,
    log_path: Path,
    early_fail: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    poll: float = 0.1,
) -> Dict[str, Any]:
    """Poll /v1/mei/status until `predicate` holds (bounded, with diagnostics)."""
    deadline = time.monotonic() + timeout
    last: Optional[Dict[str, Any]] = None
    last_err = "no attempt yet"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise TestFailure(
                f"server exited mid-test (code {proc.returncode}); server log tail:\n{tail_file(log_path)}"
            )
        try:
            last = fetch_scheduler(port, timeout=min(timeout, 10.0))
        except (OSError, TestFailure, ValueError) as exc:
            last = None
            last_err = str(exc)
        if last is not None:
            if predicate(last):
                return last
            if early_fail is not None:
                message = early_fail(last)
                if message:
                    raise TestFailure(message)
        time.sleep(poll)
    raise TestFailure(
        f"timed out after {timeout:g}s waiting for {what}; last scheduler state: "
        f"{json.dumps(last) if last is not None else f'unavailable ({last_err})'}"
    )


# ---------------------------------------------------------------------------
# System memory / swap / pressure probes (macOS; stdlib only)
# ---------------------------------------------------------------------------


def parse_vm_stat(text: str) -> Dict[str, int]:
    """Parse `vm_stat` page counts into a dict of ints (pages)."""
    pages: Dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^(.*?):\s+(\d+)\.?$", line.strip())
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            pages[key] = int(match.group(2))
    return pages


def parse_swapusage(text: str) -> int:
    """Parse `sysctl vm.swapusage` into used bytes."""
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", text)
    if not match:
        raise TestFailure(f"cannot parse vm.swapusage: {text!r}")
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return int(value * scale)


def run_capture(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TestFailure(f"could not run {cmd[0]!r}: {exc}")


def system_memory_sample() -> Dict[str, Any]:
    """One sample of system memory: reclaimable bytes, swap used, pressure."""
    page_size = int(run_capture(["sysctl", "-n", "hw.pagesize"], timeout=10).stdout.strip() or "16384")
    vm = run_capture(["vm_stat"], timeout=10).stdout
    pages = parse_vm_stat(vm)
    free = pages.get("pages_free", 0)
    inactive = pages.get("pages_inactive", 0)
    speculative = pages.get("pages_speculative", 0)
    purgeable = pages.get("pages_purgeable", 0)
    reclaimable = (free + inactive + speculative + purgeable) * page_size
    swap = parse_swapusage(run_capture(["sysctl", "-n", "vm.swapusage"], timeout=10).stdout)
    pressure_raw = run_capture(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], timeout=10).stdout.strip()
    try:
        pressure = int(pressure_raw)
    except ValueError:
        pressure = -1
    return {
        "at": time.monotonic(),
        "page_size": page_size,
        "free_bytes": free * page_size,
        "inactive_bytes": inactive * page_size,
        "speculative_bytes": speculative * page_size,
        "purgeable_bytes": purgeable * page_size,
        "reclaimable_bytes": reclaimable,
        "swap_used_bytes": swap,
        "pressure_level": pressure,  # 1 normal, 2 warning, 4 critical, -1 unknown
    }


class MemorySampler:
    """Background sampler for system memory + the server's own memory report."""

    def __init__(self, port: int, interval: float = 0.25) -> None:
        self.port = port
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mei-cc-mem", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample: Dict[str, Any] = {}
            try:
                sample = system_memory_sample()
                sample["at"] = time.monotonic()
            except Exception as exc:
                sample = {"error": str(exc), "at": time.monotonic()}
            try:
                resp = fetch(self.port, "/v1/mei/status", timeout=5.0)
                if resp.status == 200:
                    payload = resp.json()
                    sample["mlx"] = payload.get("memory")
                    sample["scheduler"] = payload.get("scheduler")
            except Exception:
                pass
            self.samples.append(sample)
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# Gate math (pure; exercised by --self-check)
# ---------------------------------------------------------------------------


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation surprises)."""
    if not values:
        raise TestFailure("percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def aggregate_tokens_per_second(tokens: int, wall_seconds: float) -> float:
    check(wall_seconds > 0, "aggregate wall time must be positive")
    return tokens / wall_seconds


def evaluate_throughput_gate(serial_tokens: int, serial_wall: float,
                             concurrent_tokens: int, concurrent_wall: float) -> Dict[str, Any]:
    serial_rate = aggregate_tokens_per_second(serial_tokens, serial_wall)
    concurrent_rate = aggregate_tokens_per_second(concurrent_tokens, concurrent_wall)
    ratio = concurrent_rate / serial_rate
    return {
        "serial_rate": serial_rate,
        "concurrent_rate": concurrent_rate,
        "ratio": ratio,
        "passed": ratio >= THROUGHPUT_RATIO_GATE,
        "gate": THROUGHPUT_RATIO_GATE,
    }


def evaluate_ttft_gate(concurrent_ttfts: List[float], alone_ttfts: List[float]) -> Dict[str, Any]:
    p95_concurrent = percentile(concurrent_ttfts, 95)
    p95_alone = percentile(alone_ttfts, 95)
    ratio = p95_concurrent / p95_alone if p95_alone > 0 else float("inf")
    return {
        "p95_concurrent": p95_concurrent,
        "p95_alone": p95_alone,
        "ratio": ratio,
        "passed": p95_concurrent <= TTFT_P95_GATE_SECONDS and ratio <= TTFT_RATIO_GATE,
        "absolute_gate_seconds": TTFT_P95_GATE_SECONDS,
        "ratio_gate": TTFT_RATIO_GATE,
    }


def evaluate_memory_gate(samples: List[Dict[str, Any]], swap_before: int, swap_after: int) -> Dict[str, Any]:
    reclaimable = [s["reclaimable_bytes"] for s in samples if "reclaimable_bytes" in s]
    pressures = [s["pressure_level"] for s in samples if s.get("pressure_level", -1) >= 0]
    min_reclaimable = min(reclaimable) if reclaimable else 0
    # "Sustained" = more than 20% of samples at warning+ or any critical.
    warned = sum(1 for p in pressures if p >= 2)
    critical = sum(1 for p in pressures if p >= 4)
    sustained = bool(pressures) and (warned / len(pressures) > 0.20 or critical > 0)
    swap_growth = swap_after - swap_before
    return {
        "min_reclaimable_bytes": min_reclaimable,
        "min_reclaimable_gib": min_reclaimable / 1024**3,
        "pressure_samples": len(pressures),
        "pressure_warned": warned,
        "pressure_critical": critical,
        "sustained_pressure": sustained,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_growth_bytes": swap_growth,
        "passed": (
            min_reclaimable >= MEMORY_FLOOR_BYTES
            and not sustained
            and swap_growth < SWAP_GROWTH_GATE_BYTES
        ),
        "floor_bytes": MEMORY_FLOOR_BYTES,
        "swap_growth_gate_bytes": SWAP_GROWTH_GATE_BYTES,
    }


def evaluate_cancellation_gate(server_ms: List[float], client_ms: List[float]) -> Dict[str, Any]:
    server_p95 = percentile(server_ms, 95) if server_ms else float("inf")
    client_p95 = percentile(client_ms, 95) if client_ms else float("inf")
    return {
        "trials": len(server_ms),
        "server_p95_ms": server_p95,
        "client_p95_ms": client_p95,
        "server_max_ms": max(server_ms) if server_ms else None,
        "passed": server_p95 <= CANCEL_RELEASE_P95_GATE_MS and client_p95 <= CANCEL_RELEASE_P95_GATE_MS,
        "gate_ms": CANCEL_RELEASE_P95_GATE_MS,
    }


# ---------------------------------------------------------------------------
# Request payloads and response extraction
# ---------------------------------------------------------------------------


def chat_payload(
    model: str,
    content: str,
    *,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    standard_priority: Optional[int] = None,
    mei_priority: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    include_usage: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if standard_priority is not None:
        payload["priority"] = standard_priority
    if mei_priority is not None:
        payload["mei_priority"] = mei_priority
    if tools is not None:
        payload["tools"] = tools
    if response_format is not None:
        payload["response_format"] = response_format
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def extract_stream_result(session: Session) -> Dict[str, Any]:
    """Everything observable about one streaming response."""
    resp = session.resp
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    stream_error: Optional[str] = None
    for frame in resp.frames:
        if isinstance(frame.get("error"), dict):
            stream_error = json.dumps(frame["error"])
        choices = frame.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    if isinstance(delta.get("content"), str):
                        text_parts.append(delta["content"])
                    if isinstance(delta.get("reasoning_content"), str):
                        reasoning_parts.append(delta["reasoning_content"])
                    for call in delta.get("tool_calls") or []:
                        if not isinstance(call, dict):
                            continue
                        index = int(call.get("index", 0))
                        entry = tool_calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                        if call.get("id"):
                            entry["id"] = call["id"]
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            entry["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["arguments"] += fn["arguments"]
        if isinstance(frame.get("usage"), dict):
            usage = frame["usage"]
    ttft = None
    if resp.first_content_at is not None and resp.sent_at is not None:
        ttft = resp.first_content_at - resp.sent_at
    total = None
    if resp.done_at is not None and resp.sent_at is not None:
        total = resp.done_at - resp.sent_at
    return {
        "status": resp.status,
        "content_type": resp.content_type,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "finish_reason": finish_reason,
        "usage": usage,
        "stream_error": stream_error,
        "saw_done": resp.saw_done,
        "error": resp.error,
        "ttft_seconds": ttft,
        "total_seconds": total,
        "sent_at": resp.sent_at,
        "done_at": resp.done_at,
    }


def extract_buffered_result(resp: HttpResponse) -> Dict[str, Any]:
    payload = safe_json(resp.body)
    message = {}
    finish_reason = None
    usage = None
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            finish_reason = choices[0].get("finish_reason")
        usage = payload.get("usage")
    total = None
    if resp.done_at is not None and resp.sent_at is not None:
        total = resp.done_at - resp.sent_at
    return {
        "status": resp.status,
        "content_type": resp.content_type,
        "text": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": finish_reason,
        "usage": usage,
        "error": resp.error,
        "total_seconds": total,
        "sent_at": resp.sent_at,
        "done_at": resp.done_at,
        "raw": payload if not isinstance(payload, dict) or "error" in payload else None,
    }


def run_streaming_request(port: int, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    session = open_stream(port, payload, timeout)
    try:
        session.wait_done(timeout)
        return extract_stream_result(session)
    finally:
        session.close()


def run_buffered_request(port: int, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    resp = fetch(port, "/v1/chat/completions", payload, timeout=timeout)
    return extract_buffered_result(resp)


def run_request(port: int, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    if payload.get("stream"):
        return run_streaming_request(port, payload, timeout)
    return run_buffered_request(port, payload, timeout)


def start_streaming_request(port: int, payload: Dict[str, Any], timeout: float) -> Session:
    return open_stream(port, payload, timeout)


# ---------------------------------------------------------------------------
# Paired runners (serialized vs concurrent arms)
# ---------------------------------------------------------------------------


def run_serialized_pair(port: int, payload_a: Dict[str, Any], payload_b: Dict[str, Any],
                        timeout: float) -> Dict[str, Any]:
    """A then B, one at a time (the reference arm)."""
    t0 = time.monotonic()
    result_a = run_request(port, payload_a, timeout)
    result_b = run_request(port, payload_b, timeout)
    t1 = time.monotonic()
    return {
        "arm": "serialized",
        "a": result_a,
        "b": result_b,
        "wall_seconds": t1 - t0,
        "overlap_seconds": 0.0,
    }


def run_concurrent_pair(port: int, payload_a: Dict[str, Any], payload_b: Dict[str, Any],
                        timeout: float) -> Dict[str, Any]:
    """A and B in flight at the same time (two connections, one process)."""
    session_a = start_streaming_request(port, payload_a, timeout)
    session_b = start_streaming_request(port, payload_b, timeout)
    try:
        session_a.wait_done(timeout)
        session_b.wait_done(timeout)
        result_a = extract_stream_result(session_a)
        result_b = extract_stream_result(session_b)
    finally:
        session_a.close()
        session_b.close()
    start = min(result_a["sent_at"] or 0, result_b["sent_at"] or 0)
    end = max(result_a["done_at"] or 0, result_b["done_at"] or 0)
    overlap = max(0.0, min(result_a["done_at"] or 0, result_b["done_at"] or 0)
                  - max(result_a["sent_at"] or 0, result_b["sent_at"] or 0))
    return {
        "arm": "concurrent",
        "a": result_a,
        "b": result_b,
        "wall_seconds": end - start,
        "overlap_seconds": overlap,
    }


def tokens_of(result: Dict[str, Any]) -> int:
    usage = result.get("usage") or {}
    value = usage.get("completion_tokens")
    check(isinstance(value, int), f"response has no usage.completion_tokens: {usage!r}")
    return int(value)


def prompt_tokens_of(result: Dict[str, Any]) -> int:
    usage = result.get("usage") or {}
    value = usage.get("prompt_tokens")
    check(isinstance(value, int), f"response has no usage.prompt_tokens: {usage!r}")
    return int(value)


# ---------------------------------------------------------------------------
# Process / environment helpers
# ---------------------------------------------------------------------------


def pick_free_port() -> int:
    """Allocate a fresh loopback port (bind port 0), excluding the usual Mei port."""
    for _ in range(8):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((LOOPBACK, 0))
            port = sock.getsockname()[1]
        if port != PREEXISTING_PORT:
            return port
    raise TestFailure(f"could not allocate a loopback port other than {PREEXISTING_PORT}")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((LOOPBACK, port)) != 0


def guard_extra_args(extra: Optional[List[str]]) -> None:
    for token in extra or []:
        for flag in OWNED_FLAGS:
            if token == flag or token.startswith(flag + "="):
                raise TestFailure(
                    f"--extra-args token {token!r} conflicts with a flag this test owns ({flag}); "
                    "use the dedicated option instead"
                )


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    binary = Path(args.binary).expanduser().resolve()
    check(binary.is_file(), f"--binary is not a file: {binary}")
    check(os.access(binary, os.X_OK), f"--binary is not executable: {binary}")
    model_dir = Path(args.model_dir).expanduser().resolve()
    check(model_dir.is_dir(), f"--model-dir is not a directory: {model_dir}")
    check((model_dir / "config.json").is_file(), f"--model-dir has no config.json: {model_dir}")
    return binary, model_dir


def preflight_binary(binary: Path) -> None:
    """Confirm the binary runs and speaks the overlap flags (no model load)."""
    version = run_capture([str(binary), "--version"], timeout=15)
    check(
        version.returncode == 0,
        f"{binary} --version failed (exit {version.returncode}): {version.stderr.strip()}",
    )
    print(f"  binary reports: {version.stdout.strip()}", flush=True)
    help_run = run_capture([str(binary), "--help"], timeout=15)
    check(help_run.returncode == 0, f"{binary} --help failed (exit {help_run.returncode})")
    for flag in ("--max-queue-depth", "--queue-wait-ms", "--max-active-generations"):
        check(
            flag in help_run.stdout,
            f"{binary} --help does not mention {flag}; this binary predates the overlap feature",
        )
    # The cap is part of the contract: >2 must be rejected at parse time.
    over = run_capture(
        [str(binary), "--model-dir", "/nonexistent", "--served-model-id", "x",
         "--max-active-generations", "3"], timeout=15)
    check(
        over.returncode != 0 and "max-active-generations" in (over.stderr + over.stdout),
        f"{binary} did not reject --max-active-generations 3 (exit {over.returncode}); "
        "the two-slot cap is part of the contract",
    )


def check_foreign_mei(allow: bool) -> None:
    """Abort when another `mei` process exists: foreign inference ownership."""
    result = run_capture(["pgrep", "-x", "mei"], timeout=15)
    pids = [p for p in result.stdout.split() if p.strip()]
    if pids and not allow:
        raise TestFailure(
            "foreign mei process(es) detected (pids: " + ", ".join(pids) + "). "
            "This test aborts on foreign inference ownership; stop them first or pass "
            "--allow-foreign-mei if you have verified they are not using the GPU."
        )


def memory_preflight() -> Dict[str, Any]:
    sample = system_memory_sample()
    check(
        sample["reclaimable_bytes"] >= MEMORY_PREFLIGHT_BYTES,
        f"refusing to start: reclaimable memory {sample['reclaimable_bytes'] / 1024**3:.1f} GiB "
        f"is below the {MEMORY_PREFLIGHT_BYTES / 1024**3:.0f} GiB preflight floor",
    )
    check(
        sample["pressure_level"] in (-1, 1),
        f"refusing to start: memory pressure level {sample['pressure_level']} (warning/critical)",
    )
    return sample


def server_command(args: argparse.Namespace, binary: Path, port: int,
                   request_log_path: Path, cache_reuse: bool) -> List[str]:
    cmd = [
        str(binary),
        "--model-dir", str(Path(args.model_dir).expanduser().resolve()),
        "--host", LOOPBACK,
        "--port", str(port),
        "--max-active-generations", "2",
        "--max-queue-depth", str(args.max_queue_depth),
        "--queue-wait-ms", str(args.queue_wait_ms),
        "--cache-reuse", "true" if cache_reuse else "false",
        "--enable-thinking", "false",
        "--request-log", str(request_log_path),
    ]
    if args.served_model_id:
        cmd += ["--served-model-id", args.served_model_id]
    if args.model_profile:
        cmd += ["--model-profile", args.model_profile]
    if args.memory_limit_bytes > 0:
        cmd += ["--memory-limit-bytes", str(args.memory_limit_bytes)]
    cmd += list(args.extra_tokens)
    return cmd


def launch_server(
    args: argparse.Namespace, binary: Path, port: int, tmp: Path,
    request_log_path: Path, cache_reuse: bool,
) -> subprocess.Popen:
    cmd = server_command(args, binary, port, request_log_path, cache_reuse)
    print("launching: " + shlex.join(cmd), flush=True)
    env = dict(os.environ)
    if args.fused_gate_up_limit_env:
        env["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"] = "0"
    log = open(tmp / "server.log", "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(tmp),
            start_new_session=True,
            env=env,
        )
    finally:
        log.close()  # the child holds its own descriptor
    return proc


def terminate(proc: subprocess.Popen, grace: float = 20.0) -> None:
    """Terminate exactly the subprocess this test launched (never a group)."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=grace)


def wait_health(port: int, proc: subprocess.Popen, timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    last = "no attempt yet"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise TestFailure(
                f"server exited during startup (code {proc.returncode}); "
                f"server log tail:\n{tail_file(log_path)}"
            )
        try:
            resp = fetch(port, "/healthz", timeout=2.0)
            if resp.status == 200 and resp.json().get("status") == "ok":
                return
            last = f"HTTP {resp.status}"
        except (OSError, TestFailure, ValueError) as exc:
            last = str(exc)
        time.sleep(0.25)
    raise TestFailure(
        f"server not healthy within {timeout:g}s (last: {last}); server log tail:\n{tail_file(log_path)}"
    )


def resolve_model_id(port: int, args: argparse.Namespace, timeout: float) -> str:
    resp = fetch(port, "/v1/models", timeout=timeout)
    check_eq(resp.status, 200, "/v1/models HTTP status")
    payload = resp.json()
    data = payload.get("data") or []
    check(bool(data) and bool(data[0].get("id")), f"/v1/models has no model entry: {payload}")
    model_id = data[0]["id"]
    if args.served_model_id:
        check_eq(model_id, args.served_model_id, "served model id reported by /v1/models")
    return model_id


def assert_error_response(
    resp: HttpResponse,
    *,
    status: int,
    code: str,
    type_: str,
    param: Optional[str] = None,
    what: str = "error response",
) -> Dict[str, Any]:
    """Assert an OpenAI-shape error envelope delivered as a real HTTP status."""
    check(resp.error is None, f"{what}: transport error: {resp.error}")
    check_eq(resp.status, status, f"{what}: HTTP status")
    check_eq(resp.content_type, "application/json", f"{what}: content-type (must be JSON, not SSE)")
    check(not resp.events, f"{what}: must not carry SSE frames")
    payload = safe_json(resp.body)
    check(isinstance(payload, dict), f"{what}: body is not a JSON object: {payload!r}")
    error = payload.get("error")
    check(isinstance(error, dict), f"{what}: body has no error envelope: {payload!r}")
    check_eq(error.get("code"), code, f"{what}: error.code")
    check_eq(error.get("type"), type_, f"{what}: error.type")
    if param is not None:
        check_eq(error.get("param"), param, f"{what}: error.param")
    return error


class SchedulerWatcher:
    """Poll /v1/mei/status during a window and record the observed overlap."""

    def __init__(self, port: int, interval: float = 0.05) -> None:
        self.port = port
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mei-cc-sched", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(fetch_scheduler(self.port, timeout=5.0))
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def max_active(self) -> int:
        return max((int(s.get("activeGenerations", 0)) for s in self.samples), default=0)

    @property
    def max_peak(self) -> int:
        return max((int(s.get("peakActiveGenerations", 0)) for s in self.samples), default=0)

    @property
    def saw_three(self) -> bool:
        return any(int(s.get("activeGenerations", 0)) > 2 for s in self.samples)


# ---------------------------------------------------------------------------
# Mode: canaries (gate G1 + overload + restart + optional cache-reuse canary)
# ---------------------------------------------------------------------------


def looks_like_pong(text: str) -> bool:
    """Sanity check that the model answered the trivial pong prompt.

    Deliberately model-agnostic (case/punctuation): the isolation guarantee
    comes from the byte-for-byte comparison against the single-flight
    reference, not from this assertion.
    """
    return "pong" in (text or "").lower()


def canary_payloads(model_id: str) -> Dict[str, Dict[str, Any]]:
    return {
        "text_buffered": chat_payload(
            model_id, PONG_PROMPT, stream=False, max_tokens=16, temperature=0.0),
        "text_stream_nonceA": chat_payload(
            model_id, nonce_prompt("NONCE_ALPHA_7Q"), stream=True, max_tokens=16, temperature=0.0),
        "text_stream_nonceB": chat_payload(
            model_id, nonce_prompt("NONCE_BRAVO_3Z"), stream=True, max_tokens=16, temperature=0.0),
        "tools_buffered": chat_payload(
            model_id, TOOLS_PROMPT, stream=False, max_tokens=128, temperature=0.0, tools=TOOLS_SCHEMA),
        "tools_stream": chat_payload(
            model_id, TOOLS_PROMPT, stream=True, max_tokens=128, temperature=0.0, tools=TOOLS_SCHEMA),
        "structured": chat_payload(
            model_id, STRUCTURED_PROMPT, stream=False, max_tokens=64, temperature=0.0,
            response_format=STRUCTURED_FORMAT),
    }


def normalized_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a tool call from either wire shape.

    The streaming path assembles a flat `{name, arguments}` dict; the
    buffered OpenAI response nests them under `function`.
    """
    function = call.get("function")
    if isinstance(function, dict):
        return {"name": function.get("name"), "arguments": function.get("arguments")}
    return {"name": call.get("name"), "arguments": call.get("arguments")}


def comparable(result: Dict[str, Any]) -> Dict[str, Any]:
    """The deterministic, comparable part of a response."""
    return {
        "text": result.get("text"),
        "finish_reason": result.get("finish_reason"),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
        "tool_calls": [normalized_tool_call(call) for call in (result.get("tool_calls") or [])],
    }


def run_pair_threaded(port: int, payload_a: Dict[str, Any], payload_b: Dict[str, Any],
                      timeout: float) -> Dict[str, Any]:
    """Two independent requests on two connections, started together."""
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    def go(key: str, payload: Dict[str, Any]) -> None:
        try:
            out[key] = run_request(port, payload, timeout)
        except Exception as exc:  # noqa: BLE001 - reported verbatim
            errors[key] = f"{type(exc).__name__}: {exc}"

    t1 = threading.Thread(target=go, args=("a", payload_a), daemon=True)
    t2 = threading.Thread(target=go, args=("b", payload_b), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout + 30)
    t2.join(timeout + 30)
    if errors:
        raise TestFailure(f"concurrent pair raised: {errors}")
    result_a, result_b = out["a"], out["b"]
    start = min(result_a["sent_at"] or 0, result_b["sent_at"] or 0)
    end = max(result_a["done_at"] or 0, result_b["done_at"] or 0)
    overlap = max(0.0, min(result_a["done_at"] or 0, result_b["done_at"] or 0)
                  - max(result_a["sent_at"] or 0, result_b["sent_at"] or 0))
    return {"a": result_a, "b": result_b, "wall_seconds": end - start,
            "overlap_seconds": overlap}


def run_canaries(args: argparse.Namespace, binary: Path, model_dir: Path,
                 artifacts: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    evidence: Dict[str, Any] = {"mode": "canaries", "gate": "G1 (+overload, restart)"}

    def ok(name: str, detail: str = "") -> None:
        checks.append({"name": name, "detail": detail, "passed": True})
        print(f"   PASS  {name}" + (f" — {detail}" if detail else ""), flush=True)

    def step(title: str) -> None:
        print(f"\n-- {title}", flush=True)

    port = pick_free_port()
    check(port != PREEXISTING_PORT, "the allocated port must never be 8024")
    request_log = artifacts / "canaries-requests.jsonl"
    proc = launch_server(args, binary, port, artifacts, request_log, cache_reuse=False)
    try:
        wait_health(port, proc, args.startup_timeout, artifacts / "server.log")
        model_id = resolve_model_id(port, args, args.timeout)
        evidence["model_id"] = model_id

        # --- 1. initial state -------------------------------------------------
        step("1. initial scheduler state (overlap enabled, nothing submitted)")
        sched = fetch_scheduler(port, args.timeout)
        evidence["status_initial"] = sched
        check_eq(sched["maxActiveGenerations"], 2, "advertised maxActiveGenerations")
        check_eq(sched["activeGenerations"], 0, "initial activeGenerations")
        check_eq(sched["running"], False, "initial running")
        check_eq(sched["peakActiveGenerations"], 0, "initial peakActiveGenerations")
        for counter in ("admitted", "completed", "failed", "cancelledQueued",
                        "cancelledActive", "rejectedQueueFull", "rejectedTimeout", "slotReleases"):
            check_eq(sched[counter], 0, f"initial {counter}")
        ok("initial state: overlap enabled (max=2), idle, zero counters")

        # --- 2. single-flight references --------------------------------------
        step("2. single-flight references for the greedy canaries (run alone)")
        payloads = canary_payloads(model_id)
        references: Dict[str, Dict[str, Any]] = {}
        for name, payload in payloads.items():
            references[name] = run_request(port, payload, args.timeout)
        # Reference stability: a second identical run must be byte-identical.
        for name, payload in payloads.items():
            again = run_request(port, payload, args.timeout)
            check_eq(comparable(again), comparable(references[name]),
                     f"reference stability for {name} (greedy decode must be deterministic)")
        evidence["references"] = {name: comparable(r) for name, r in references.items()}
        check(looks_like_pong(references["text_buffered"]["text"]),
              f"text canary answer (single-flight reference): {references['text_buffered']['text']!r}")
        check(
            "NONCE_ALPHA_7Q" in references["text_stream_nonceA"]["text"],
            f"nonce A reference missing its nonce: {references['text_stream_nonceA']['text']!r}",
        )
        check_eq(len(references["tools_buffered"]["tool_calls"]), 1,
                 "tools canary must emit exactly one tool call (single-flight reference)")
        check_eq(len(references["tools_stream"]["tool_calls"]), 1,
                 "streaming tools canary must emit exactly one tool call (single-flight reference)")
        for tools_name in ("tools_buffered", "tools_stream"):
            normalized = normalized_tool_call(references[tools_name]["tool_calls"][0])
            check_eq(normalized["name"], "add_numbers", f"{tools_name}: tool call name")
            arguments = json.loads(normalized["arguments"] or "{}")
            check(arguments.get("a") == 17 and arguments.get("b") == 25,
                  f"{tools_name}: tool call arguments must be {{a: 17, b: 25}}, got {normalized['arguments']!r}")
        structured = safe_json(references["structured"]["text"].encode())
        check(isinstance(structured, dict) and structured.get("word") == "pong"
              and structured.get("count") == 2,
              f"structured canary reference: {references['structured']['text']!r}")
        ok("single-flight references captured and stable (text/stream/tools/structured)")

        # --- 3. concurrent pairs: each request must match its own reference ---
        step("3. two-in-flight pairs match their single-flight references exactly")
        pairs = [
            ("text_buffered", "text_stream_nonceA"),
            ("tools_buffered", "tools_stream"),
            ("structured", "text_stream_nonceB"),
        ]
        pair_evidence = []
        for left, right in pairs:
            watcher = SchedulerWatcher(port)
            watcher.start()
            pair = run_pair_threaded(port, payloads[left], payloads[right], args.timeout)
            watcher.stop()
            entry = {"pair": [left, right], "wall_seconds": pair["wall_seconds"],
                     "overlap_seconds": pair["overlap_seconds"],
                     "observed_max_active": watcher.max_active,
                     "observed_peak": watcher.max_peak,
                     "saw_three": watcher.saw_three,
                     "left": comparable(pair["a"]), "right": comparable(pair["b"])}
            pair_evidence.append(entry)
            for key, name in (("a", left), ("b", right)):
                result = pair[key]
                check(result.get("error") is None, f"{name}: transport error: {result.get('error')}")
                check(result.get("stream_error") is None, f"{name}: stream_error frame: {result.get('stream_error')}")
                check_eq(comparable(result), comparable(references[name]),
                         f"{name} output while two-in-flight vs its single-flight reference")
            check(watcher.max_active == 2,
                  f"pair {left}/{right}: two-in-flight was not observed (max active={watcher.max_active})")
            check(not watcher.saw_three, f"pair {left}/{right}: more than two active generations observed")
            check(pair["overlap_seconds"] > 0.05,
                  f"pair {left}/{right}: requests did not overlap (overlap={pair['overlap_seconds']:.3f}s)")
        evidence["pairs"] = pair_evidence
        ok("all three concurrent pairs matched their single-flight references byte-for-byte",
           f"observed max active=2 in every pair; peak never exceeded 2")

        # --- 4. nonce isolation ----------------------------------------------
        step("4. nonce isolation: neither session may see the other's content")
        watcher = SchedulerWatcher(port)
        watcher.start()
        pair = run_pair_threaded(port, payloads["text_stream_nonceA"],
                                 payloads["text_stream_nonceB"], args.timeout)
        watcher.stop()
        a_text = (pair["a"]["text"] or "") + (pair["a"].get("reasoning") or "")
        b_text = (pair["b"]["text"] or "") + (pair["b"].get("reasoning") or "")
        check("NONCE_ALPHA_7Q" in a_text and "NONCE_BRAVO_3Z" not in a_text,
              f"nonce A session leaked or missed its nonce: {a_text!r}")
        check("NONCE_BRAVO_3Z" in b_text and "NONCE_ALPHA_7Q" not in b_text,
              f"nonce B session leaked or missed its nonce: {b_text!r}")
        evidence["nonce_isolation"] = {"a": a_text, "b": b_text, "overlap": pair["overlap_seconds"]}
        ok("nonce isolation held (each session saw exactly its own nonce)")

        # --- 5. overload under overlap ----------------------------------------
        step("5. overload with both slots busy: queue bound + 429 before any SSE")
        long_a = chat_payload(model_id, COUNT_PROMPT, stream=True, max_tokens=1024,
                              temperature=0.0, mei_priority="background")
        long_b = chat_payload(model_id, COUNT_PROMPT, stream=True, max_tokens=1024,
                              temperature=0.0, mei_priority="background")
        sa = start_streaming_request(port, long_a, args.timeout)
        sb = start_streaming_request(port, long_b, args.timeout)
        wait_scheduler(port, lambda s: s["activeGenerations"] == 2, args.timeout,
                       "both long requests active", proc, artifacts / "server.log")
        queued = []
        for _ in range(args.max_queue_depth):
            queued.append(start_streaming_request(
                port, chat_payload(model_id, PONG_PROMPT, stream=True, max_tokens=8,
                                   temperature=0.0), args.timeout))
        wait_scheduler(port, lambda s: s["queueDepth"] == args.max_queue_depth, args.timeout,
                       "queue filled to its bound", proc, artifacts / "server.log")
        refusal = fetch(port, "/v1/chat/completions",
                        chat_payload(model_id, PONG_PROMPT, stream=True, max_tokens=8,
                                     temperature=0.0), timeout=args.timeout)
        error = assert_error_response(refusal, status=429, code="queue_full",
                                      type_="rate_limit_error", what="overlap queue-full refusal")
        sched = fetch_scheduler(port, args.timeout)
        check_eq(sched["rejectedQueueFull"], 1, "rejectedQueueFull after the refusal")
        check_eq(sched["activeGenerations"], 2, "active count unchanged by the refusal")
        evidence["overload"] = {"refusal": error, "scheduler": sched}
        sa.abort()
        sb.abort()
        for session in queued:
            session.abort()
        wait_scheduler(port, lambda s: s["activeGenerations"] == 0 and s["queueDepth"] == 0,
                       args.timeout, "all slots and queue to drain", proc, artifacts / "server.log")
        ok("overload refusal delivered as HTTP 429 JSON while both slots were busy",
           "the refused request was never enqueued or started")

        # --- 6. restart --------------------------------------------------------
        step("6. restart: clean state, canary passes on the new process")
        terminate(proc)
        check(proc.poll() is not None, "the server must exit on SIGTERM")
        restart_log = artifacts / "server-restart.log"
        proc = launch_server(args, binary, port, artifacts, artifacts / "canaries-restart-requests.jsonl",
                             cache_reuse=False)
        try:
            wait_health(port, proc, args.startup_timeout, artifacts / "server.log")
            sched = fetch_scheduler(port, args.timeout)
            check_eq(sched["activeGenerations"], 0, "post-restart activeGenerations")
            check_eq(sched["peakActiveGenerations"], 0, "post-restart peakActiveGenerations")
            check_eq(sched["admitted"], 0, "post-restart admitted")
            canary = run_request(port, payloads["text_buffered"], args.timeout)
            check(looks_like_pong(canary["text"]), "post-restart canary answer")
            evidence["restart"] = {"scheduler": sched, "canary": comparable(canary)}
            ok("restart came back clean (counters zeroed) and the canary passed")
        finally:
            pass  # the outer finally terminates the (possibly relaunched) process

        # --- 7. cache-reuse canary (coordinator path under overlap) -----------
        if not args.skip_cache_reuse_canary:
            step("7. cache-reuse canary: two sessions, two turns, overlap with the KV coordinator")
            terminate(proc)
            kv_dir = artifacts / "kv-cache-canary"
            kv_dir.mkdir(parents=True, exist_ok=True)
            reuse_log = artifacts / "server-cache-reuse.log"
            args.extra_tokens = list(args.extra_tokens) + ["--kv-cache-dir", str(kv_dir)]
            proc = launch_server(args, binary, port, artifacts, reuse_log, cache_reuse=True)
            args.extra_tokens = [t for t in args.extra_tokens
                                 if t not in ("--kv-cache-dir", str(kv_dir))]
            try:
                wait_health(port, proc, args.startup_timeout, artifacts / "server.log")
                reuse_model_id = resolve_model_id(port, args, args.timeout)

                def conversation(nonce_sys: str, nonce_turn1: str, nonce_turn2: str) -> List[Dict[str, Any]]:
                    system = {"role": "system", "content": f"You are a test session. Session token: {nonce_sys}."}
                    turn1 = {"role": "user", "content": nonce_prompt(nonce_turn1)}
                    turn2 = {"role": "user", "content": nonce_prompt(nonce_turn2)}
                    return [system, turn1, turn2]

                convo_a = conversation("SYS_ALPHA", "NONCE_A1", "NONCE_A2")
                convo_b = conversation("SYS_BRAVO", "NONCE_B1", "NONCE_B2")

                def two_turn_request(messages: List[Dict[str, Any]], up_to: int,
                                     model: str) -> Dict[str, Any]:
                    payload = {
                        "model": model,
                        "messages": messages[:up_to],
                        "stream": True,
                        "max_tokens": 16,
                        "temperature": 0.0,
                        "stream_options": {"include_usage": True},
                    }
                    return run_streaming_request(port, payload, args.timeout)

                # Serial reference: A turn1, B turn1, A turn2, B turn2.
                ref = {}
                ref["a1"] = two_turn_request(convo_a, 2, reuse_model_id)
                ref["b1"] = two_turn_request(convo_b, 2, reuse_model_id)
                ref["a2"] = two_turn_request(convo_a, 3, reuse_model_id)
                ref["b2"] = two_turn_request(convo_b, 3, reuse_model_id)

                # Overlapped: A1 ∥ B1, then A2 ∥ B2.
                watcher = SchedulerWatcher(port)
                watcher.start()
                ov_a1 = threading.Thread(
                    target=lambda: None, daemon=True)
                results: Dict[str, Any] = {}
                errors: Dict[str, str] = {}

                def go(key: str, messages: List[Dict[str, Any]], up_to: int) -> None:
                    try:
                        results[key] = two_turn_request(messages, up_to, reuse_model_id)
                    except Exception as exc:  # noqa: BLE001
                        errors[key] = f"{type(exc).__name__}: {exc}"

                threads = [
                    threading.Thread(target=go, args=("a1", convo_a, 2), daemon=True),
                    threading.Thread(target=go, args=("b1", convo_b, 2), daemon=True),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(args.timeout + 30)
                check(not errors, f"cache-reuse turn 1 raised: {errors}")
                for key, ref_key in (("a1", "a1"), ("b1", "b1")):
                    check(results[key].get("stream_error") is None,
                          f"cache-reuse {key} stream_error: {results[key].get('stream_error')}")
                    check_eq(comparable(results[key]), comparable(ref[ref_key]),
                             f"cache-reuse {key} (overlap) vs serial reference")
                threads = [
                    threading.Thread(target=go, args=("a2", convo_a, 3), daemon=True),
                    threading.Thread(target=go, args=("b2", convo_b, 3), daemon=True),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(args.timeout + 30)
                watcher.stop()
                check(not errors, f"cache-reuse turn 2 raised: {errors}")
                # The isolation property is: (a) each session's output must
                # equal its own serial reference byte-for-byte (asserted just
                # below), and (b) neither session may ever see the other's
                # tokens. What the model actually answers at turn 2 is
                # model-quality-dependent (the 4B echoes the earlier token
                # instead of the newest one) and is pinned by the reference
                # comparison, not by an expectation about the answer text.
                a2_text = (results["a2"]["text"] or "") + (results["a2"].get("reasoning") or "")
                b2_text = (results["b2"]["text"] or "") + (results["b2"].get("reasoning") or "")
                check(all(token not in a2_text for token in ("SYS_BRAVO", "NONCE_B1", "NONCE_B2")),
                      f"cache-reuse session A turn2 saw session B's tokens: {a2_text!r}")
                check(all(token not in b2_text for token in ("SYS_ALPHA", "NONCE_A1", "NONCE_A2")),
                      f"cache-reuse session B turn2 saw session A's tokens: {b2_text!r}")
                check_eq(comparable(results["a2"]), comparable(ref["a2"]),
                         "cache-reuse A turn2 (overlap) vs serial reference")
                check_eq(comparable(results["b2"]), comparable(ref["b2"]),
                         "cache-reuse B turn2 (overlap) vs serial reference")
                evidence["cache_reuse_canary"] = {
                    "observed_max_active": watcher.max_active,
                    "reference": {k: comparable(v) for k, v in ref.items()},
                    "overlapped": {k: comparable(v) for k, v in results.items()},
                }
                ok("cache-reuse canary: two sessions, two turns, overlap matched the serial reference",
                   f"observed max active={watcher.max_active}")
            finally:
                terminate(proc)
    except TestFailure:
        # Preserve whatever was measured before the failing assertion so a
        # failed canary run is still diagnosable from its evidence file.
        evidence["checks"] = checks
        try:
            write_evidence(artifacts, "canaries-partial", evidence)
        except OSError:
            pass
        raise
    finally:
        terminate(proc)

    evidence["checks"] = checks
    return evidence


# ---------------------------------------------------------------------------
# Mode: benchmark (gates G2 + G3 + G4)
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace, binary: Path, model_dir: Path,
                  artifacts: Path) -> Dict[str, Any]:
    print("\n-- benchmark: paired counterbalanced serialized/concurrent trials", flush=True)
    port = pick_free_port()
    request_log = artifacts / "benchmark-requests.jsonl"
    proc = launch_server(args, binary, port, artifacts, request_log, cache_reuse=False)
    evidence: Dict[str, Any] = {
        "mode": "benchmark",
        "gates": ["G2 throughput", "G3 interactive TTFT", "G4 memory"],
        "trials": args.trials,
        "warmup": args.warmup,
        "filler_repetitions": args.filler_reps,
        "p_max_tokens": args.p_maxtokens,
        "d_max_tokens": args.d_maxtokens,
    }
    try:
        wait_health(port, proc, args.startup_timeout, artifacts / "server.log")
        model_id = resolve_model_id(port, args, args.timeout)
        evidence["model_id"] = model_id

        payload_p = chat_payload(
            model_id, prompt_heavy_prompt(args.filler_reps), stream=True,
            max_tokens=args.p_maxtokens, temperature=0.0, mei_priority="interactive")
        payload_d = chat_payload(
            model_id, COUNT_PROMPT, stream=True,
            max_tokens=args.d_maxtokens, temperature=0.0, mei_priority="background")

        swap_before = system_memory_sample()["swap_used_bytes"]
        sampler = MemorySampler(port, interval=0.25)
        sampler.start()
        peak_watcher = SchedulerWatcher(port, interval=0.1)
        peak_watcher.start()

        def run_pair_arm(arm: str) -> Dict[str, Any]:
            if arm == "serialized":
                pair = run_serialized_pair(port, payload_p, payload_d, args.timeout)
            else:
                pair = run_concurrent_pair(port, payload_p, payload_d, args.timeout)
            for key in ("a", "b"):
                result = pair[key]
                check(result.get("error") is None,
                      f"{arm} arm {key}: transport error: {result.get('error')}")
                check(result.get("stream_error") is None,
                      f"{arm} arm {key}: stream_error: {result.get('stream_error')}")
                check(result.get("finish_reason") in ("stop", "length"),
                      f"{arm} arm {key}: unexpected finish_reason {result.get('finish_reason')!r}")
            pair["tokens_a"] = tokens_of(pair["a"])
            pair["tokens_b"] = tokens_of(pair["b"])
            pair["prompt_tokens_a"] = prompt_tokens_of(pair["a"])
            pair["prompt_tokens_b"] = prompt_tokens_of(pair["b"])
            return pair

        # Warmup (discarded): both arms once per warmup pair.
        for i in range(args.warmup):
            run_pair_arm("serialized")
            run_pair_arm("concurrent")
            print(f"   warmup pair {i + 1}/{args.warmup} done", flush=True)

        serialized_runs: List[Dict[str, Any]] = []
        concurrent_runs: List[Dict[str, Any]] = []
        for i in range(args.trials):
            order = ["serialized", "concurrent"] if i % 2 == 0 else ["concurrent", "serialized"]
            pair_record: Dict[str, Any] = {"index": i, "order": order}
            for arm in order:
                result = run_pair_arm(arm)
                pair_record[arm] = {
                    "wall_seconds": result["wall_seconds"],
                    "overlap_seconds": result["overlap_seconds"],
                    "tokens_a": result["tokens_a"],
                    "tokens_b": result["tokens_b"],
                    "prompt_tokens_a": result["prompt_tokens_a"],
                    "prompt_tokens_b": result["prompt_tokens_b"],
                    "ttft_a": result["a"].get("ttft_seconds"),
                    "ttft_b": result["b"].get("ttft_seconds"),
                }
                if arm == "serialized":
                    serialized_runs.append(result)
                else:
                    concurrent_runs.append(result)
            # Determinism guard: the two arms must have produced the same
            # token counts; otherwise the throughput comparison is invalid.
            check_eq(pair_record["serialized"]["tokens_a"], pair_record["concurrent"]["tokens_a"],
                     f"trial {i}: interactive completion tokens must match across arms")
            check_eq(pair_record["serialized"]["tokens_b"], pair_record["concurrent"]["tokens_b"],
                     f"trial {i}: background completion tokens must match across arms")
            if (i + 1) % 5 == 0 or i == args.trials - 1:
                print(f"   trial {i + 1}/{args.trials} done", flush=True)

        sampler.stop()
        peak_watcher.stop()
        swap_after = system_memory_sample()["swap_used_bytes"]

        # G3 scope guard: the interactive prompt must be <= 1000 tokens.
        max_prompt = max(run["prompt_tokens_a"] for run in serialized_runs)
        check(max_prompt <= TTFT_PROMPT_TOKEN_CAP,
              f"interactive prompt is {max_prompt} tokens; gate G3 only covers <= {TTFT_PROMPT_TOKEN_CAP}")

        serial_tokens = sum(run["tokens_a"] + run["tokens_b"] for run in serialized_runs)
        serial_wall = sum(run["wall_seconds"] for run in serialized_runs)
        concurrent_tokens = sum(run["tokens_a"] + run["tokens_b"] for run in concurrent_runs)
        concurrent_wall = sum(run["wall_seconds"] for run in concurrent_runs)

        gate2 = evaluate_throughput_gate(serial_tokens, serial_wall, concurrent_tokens, concurrent_wall)
        gate3 = evaluate_ttft_gate(
            [run["a"]["ttft_seconds"] for run in concurrent_runs],
            [run["a"]["ttft_seconds"] for run in serialized_runs])
        gate4 = evaluate_memory_gate(sampler.samples, swap_before, swap_after)

        # Two-in-flight sanity: the concurrent arm must actually have overlapped.
        overlaps = [run["overlap_seconds"] for run in concurrent_runs]
        check(min(overlaps) > 0.05,
              f"a concurrent trial did not overlap (min overlap {min(overlaps):.3f}s)")
        check(peak_watcher.max_active == 2,
              f"two-in-flight was not observed during the benchmark (max active={peak_watcher.max_active})")
        check(not peak_watcher.saw_three, "more than two active generations observed during the benchmark")

        # TTFT is meaningful only if the interactive request really was
        # prefill-bound; report its share for the record.
        evidence.update({
            "serialized": {
                "tokens": serial_tokens, "wall_seconds": serial_wall,
                "aggregate_tokens_per_second": gate2["serial_rate"],
                "p50_wall_seconds": statistics.median(r["wall_seconds"] for r in serialized_runs),
            },
            "concurrent": {
                "tokens": concurrent_tokens, "wall_seconds": concurrent_wall,
                "aggregate_tokens_per_second": gate2["concurrent_rate"],
                "p50_wall_seconds": statistics.median(r["wall_seconds"] for r in concurrent_runs),
            },
            "gate2": gate2,
            "gate3": gate3,
            "gate4": gate4,
            "per_trial": [
                {
                    "index": i,
                    "serialized_wall": serialized_runs[i]["wall_seconds"],
                    "concurrent_wall": concurrent_runs[i]["wall_seconds"],
                    "wall_ratio": serialized_runs[i]["wall_seconds"] / concurrent_runs[i]["wall_seconds"],
                    "serialized_ttft_a": serialized_runs[i]["a"]["ttft_seconds"],
                    "concurrent_ttft_a": concurrent_runs[i]["a"]["ttft_seconds"],
                    "tokens_a": serialized_runs[i]["tokens_a"],
                    "tokens_b": serialized_runs[i]["tokens_b"],
                    "overlap_seconds": concurrent_runs[i]["overlap_seconds"],
                }
                for i in range(args.trials)
            ],
            "max_prompt_tokens": max_prompt,
            "observed_max_active": peak_watcher.max_active,
        })

        print("\n   results:")
        print(f"   G2 aggregate: serialized {gate2['serial_rate']:.2f} tok/s, "
              f"concurrent {gate2['concurrent_rate']:.2f} tok/s, "
              f"ratio {gate2['ratio']:.3f} (gate >= {THROUGHPUT_RATIO_GATE}) "
              f"-> {'PASS' if gate2['passed'] else 'FAIL'}")
        print(f"   G3 TTFT: p95 alone {gate3['p95_alone']:.3f}s, p95 concurrent {gate3['p95_concurrent']:.3f}s, "
              f"ratio {gate3['ratio']:.3f} (gate <= {TTFT_RATIO_GATE} and <= {TTFT_P95_GATE_SECONDS}s) "
              f"-> {'PASS' if gate3['passed'] else 'FAIL'}")
        print(f"   G4 memory: min reclaimable {gate4['min_reclaimable_gib']:.2f} GiB "
              f"(gate >= {MEMORY_FLOOR_BYTES / 1024**3:.0f} GiB), warned {gate4['pressure_warned']}/"
              f"{gate4['pressure_samples']}, swap growth {gate4['swap_growth_bytes'] / 1024**2:.1f} MiB "
              f"-> {'PASS' if gate4['passed'] else 'FAIL'}")
    finally:
        terminate(proc)
    return evidence


# ---------------------------------------------------------------------------
# Mode: cancellation (gate G5)
# ---------------------------------------------------------------------------


def run_cancellation(args: argparse.Namespace, binary: Path, model_dir: Path,
                     artifacts: Path) -> Dict[str, Any]:
    print("\n-- cancellation: active cancel-to-slot-release with both slots busy", flush=True)
    port = pick_free_port()
    request_log = artifacts / "cancellation-requests.jsonl"
    proc = launch_server(args, binary, port, artifacts, request_log, cache_reuse=False)
    evidence: Dict[str, Any] = {"mode": "cancellation", "gate": "G5", "trials": args.cancel_trials}
    try:
        wait_health(port, proc, args.startup_timeout, artifacts / "server.log")
        model_id = resolve_model_id(port, args, args.timeout)
        evidence["model_id"] = model_id

        long_payload = chat_payload(model_id, COUNT_PROMPT, stream=True, max_tokens=1024,
                                    temperature=0.0, mei_priority="background")
        server_ms: List[float] = []
        client_ms: List[float] = []
        trial_records: List[Dict[str, Any]] = []

        def cancel_and_measure(session: Session, label: str) -> Dict[str, float]:
            before = fetch_scheduler(port, args.timeout)
            releases_before = int(before["slotReleases"])
            t_abort = time.monotonic()
            session.abort()
            deadline = time.monotonic() + args.timeout
            observed: Optional[Dict[str, Any]] = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise TestFailure(
                        f"server exited during cancellation; log tail:\n{tail_file(artifacts / 'server.log')}")
                try:
                    snapshot = fetch_scheduler(port, args.timeout)
                except (OSError, TestFailure, ValueError):
                    continue
                if int(snapshot["slotReleases"]) >= releases_before + 1:
                    observed = snapshot
                    break
                time.sleep(0.02)
            check(observed is not None, f"{label}: slot release was not observed within {args.timeout:g}s")
            client_latency = (time.monotonic() - t_abort) * 1000.0
            server_latency = observed.get("lastCancelToReleaseMs")
            check(server_latency is not None,
                  f"{label}: server did not record cancel-to-release timing")
            return {"client_ms": client_latency, "server_ms": float(server_latency)}

        for i in range(args.cancel_trials):
            s1 = start_streaming_request(port, long_payload, args.timeout)
            s2 = start_streaming_request(port, long_payload, args.timeout)
            wait_scheduler(port, lambda s: s["activeGenerations"] == 2, args.timeout,
                           "both long requests active", proc, artifacts / "server.log")
            # Wait until both are visibly producing tokens (first content frame).
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if s1.resp.first_content_at is not None and s2.resp.first_content_at is not None:
                    break
                if s1.resp.done or s2.resp.done:
                    break
                time.sleep(0.01)
            check(s1.resp.first_content_at is not None and s2.resp.first_content_at is not None,
                  f"trial {i}: both requests must produce tokens before cancellation")
            check(not s1.resp.saw_done and not s2.resp.saw_done,
                  f"trial {i}: a long request finished before it could be cancelled; raise max_tokens")
            r1 = cancel_and_measure(s1, f"trial {i} cancel #1")
            r2 = cancel_and_measure(s2, f"trial {i} cancel #2")
            server_ms += [r1["server_ms"], r2["server_ms"]]
            client_ms += [r1["client_ms"], r2["client_ms"]]
            trial_records.append({"trial": i, "cancel1": r1, "cancel2": r2})
            # No orphaned generation: everything must drain back to zero, and a
            # follow-up request must run on a clean scheduler.
            wait_scheduler(port, lambda s: s["activeGenerations"] == 0 and s["queueDepth"] == 0,
                           args.timeout, "scheduler to drain after cancellation", proc,
                           artifacts / "server.log")
            follow_up = run_request(
                port, chat_payload(model_id, PONG_PROMPT, stream=True, max_tokens=8,
                                   temperature=0.0), args.timeout)
            check(looks_like_pong(follow_up["text"]),
                  f"trial {i}: follow-up request after cancellation")
            print(f"   cancel trial {i + 1}/{args.cancel_trials}: "
                  f"server {r1['server_ms']:.0f}/{r2['server_ms']:.0f} ms, "
                  f"client {r1['client_ms']:.0f}/{r2['client_ms']:.0f} ms", flush=True)

        gate5 = evaluate_cancellation_gate(server_ms, client_ms)
        evidence["gate5"] = gate5
        evidence["trials_detail"] = trial_records
        print(f"\n   G5 cancellation: server p95 {gate5['server_p95_ms']:.0f} ms, "
              f"client p95 {gate5['client_p95_ms']:.0f} ms (gate <= {CANCEL_RELEASE_P95_GATE_MS:.0f} ms) "
              f"-> {'PASS' if gate5['passed'] else 'FAIL'}")
    finally:
        terminate(proc)
    return evidence


# ---------------------------------------------------------------------------
# Dry run and self-check (both launch nothing)
# ---------------------------------------------------------------------------


def dry_run(args: argparse.Namespace, binary: Path, model_dir: Path) -> None:
    port = pick_free_port()
    request_log = Path("<artifacts>/<mode>-requests.jsonl")
    print("dry run — nothing will be launched\n")
    print(f"  binary:    {binary}")
    print(f"  model dir: {model_dir}")
    print(f"  allocated port would be: {port} (never {PREEXISTING_PORT})")
    print(f"  artifacts dir: {args.artifacts_dir or '<default under ~/.local/share/mei-concurrent-generations>'}")
    print("\n  server command (canaries/benchmark/cancellation):")
    print("    " + shlex.join(server_command(args, binary, port, request_log, cache_reuse=False)))
    print("\n  gates checked by --mode all:")
    print(f"    G1 isolation canaries + overload + restart + cache-reuse canary")
    print(f"    G2 throughput: >= {args.trials} paired trials/arm, ratio >= {THROUGHPUT_RATIO_GATE}")
    print(f"    G3 TTFT: p95 <= {TTFT_P95_GATE_SECONDS}s and ratio <= {TTFT_RATIO_GATE} for prompts <= {TTFT_PROMPT_TOKEN_CAP} tokens")
    print(f"    G4 memory: >= {MEMORY_FLOOR_BYTES / 1024**3:.0f} GiB reclaimable, no sustained pressure, "
          f"swap growth < {SWAP_GROWTH_GATE_BYTES / 1024**2:.0f} MiB")
    print(f"    G5 cancellation: p95 <= {CANCEL_RELEASE_P95_GATE_MS:.0f} ms, no orphan")
    print("    G6 defaults: overlap only via --max-active-generations 2 (runtime rejects >2)")
    print("\n  memory preflight:")
    try:
        sample = memory_preflight()
        print(f"    reclaimable {sample['reclaimable_bytes'] / 1024**3:.1f} GiB, "
              f"pressure level {sample['pressure_level']}, swap {sample['swap_used_bytes'] / 1024**2:.0f} MiB")
    except TestFailure as exc:
        print(f"    WOULD REFUSE TO START: {exc}")
    print("\n  foreign-mei check:")
    try:
        check_foreign_mei(args.allow_foreign_mei)
        print("    no foreign mei process")
    except TestFailure as exc:
        print(f"    WOULD ABORT: {exc}")


def self_check() -> int:
    print("self-check: offline checks — loopback stub + parsers/gate math; no mei process, no model\n")
    results: List[Tuple[str, bool, str]] = []

    def case(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report any failure verbatim
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
        else:
            results.append((name, True, ""))

    def expect_failure(fn: Callable[[], Any]) -> None:
        try:
            fn()
        except TestFailure:
            return
        raise AssertionError("expected TestFailure, but the check passed")

    def head_case() -> None:
        status, reason, headers = parse_head(
            b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\ncontent-length: 2\r\n"
        )
        check_eq(status, 429, "status")
        check_eq(reason, "Too Many Requests", "reason")
        check_eq(headers["content-type"], "application/json", "content-type header")

    def chunked_case() -> None:
        decoder = ChunkedDecoder()
        out = decoder.feed(b"4\r\nda")
        check_eq(out, b"da", "first split feed")
        out = decoder.feed(b"ta\r\n4\r\n: x\n\r\n0\r\n\r\n")
        check_eq(out, b"ta: x\n", "second split feed")
        check(decoder.complete, "decoder must report completion at the 0-chunk")
        check(decoder.error is None, f"decoder error: {decoder.error}")

    def sse_case() -> None:
        parser = SSEParser()
        check_eq(parser.feed(b'data: {"a":1}'), [], "partial frame yields nothing")
        payloads = parser.feed(b'\n\ndata: [DONE]\n\n')
        check_eq(payloads, ['{"a":1}', "[DONE]"], "complete frames")

    def envelope_case() -> None:
        good = HttpResponse()
        good.status = 429
        good.content_type = "application/json"
        good.body = json.dumps(
            {"error": {"message": "admission queue is full (2 waiting)",
                       "type": "rate_limit_error", "code": "queue_full"}}).encode()
        error = assert_error_response(good, status=429, code="queue_full", type_="rate_limit_error")
        check_eq(error.get("code"), "queue_full", "extracted code")

    def envelope_reject_case() -> None:
        bad = HttpResponse()
        bad.status = 429
        bad.content_type = "application/json"
        bad.body = json.dumps({"error": {"message": "x", "type": "rate_limit_error", "code": "wrong"}}).encode()
        expect_failure(lambda: assert_error_response(bad, status=429, code="queue_full",
                                                     type_="rate_limit_error"))

    def extra_args_case() -> None:
        guard_extra_args(shlex.split("--load-mmap false"))
        expect_failure(lambda: guard_extra_args(shlex.split("--port 8024")))
        expect_failure(lambda: guard_extra_args(shlex.split("--max-active-generations 3")))
        expect_failure(lambda: guard_extra_args(shlex.split("--request-log=/tmp/x")))

    def payload_case() -> None:
        payload = chat_payload("m", "hi", stream=True, max_tokens=8, standard_priority=7)
        check_eq(payload["priority"], 7, "standard priority passed through as given")
        check("mei_priority" not in payload, "mei_priority only present when requested")
        payload = chat_payload("m", "hi", mei_priority="background")
        check_eq(payload["mei_priority"], "background", "mei_priority value")
        check(payload["stream"] is True, "stream flag")
        check_eq(payload["stream_options"], {"include_usage": True}, "include_usage on by default")

    def port_case() -> None:
        port = pick_free_port()
        check(port != PREEXISTING_PORT, f"port picker returned {PREEXISTING_PORT}")
        check(1024 <= port <= 65535, f"port picker returned {port}")

    def vm_stat_case() -> None:
        pages = parse_vm_stat("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                              "Pages free:                                   411050.\n"
                              "Pages inactive:                               762033.\n"
                              "Pages speculative:                               282.\n"
                              "Pages purgeable:                               18028.\n")
        check_eq(pages["pages_free"], 411050, "pages_free")
        check_eq(pages["pages_inactive"], 762033, "pages_inactive")
        check_eq(pages["pages_purgeable"], 18028, "pages_purgeable")

    def swap_case() -> None:
        check_eq(parse_swapusage("total = 2048.00M  used = 256.50M  free = 1791.50M  (encrypted)"),
                 256 * 1024**2 + 512 * 1024, "swap used")
        check_eq(parse_swapusage("total = 4.00G  used = 1.25G  free = 2.75G  (encrypted)"),
                 int(1.25 * 1024**3), "swap used (G)")

    def percentile_case() -> None:
        values = [float(i) for i in range(1, 21)]  # 1..20
        check_eq(percentile(values, 95), 19.0, "nearest-rank p95 of 1..20")
        check_eq(percentile([5.0], 95), 5.0, "single sample")
        expect_failure(lambda: percentile([], 95))

    def throughput_gate_case() -> None:
        gate = evaluate_throughput_gate(1000, 100.0, 1000, 83.33)
        check(gate["passed"], f"ratio {gate['ratio']} must pass the 1.2 gate")
        gate = evaluate_throughput_gate(1000, 100.0, 1000, 84.0)
        check(not gate["passed"], f"ratio {gate['ratio']} must fail the 1.2 gate")

    def ttft_gate_case() -> None:
        alone = [2.0] * 20
        gate = evaluate_ttft_gate([2.3] * 20, alone)
        check(gate["passed"], f"ttft {gate} must pass (2.3 <= 1.2*2.0)")
        gate = evaluate_ttft_gate([2.5] * 20, alone)
        check(not gate["passed"], f"ttft {gate} must fail the ratio")
        gate = evaluate_ttft_gate([6.0] * 20, [6.0] * 20)
        check(not gate["passed"], f"ttft {gate} must fail the absolute 5 s gate")

    def memory_gate_case() -> None:
        samples = [{"reclaimable_bytes": 5 * 1024**3, "pressure_level": 1} for _ in range(10)]
        gate = evaluate_memory_gate(samples, 0, 0)
        check(gate["passed"], f"memory {gate} must pass")
        samples[0]["reclaimable_bytes"] = 3 * 1024**3
        check(not evaluate_memory_gate(samples, 0, 0)["passed"], "floor breach must fail")
        samples = [{"reclaimable_bytes": 5 * 1024**3, "pressure_level": 2} for _ in range(10)]
        check(not evaluate_memory_gate(samples, 0, 0)["passed"], "sustained warning must fail")
        samples = [{"reclaimable_bytes": 5 * 1024**3, "pressure_level": 1} for _ in range(10)]
        check(not evaluate_memory_gate(samples, 0, 1024**3)["passed"], "swap growth must fail")

    def cancellation_gate_case() -> None:
        gate = evaluate_cancellation_gate([1500.0] * 10, [1600.0] * 10)
        check(gate["passed"], f"cancellation {gate} must pass")
        gate = evaluate_cancellation_gate([2500.0] * 10, [100.0] * 10)
        check(not gate["passed"], f"cancellation {gate} must fail the server p95")

    def content_frame_case() -> None:
        check(_is_content_frame({"choices": [{"delta": {"content": "x"}}]}), "content delta")
        check(_is_content_frame({"choices": [{"delta": {"reasoning_content": "x"}}]}), "reasoning delta")
        check(_is_content_frame({"choices": [{"delta": {"tool_calls": [{}]}}]}), "tool call delta")
        check(not _is_content_frame({"choices": [{"delta": {}}]}), "empty delta")
        check(not _is_content_frame({"choices": []}), "usage-only frame")

    def pong_case() -> None:
        check(looks_like_pong("pong"), "lowercase pong")
        check(looks_like_pong("Pong."), "capitalized pong")
        check(not looks_like_pong("ping"), "wrong word must not pass")
        check(not looks_like_pong(""), "empty must not pass")

    def stream_result_case() -> None:
        session = Session.__new__(Session)  # no socket needed
        session.resp = HttpResponse()
        session.resp.status = 200
        session.resp.sent_at = 100.0
        session.resp.first_content_at = 100.5
        session.resp.done_at = 101.0
        session.resp.saw_done = True
        session.resp.frames = [
            {"choices": [{"delta": {"content": "po"}}]},
            {"choices": [{"delta": {"content": "ng"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "add_numbers", "arguments": "{\"a\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "17}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        ]
        result = extract_stream_result(session)
        check_eq(result["text"], "pong", "assembled text")
        check_eq(result["finish_reason"], "tool_calls", "finish reason")
        check_eq(result["ttft_seconds"], 0.5, "ttft")
        check_eq(result["usage"]["completion_tokens"], 4, "usage")
        check_eq(result["tool_calls"][0]["arguments"], '{"a":17}', "tool call assembly")
        check_eq(tokens_of(result), 4, "tokens_of")
        check_eq(prompt_tokens_of(result), 4 + 5, "prompt_tokens_of")

    def comparable_case() -> None:
        left = {"text": "x", "finish_reason": "stop", "usage": {"completion_tokens": 3, "prompt_tokens": 9},
                "tool_calls": [{"name": "n", "arguments": "{}"}]}
        right = {"text": "x", "finish_reason": "stop", "usage": {"completion_tokens": 3, "prompt_tokens": 9},
                 "tool_calls": [{"name": "n", "arguments": "{}"}]}
        check_eq(comparable(left), comparable(right), "identical responses compare equal")
        right["text"] = "y"
        check(comparable(left) != comparable(right), "different text must differ")
        # Both wire shapes must normalize to the same comparable form.
        flat = {"tool_calls": [{"name": "add_numbers", "arguments": '{"a":1}'}]}
        nested = {"tool_calls": [{"id": "call_1", "type": "function",
                                  "function": {"name": "add_numbers", "arguments": '{"a":1}'}}]}
        check_eq(comparable(flat)["tool_calls"], comparable(nested)["tool_calls"],
                 "flat and nested tool-call shapes must compare equal")

    def stub_case() -> None:
        """Exercise the real client against a throwaway loopback stub.

        Two concurrent SSE connections (deferred head, split chunks, usage
        frame, [DONE]) plus a scheduler-status surface, proving the pair
        runner, extraction and overlap accounting work end to end without a
        model.
        """

        def chunk(payload: bytes) -> bytes:
            return f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n"

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((LOOPBACK, 0))
        listener.listen(8)
        listener.settimeout(1.0)
        stub_port = listener.getsockname()[1]
        sched_state = {"activeGenerations": 2, "maxActiveGenerations": 2,
                       "peakActiveGenerations": 2, "running": True, "queueDepth": 0,
                       "queuedInteractive": 0, "queuedBackground": 0, "maxQueueDepth": 2,
                       "queueWaitTimeoutMs": 120000, "admitted": 2, "completed": 0, "failed": 0,
                       "cancelledQueued": 0, "cancelledActive": 0, "rejectedQueueFull": 0,
                       "rejectedTimeout": 0, "slotReleases": 0, "lastQueueWaitMs": 0.0,
                       "maxQueueWaitMs": 0.0}

        def serve_one(conn: socket.socket) -> None:
            try:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    piece = conn.recv(4096)
                    if not piece:
                        return
                    buf += piece
                head, _, rest = buf.partition(b"\r\n\r\n")
                path = head.split(b" ")[1].decode("ascii")
                content_length = 0
                for line in head.split(b"\r\n")[1:]:
                    if line.lower().startswith(b"content-length:"):
                        content_length = int(line.split(b":", 1)[1])
                while len(rest) < content_length:
                    rest += conn.recv(4096)
                if path == "/healthz":
                    body = b'{"status":"ok"}'
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                        b"content-length: %d\r\n\r\n" % len(body) + body)
                elif path == "/v1/mei/status":
                    body = json.dumps({"scheduler": sched_state}).encode()
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                        b"content-length: %d\r\n\r\n" % len(body) + body)
                elif path == "/v1/chat/completions":
                    time.sleep(0.1)  # deferred head, like the engine's first event
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                        b"connection: keep-alive\r\n\r\n")
                    frames = [
                        b'data: {"choices":[{"delta":{"content":"po"}}]}\n\n',
                        b'data: {"choices":[{"delta":{"content":"ng"}}]}\n\n',
                        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                        b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\n',
                        b"data: [DONE]\n\n",
                    ]
                    first = chunk(frames[0])
                    conn.sendall(first[:4])  # split mid-chunk, mid-frame
                    time.sleep(0.05)
                    conn.sendall(first[4:])
                    for frame in frames[1:]:
                        conn.sendall(chunk(frame))
                    conn.sendall(b"0\r\n\r\n")
                elif path == "/refuse":
                    body = b'{"error":{"message":"full","type":"rate_limit_error","code":"queue_full"}}'
                    conn.sendall(
                        b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\n"
                        b"content-length: %d\r\n\r\n" % len(body) + body)
                else:
                    conn.sendall(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n")
            finally:
                conn.close()

        def accept_loop() -> None:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return  # listener closed: nothing left to serve
                threading.Thread(target=serve_one, args=(conn,), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        try:
            health = fetch(stub_port, "/healthz", timeout=5)
            check_eq(health.status, 200, "stub health status")
            check_eq(health.json().get("status"), "ok", "stub health body")

            sched = fetch_scheduler(stub_port, timeout=5)
            check_eq(sched["activeGenerations"], 2, "stub scheduler activeGenerations")

            stream = open_stream(stub_port, {"stub": True}, timeout=5)
            stream.wait_head(5, "stub stream head")
            check_eq(stream.resp.status, 200, "stub stream status")
            check_eq(stream.resp.content_type, "text/event-stream", "stub stream content-type")
            stream.wait_done(5, "stub stream done")
            check_eq(stream.resp.events[-1], "[DONE]", "stub stream last event")
            check(stream.resp.saw_done, "stub stream must see [DONE]")
            result = extract_stream_result(stream)
            check_eq(result["text"], "pong", "stub stream assembled text")
            check_eq(tokens_of(result), 2, "stub stream tokens")
            check(result["ttft_seconds"] is not None and result["ttft_seconds"] > 0.09,
                  f"stub stream ttft {result['ttft_seconds']}")
            stream.close()

            # Two concurrent stub streams: the pair runner must overlap them.
            pair = run_pair_threaded(stub_port, {"stream": True}, {"stream": True}, 5)
            check(pair["overlap_seconds"] > 0.05,
                  f"stub pair must overlap (got {pair['overlap_seconds']:.3f}s)")
            check_eq(pair["a"]["text"], "pong", "stub pair left text")
            check_eq(pair["b"]["text"], "pong", "stub pair right text")

            watcher = SchedulerWatcher(stub_port, interval=0.01)
            watcher.start()
            time.sleep(0.05)
            watcher.stop()
            check_eq(watcher.max_active, 2, "stub watcher max_active")
            check(not watcher.saw_three, "stub watcher saw_three")

            refusal = fetch(stub_port, "/refuse", timeout=5)
            assert_error_response(refusal, status=429, code="queue_full",
                                  type_="rate_limit_error", what="stub refusal")
        finally:
            listener.close()

    case("parse_head parses status line and headers", head_case)
    case("ChunkedDecoder decodes split chunked bodies", chunked_case)
    case("SSEParser extracts data frames across split reads", sse_case)
    case("error-envelope assertions accept the contract shape", envelope_case)
    case("error-envelope assertions reject a wrong code", envelope_reject_case)
    case("extra server args guard refuses owned flags (incl. --max-active-generations)", extra_args_case)
    case("chat payload carries priority/tools/response_format correctly", payload_case)
    case("port picker returns a non-8024 loopback port", port_case)
    case("vm_stat parsing extracts page counts", vm_stat_case)
    case("vm.swapusage parsing converts units to bytes", swap_case)
    case("nearest-rank percentile is deterministic", percentile_case)
    case("throughput gate enforces the 1.20 ratio boundary", throughput_gate_case)
    case("TTFT gate enforces absolute and ratio boundaries", ttft_gate_case)
    case("memory gate enforces floor, pressure and swap rules", memory_gate_case)
    case("cancellation gate enforces the 2 s p95 boundary", cancellation_gate_case)
    case("content-frame detection drives TTFT measurement", content_frame_case)
    case("pong sanity check is model-agnostic", pong_case)
    case("stream result extraction assembles text/tools/usage", stream_result_case)
    case("comparable() pins the deterministic response surface", comparable_case)
    case("loopback stub: two concurrent SSE streams + scheduler + refusal", stub_case)

    failed = [entry for entry in results if not entry[1]]
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    print(f"\nself-check: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Artifacts and entry point
# ---------------------------------------------------------------------------


def default_artifacts_dir() -> Path:
    return Path.home() / ".local" / "share" / "mei-concurrent-generations" / "artifacts"


def write_evidence(artifacts: Path, name: str, evidence: Dict[str, Any]) -> Path:
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"{name}.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str))
    print(f"\n  evidence written: {path}")
    return path


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_concurrent_generations_integration.py",
        description=(
            "Opt-in, model-backed live acceptance test + benchmark for Mei bounded-overlap "
            "concurrency (two active generations; Kiem plan 820814e1 stage 5+). Launches only "
            "the given binary + model on a fresh loopback port; never touches an already-running "
            "server; terminates only its own child process."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--binary", help="path to the mei executable to launch (required; no default)")
    parser.add_argument("--model-dir", help="model bundle directory to serve (required; must contain config.json)")
    parser.add_argument("--served-model-id", help="passed through to the server; default: derived from the dir name")
    parser.add_argument("--model-profile", help="optional --model-profile NAME passed through to the server")
    parser.add_argument("--mode", choices=["canaries", "benchmark", "cancellation", "all"],
                        default="all", help="which acceptance mode(s) to run (default: all)")
    parser.add_argument("--artifacts-dir", help="directory for evidence/logs (default: ~/.local/share/mei-concurrent-generations/artifacts)")
    parser.add_argument("--startup-timeout", type=float, default=600.0,
                        help="seconds to wait for model load + /healthz (default: 600)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-step wait budget in seconds (default: 300)")
    parser.add_argument("--trials", type=int, default=TRIALS_DEFAULT,
                        help=f"paired counterbalanced benchmark trials per arm (default: {TRIALS_DEFAULT})")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT,
                        help=f"discarded warmup pairs, both arms (default: {WARMUP_DEFAULT})")
    parser.add_argument("--cancel-trials", type=int, default=10,
                        help="cancellation trials (two cancels each) (default: 10)")
    parser.add_argument("--filler-reps", type=int, default=30,
                        help="repetitions of the filler sentence in the interactive prompt (default: 30)")
    parser.add_argument("--p-max-tokens", type=int, default=96, dest="p_maxtokens",
                        help="max_tokens for the interactive request (default: 96)")
    parser.add_argument("--d-max-tokens", type=int, default=256, dest="d_maxtokens",
                        help="max_tokens for the background request (default: 256)")
    parser.add_argument("--max-queue-depth", type=int, default=2,
                        help="server --max-queue-depth for the live runs (default: 2)")
    parser.add_argument("--queue-wait-ms", type=int, default=120_000,
                        help="server --queue-wait-ms for the live runs (default: 120000)")
    parser.add_argument("--memory-limit-bytes", type=int, default=27_917_287_424,
                        help="server --memory-limit-bytes for the live runs (default: 26 GiB; 0 = do not pass)")
    parser.add_argument("--skip-cache-reuse-canary", action="store_true",
                        help="skip the cache-reuse ON canary in --mode canaries")
    parser.add_argument("--allow-foreign-mei", action="store_true",
                        help="do not abort when another mei process exists (you verified it is idle)")
    parser.add_argument("--no-fused-gate-up-limit-env", action="store_false",
                        dest="fused_gate_up_limit_env",
                        help="do not set VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 for the child")
    parser.add_argument("--extra-args", metavar="STRING",
                        help="extra server arguments as one shell-quoted string; refuses flags this test owns")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; launch nothing")
    parser.add_argument("--self-check", action="store_true",
                        help="run the offline checks (loopback stub + parsers/gate math); launch nothing")
    args = parser.parse_args(argv)

    if args.self_check:
        args.extra_tokens = []
        return args

    if not args.binary:
        parser.error("--binary PATH is required (explicit; there is no default and no fallback)")
    if not args.model_dir:
        parser.error("--model-dir DIR is required (explicit; there is no default and no fallback)")
    if args.startup_timeout <= 0 or args.timeout <= 0:
        parser.error("--startup-timeout and --timeout must be > 0")
    if args.trials < 1:
        parser.error("--trials must be >= 1 (the settled gate requires >= 30)")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.max_queue_depth < 1:
        parser.error("--max-queue-depth must be >= 1 for the overload canary")
    try:
        args.extra_tokens = shlex.split(args.extra_args) if args.extra_args else []
    except ValueError as exc:
        parser.error(f"--extra-args is not a valid shell-quoted string: {exc}")
    try:
        guard_extra_args(args.extra_tokens)
    except TestFailure as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_check:
        return self_check()

    try:
        binary, model_dir = resolve_paths(args)
    except TestFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        dry_run(args, binary, model_dir)
        return 0

    print("mei bounded-overlap concurrency acceptance client (MODEL-BACKED, opt-in)")
    print(f"  binary:    {binary}")
    print(f"  model dir: {model_dir}")
    print("  launches only this binary + model on a fresh loopback port; never touches another server")

    artifacts = (Path(args.artifacts_dir).expanduser().resolve()
                 if args.artifacts_dir
                 else default_artifacts_dir() / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    artifacts.mkdir(parents=True, exist_ok=True)
    print(f"  artifacts: {artifacts}")
    run_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        preflight_binary(binary)
        foreign = memory_preflight()
        print(f"  memory preflight: reclaimable {foreign['reclaimable_bytes'] / 1024**3:.1f} GiB, "
              f"pressure level {foreign['pressure_level']}")
        check_foreign_mei(args.allow_foreign_mei)
        print("  no foreign mei process")
    except TestFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    modes = [args.mode] if args.mode != "all" else ["canaries", "benchmark", "cancellation"]
    evidence: Dict[str, Any] = {
        "run_started": run_started,
        "binary": str(binary),
        "model_dir": str(model_dir),
        "served_model_id": args.served_model_id,
        "extra_args": args.extra_tokens,
        "memory_limit_bytes": args.memory_limit_bytes,
        "modes": modes,
        "trials": args.trials,
        "warmup": args.warmup,
        "gates": {
            "G1": "isolation canaries + overload + restart + cache-reuse canary",
            "G2": f"aggregate tokens/s ratio >= {THROUGHPUT_RATIO_GATE} over >= {args.trials} paired trials/arm",
            "G3": f"p95 TTFT <= {TTFT_P95_GATE_SECONDS}s and <= {TTFT_RATIO_GATE}x alone",
            "G4": f">= {MEMORY_FLOOR_BYTES / 1024**3:.0f} GiB reclaimable, no sustained pressure, no swap growth",
            "G5": f"cancel-to-release p95 <= {CANCEL_RELEASE_P95_GATE_MS:.0f} ms",
            "G6": "default one slot; opt-in max 2; >2 rejected",
        },
        "gate_results": {},
    }
    exit_code = 0
    try:
        for mode in modes:
            try:
                if mode == "canaries":
                    evidence["canaries"] = run_canaries(args, binary, model_dir, artifacts)
                    evidence["gate_results"]["G1"] = "PASS"
                elif mode == "benchmark":
                    result = run_benchmark(args, binary, model_dir, artifacts)
                    evidence["benchmark"] = result
                    evidence["gate_results"]["G2"] = "PASS" if result["gate2"]["passed"] else "FAIL"
                    evidence["gate_results"]["G3"] = "PASS" if result["gate3"]["passed"] else "FAIL"
                    evidence["gate_results"]["G4"] = "PASS" if result["gate4"]["passed"] else "FAIL"
                    if not all(result[g]["passed"] for g in ("gate2", "gate3", "gate4")):
                        exit_code = 1
                elif mode == "cancellation":
                    result = run_cancellation(args, binary, model_dir, artifacts)
                    evidence["cancellation"] = result
                    evidence["gate_results"]["G5"] = "PASS" if result["gate5"]["passed"] else "FAIL"
                    if not result["gate5"]["passed"]:
                        exit_code = 1
            except TestFailure as exc:
                print(f"\nFAIL ({mode}): {exc}", file=sys.stderr)
                evidence.setdefault("failures", []).append({"mode": mode, "error": str(exc)})
                evidence["gate_results"][mode] = "FAIL"
                exit_code = 1
                break
            except Exception as exc:  # noqa: BLE001 - unexpected: report with traceback
                traceback.print_exc()
                evidence.setdefault("failures", []).append(
                    {"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
                evidence["gate_results"][mode] = "ERROR"
                exit_code = 1
                break
    finally:
        write_evidence(artifacts, "acceptance-summary", evidence)
        print("\n  gate summary: " + json.dumps(evidence["gate_results"]))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())



