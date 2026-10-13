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
  scripts/test_concurrent_generations_integration.py --mode canaries|benchmark|cancellation|peix-like|all \
      --binary /path/to/mei --model-dir /path/to/model-bundle \
      [--served-model-id ID] [--artifacts-dir DIR] [--trials N] [--warmup N]
      [--timeout S] [--startup-timeout S] [--extra-args STRING]
      [--memory-limit-bytes N] [--keep-temp]

Modes:
  canaries      Gate G1 isolation/lifecycle canaries + overload + restart.
  benchmark     Gates G2/G3/G4: paired counterbalanced throughput/TTFT/memory.
  cancellation  Gate G5: active cancel-to-slot-release with both slots busy.
  peix-like     Peix-style independent logical-agent workload (Kiem plan
                820814e1 stages 7-8): the same stand-in task set (a
                synchronized 2-task pair, a staged 4-task burst, and a
                cancellation scenario with a deliberately invalid request)
                is run against one explicit Mei binary/model in BOTH arms —
                opt-in overlap (--max-active-generations 2) and one-slot
                serial mode (--max-active-generations 1, the default) —
                launching one server process at a time, never two. The
                burst and cancellation scenarios are staged: their long
                first tasks must verifiably occupy every slot (checked from
                /v1/mei/status), only then are the remaining tasks
                submitted, and they must be observed queued above capacity
                before any first task releases — a race there fails the run
                loudly instead of silently degrading the queue evidence.
                Per-task TTFT, completion, queue wait (an estimate derived
                from observed slot-release transitions, plus the engine's
                exact request-log value when the line-to-task match is
                unique; the exact value is authoritative and any
                disagreement is reported), makespan, aggregate throughput,
                memory/pressure/swap, and cancellation data are recorded in
                the artifacts directory.
                Active-workload acceptance gates: every task completes and
                maps to its own task ID, observed peak active == the
                configured slots and never > 2, queueing above capacity is
                observed, cancelling one task leaves its peers healthy,
                interactive p95 TTFT <= 5 s, >= 4 GiB reclaimable at peak,
                no sustained yellow/red pressure and no test-induced swap
                growth. The legacy >= 1.20x aggregate-throughput and
                <= 1.20x-alone TTFT-ratio gates are measurements only here
                (plan stage 7), never functional pass gates. This validates
                Mei-facing request behavior for Peix-like workloads; it is
                NOT Peix end-to-end verification (the Peix harness does not
                exist yet). `--mode all` keeps its original three modes.
  all           canaries + benchmark + cancellation (unchanged; peix-like
                is opt-in and not part of `all`).

The peix-like fixtures/helpers live in `scripts/integration/peix_like.py`
(kept cohesive there; this client stays canonical for launching, live
requests, and artifacts). If that module is missing, the other modes are
unaffected and `--self-check` reports the failure.

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
# Peix-like fixtures (scripts/integration/peix_like.py)
# ---------------------------------------------------------------------------
#
# The peix-like mode's task envelopes, stand-in workloads, offline gate math
# and the self-check stub server live in a cohesive fixtures module next to
# this script. Import failure must never break the other modes: the module is
# loaded best-effort and its absence is reported by --self-check and by
# --mode peix-like only.

_SCRIPT_DIR = Path(__file__).resolve().parent
_INTEGRATION_DIR = _SCRIPT_DIR / "integration"
try:
    if str(_INTEGRATION_DIR) not in sys.path:
        sys.path.insert(0, str(_INTEGRATION_DIR))
    import peix_like  # noqa: E402  # type: ignore[import-not-found]

    PEIX_LIKE_IMPORT_ERROR: Optional[str] = None
except Exception as _peix_exc:  # noqa: BLE001 - reported verbatim
    peix_like = None  # type: ignore[assignment]
    PEIX_LIKE_IMPORT_ERROR = f"{type(_peix_exc).__name__}: {_peix_exc}"

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

# Peix-like staged-orchestration bounds (see peix_execute_burst_scenario).
# A staged scheduler state (all slots occupied, then all remaining tasks
# queued before any release) must appear promptly; if it cannot, the run
# fails loudly instead of degrading the queue-depth evidence.
STAGED_WAIT_CAP_SECONDS = 15.0
STAGED_WAIT_POLL_SECONDS = 0.01
# Between scenarios, wait for the previous scenario's slots to drain before
# submitting anything new (a slot release lags client-visible completion).
DRAIN_WAIT_CAP_SECONDS = 30.0

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
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    early_fail: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    poll: float = 0.1,
) -> Dict[str, Any]:
    """Poll /v1/mei/status until `predicate` holds (bounded, with diagnostics).

    `proc` is the launched server (when this client owns one); offline
    callers (the self-check stub) pass None.
    """
    deadline = time.monotonic() + timeout
    last: Optional[Dict[str, Any]] = None
    last_err = "no attempt yet"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise TestFailure(
                f"server exited mid-test (code {proc.returncode}); server log tail:\n"
                f"{tail_file(log_path) if log_path is not None else '<no log>'}"
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
    content: str = "",
    *,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    standard_priority: Optional[int] = None,
    mei_priority: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    include_usage: bool = True,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a chat request payload.

    `messages` (peix-like task envelopes) supplies an explicit isolated
    message history; when it is None the single `content` user turn is used,
    exactly as before.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages) if messages is not None else [{"role": "user", "content": content}],
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
                   request_log_path: Path, cache_reuse: bool,
                   max_active: int = 2,
                   max_queue_depth: Optional[int] = None) -> List[str]:
    """The exact server argv this client launches.

    `max_active` must stay 1 or 2 (the settled cap; no third slot), and the
    queue depth may only be sized up to fit a workload — never left
    unvalidated.
    """
    check(max_active in (1, 2),
          f"this client launches only 1 or 2 active generations (got {max_active}); no third slot")
    queue_depth = args.max_queue_depth if max_queue_depth is None else int(max_queue_depth)
    check(queue_depth >= 1, f"--max-queue-depth must be >= 1 (got {queue_depth})")
    cmd = [
        str(binary),
        "--model-dir", str(Path(args.model_dir).expanduser().resolve()),
        "--host", LOOPBACK,
        "--port", str(port),
        "--max-active-generations", str(max_active),
        "--max-queue-depth", str(queue_depth),
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
    max_active: int = 2,
    max_queue_depth: Optional[int] = None,
) -> subprocess.Popen:
    cmd = server_command(args, binary, port, request_log_path, cache_reuse,
                         max_active=max_active, max_queue_depth=max_queue_depth)
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
                sample = fetch_scheduler(self.port, timeout=5.0)
                # Monotonic observation time: the peix-like runner derives
                # slot-release/admission timelines from consecutive samples.
                sample["at"] = time.monotonic()
                self.samples.append(sample)
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
# Mode: peix-like (independent logical-agent tasks; Kiem 820814e1 stages 7-8)
# ---------------------------------------------------------------------------
#
# Stand-in workload for Peix-style independent logical agents: task envelopes
# (task/agent ID, isolated history, role/instructions, priority class, stream
# mode, optional tools/response format, token budget) sent to ONE explicitly
# launched Mei server/model — never a duplicate process, one arm at a time.
# The same task set runs in both arms (opt-in overlap with
# --max-active-generations 2, and one-slot serial mode with the default 1).
# The fixtures, workload, admission attribution, gate evaluation and the
# offline stub live in scripts/integration/peix_like.py.


class PeixTaskHandle:
    """One live task: its envelope, session (streaming), result and error."""

    def __init__(self, envelope: Any) -> None:
        self.envelope = envelope
        self.session: Optional[Session] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancelled = False
        self.submitted_at: Optional[float] = None
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None


def peix_task_payload(model_id: str, envelope: Any) -> Dict[str, Any]:
    """The wire payload for one task envelope (isolated history included)."""
    return chat_payload(
        model_id,
        stream=envelope.stream,
        max_tokens=envelope.max_tokens,
        temperature=envelope.temperature,
        mei_priority=envelope.priority,
        tools=envelope.tools,
        response_format=envelope.response_format,
        messages=envelope.messages(),
    )


def peix_start_task(handle: PeixTaskHandle, payload: Dict[str, Any], port: int,
                    timeout: float, barrier: Optional[threading.Barrier],
                    delay_s: float, lock: threading.Lock) -> None:
    """Start one task on its own connection/thread (barrier-synchronized)."""

    def body() -> None:
        try:
            if barrier is not None:
                barrier.wait(timeout=timeout + 30)
            if delay_s > 0:
                time.sleep(delay_s)
            handle.submitted_at = time.monotonic()
            if handle.envelope.stream:
                session = open_stream(port, payload, timeout)
                with lock:
                    handle.session = session
                session.wait_done(timeout)
                result = extract_stream_result(session)
                result["aborted"] = session.resp.aborted
            else:
                resp = fetch(port, "/v1/chat/completions", payload, timeout=timeout)
                result = extract_buffered_result(resp)
                result["aborted"] = resp.aborted
            handle.result = result
        except Exception as exc:  # noqa: BLE001 - reported per task
            handle.error = f"{type(exc).__name__}: {exc}"
        finally:
            handle.done.set()

    handle.thread = threading.Thread(
        target=body, name=f"peix-{handle.envelope.task_id}", daemon=True)
    handle.thread.start()


def peix_wait_task_done(handle: PeixTaskHandle, timeout: float, what: str) -> None:
    if not handle.done.wait(timeout):
        raise TestFailure(
            f"{what}: task {handle.envelope.task_id} did not finish within {timeout:g}s")


def peix_wait_first_content(handle: PeixTaskHandle, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if handle.error is not None:
            raise TestFailure(
                f"{what}: task {handle.envelope.task_id} failed before producing tokens: {handle.error}")
        session = handle.session
        if session is not None and session.resp.first_content_at is not None:
            return
        if handle.done.is_set():
            break
        time.sleep(0.01)
    raise TestFailure(
        f"{what}: task {handle.envelope.task_id} produced no content within {timeout:g}s")


def peix_wait_drained(port: int, timeout: float, proc: Optional[subprocess.Popen],
                      log_path: Optional[Path], what: str) -> None:
    """Wait until the scheduler shows no active or queued generations.

    Scenarios run back-to-back against one server and a slot release lags the
    client-visible completion (post-token drain + cache store), so every
    scenario starts from a clean scheduler instead of racing the previous
    scenario's tail; otherwise a new scenario's first requests would queue
    behind the tail and break the staged admission assumptions.
    """
    wait_scheduler(
        port,
        lambda s: int(s["activeGenerations"]) == 0 and int(s["queueDepth"]) == 0,
        min(timeout, DRAIN_WAIT_CAP_SECONDS),
        f"{what}: the scheduler to drain before submitting tasks",
        proc, log_path, poll=0.05)


def request_log_lines(path: Path) -> List[str]:
    """Complete JSONL lines currently in the request log (a partial tail is ignored)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    if "\n" not in text:
        return []
    return text[: text.rfind("\n")].split("\n")


def wait_request_log_records(path: Path, minimum: int, timeout: float) -> List[str]:
    deadline = time.monotonic() + timeout
    lines = request_log_lines(path)
    while len(lines) < minimum and time.monotonic() < deadline:
        time.sleep(0.1)
        lines = request_log_lines(path)
    return lines


def parse_request_log_lines(lines: List[str]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            entry = {"unparsed": line}
        if isinstance(entry, dict):
            parsed.append(entry)
    return parsed


def peix_execute_burst_scenario(scenario: Dict[str, Any], model_id: str, port: int,
                                timeout: float, watcher_interval: float,
                                proc: Optional[subprocess.Popen],
                                log_path: Optional[Path]) -> Dict[str, Any]:
    """Run one burst scenario (the pair, or the staged 4-task burst).

    A non-staged scenario (the pair) submits every task as one synchronized
    burst, exactly as before. A staged scenario (the burst) makes the
    queue-above-capacity state deterministic instead of racing the first
    tasks' completions:

      1. submit the first `peak_active` tasks (the fixture's long hold
         tasks) together and require the scheduler to show every slot
         occupied and the queue empty;
      2. check none of them has finished, then submit the remaining tasks
         while every slot is verifiably held;
      3. require the scheduler to show `active == peak_active` AND
         `queue == expected queue` in one snapshot with no slot release
         since step 1 (the queued state must exist before any first task
         releases). If a first task releases first, the run fails loudly.

    The step-3 snapshot is added to the scenario's samples, so the queue
    gate is backed by an asserted observation, not only by best-effort
    watcher sampling.
    """
    envelopes = scenario["tasks"]
    payloads = {env.task_id: peix_task_payload(model_id, env) for env in envelopes}
    handles = [PeixTaskHandle(env) for env in envelopes]
    by_id = {handle.envelope.task_id: handle for handle in handles}
    lock = threading.Lock()
    stagger = float(scenario.get("stagger_seconds", 0.0))
    expected_peak = int(scenario["expected"]["peak_active"])
    queue_target = scenario["expected"].get("queue_depth")
    expected_queue = int(queue_target) if queue_target is not None else 0
    staged = bool(scenario.get("staged"))
    first: List[Any] = []
    rest: List[Any] = []
    if staged:
        check(queue_target is not None and expected_queue > 0,
              f"staged scenario {scenario['name']} must expect queueing above capacity")
        check(expected_queue == len(envelopes) - expected_peak,
              f"staged scenario {scenario['name']}: queue target {queue_target} != "
              f"{len(envelopes)} tasks - {expected_peak} slots")
        first = list(envelopes[:expected_peak])
        rest = list(envelopes[expected_peak:])

    peix_wait_drained(port, timeout, proc, log_path, f"scenario {scenario['name']}")
    watcher = SchedulerWatcher(port, interval=watcher_interval)
    watcher.start()
    t0 = time.monotonic()
    observation_samples: List[Dict[str, Any]] = []
    staged_evidence: Optional[Dict[str, Any]] = None

    if not staged:
        barrier = threading.Barrier(len(envelopes))
        for index, handle in enumerate(handles):
            peix_start_task(handle, payloads[handle.envelope.task_id], port, timeout,
                            barrier, index * stagger, lock)
    else:
        # Stage 1: the long hold tasks; every slot must be occupied before
        # anything else is submitted.
        barrier = threading.Barrier(len(first))
        for index, env in enumerate(first):
            peix_start_task(by_id[env.task_id], payloads[env.task_id], port, timeout,
                            barrier, index * stagger, lock)
        baseline = wait_scheduler(
            port,
            lambda s: (int(s["activeGenerations"]) == expected_peak
                       and int(s["queueDepth"]) == 0),
            min(timeout, STAGED_WAIT_CAP_SECONDS),
            f"scenario {scenario['name']}: the first {expected_peak} tasks to occupy every slot",
            proc, log_path, poll=STAGED_WAIT_POLL_SECONDS)
        for env in first:
            handle = by_id[env.task_id]
            check(not handle.done.is_set(),
                  f"scenario {scenario['name']}: stage-1 task {env.task_id} finished before "
                  f"the remaining tasks could be submitted (its budget is too small to hold "
                  f"its slot across the stage-2 window on this host)")
        releases_before = int(baseline["slotReleases"])
        stage2_submitted_at = time.monotonic()

        def released_early(s: Dict[str, Any]) -> Optional[str]:
            if int(s["slotReleases"]) > releases_before:
                return (
                    f"scenario {scenario['name']}: a first task released (slotReleases "
                    f"{s['slotReleases']} > {releases_before}) before all "
                    f"{expected_queue} remaining tasks were observed queued; the "
                    f"queueing above capacity could not be verified with this timing")
            return None

        # Stage 2: submit the rest while both slots are verifiably held, then
        # require every one of them queued before any first task releases.
        barrier = threading.Barrier(len(rest))
        for index, env in enumerate(rest):
            peix_start_task(by_id[env.task_id], payloads[env.task_id], port, timeout,
                            barrier, index * stagger, lock)
        observed = wait_scheduler(
            port,
            lambda s: (int(s["activeGenerations"]) == expected_peak
                       and int(s["queueDepth"]) == expected_queue),
            min(timeout, STAGED_WAIT_CAP_SECONDS),
            f"scenario {scenario['name']}: all {expected_queue} remaining tasks queued "
            f"while every slot is still busy (before any first task releases)",
            proc, log_path, early_fail=released_early, poll=STAGED_WAIT_POLL_SECONDS)
        observed_at = time.monotonic()
        if int(observed["slotReleases"]) != releases_before:
            raise TestFailure(
                f"scenario {scenario['name']}: the queued state was observed after a slot "
                f"release (slotReleases {observed['slotReleases']} != {releases_before}); "
                f"the queue-depth evidence is not attributable")
        snapshot = dict(observed)
        snapshot["at"] = observed_at
        snapshot["staged_observation"] = True
        observation_samples.append(snapshot)
        staged_evidence = {
            "stage1_task_ids": [env.task_id for env in first],
            "stage2_task_ids": [env.task_id for env in rest],
            "expected_active": expected_peak,
            "expected_queue": expected_queue,
            "releases_before_stage2": releases_before,
            "stage2_submitted_at": stage2_submitted_at,
            "observed_at": observed_at,
            "observed": {key: observed.get(key) for key in (
                "activeGenerations", "queueDepth", "slotReleases",
                "peakActiveGenerations", "maxActiveGenerations")},
            "queue_observed_before_any_release": True,
        }

    for handle in handles:
        peix_wait_task_done(handle, timeout + 60, f"scenario {scenario['name']}")
    t1 = time.monotonic()
    watcher.stop()
    return {"handles": handles, "samples": watcher.samples, "t0": t0, "t1": t1,
            "cancel_to_release_ms": None, "staged": staged_evidence,
            "observation_samples": observation_samples}


def peix_execute_cancel_scenario(scenario: Dict[str, Any], model_id: str, port: int,
                                 timeout: float, watcher_interval: float,
                                 proc: Optional[subprocess.Popen],
                                 log_path: Optional[Path]) -> Dict[str, Any]:
    """Cancellation scenario: long victim + long peer, two queued tasks, one
    deliberately invalid request; cancel the victim and watch everyone else.

    Staged like the burst: the two long tasks must verifiably occupy every
    slot (and the victim must be visibly producing tokens) before the queue
    is filled, and the queued tasks must be observed queued above capacity
    before any slot release.
    """
    envelopes = scenario["tasks"]
    check(len(envelopes) >= 4,
          f"cancel scenario {scenario['name']} needs victim/peer/queued tasks")
    victim_id = scenario["cancel_task_id"]
    first = envelopes[:2]
    queued = [env for env in envelopes[2:] if env.expect_rejection is None]
    invalid = [env for env in envelopes if env.expect_rejection is not None]
    payloads = {env.task_id: peix_task_payload(model_id, env) for env in envelopes}
    handles = {env.task_id: PeixTaskHandle(env) for env in envelopes}
    lock = threading.Lock()
    peix_wait_drained(port, timeout, proc, log_path, f"scenario {scenario['name']}")
    watcher = SchedulerWatcher(port, interval=watcher_interval)
    watcher.start()
    t0 = time.monotonic()
    stagger = float(scenario.get("stagger_seconds", 0.0))
    expected_peak = int(scenario["expected"]["peak_active"])
    expected_queue = int(scenario["expected"]["queue_depth"] or 0)
    observation_samples: List[Dict[str, Any]] = []
    staged_evidence: Optional[Dict[str, Any]] = None

    # 1. The long tasks go first; the victim must be visibly producing tokens
    #    (and, under overlap, the peer too) before the queue is filled.
    barrier = threading.Barrier(len(first))
    for index, env in enumerate(first):
        peix_start_task(handles[env.task_id], payloads[env.task_id], port, timeout,
                        barrier, index * stagger, lock)
    wait_scheduler(
        port,
        lambda s: (s["activeGenerations"] == expected_peak
                   and s["queueDepth"] >= len(first) - expected_peak),
        timeout, "the long tasks to start", proc, log_path)
    victim = handles[victim_id]
    peix_wait_first_content(victim, timeout, f"cancel scenario {scenario['name']}")
    if scenario.get("peer_active_before_cancel"):
        peix_wait_first_content(handles[first[1].task_id], timeout,
                                f"cancel scenario {scenario['name']} (peer)")
    for env in first:
        handle = handles[env.task_id]
        check(handle.session is not None and not handle.session.resp.saw_done,
              f"cancel scenario: {env.task_id} finished before it could be cancelled "
              f"(raise its max_tokens)")

    # 2. Fill the queue above capacity while the long tasks still hold the
    #    slots; a slot release before the queue is observed means the long
    #    tasks are too short and the run fails loudly.
    queue_releases_before = int(fetch_scheduler(port, timeout)["slotReleases"])
    stage2_submitted_at = time.monotonic()

    def released_early(s: Dict[str, Any]) -> Optional[str]:
        if int(s["slotReleases"]) > queue_releases_before:
            return (
                f"cancel scenario {scenario['name']}: a long task released (slotReleases "
                f"{s['slotReleases']} > {queue_releases_before}) before all {expected_queue} "
                f"queued tasks were observed queued; the queueing above capacity could not "
                f"be verified with this timing")
        return None

    barrier = threading.Barrier(len(queued))
    for index, env in enumerate(queued):
        peix_start_task(handles[env.task_id], payloads[env.task_id], port, timeout,
                        barrier, index * stagger, lock)
    observed = wait_scheduler(port, lambda s: s["queueDepth"] == expected_queue, timeout,
                              "the queue to fill above capacity", proc, log_path,
                              early_fail=released_early, poll=STAGED_WAIT_POLL_SECONDS)
    observed_at = time.monotonic()
    if int(observed["slotReleases"]) != queue_releases_before:
        raise TestFailure(
            f"cancel scenario {scenario['name']}: the queued state was observed after a "
            f"slot release (slotReleases {observed['slotReleases']} != {queue_releases_before})")
    snapshot = dict(observed)
    snapshot["at"] = observed_at
    snapshot["staged_observation"] = True
    observation_samples.append(snapshot)
    staged_evidence = {
        "stage1_task_ids": [env.task_id for env in first],
        "stage2_task_ids": [env.task_id for env in queued],
        "expected_active": expected_peak,
        "expected_queue": expected_queue,
        "releases_before_stage2": queue_releases_before,
        "stage2_submitted_at": stage2_submitted_at,
        "observed_at": observed_at,
        "observed": {key: observed.get(key) for key in (
            "activeGenerations", "queueDepth", "slotReleases",
            "peakActiveGenerations", "maxActiveGenerations")},
        "queue_observed_before_any_release": True,
    }

    # 3. The deliberately invalid request must be refused without disturbing
    #    the active/queued state (failure isolation for one logical agent).
    for env in invalid:
        handle = handles[env.task_id]
        handle.submitted_at = time.monotonic()
        refusal = fetch(port, "/v1/chat/completions", payloads[env.task_id], timeout=timeout)
        result = extract_buffered_result(refusal)
        result["aborted"] = refusal.aborted
        handle.result = result
        handle.done.set()
    wait_scheduler(
        port,
        lambda s: s["activeGenerations"] == expected_peak and s["queueDepth"] == expected_queue,
        timeout, "the scheduler to be unchanged by the refused request", proc, log_path)

    # 4. Cancel exactly the victim; measure the client-observed release.
    check(not victim.done.is_set(),
          "cancellation scenario: the long victim finished before cancellation "
          "(raise its max_tokens)")
    victim_session = victim.session
    if victim_session is None:
        raise TestFailure("cancellation scenario: the victim has no open session")
    releases_before = int(fetch_scheduler(port, timeout)["slotReleases"])
    t_abort = time.monotonic()
    victim.cancelled = True
    victim_session.abort()
    # abort() is the client-side terminal event for this response. On some
    # hosts shutdown()/close() does not wake a reader blocked in recv() on a
    # timeout socket, so mark the response done here to let the task thread
    # finish deterministically (the reader thread's own finally is
    # idempotent and `aborted` records why the stream ended).
    if not victim_session.resp.done:
        victim_session.resp.done_at = time.monotonic()
        victim_session.resp.done = True
    cancel_to_release_ms: Optional[float] = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise TestFailure(
                "server exited during the cancellation scenario; log tail:\n"
                f"{tail_file(log_path) if log_path is not None else '<no log>'}")
        try:
            snapshot = fetch_scheduler(port, timeout)
        except (OSError, TestFailure, ValueError):
            continue
        if int(snapshot["slotReleases"]) >= releases_before + 1:
            cancel_to_release_ms = (time.monotonic() - t_abort) * 1000.0
            break
        time.sleep(0.02)
    check(cancel_to_release_ms is not None,
          "cancellation scenario: the victim's slot release was not observed")
    for env in envelopes:
        peix_wait_task_done(handles[env.task_id], timeout + 60,
                            f"cancel scenario {scenario['name']}")
    wait_scheduler(port, lambda s: s["activeGenerations"] == 0 and s["queueDepth"] == 0,
                   timeout, "the scheduler to drain after cancellation", proc, log_path)
    t1 = time.monotonic()
    watcher.stop()
    return {"handles": [handles[env.task_id] for env in envelopes],
            "samples": watcher.samples, "t0": t0, "t1": t1,
            "cancel_to_release_ms": cancel_to_release_ms,
            "staged": staged_evidence, "observation_samples": observation_samples}


def peix_assemble_scenario(scenario: Dict[str, Any], execution: Dict[str, Any],
                           log_path: Optional[Path],
                           log_lines_before: int) -> Dict[str, Any]:
    """Turn one scenario's execution into per-task + observed evidence."""
    module = peix_like
    if module is None:
        raise TestFailure(
            "scripts/integration/peix_like.py is required for --mode peix-like "
            f"(import failed: {PEIX_LIKE_IMPORT_ERROR})")
    handles = execution["handles"]
    envelopes = scenario["tasks"]
    # The staged observations (asserted scheduler states from the orchestrator)
    # join the watcher samples, so the observed peak/queue metrics rest on
    # asserted evidence as well as best-effort sampling.
    samples = list(execution["samples"]) + list(execution.get("observation_samples") or [])
    samples.sort(key=lambda s: float(s.get("at", 0.0)))
    observations = module.observations_from_samples(samples)
    submissions: List[Dict[str, Any]] = []
    for handle in handles:
        # A deliberately invalid request is refused before admission: it is
        # never queued and carries no queue wait.
        if handle.envelope.expect_rejection is not None:
            continue
        sent_at = None
        if handle.result is not None:
            sent_at = handle.result.get("sent_at")
        if sent_at is None:
            sent_at = handle.submitted_at if handle.submitted_at is not None else execution["t0"]
        submissions.append({"task_id": handle.envelope.task_id,
                            "priority": handle.envelope.priority,
                            "sent_at": sent_at})
    admissions = module.attribute_admissions(
        submissions, int(scenario["expected"]["peak_active"]),
        observations["slot_release_times"])

    results: Dict[str, Optional[Dict[str, Any]]] = {}
    errors: Dict[str, Optional[str]] = {}
    cancelled_ids: List[str] = []
    for handle in handles:
        task_id = handle.envelope.task_id
        results[task_id] = handle.result
        errors[task_id] = handle.error
        if handle.cancelled:
            cancelled_ids.append(task_id)
    records = module.build_task_records(envelopes, results, errors, cancelled_ids, admissions)
    summary = module.summarize_records(records)

    # The window during which every submitted task was still awaiting its
    # response (all in flight at once).
    all_in_flight = None
    sent = [r["sent_at"] for r in records if r.get("sent_at") is not None]
    done = [r["done_at"] for r in records if r.get("done_at") is not None]
    if sent and done:
        all_in_flight = max(0.0, min(done) - max(sent))

    # Server-side request-log slice for this scenario (exact queue waits when
    # the line-to-task attribution is unambiguous).
    request_log: Optional[Dict[str, Any]] = None
    if log_path is not None:
        expected_total = log_lines_before + sum(1 for e in envelopes if e.expect_rejection is None)
        lines = wait_request_log_records(log_path, expected_total, 15.0)
        slice_lines = parse_request_log_lines(lines[log_lines_before:expected_total])
        attribution = module.attribute_request_log(records, slice_lines)
        for record in records:
            entry = attribution["per_task"].get(record["task_id"])
            if entry:
                record["queue_wait_ms_server"] = entry.get("queue_wait_ms")
                record["request_log"] = entry
        request_log = {
            "path": str(log_path),
            "lines": len(slice_lines),
            "attributed": attribution["attributed"],
            "notes": attribution["notes"],
        }

    # Pick the authoritative queue wait per task (the exact server value when
    # the match is unique, else the labeled estimate) and audit the two
    # sources against each other.
    queue_wait_audit = module.resolve_queue_waits(records)

    entry: Dict[str, Any] = {
        "name": scenario["name"],
        "tasks": records,
        "expected": dict(scenario["expected"]),
        "observed": observations,
        "all_in_flight_seconds": all_in_flight,
        "makespan_seconds": summary["makespan_seconds"],
        "aggregate_completion_tokens": summary["aggregate_completion_tokens"],
        "aggregate_tokens_per_second": summary["aggregate_tokens_per_second"],
        "tasks_completed": summary["tasks_completed"],
        "tasks_cancelled": summary["tasks_cancelled"],
        "tasks_rejected": summary["tasks_rejected"],
        "request_log": request_log,
        "queue_wait_audit": queue_wait_audit,
        "staged": execution.get("staged"),
        "t0": execution["t0"],
        "t1": execution["t1"],
        "memory": None,
    }
    if scenario.get("cancel_task_id"):
        entry["cancel"] = {
            "victim": scenario["cancel_task_id"],
            "cancel_to_release_ms": execution.get("cancel_to_release_ms"),
        }
    return entry


def peix_run_arm(arm_name: str, max_active: int, scenarios: List[Dict[str, Any]],
                 model_id: str, port: int, timeout: float, *,
                 watcher_interval: float = 0.05,
                 proc: Optional[subprocess.Popen] = None,
                 log_path: Optional[Path] = None,
                 sample_memory: bool = True) -> Dict[str, Any]:
    """Run one arm (overlap or serial) of the peix-like workload."""
    arm: Dict[str, Any] = {"arm": arm_name, "max_active": max_active, "scenarios": []}
    sampler: Optional[MemorySampler] = None
    arm_swap_before: Optional[int] = None
    if sample_memory:
        arm_swap_before = system_memory_sample()["swap_used_bytes"]
        sampler = MemorySampler(port, interval=0.25)
        sampler.start()
    log_lines_before = len(request_log_lines(log_path)) if log_path is not None else 0
    try:
        for scenario in scenarios:
            swap_before = system_memory_sample()["swap_used_bytes"] if sample_memory else None
            print(f"   [{arm_name}] scenario {scenario['name']}: "
                  f"{len(scenario['tasks'])} tasks, peak target "
                  f"{scenario['expected']['peak_active']}, queue target "
                  f"{scenario['expected']['queue_depth']}", flush=True)
            if scenario.get("cancel_task_id"):
                execution = peix_execute_cancel_scenario(
                    scenario, model_id, port, timeout, watcher_interval, proc, log_path)
            else:
                execution = peix_execute_burst_scenario(
                    scenario, model_id, port, timeout, watcher_interval, proc, log_path)
            swap_after = system_memory_sample()["swap_used_bytes"] if sample_memory else None
            entry = peix_assemble_scenario(scenario, execution, log_path, log_lines_before)
            log_lines_before += sum(
                1 for e in scenario["tasks"] if e.expect_rejection is None)
            if sample_memory and sampler is not None:
                window = [s for s in sampler.samples
                          if execution["t0"] <= float(s.get("at", 0.0)) <= execution["t1"]]
                entry["memory"] = evaluate_memory_gate(window, int(swap_before or 0), int(swap_after or 0))
                entry["memory_window_samples"] = len(window)
            arm["scenarios"].append(entry)
            print(f"   [{arm_name}] {scenario['name']}: peak={entry['observed']['peak_active']} "
                  f"queue={entry['observed']['max_queue_depth']} "
                  f"makespan={_fmt_seconds(entry['makespan_seconds'])} "
                  f"tok/s={_fmt_rate(entry['aggregate_tokens_per_second'])} "
                  f"completed={entry['tasks_completed']}/{len(entry['tasks'])}", flush=True)
            staged = entry.get("staged") or {}
            if staged:
                print(f"   [{arm_name}] {scenario['name']}: staged — "
                      f"stage1={','.join(staged['stage1_task_ids'])} held with "
                      f"active={staged['observed'].get('activeGenerations')}, then "
                      f"stage2={','.join(staged['stage2_task_ids'])} observed queued "
                      f"(queue={staged['observed'].get('queueDepth')}) before any slot "
                      f"release (slotReleases {staged['releases_before_stage2']})", flush=True)
            audit = entry.get("queue_wait_audit") or {}
            if audit.get("compared"):
                note = (" (disagreement beyond tolerance; the exact server request-log "
                        "values are authoritative)" if audit.get("disagreement") else "")
                print(f"   [{arm_name}] {scenario['name']}: queue-wait audit — "
                      f"{audit['compared']} task(s) with both sources, max "
                      f"|derived-server| {audit['max_abs_delta_ms']:.0f} ms{note}", flush=True)
    finally:
        if sampler is not None:
            sampler.stop()
    if sample_memory and sampler is not None:
        arm["memory_gate"] = evaluate_memory_gate(
            sampler.samples, int(arm_swap_before or 0),
            system_memory_sample()["swap_used_bytes"])
        arm["memory_samples"] = len(sampler.samples)
    else:
        arm["memory_gate"] = None
    arm["interactive_ttft_gated_seconds"] = [
        float(task["ttft_seconds"])
        for scenario in arm["scenarios"]
        if scenario["expected"].get("gate_interactive_ttft")
        for task in scenario["tasks"]
        if task["priority"] == "interactive"
        and not task["expected_cancelled"]
        and not task["expected_rejected"]
        and task["finish_reason"] in ("stop", "length", "tool_calls")
        and task["ttft_seconds"] is not None
    ]
    return arm


def _fmt_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def run_peix_like(args: argparse.Namespace, binary: Path, model_dir: Path,
                  artifacts: Path) -> Dict[str, Any]:
    """--mode peix-like: the stand-in independent-agent workload in both arms."""
    print("\n-- peix-like: independent logical-agent task envelopes against one "
          "Mei server/model", flush=True)
    if peix_like is None:
        raise TestFailure(
            "scripts/integration/peix_like.py is required for --mode peix-like "
            f"(import failed: {PEIX_LIKE_IMPORT_ERROR})")
    evidence: Dict[str, Any] = {
        "mode": "peix-like",
        "note": ("stand-in workload for Peix-style independent logical agents; the real Peix "
                 "harness does not exist, so this is Mei-facing request behavior verification, "
                 "not Peix end-to-end verification"),
        "arms": {},
    }
    arms: Dict[str, Dict[str, Any]] = {}
    try:
        for arm_name, max_active in (("overlap", 2), ("serial", 1)):
            scenarios = peix_like.peix_scenarios(max_active)
            queue_depth = max(args.max_queue_depth,
                              peix_like.required_queue_depth(scenarios, max_active))
            port = pick_free_port()
            check(port != PREEXISTING_PORT, "the allocated port must never be 8024")
            arm_dir = artifacts / f"peix-like-{arm_name}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            log_path = arm_dir / "requests.jsonl"
            print(f"\n   [{arm_name}] one server, --max-active-generations {max_active}, "
                  f"--max-queue-depth {queue_depth} (queue sized to fit the burst; "
                  f"request log: {log_path.name})", flush=True)
            proc = launch_server(args, binary, port, arm_dir, log_path, cache_reuse=False,
                                 max_active=max_active, max_queue_depth=queue_depth)
            try:
                wait_health(port, proc, args.startup_timeout, arm_dir / "server.log")
                model_id = resolve_model_id(port, args, args.timeout)
                arm = peix_run_arm(arm_name, max_active, scenarios, model_id, port,
                                   args.timeout, proc=proc, log_path=log_path,
                                   sample_memory=True)
                arm["model_id"] = model_id
                arm["port"] = port
                arm["queue_depth"] = queue_depth
                arm["workload"] = peix_like.workload_summary(scenarios)
                arms[arm_name] = arm
            finally:
                terminate(proc)
        evidence["arms"] = arms
        verdict = peix_like.peix_verdict(arms)
        evidence["verdict"] = verdict

        print("\n   peix-like gate results ([gate] counts toward acceptance):")
        for arm_name, gates in verdict["gates"].items():
            for gate_name, gate in gates.items():
                marker = "gate" if gate["acceptance"] else "measurement"
                print(f"   {arm_name:>7}  {'PASS' if gate['passed'] else 'FAIL':>4}  "
                      f"{gate_name} [{marker}]  {gate['detail']}", flush=True)
        measurements = verdict["measurements"]
        burst = measurements.get("burst_throughput")
        if burst:
            print(f"\n   measurement: burst aggregate tokens/s — overlap "
                  f"{burst['overlap_tokens_per_second']:.2f}, serial "
                  f"{burst['serial_tokens_per_second']:.2f}, ratio "
                  f"{burst['ratio_overlap_over_serial']:.3f} (not a gate)")
        makespan = measurements.get("burst_makespan")
        if makespan:
            print(f"   measurement: burst makespan — overlap "
                  f"{makespan['overlap_seconds']:.2f}s, serial "
                  f"{makespan['serial_seconds']:.2f}s, serial/overlap "
                  f"{makespan['serial_over_overlap_ratio']:.3f}")
        ttft = measurements.get("interactive_ttft")
        if ttft:
            for arm_name, summary in ttft.items():
                print(f"   measurement: interactive TTFT p95 ({arm_name}) "
                      f"{summary['p95_seconds']:.3f}s over {summary['samples']} task(s)")
        print(f"\n   peix-like verdict: {'PASS' if verdict['passed'] else 'FAIL'}"
              + (f" (failed: {', '.join(verdict['failed'])})" if verdict["failed"] else ""),
              flush=True)
        write_evidence(artifacts, "peix-like", evidence)
    except TestFailure:
        evidence["arms"] = arms
        try:
            write_evidence(artifacts, "peix-like-partial", evidence)
        except OSError:
            pass
        raise
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
    print("\n  peix-like mode (--mode peix-like; one server process at a time, per arm):")
    if peix_like is None:
        print(f"    UNAVAILABLE: scripts/integration/peix_like.py did not import "
              f"({PEIX_LIKE_IMPORT_ERROR})")
    else:
        for arm_name, max_active in (("overlap", 2), ("serial", 1)):
            scenarios = peix_like.peix_scenarios(max_active)
            depth = max(args.max_queue_depth, peix_like.required_queue_depth(scenarios, max_active))
            tasks = sum(len(s["tasks"]) for s in scenarios)
            print(f"    {arm_name}: --max-active-generations {max_active}, "
                  f"--max-queue-depth {depth}, {tasks} tasks in "
                  + ", ".join(f"{s['name']}({len(s['tasks'])})" for s in scenarios))
        print("    acceptance gates: tasks complete+map, peak==slots and never >2, queueing above")
        print("      capacity, cancel peer isolation, interactive p95 TTFT <= 5s, >= 4 GiB")
        print("      reclaimable, no sustained pressure/swap growth")
        print("    measurements only (not gates): overlap/serial throughput + makespan; the legacy")
        print("      +20% throughput and 1.20x TTFT ratios are not applied in peix-like mode")
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

    # -- peix-like offline cases -------------------------------------------

    def expect_value_error(fn: Callable[[], Any]) -> None:
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError, but the call succeeded")

    def peix_fixtures_case() -> None:
        """Envelopes validate; the stand-in workloads carry the required mix."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")
        pair = module.pair_envelopes()
        burst = module.burst_envelopes()
        cancel = module.cancel_envelopes()
        for workload in (pair, burst, cancel):
            module.validate_workload(workload)
        check_eq(len(pair), 2, "pair task count")
        check_eq({env.priority for env in pair}, {"interactive", "background"}, "pair mixes classes")
        check_eq(len(burst), 4, "burst task count")
        check_eq([env.task_id for env in burst],
                 ["burst-topic-long", "burst-tool-runner",
                  "burst-general-quick", "burst-structured-report"],
                 "burst order: the long stage-1 hold tasks first, the short general + "
                 "structured tasks queued")
        check(burst[0].kind == "background-long" and burst[0].max_tokens >= 128,
              "the burst's first task is the long hold task")
        check(burst[1].expect_tool is not None,
              "the burst's second stage-1 task is the tool task (its slot must be held too)")
        check(burst[2].kind == "interactive-short" and burst[2].priority == "interactive",
              "the short General-like interactive task is in the queued stage")
        check(burst[3].expect_structured is not None,
              "the structured task is in the queued stage")
        check(sum(1 for env in burst if env.priority == "background") == 2, "burst has two background tasks")
        check(any(env.expect_tool for env in burst), "burst includes a tool task")
        check(any(env.expect_structured for env in burst), "burst includes a structured task")
        check(any(not env.stream for env in burst), "burst includes a buffered task")
        check(all(env.max_tokens >= 1 for env in pair + burst + cancel), "token budgets present")
        nonces = [env.nonce for env in pair + burst + cancel if env.nonce]
        check_eq(len(nonces), len(set(nonces)), "all nonces are distinct")
        tokens = [env.session_token for env in pair + burst + cancel]
        check_eq(len(tokens), len(set(tokens)), "all session tokens are distinct")
        check_eq([env.describe() for env in burst],
                 [env.describe() for env in module.burst_envelopes()],
                 "workload envelopes are deterministic across calls")
        for env in pair + burst + cancel:
            messages = env.messages()
            check(messages[0]["role"] == "system", f"{env.task_id}: history starts with its role")
            check(messages[-1]["role"] == "user" and env.instructions in messages[-1]["content"],
                  f"{env.task_id}: instructions are its own user turn")
        # malformed envelopes are rejected
        expect_value_error(lambda: module.validate_envelope(module.TaskEnvelope(
            task_id="bad-priority", agent_id="a", kind="k", role="r", instructions="x",
            priority="urgent", max_tokens=8)))
        expect_value_error(lambda: module.validate_envelope(module.TaskEnvelope(
            task_id="no-nonce", agent_id="a", kind="k", role="r", instructions="x",
            expect_own_nonce=True)))
        expect_value_error(lambda: module.validate_envelope(module.TaskEnvelope(
            task_id="tool-without-schema", agent_id="a", kind="k", role="r", instructions="x",
            expect_tool={"name": "add_numbers"})))
        expect_value_error(lambda: module.validate_workload(pair + [pair[0]]))
        # both arms run the same task set; only the expectations differ
        overlap_scenarios = module.peix_scenarios(2)
        serial_scenarios = module.peix_scenarios(1)
        check_eq([s["name"] for s in overlap_scenarios], ["pair", "burst", "cancel"],
                 "scenario names")
        check_eq([[t.task_id for t in s["tasks"]] for s in overlap_scenarios],
                 [[t.task_id for t in s["tasks"]] for s in serial_scenarios],
                 "the serial and overlap arms run the identical task set")
        check_eq(overlap_scenarios[1]["expected"]["queue_depth"], 2, "overlap burst queues 2")
        check_eq(serial_scenarios[1]["expected"]["queue_depth"], 3, "serial burst queues 3")
        check_eq(module.required_queue_depth(serial_scenarios, 1), 3,
                 "serial arm needs a queue depth of 3")
        check_eq(overlap_scenarios[0]["stagger_seconds"], 0.0,
                 "the pair is the synchronized test (no stagger)")
        check(overlap_scenarios[1]["stagger_seconds"] > 0 and overlap_scenarios[2]["stagger_seconds"] > 0,
              "burst and cancel submissions are staggered for a deterministic arrival order")
        check_eq(overlap_scenarios[0]["staged"], False,
                 "the pair stays a single synchronized burst (no staging)")
        check_eq(overlap_scenarios[1]["staged"], True,
                 "the burst is staged (hold the first tasks, then assert the queue)")
        check_eq(overlap_scenarios[2]["staged"], True,
                 "the cancellation scenario is staged too")
        check_eq(serial_scenarios[1]["staged"], True, "the serial burst is staged too")
        check_eq(module.workload_summary(overlap_scenarios)[1]["staged"], True,
                 "the workload summary carries the staging contract")
        expect_value_error(lambda: module.peix_scenarios(3))
        # staged submission without queueing is a misconfiguration
        with_scenario = module.peix_scenarios(2)
        check_eq(with_scenario[1]["expected"]["queue_depth"], 2,
                 "the staged burst's queue target matches tasks - slots")
        check_eq(with_scenario[1]["expected"]["queue_depth"],
                 len(with_scenario[1]["tasks"]) - with_scenario[1]["expected"]["peak_active"],
                 "stage-2 count == tasks - stage-1 count")

    def peix_attribution_case() -> None:
        """Admission/queue-wait attribution from observed slot releases."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")
        submissions = [
            {"task_id": "a", "priority": "interactive", "sent_at": 0.0},
            {"task_id": "b", "priority": "interactive", "sent_at": 0.01},
            {"task_id": "bg", "priority": "background", "sent_at": 0.02},
            {"task_id": "fg", "priority": "interactive", "sent_at": 0.03},
        ]
        admissions = module.attribute_admissions(submissions, 2, [5.0, 6.0])
        check_eq(admissions["a"]["basis"], "immediate", "first arrival admitted on its slot")
        check_eq(admissions["a"]["queue_wait_ms"], 0.0, "immediate queue wait is 0")
        check_eq(admissions["b"]["basis"], "immediate", "second arrival admitted on its slot")
        check_eq(admissions["fg"]["basis"], "slot-release", "the queued interactive task waited")
        check(abs(admissions["fg"]["queue_wait_ms"] - 4970.0) < 1.0,
              "queued interactive outranks the queued background task (first release)")
        check(abs(admissions["bg"]["queue_wait_ms"] - 5980.0) < 1.0,
              "background task admitted on the second release")
        admissions = module.attribute_admissions(submissions, 2, [5.0])
        check_eq(admissions["bg"]["basis"], "unattributed", "missing release -> unattributed")
        # observations derived from watcher samples
        samples = [
            {"at": 0.0, "activeGenerations": 1, "peakActiveGenerations": 1,
             "queueDepth": 0, "slotReleases": 0},
            {"at": 0.5, "activeGenerations": 2, "peakActiveGenerations": 2,
             "queueDepth": 1, "slotReleases": 0},
            {"at": 1.0, "activeGenerations": 2, "peakActiveGenerations": 2,
             "queueDepth": 2, "slotReleases": 1},
            {"at": 1.5, "activeGenerations": 3, "peakActiveGenerations": 3,
             "queueDepth": 0, "slotReleases": 2},
        ]
        observed = module.observations_from_samples(samples)
        check_eq(observed["peak_active"], 3, "peak active from samples")
        check(observed["saw_above_two"], "three active generations must be flagged")
        check_eq(observed["max_queue_depth"], 2, "max queue depth from samples")
        check(abs(observed["two_active_seconds"] - 1.0) < 1e-9, "two-active window from samples")
        check_eq(len(observed["slot_release_times"]), 2, "two slot releases observed")

    def peix_gate_case() -> None:
        """The peix-like gate evaluation accepts a healthy arm, catches every
        failure mode, and keeps the serial arm's slow-but-expected values as
        measurements."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")

        def task_record(task_id, priority="interactive", finish="stop", mapping=True,
                        ttft=1.0, cancelled=False, rejected=False):
            return {
                "task_id": task_id, "priority": priority,
                "finish_reason": None if (cancelled or rejected) else finish,
                "mapping_ok": mapping, "mapping_detail": "ok" if mapping else "mismatch",
                "expected_cancelled": cancelled, "expected_rejected": rejected,
                "cancelled_observed": cancelled, "error": None, "stream_error": None,
                "ttft_seconds": ttft,
            }

        def make_arm(arm_name, max_active, *, peak=None, saw_above=False, queue_depth=None,
                     mapping=True, ttft=1.0, cancel_clean=True, memory_ok=True,
                     two_active=None):
            peak = max_active if peak is None else peak
            if queue_depth is None:
                queue_depth = 4 - max_active
            if two_active is None:
                two_active = 1.0 if max_active > 1 else 0.0
            victim = task_record("c-victim", priority="background", cancelled=True)
            if not cancel_clean:
                victim["finish_reason"] = "stop"

            def scenario(name, tasks, expected_queue, require_overlap, gate_ttft, cancel=None):
                return {
                    "name": name, "tasks": tasks,
                    "expected": {"peak_active": max_active, "queue_depth": expected_queue,
                                 "require_overlap": require_overlap,
                                 "gate_interactive_ttft": gate_ttft},
                    "observed": {"peak_active": peak, "peak_counter": peak,
                                 "max_queue_depth": queue_depth if expected_queue else 0,
                                 "saw_above_two": saw_above, "two_active_seconds": two_active},
                    "all_in_flight_seconds": 1.0,
                    "cancel": cancel,
                }

            return {
                "arm": arm_name, "max_active": max_active,
                "scenarios": [
                    scenario("pair", [task_record("p1", ttft=ttft),
                                      task_record("p2", priority="background")],
                             None, max_active > 1, True),
                    scenario("burst", [task_record("b1", ttft=ttft),
                                       task_record("b2", priority="background"),
                                       task_record("b3", priority="background"),
                                       task_record("b4", priority="background")],
                             4 - max_active, max_active > 1, True),
                    scenario("cancel", [victim,
                                        task_record("c-peer", priority="background"),
                                        task_record("c-queued1", mapping=mapping),
                                        task_record("c-queued2", priority="background")],
                             4 - max_active, False, False,
                             cancel={"victim": "c-victim", "cancel_to_release_ms": 500.0}),
                ],
                "memory_gate": {"passed": memory_ok},
                "interactive_ttft_gated_seconds": [ttft, ttft],
            }

        healthy = {"overlap": make_arm("overlap", 2),
                   "serial": make_arm("serial", 1, ttft=20.0)}
        verdict = module.peix_verdict(healthy)
        check(verdict["passed"],
              f"a healthy arm pair must pass (failed: {verdict['failed']})")
        check(all(gate["passed"] for gate in verdict["gates"]["overlap"].values()),
              "every overlap-arm gate passes on the healthy arm")
        check(verdict["gates"]["serial"]["interactive_ttft_p95_within_ceiling"]["acceptance"] is False,
              "serial-arm TTFT is a measurement, not an acceptance gate")
        check(not verdict["gates"]["serial"]["interactive_ttft_p95_within_ceiling"]["passed"],
              "the serial arm's slow TTFT is still reported")

        def expect_gate_failure(arms, arm_name, gate_name):
            result = module.peix_verdict(arms)
            check(not result["passed"], f"{arm_name}/{gate_name} must fail the verdict")
            check(not result["gates"][arm_name][gate_name]["passed"],
                  f"{arm_name}/{gate_name} must be reported as failed")
            check(f"{arm_name}/{gate_name}" in result["failed"],
                  f"{arm_name}/{gate_name} must be named in the failed list")

        overlap_only = lambda **kwargs: {"overlap": make_arm("overlap", 2, **kwargs),
                                         "serial": make_arm("serial", 1, ttft=20.0)}
        expect_gate_failure(overlap_only(peak=3, saw_above=True), "overlap",
                            "peak_active_exactly_max_and_never_above")
        expect_gate_failure(overlap_only(queue_depth=1), "overlap", "queueing_above_capacity")
        expect_gate_failure(overlap_only(two_active=0.0), "overlap", "overlap_observed")
        expect_gate_failure(overlap_only(mapping=False), "overlap", "task_mapping_correct")
        expect_gate_failure(overlap_only(ttft=5.5), "overlap", "interactive_ttft_p95_within_ceiling")
        expect_gate_failure(overlap_only(cancel_clean=False), "overlap", "cancellation_peer_isolation")
        expect_gate_failure(overlap_only(memory_ok=False), "overlap", "memory_envelope")
        # a wrong serial-arm peak is a serial acceptance failure
        wrong_serial = {"overlap": make_arm("overlap", 2),
                        "serial": make_arm("serial", 1, peak=2)}
        expect_gate_failure(wrong_serial, "serial", "peak_active_exactly_max_and_never_above")
        # and an incomplete task fails the completion gate
        broken = overlap_only()
        broken["overlap"]["scenarios"][1]["tasks"][0]["finish_reason"] = "length_error"
        expect_gate_failure(broken, "overlap", "all_tasks_complete")

    def peix_request_log_case() -> None:
        """Request-log line attribution: exact matches only, cancelled runs by
        their finish, ambiguity and leftovers reported and never trusted."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")

        def record(task_id, priority, finish, tokens, cancelled=False, rejected=False):
            return {"task_id": task_id, "priority": priority, "finish_reason": None if cancelled else finish,
                    "completion_tokens": tokens, "expected_cancelled": cancelled,
                    "expected_rejected": rejected}

        records = [
            record("t1", "interactive", "stop", 2),
            record("t2", "background", "stop", 50),
            record("t3", "background", "stop", 50),   # ambiguous twin of t2
            record("t4", "background", "stop", 7),
            record("victim", "background", "stop", 3, cancelled=True),
            record("bad", "urgent", None, None, rejected=True),
        ]
        lines = [
            {"priority": "interactive", "finish": "stop", "completion_tokens": 2, "queue_wait_ms": 0.0},
            {"priority": "background", "finish": "stop", "completion_tokens": 50, "queue_wait_ms": 120.0},
            {"priority": "background", "finish": "stop", "completion_tokens": 50, "queue_wait_ms": 0.0},
            {"priority": "background", "finish": "stop", "completion_tokens": 7, "queue_wait_ms": 40.0},
            {"priority": "background", "finish": "cancelled", "completion_tokens": 3, "queue_wait_ms": 5.0},
        ]
        attribution = module.attribute_request_log(records, lines)
        check_eq(attribution["per_task"]["t1"]["queue_wait_ms"], 0.0, "t1 matched by exact tuple")
        check_eq(attribution["per_task"]["victim"]["queue_wait_ms"], 5.0,
                 "the cancelled run matched by its cancelled finish")
        check(not attribution["attributed"], "ambiguous twins must not count as attributed")
        check(any("t2" in note or "t3" in note for note in attribution["notes"]),
              f"the ambiguity must be reported: {attribution['notes']}")
        # a clean, unambiguous slice is attributed
        clean = module.attribute_request_log(records[:2] + [records[4]],
                                            [lines[0], lines[1], lines[4]])
        check(clean["attributed"], f"clean slice must be attributed: {clean['notes']}")
        check_eq(clean["unused_lines"], 0, "no leftover lines")
        # a missing line is reported, never invented
        missing = module.attribute_request_log(records[:1], [])
        check(not missing["attributed"], "missing line must not be attributed")
        check_eq(missing["per_task"], {}, "no fabricated attribution")

    def peix_queue_wait_audit_case() -> None:
        """The queue-wait audit prefers the exact server value, labels the
        derived estimate, and reports disagreement between the two sources."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")
        records = [
            {"task_id": "a", "queue_wait_ms_derived": 910.3, "queue_wait_ms_server": 177.586},
            {"task_id": "b", "queue_wait_ms_derived": 120.0, "queue_wait_ms_server": 180.0},
            {"task_id": "c", "queue_wait_ms_derived": 50.0, "queue_wait_ms_server": None},
            {"task_id": "bad", "queue_wait_ms_derived": None, "queue_wait_ms_server": None},
        ]
        audit = module.resolve_queue_waits(records)
        check_eq(records[0]["queue_wait_ms"], 177.586,
                 "the exact server request-log value is authoritative")
        check_eq(records[0]["queue_wait_ms_source"], "server-request-log", "server source labeled")
        check(records[0]["queue_wait_ms_exact"] is True, "the server value is exact")
        check_eq(records[1]["queue_wait_ms"], 180.0, "a within-tolerance server value still wins")
        check_eq(records[2]["queue_wait_ms"], 50.0, "the estimate is used when no server value exists")
        check_eq(records[2]["queue_wait_ms_source"], "slot-release-estimate",
                 "the estimate source is labeled")
        check(records[2]["queue_wait_ms_exact"] is False, "an estimate is never exact")
        check_eq(records[3]["queue_wait_ms"], None, "no source -> no value")
        check_eq(records[3]["queue_wait_ms_source"], "unattributed", "unattributed labeled")
        check_eq(audit["compared"], 2, "two tasks carry both sources")
        check(abs(audit["max_abs_delta_ms"] - 732.714) < 1e-6,
              f"max |delta| must be the 733 ms gap, got {audit['max_abs_delta_ms']}")
        check(audit["disagreement"], "a 733 ms gap must be flagged as a disagreement")
        check(any("authoritative" in note for note in audit["notes"]),
              f"the disagreement note must name the authoritative source: {audit['notes']}")
        check(any("estimate" in note for note in audit["notes"]),
              "the sources note must label the derived value as an estimate")
        check_eq(audit["tolerance_ms"], module.QUEUE_WAIT_AUDIT_TOLERANCE_MS,
                 "the tolerance is reported with the audit")
        clean = [{"task_id": "x", "queue_wait_ms_derived": 100.0, "queue_wait_ms_server": 120.0}]
        clean_audit = module.resolve_queue_waits(clean)
        check(not clean_audit["disagreement"], "a within-tolerance gap is not a disagreement")
        check_eq(clean_audit["compared"], 1, "one compared task")

    def peix_staged_strictness_case() -> None:
        """The staged burst is a real assertion: when the required scheduler
        state cannot be reached (one slot; or a queue too small to hold the
        staged tasks), the run must fail loudly, never degrade silently."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")

        def run_burst_against_stub(stub_kwargs, timeout):
            stub = module.StubMeiServer(**stub_kwargs)
            stub_port = stub.start()
            try:
                scenarios = [s for s in module.peix_scenarios(2) if s["name"] == "burst"]
                for scenario in scenarios:
                    scenario["stagger_seconds"] = 0.02  # keep the offline run quick
                return peix_run_arm(
                    "overlap", 2, scenarios, "stub-model", stub_port,
                    timeout=timeout, watcher_interval=0.01,
                    proc=None, log_path=None, sample_memory=False)
            finally:
                stub.stop()

        # One slot can never show the two active generations the staged
        # scenario requires: the stage-1 wait must time out, not pass.
        expect_failure(lambda: run_burst_against_stub(
            {"max_active": 1, "prefill_delay": 0.05, "per_token_delay": 0.01,
             "count_cap": 30, "min_stream_seconds": 0.3}, 2.0))
        # A queue that can hold only one request can never show both remaining
        # tasks queued: the stage-2 wait must fail (a release happens first).
        expect_failure(lambda: run_burst_against_stub(
            {"max_active": 2, "max_queue_depth": 1, "prefill_delay": 0.05,
             "per_token_delay": 0.01, "count_cap": 30, "min_stream_seconds": 0.3}, 2.0))

    def peix_stub_case() -> None:
        """The real peix-like scenario machinery against the offline stub:
        pair overlap, burst queueing, cancellation peer isolation, gates."""
        module = peix_like
        if module is None:
            raise AssertionError(f"scripts/integration/peix_like.py did not import: {PEIX_LIKE_IMPORT_ERROR}")

        def run_against_stub(max_active, scenario_names):
            stub = module.StubMeiServer(
                max_active=max_active, prefill_delay=0.3, per_token_delay=0.01,
                count_cap=60, min_stream_seconds=0.45)
            stub_port = stub.start()
            try:
                scenarios = [s for s in module.peix_scenarios(max_active)
                             if s["name"] in scenario_names]
                for scenario in scenarios:
                    if scenario["name"] != "pair":
                        scenario["stagger_seconds"] = 0.02  # keep the offline run quick
                return peix_run_arm(
                    "overlap" if max_active == 2 else "serial", max_active, scenarios,
                    "stub-model", stub_port, timeout=30, watcher_interval=0.01,
                    proc=None, log_path=None, sample_memory=False)
            finally:
                stub.stop()

        overlap_arm = run_against_stub(2, ("pair", "burst", "cancel"))
        serial_arm = run_against_stub(1, ("burst",))
        verdict = module.peix_verdict({"overlap": overlap_arm, "serial": serial_arm})
        check(verdict["passed"],
              "stub verdict must pass; failed: " + json.dumps(
                  {arm: {name: gate["detail"] for name, gate in gates.items() if not gate["passed"]}
                   for arm, gates in verdict["gates"].items()}))

        pair, burst, cancel = overlap_arm["scenarios"]
        check_eq(pair["observed"]["peak_active"], 2, "stub pair observed peak active 2")
        check(pair["observed"]["two_active_seconds"] > 0.05, "stub pair observed a two-active window")
        check(pair["staged"] is None, "stub pair is not staged (single synchronized burst)")
        check(burst["observed"]["peak_active"] == 2, "stub burst peak active 2")
        check(not burst["observed"]["saw_above_two"], "stub burst never exceeded two active")
        check(burst["observed"]["max_queue_depth"] >= 2, "stub burst queued above capacity")
        check_eq(burst["tasks_completed"], 4, "stub burst completed every task")
        queued = [task for task in burst["tasks"] if task["admission_basis"] == "slot-release"]
        check_eq(len(queued), 2, "stub burst: two tasks were admitted from the queue")
        check(all((task["queue_wait_ms_derived"] or 0.0) > 0 for task in queued),
              "stub burst: queued tasks have a positive derived queue wait")
        check(all(task["mapping_ok"] for task in burst["tasks"]),
              "stub burst: every response mapped to its own task")
        # The staged orchestration evidence: long first tasks held every slot,
        # the remaining two were asserted queued before any release.
        staged = burst.get("staged") or {}
        check_eq(staged.get("stage1_task_ids"),
                 ["burst-topic-long", "burst-tool-runner"], "stub burst stage-1 hold tasks")
        check_eq(staged.get("stage2_task_ids"),
                 ["burst-general-quick", "burst-structured-report"],
                 "stub burst stage-2 queued tasks")
        check_eq(staged.get("observed", {}).get("activeGenerations"), 2,
                 "stub burst: two active generations at the staged observation")
        check_eq(staged.get("observed", {}).get("queueDepth"), 2,
                 "stub burst: both remaining tasks observed queued")
        check(staged.get("queue_observed_before_any_release") is True,
              "stub burst: the queue was observed before any first task released")
        check_eq(staged.get("observed", {}).get("slotReleases"),
                 staged.get("releases_before_stage2"),
                 "stub burst: no slot release between the stage-1 hold and the observation")
        check(staged.get("observed_at", 0) >= staged.get("stage2_submitted_at", 0),
              "stub burst: the staged observation follows the stage-2 submission")
        cancel_staged = cancel.get("staged") or {}
        check_eq(cancel_staged.get("stage1_task_ids"),
                 ["cancel-long-victim", "cancel-long-peer"],
                 "stub cancel: the long tasks are the staged stage-1")
        check_eq(cancel_staged.get("stage2_task_ids"),
                 ["cancel-queued-interactive", "cancel-queued-background"],
                 "stub cancel: the queued tasks are the staged stage-2")
        check_eq(cancel_staged.get("observed", {}).get("queueDepth"), 2,
                 "stub cancel: the queue was observed above capacity")
        check(cancel["cancel"]["cancel_to_release_ms"] is not None,
              "stub cancel: slot release observed")
        check_eq(cancel["tasks_rejected"], 1, "stub cancel: the invalid request was refused")
        check_eq(cancel["tasks_cancelled"], 1, "stub cancel: exactly one cancellation")
        serial_burst = serial_arm["scenarios"][0]
        check_eq(serial_burst["observed"]["peak_active"], 1, "stub serial burst peak active 1")
        check(serial_burst["observed"]["max_queue_depth"] >= 3,
              "stub serial burst queued the other three tasks")
        check_eq(serial_burst["tasks_completed"], 4, "stub serial burst completed every task")
        serial_staged = serial_burst.get("staged") or {}
        check_eq(serial_staged.get("stage1_task_ids"), ["burst-topic-long"],
                 "stub serial burst stage-1 is the single long task")
        check_eq(serial_staged.get("stage2_task_ids"),
                 ["burst-tool-runner", "burst-general-quick", "burst-structured-report"],
                 "stub serial burst stage-2 is the other three tasks")
        check_eq(serial_staged.get("observed", {}).get("queueDepth"), 3,
                 "stub serial burst: all three remaining tasks observed queued")
        # Without a request log the queue waits are labeled estimates, and the
        # audit reports no derived-vs-server comparison.
        check_eq(burst["queue_wait_audit"]["compared"], 0,
                 "stub burst: no request log means no derived-vs-server comparison")
        check(all(task["queue_wait_ms_exact"] is False for task in burst["tasks"]),
              "stub burst: queue waits are labeled estimates without a request log")
        check(any(task["queue_wait_ms_source"] == "slot-release-estimate"
                  for task in burst["tasks"] if not task["expected_rejected"]),
              "stub burst: the estimate source is labeled")

    def chat_payload_messages_case() -> None:
        payload = chat_payload("m", "ignored", messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ])
        check_eq(payload["messages"],
                 [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
                 "an explicit message history is used as given")
        payload = chat_payload("m", "hello")
        check_eq(payload["messages"], [{"role": "user", "content": "hello"}],
                 "the content-only payload is unchanged")
        check_eq(payload["stream_options"], {"include_usage": True}, "include_usage unchanged")

    def server_command_case() -> None:
        args = argparse.Namespace(model_dir="/tmp/model", served_model_id="x", model_profile=None,
                                  max_queue_depth=2, queue_wait_ms=1000, memory_limit_bytes=0,
                                  extra_tokens=[], fused_gate_up_limit_env=True)
        cmd = server_command(args, Path("/bin/mei"), 1234, Path("/tmp/log"), cache_reuse=False)
        check_eq(cmd[cmd.index("--max-active-generations") + 1], "2",
                 "the default server command keeps the two-slot opt-in")
        cmd = server_command(args, Path("/bin/mei"), 1234, Path("/tmp/log"), cache_reuse=False,
                             max_active=1)
        check_eq(cmd[cmd.index("--max-active-generations") + 1], "1",
                 "the serial arm launches exactly one slot")
        cmd = server_command(args, Path("/bin/mei"), 1234, Path("/tmp/log"), cache_reuse=False,
                             max_queue_depth=3)
        check_eq(cmd[cmd.index("--max-queue-depth") + 1], "3", "the queue depth can be sized up")
        check_eq(cmd[cmd.index("--port") + 1], "1234", "the port is the one this client allocated")
        expect_failure(lambda: server_command(args, Path("/bin/mei"), 1234, Path("/tmp/log"),
                                               cache_reuse=False, max_active=3))
        expect_failure(lambda: server_command(args, Path("/bin/mei"), 1234, Path("/tmp/log"),
                                               cache_reuse=False, max_queue_depth=0))

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
    case("chat payload accepts an explicit isolated message history", chat_payload_messages_case)
    case("server command pins 1-2 slots and sizes the queue explicitly", server_command_case)
    case("peix fixtures validate, stay deterministic, and share one task set", peix_fixtures_case)
    case("peix admission attribution maps queued tasks to slot releases", peix_attribution_case)
    case("peix gate evaluation enforces the active-workload acceptance set", peix_gate_case)
    case("peix request-log attribution matches exactly and reports ambiguity", peix_request_log_case)
    case("peix queue-wait audit prefers exact server values and reports disagreement", peix_queue_wait_audit_case)
    case("peix staged burst fails loudly when the required queue state is unreachable", peix_staged_strictness_case)
    case("peix stub: pair+burst+cancel run end to end offline (staged)", peix_stub_case)

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
    parser.add_argument("--mode", choices=["canaries", "benchmark", "cancellation", "peix-like", "all"],
                        default="all", help="which acceptance mode(s) to run (default: all; "
                        "'all' stays canaries+benchmark+cancellation — peix-like is opt-in)")
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

    if args.mode == "peix-like" and peix_like is None:
        print(f"error: --mode peix-like requires scripts/integration/peix_like.py next to this "
              f"script (import failed: {PEIX_LIKE_IMPORT_ERROR})", file=sys.stderr)
        return 2

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
                elif mode == "peix-like":
                    result = run_peix_like(args, binary, model_dir, artifacts)
                    evidence["peix_like"] = result
                    evidence["gate_results"]["peix-like"] = (
                        "PASS" if result["verdict"]["passed"] else "FAIL")
                    if not result["verdict"]["passed"]:
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



