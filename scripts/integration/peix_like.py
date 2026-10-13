#!/usr/bin/env python3
"""Peix-like workload fixtures, offline helpers, and a stub Mei server.

This module is the fixture/helper home for `--mode peix-like` of
`scripts/test_concurrent_generations_integration.py` (Kiem plan
`820814e1`, stages 7-8: the Peix-like stand-in workload and its
latency/resource envelope). It is deliberately stdlib-only and free of any
process-launching code: the client stays canonical for launching, safety
checks, live requests, and artifacts.

What lives here:

  * `TaskEnvelope` — one independent logical-agent task (task/agent ID,
    isolated message history with its own role/instructions, scheduling
    class, stream mode, optional tools / structured response format, token
    budget, isolation nonce/session token). These are stand-in "Peix task
    envelopes", not a Peix integration: the real Peix harness does not
    exist yet, so nothing here claims Peix-side orchestration semantics.
  * The stand-in workloads: a synchronized 2-task pair, a staged 4-task
    burst whose two long first tasks are held in the slots (verified from
    `/v1/mei/status`) before the remaining two are submitted and must then be
    observed queued above the two-slot capacity, and a cancellation scenario
    (two long tasks, two queued, one deliberately invalid request).
  * Pure helpers used live and offline: admission/queue-wait attribution
    from observed slot-release transitions (an ESTIMATE, labeled as such),
    per-task record building and mapping verification (own nonce must
    return, no foreign token may appear, tool/structured payloads must
    match), scenario summarization, request-log line attribution, the
    derived-vs-server queue-wait audit (`resolve_queue_waits`; the exact
    server value is authoritative when uniquely matched), and the
    peix-like gate evaluation.
  * `StubMeiServer` — a small in-process stand-in used by the client's
    `--self-check` to exercise the whole peix-like scenario machinery
    (pair, burst, cancellation, mapping, watcher attribution, gates)
    without a model or a real server process.

Gate vocabulary (settled for the active workload; see the client docstring):
all tasks complete and map correctly, observed peak active == the configured
slot count and never > 2, queueing above capacity observed, cancellation
peer isolation, interactive p95 TTFT <= 5 s, >= 4 GiB reclaimable with no
sustained pressure or swap growth. The legacy >= 1.20x aggregate-throughput
and <= 1.20x-alone TTFT-ratio gates are NOT applied here: they are recorded
as measurements only (plan stage 7).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import select
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vocabulary (mirrored from the client so this module stays self-contained)
# ---------------------------------------------------------------------------

INTERACTIVE = "interactive"
BACKGROUND = "background"
VALID_PRIORITIES = (INTERACTIVE, BACKGROUND)

#: Interactive TTFT ceiling for the active workload (absolute; gate).
INTERACTIVE_TTFT_GATE_SECONDS = 5.0
#: Cancellation slot-release bound (gate; one sample in the cancel scenario).
CANCEL_TO_RELEASE_GATE_MS = 2000.0
#: Minimum client-observed overlap for a scenario that must overlap.
OVERLAP_MIN_SECONDS = 0.05

COUNT_INSTRUCTIONS = (
    "Count upward from 1, printing one number per line. "
    "Keep counting as long as you can; do not summarize or stop early."
)
TOOL_INSTRUCTIONS = "Use the add_numbers tool to add 17 and 25. Do not answer in text."
STRUCTURED_INSTRUCTIONS = 'Reply as JSON with the word "pong" and the number 2.'

ADD_NUMBERS_TOOL = {
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
ANSWER_FORMAT = {
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
EXPECTED_TOOL_ARGUMENTS = {"a": 17, "b": 25}
EXPECTED_STRUCTURED = {"word": "pong", "count": 2}


def nonce_instructions(nonce: str) -> str:
    return f"Reply with exactly this token and nothing else: {nonce}"


# ---------------------------------------------------------------------------
# Task envelopes
# ---------------------------------------------------------------------------


class TaskEnvelope:
    """One independent logical-agent task envelope.

    Models what a Peix logical agent would send to Mei: its own agent
    identity, an isolated message history (system role + private turns + the
    task instruction), a scheduling class, a stream mode, an optional tool
    schema or structured response format, and a token budget.

    `nonce` / `session_token` are the isolation markers: when
    `expect_own_nonce` is set the task's own nonce must come back in the
    response, and no other task's nonce or session token may ever appear in
    it (cross-agent leakage check).
    """

    def __init__(
        self,
        task_id: str,
        agent_id: str,
        kind: str,
        role: str,
        instructions: str,
        priority: str = INTERACTIVE,
        stream: bool = True,
        max_tokens: int = 64,
        temperature: float = 0.0,
        nonce: Optional[str] = None,
        session_token: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        expect_own_nonce: bool = False,
        expect_tool: Optional[Dict[str, Any]] = None,
        expect_structured: Optional[Dict[str, Any]] = None,
        expect_rejection: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_id = task_id
        self.agent_id = agent_id
        self.kind = kind
        self.role = role
        self.instructions = instructions
        self.priority = priority
        self.stream = stream
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.nonce = nonce
        self.session_token = session_token or ("PEIXSESS-" + task_id.upper().replace("-", ""))
        self.history = [dict(turn) for turn in (history or [])]
        self.tools = tools
        self.response_format = response_format
        self.expect_own_nonce = expect_own_nonce
        self.expect_tool = expect_tool
        self.expect_structured = expect_structured
        #: A deliberately invalid request (e.g. bad mei_priority) that must be
        #: refused with a real HTTP error envelope and never admitted.
        self.expect_rejection = expect_rejection

    def messages(self) -> List[Dict[str, str]]:
        """The task's isolated message history (system role + private turns + task)."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.role}]
        messages.extend(dict(turn) for turn in self.history)
        messages.append({"role": "user", "content": self.instructions})
        return messages

    def describe(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "kind": self.kind,
            "priority": self.priority,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "nonce": self.nonce,
            "session_token": self.session_token,
            "tool_names": [t.get("function", {}).get("name") for t in (self.tools or [])],
            "response_format_type": (self.response_format or {}).get("type"),
            "expect_own_nonce": self.expect_own_nonce,
            "expect_tool": self.expect_tool,
            "expect_structured": self.expect_structured,
            "expect_rejection": self.expect_rejection,
            "history_turns": len(self.history),
        }


def validate_envelope(env: TaskEnvelope) -> None:
    """Reject a malformed task envelope (raises ValueError)."""
    if not isinstance(env.task_id, str) or not env.task_id:
        raise ValueError("task envelope needs a non-empty task_id")
    if not isinstance(env.agent_id, str) or not env.agent_id:
        raise ValueError(f"{env.task_id}: task envelope needs a non-empty agent_id")
    if env.expect_rejection is None:
        if env.priority not in VALID_PRIORITIES:
            raise ValueError(f"{env.task_id}: invalid priority {env.priority!r}")
    elif env.priority in VALID_PRIORITIES:
        raise ValueError(
            f"{env.task_id}: expect_rejection envelopes must carry a deliberately invalid priority"
        )
    if not isinstance(env.stream, bool):
        raise ValueError(f"{env.task_id}: stream must be a bool")
    if not isinstance(env.max_tokens, int) or env.max_tokens < 1:
        raise ValueError(f"{env.task_id}: max_tokens must be a positive int")
    if env.temperature != 0.0:
        raise ValueError(f"{env.task_id}: task envelopes must be greedy (temperature 0.0)")
    if env.expect_own_nonce:
        if not env.nonce:
            raise ValueError(f"{env.task_id}: expect_own_nonce without a nonce")
        if env.nonce not in env.instructions:
            raise ValueError(f"{env.task_id}: nonce {env.nonce!r} is not in the instructions")
    if env.expect_tool:
        if not env.tools:
            raise ValueError(f"{env.task_id}: tool expectation without a tools schema")
        names = [t.get("function", {}).get("name") for t in env.tools]
        if env.expect_tool.get("name") not in names:
            raise ValueError(f"{env.task_id}: expected tool {env.expect_tool.get('name')!r} not in {names}")
    if env.expect_structured:
        if env.response_format is None:
            raise ValueError(f"{env.task_id}: structured expectation without a response_format")
    messages = env.messages()
    if not messages:
        raise ValueError(f"{env.task_id}: empty message history")
    for turn in messages:
        if turn.get("role") not in ("system", "user", "assistant"):
            raise ValueError(f"{env.task_id}: invalid message role {turn.get('role')!r}")
        if not isinstance(turn.get("content"), str):
            raise ValueError(f"{env.task_id}: message content must be a string")


def validate_workload(envelopes: List[TaskEnvelope]) -> None:
    """Reject a malformed workload: bad envelopes, duplicate IDs/nonces/tokens."""
    if not envelopes:
        raise ValueError("empty workload")
    seen_ids: set = set()
    seen_nonces: set = set()
    seen_tokens: set = set()
    for env in envelopes:
        validate_envelope(env)
        if env.task_id in seen_ids:
            raise ValueError(f"duplicate task id {env.task_id!r}")
        seen_ids.add(env.task_id)
        if env.nonce:
            if env.nonce in seen_nonces:
                raise ValueError(f"duplicate nonce {env.nonce!r}")
            seen_nonces.add(env.nonce)
        if env.session_token in seen_tokens:
            raise ValueError(f"duplicate session token {env.session_token!r}")
        seen_tokens.add(env.session_token)


# ---------------------------------------------------------------------------
# Stand-in workloads (identical task sets in both arms; only expectations differ)
# ---------------------------------------------------------------------------

_GENERAL_ROLE = (
    "You are the General agent of a local assistant. Answer the user's quick "
    "question directly and stop."
)
_TOPIC_ROLE = (
    "You are a background topic-index agent working through durable work. "
    "Produce the enumeration the user asks for without commentary."
)
_TOOL_ROLE = "You are a background tool-running agent. Call the requested tool."
_REPORT_ROLE = "You are a background report-writing agent. Answer with the requested JSON object."


def pair_envelopes() -> List[TaskEnvelope]:
    """The synchronized 2-task pair: short interactive + long background."""
    quick = TaskEnvelope(
        task_id="pair-general-quick",
        agent_id="agent-general",
        kind="interactive-short",
        role=_GENERAL_ROLE,
        instructions=nonce_instructions("PEIX-PAIR-ALPHA-7Q"),
        priority=INTERACTIVE,
        stream=True,
        max_tokens=16,
        nonce="PEIX-PAIR-ALPHA-7Q",
        expect_own_nonce=True,
    )
    long = TaskEnvelope(
        task_id="pair-topic-long",
        agent_id="agent-topic-index",
        kind="background-long",
        role=_TOPIC_ROLE,
        instructions=COUNT_INSTRUCTIONS,
        priority=BACKGROUND,
        stream=True,
        max_tokens=192,
    )
    return [quick, long]


def burst_envelopes() -> List[TaskEnvelope]:
    """The staged 4-task burst.

    Order is the orchestration contract: the first two tasks are the stage-1
    hold (a long background enumeration and the interactive tool call), which
    the staged runner submits together and must see occupy every slot before
    submitting the remaining two (the short General-like nonce task and the
    structured report). The two remaining tasks must then be observed queued
    above the two-slot capacity before any first task releases, so the hold
    tasks must outlive the stage-2 submission window; the staged runner fails
    loudly if one releases first instead of degrading the queue evidence.
    """
    long = TaskEnvelope(
        task_id="burst-topic-long",
        agent_id="agent-topic-index",
        kind="background-long",
        role=_TOPIC_ROLE,
        instructions=COUNT_INSTRUCTIONS,
        priority=BACKGROUND,
        stream=True,
        max_tokens=192,
    )
    tool = TaskEnvelope(
        task_id="burst-tool-runner",
        agent_id="agent-tool-runner",
        kind="tool-call",
        role=_TOOL_ROLE,
        instructions=TOOL_INSTRUCTIONS,
        priority=INTERACTIVE,
        stream=True,
        max_tokens=128,
        tools=[ADD_NUMBERS_TOOL],
        expect_tool={"name": "add_numbers", "arguments": dict(EXPECTED_TOOL_ARGUMENTS)},
    )
    quick = TaskEnvelope(
        task_id="burst-general-quick",
        agent_id="agent-general",
        kind="interactive-short",
        role=_GENERAL_ROLE,
        instructions=nonce_instructions("PEIX-BURST-ALPHA-2M"),
        priority=INTERACTIVE,
        stream=True,
        max_tokens=16,
        nonce="PEIX-BURST-ALPHA-2M",
        expect_own_nonce=True,
    )
    structured = TaskEnvelope(
        task_id="burst-structured-report",
        agent_id="agent-report-writer",
        kind="structured-report",
        role=_REPORT_ROLE,
        instructions=STRUCTURED_INSTRUCTIONS,
        priority=BACKGROUND,
        stream=False,
        max_tokens=64,
        response_format=ANSWER_FORMAT,
        expect_structured=dict(EXPECTED_STRUCTURED),
    )
    return [long, tool, quick, structured]


def cancel_envelopes() -> List[TaskEnvelope]:
    """The cancellation scenario: a long victim, a long peer, two queued
    tasks, and one deliberately invalid request (bad mei_priority) that must
    be refused without disturbing anyone else."""
    victim = TaskEnvelope(
        task_id="cancel-long-victim",
        agent_id="agent-topic-index",
        kind="background-long",
        role=_TOPIC_ROLE,
        instructions=COUNT_INSTRUCTIONS,
        priority=BACKGROUND,
        stream=True,
        max_tokens=1024,
    )
    peer = TaskEnvelope(
        task_id="cancel-long-peer",
        agent_id="agent-topic-index-2",
        kind="background-long",
        role=_TOPIC_ROLE,
        instructions=COUNT_INSTRUCTIONS,
        priority=BACKGROUND,
        stream=True,
        max_tokens=512,
    )
    queued_interactive = TaskEnvelope(
        task_id="cancel-queued-interactive",
        agent_id="agent-general",
        kind="interactive-short",
        role=_GENERAL_ROLE,
        instructions=nonce_instructions("PEIX-CANCEL-ALPHA-9F"),
        priority=INTERACTIVE,
        stream=True,
        max_tokens=16,
        nonce="PEIX-CANCEL-ALPHA-9F",
        expect_own_nonce=True,
    )
    queued_background = TaskEnvelope(
        task_id="cancel-queued-background",
        agent_id="agent-report-writer",
        kind="background-short",
        role=_REPORT_ROLE,
        instructions=nonce_instructions("PEIX-CANCEL-BRAVO-4T"),
        priority=BACKGROUND,
        stream=True,
        max_tokens=16,
        nonce="PEIX-CANCEL-BRAVO-4T",
        expect_own_nonce=True,
    )
    bad_request = TaskEnvelope(
        task_id="cancel-bad-request",
        agent_id="agent-misconfigured",
        kind="invalid-request",
        role=_REPORT_ROLE,
        instructions="This request must be refused before admission.",
        priority="urgent",  # deliberately invalid: must never be admitted
        stream=False,
        max_tokens=8,
        expect_rejection={
            "status": 400,
            "type": "invalid_request_error",
            "code": "invalid_priority",
            "param": "mei_priority",
        },
    )
    return [victim, peer, queued_interactive, queued_background, bad_request]


def peix_scenarios(max_active: int) -> List[Dict[str, Any]]:
    """The stand-in workload scenarios for one arm (same task set per arm).

    `max_active` only shapes the expectations (peak active, queue depth,
    whether overlap is required) — the envelopes are identical in both arms
    so the serial and overlap runs compare the same task set.
    """
    if max_active not in (1, 2):
        raise ValueError(f"max_active must be 1 or 2, got {max_active!r}")

    def scenario(name: str, tasks: List[TaskEnvelope], *, queue_expected: bool,
                 require_overlap: bool, gate_ttft: bool,
                 cancel_task_id: Optional[str] = None,
                 peer_active_before_cancel: bool = False,
                 stagger_seconds: float = 0.0,
                 staged: bool = False) -> Dict[str, Any]:
        validate_workload(tasks)
        admittable = [t for t in tasks if t.expect_rejection is None]
        expected_queue = (len(admittable) - max_active) if queue_expected else None
        if staged and not queue_expected:
            raise ValueError(f"scenario {name!r}: staged submission requires queueing above capacity")
        return {
            "name": name,
            "tasks": tasks,
            "cancel_task_id": cancel_task_id,
            "peer_active_before_cancel": peer_active_before_cancel,
            # Submissions are staggered in fixture order so the arrival order
            # (and therefore the admission order) is deterministic; the pair
            # keeps 0.0 because it is the deliberately synchronized overlap
            # test (both requests fire together).
            "stagger_seconds": stagger_seconds,
            # A staged scenario holds its first `max_active` tasks in the
            # slots (verified from /v1/mei/status) before submitting the
            # rest, so the queue-above-capacity state is deterministic
            # instead of a race against the first tasks' completions.
            "staged": staged,
            "expected": {
                "peak_active": max_active,
                "queue_depth": expected_queue,
                "require_overlap": require_overlap,
                "gate_interactive_ttft": gate_ttft,
            },
        }

    pair = scenario("pair", pair_envelopes(), queue_expected=False,
                    require_overlap=max_active > 1, gate_ttft=True,
                    stagger_seconds=0.0, staged=False)
    burst = scenario("burst", burst_envelopes(), queue_expected=True,
                     require_overlap=max_active > 1, gate_ttft=True,
                     stagger_seconds=0.05, staged=True)
    cancel = scenario("cancel", cancel_envelopes(), queue_expected=True,
                      require_overlap=False, gate_ttft=False,
                      cancel_task_id="cancel-long-victim",
                      peer_active_before_cancel=max_active > 1,
                      stagger_seconds=0.05, staged=True)
    return [pair, burst, cancel]


def required_queue_depth(scenarios: List[Dict[str, Any]], max_active: int) -> int:
    """The smallest server --max-queue-depth that fits the whole workload."""
    needed = 0
    for scenario in scenarios:
        admittable = [t for t in scenario["tasks"] if t.expect_rejection is None]
        needed = max(needed, len(admittable) - max_active)
    return max(needed, 0)


def workload_summary(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A JSON-safe description of the workload (for dry-run and evidence)."""
    out = []
    for scenario in scenarios:
        out.append({
            "name": scenario["name"],
            "expected": dict(scenario["expected"]),
            "cancel_task_id": scenario.get("cancel_task_id"),
            "staged": bool(scenario.get("staged")),
            "tasks": [t.describe() for t in scenario["tasks"]],
        })
    return out


# ---------------------------------------------------------------------------
# Pure helpers: observations, attribution, records, summaries
# ---------------------------------------------------------------------------


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile (same semantics as the client's)."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(math.ceil(pct / 100.0 * len(ordered)))))
    return ordered[rank - 1]


def observations_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive the scenario's observed scheduler behaviour from watcher samples.

    Samples are `/v1/mei/status` scheduler blocks with a monotonic `at`
    timestamp (the client's SchedulerWatcher adds it). Slot-release times are
    the sample timestamps at which `slotReleases` was first seen to advance;
    each observed increment yields one (approximate, +/- one poll interval)
    release time.
    """
    peak = max((int(s.get("activeGenerations", 0)) for s in samples), default=0)
    peak_counter = max((int(s.get("peakActiveGenerations", 0)) for s in samples), default=0)
    max_queue = max((int(s.get("queueDepth", 0)) for s in samples), default=0)
    saw_above_two = any(int(s.get("activeGenerations", 0)) > 2 for s in samples)
    two_active_seconds = 0.0
    for previous_sample, sample in zip(samples, samples[1:]):
        if int(previous_sample.get("activeGenerations", 0)) >= 2:
            two_active_seconds += max(
                0.0, float(sample.get("at", 0.0)) - float(previous_sample.get("at", 0.0)))
    release_times: List[float] = []
    previous: Optional[int] = None
    for sample in samples:
        current = int(sample.get("slotReleases", 0))
        if previous is not None and current > previous:
            release_times.extend([float(sample.get("at", 0.0))] * (current - previous))
        previous = current
    return {
        "samples": len(samples),
        "peak_active": peak,
        "peak_counter": peak_counter,
        "max_queue_depth": max_queue,
        "saw_above_two": saw_above_two,
        "two_active_seconds": two_active_seconds,
        "slot_release_times": release_times,
        "initial_slot_releases": int(samples[0].get("slotReleases", 0)) if samples else None,
        "final_slot_releases": int(samples[-1].get("slotReleases", 0)) if samples else None,
    }


def attribute_admissions(
    submissions: List[Dict[str, Any]],
    max_active: int,
    release_times: List[float],
) -> Dict[str, Dict[str, Any]]:
    """Attribute each task's admission time from the observed scheduler timeline.

    This is an ESTIMATE, never a measurement: `admitted_at` is a watcher
    sample time (approximate to +/- one poll interval, and a release
    transition can be missed entirely when samples are slow under load), and
    `queue_wait_ms` additionally carries the client-send -> server-enqueue
    latency, which no client-side observation can see. It is structurally
    valid only for the controlled stand-in scenarios: submissions are staged
    so the first `max_active` arrivals find free slots (immediate admission,
    queue wait 0), and every later arrival is enqueued while every slot is
    verifiably held (the staged runner asserts that), so the k-th observed
    release admits the k-th queued request in interactive-first-then-FIFO
    order. The engine's own request-log `queue_wait_ms` is exact and is the
    authoritative value whenever the line-to-task match is unique (see
    `attribute_request_log` and `resolve_queue_waits`); these derived values
    are corroboration and are never presented as exact. (The synchronized
    pair scenario fires both requests together by design: under one slot its
    arrival order is inherently racy, so its derived values are corroborated
    by the request log and never used as a gate.)

    `submissions`: [{"task_id", "priority", "sent_at"}, ...]
    """
    order = sorted(submissions, key=lambda s: (s["sent_at"], s["task_id"]))
    out: Dict[str, Dict[str, Any]] = {}
    for submission in order[:max_active]:
        out[submission["task_id"]] = {
            "admitted_at": submission["sent_at"],
            "queue_wait_ms": 0.0,
            "basis": "immediate",
        }
    waiting = order[max_active:]
    ranked = sorted(
        waiting,
        key=lambda s: (0 if s["priority"] == INTERACTIVE else 1, s["sent_at"], s["task_id"]),
    )
    for index, submission in enumerate(ranked):
        if index < len(release_times):
            admitted = float(release_times[index])
            out[submission["task_id"]] = {
                "admitted_at": admitted,
                "queue_wait_ms": max(0.0, (admitted - submission["sent_at"]) * 1000.0),
                "basis": "slot-release",
            }
        else:
            out[submission["task_id"]] = {
                "admitted_at": None,
                "queue_wait_ms": None,
                "basis": "unattributed",
            }
    return out


def _normalized_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a tool call from either wire shape (flat or nested)."""
    function = call.get("function")
    if isinstance(function, dict):
        return {"name": function.get("name"), "arguments": function.get("arguments")}
    return {"name": call.get("name"), "arguments": call.get("arguments")}


def result_fingerprint(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """A hash of the deterministic response surface (cross-arm comparison)."""
    if result is None:
        return None
    surface = {
        "text": result.get("text") or "",
        "finish_reason": result.get("finish_reason"),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
        "tool_calls": [_normalized_tool_call(c) for c in (result.get("tool_calls") or [])],
    }
    return hashlib.sha1(json.dumps(surface, sort_keys=True).encode("utf-8")).hexdigest()


def _response_text(result: Dict[str, Any]) -> str:
    return (result.get("text") or "") + "\n" + (result.get("reasoning") or "")


def verify_task_result(
    env: TaskEnvelope,
    result: Optional[Dict[str, Any]],
    error: Optional[str],
    foreign_tokens: List[str],
    expected_cancelled: bool = False,
) -> Tuple[bool, str]:
    """Check that one task's response belongs to that task and is healthy."""
    if env.expect_rejection is not None:
        if error:
            return False, f"rejected request raised: {error}"
        if result is None:
            return False, "no result collected for the deliberately invalid request"
        expected = env.expect_rejection
        if result.get("status") != expected.get("status"):
            return False, f"rejection status {result.get('status')!r} != {expected.get('status')!r}"
        payload = result.get("raw")
        envelope = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(envelope, dict):
            return False, f"rejection body has no error envelope: {payload!r}"
        for key in ("code", "type", "param"):
            if envelope.get(key) != expected.get(key):
                return False, f"rejection error.{key} {envelope.get(key)!r} != {expected.get(key)!r}"
        return True, "refused before admission with the expected error envelope"
    if expected_cancelled:
        if error:
            return False, f"cancelled task raised: {error}"
        if result is None:
            return False, "no result collected for the cancelled task"
        if result.get("finish_reason") is not None:
            return False, f"cancelled task reported finish_reason {result['finish_reason']!r} (false success)"
        if result.get("saw_done"):
            return False, "cancelled task reached [DONE] before cancellation took effect"
        return True, "cancelled without a false success finish"
    if error:
        return False, f"transport/runtime error: {error}"
    if result is None:
        return False, "no result"
    if result.get("stream_error"):
        return False, f"stream_error frame: {result['stream_error']}"
    finish = result.get("finish_reason")
    if finish not in ("stop", "length", "tool_calls"):
        return False, f"unexpected finish_reason {finish!r}"
    usage = result.get("usage") or {}
    completion = usage.get("completion_tokens")
    if not isinstance(completion, int) or completion < 1:
        return False, f"missing/zero completion_tokens: {usage!r}"
    text = _response_text(result)
    if env.expect_own_nonce and env.nonce and env.nonce not in text:
        return False, f"own nonce {env.nonce!r} missing from the response"
    for token in foreign_tokens:
        if token in text:
            return False, f"foreign token {token!r} leaked into the response"
    if env.expect_tool:
        calls = result.get("tool_calls") or []
        if len(calls) != 1:
            return False, f"expected exactly one tool call, got {len(calls)}"
        normalized = _normalized_tool_call(calls[0])
        if normalized.get("name") != env.expect_tool.get("name"):
            return False, f"tool name {normalized.get('name')!r} != {env.expect_tool.get('name')!r}"
        try:
            arguments = json.loads(normalized.get("arguments") or "{}")
        except ValueError:
            return False, f"tool arguments are not JSON: {normalized.get('arguments')!r}"
        if arguments != env.expect_tool.get("arguments"):
            return False, f"tool arguments {arguments!r} != {env.expect_tool.get('arguments')!r}"
    if env.expect_structured:
        try:
            payload = json.loads(result.get("text") or "")
        except ValueError:
            return False, f"structured response is not JSON: {(result.get('text') or '')[:120]!r}"
        if payload != env.expect_structured:
            return False, f"structured payload {payload!r} != {env.expect_structured!r}"
    return True, "mapped to its own task"


def build_task_records(
    envelopes: List[TaskEnvelope],
    results: Dict[str, Optional[Dict[str, Any]]],
    errors: Dict[str, Optional[str]],
    cancelled_ids: List[str],
    admissions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Per-task records: timings, usage, verification outcome, queue wait."""
    records: List[Dict[str, Any]] = []
    for env in envelopes:
        result = results.get(env.task_id)
        error = errors.get(env.task_id)
        foreign = [
            token
            for other in envelopes
            if other.task_id != env.task_id
            for token in (([other.nonce] if other.nonce else []) + [other.session_token])
        ]
        ok, detail = verify_task_result(env, result, error, foreign, env.task_id in cancelled_ids)
        usage = (result or {}).get("usage") or {}
        record: Dict[str, Any] = {
            "task_id": env.task_id,
            "agent_id": env.agent_id,
            "kind": env.kind,
            "priority": env.priority,
            "stream": env.stream,
            "max_tokens": env.max_tokens,
            "nonce": env.nonce,
            "expected_cancelled": env.task_id in cancelled_ids,
            "expected_rejected": env.expect_rejection is not None,
            "cancelled_observed": bool((result or {}).get("aborted")) if result else False,
            "error": error,
            "stream_error": (result or {}).get("stream_error"),
            "status": (result or {}).get("status"),
            "finish_reason": (result or {}).get("finish_reason"),
            "saw_done": bool((result or {}).get("saw_done")) if result else False,
            "ttft_seconds": (result or {}).get("ttft_seconds"),
            "total_seconds": (result or {}).get("total_seconds"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "sent_at": (result or {}).get("sent_at"),
            "done_at": (result or {}).get("done_at"),
            "mapping_ok": ok,
            "mapping_detail": detail,
            "fingerprint": result_fingerprint(result),
            "text_preview": ((result or {}).get("text") or "")[:400],
            "reasoning_preview": ((result or {}).get("reasoning") or "")[:200],
            "tool_calls": (result or {}).get("tool_calls") or [],
        }
        admission = (admissions or {}).get(env.task_id)
        record["queue_wait_ms_derived"] = admission["queue_wait_ms"] if admission else None
        record["admission_basis"] = admission["basis"] if admission else None
        record["queue_wait_ms_server"] = None
        records.append(record)
    return records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scenario-level makespan and aggregate completion throughput."""
    sent = [r["sent_at"] for r in records if r.get("sent_at") is not None]
    done = [r["done_at"] for r in records if r.get("done_at") is not None]
    makespan = (max(done) - min(sent)) if (sent and done) else None
    tokens = sum(int(r.get("completion_tokens") or 0) for r in records if not r["expected_cancelled"])
    rate = (tokens / makespan) if (makespan and makespan > 0) else None
    return {
        "makespan_seconds": makespan,
        "aggregate_completion_tokens": tokens,
        "aggregate_tokens_per_second": rate,
        "tasks_completed": sum(
            1 for r in records
            if not r["expected_cancelled"] and not r["expected_rejected"]
            and r.get("finish_reason") in ("stop", "length", "tool_calls")
        ),
        "tasks_cancelled": sum(1 for r in records if r["expected_cancelled"]),
        "tasks_rejected": sum(1 for r in records if r["expected_rejected"]),
    }


def attribute_request_log(
    records: List[Dict[str, Any]], log_lines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Match server request-log lines to tasks (exact per-run queue wait).

    Lines are written one per finalized run, in completion order; the log has
    no task ID, so a line is attributed to a task only when the match is
    unambiguous on (priority, completion_tokens, finish) — the deliberately
    rejected request never appears in the log. Anything else is reported as
    unattributed and never used as evidence.
    """
    used: set = set()
    per_task: Dict[str, Any] = {}
    notes: List[str] = []
    for record in records:
        if record.get("expected_rejected"):
            continue
        candidates = []
        for index, line in enumerate(log_lines):
            if index in used:
                continue
            if record["expected_cancelled"]:
                if line.get("finish") == "cancelled":
                    candidates.append(index)
            elif (
                line.get("finish") == record.get("finish_reason")
                and line.get("priority") == record.get("priority")
                and line.get("completion_tokens") == record.get("completion_tokens")
            ):
                candidates.append(index)
        if len(candidates) == 1:
            used.add(candidates[0])
            line = log_lines[candidates[0]]
            per_task[record["task_id"]] = {
                "queue_wait_ms": line.get("queue_wait_ms"),
                "wall_ms": line.get("wall_ms"),
                "prefill_ms": line.get("prefill_ms"),
                "generate_ms": line.get("generate_ms"),
                "priority": line.get("priority"),
                "finish": line.get("finish"),
                "completion_tokens": line.get("completion_tokens"),
            }
        elif not candidates:
            notes.append(f"{record['task_id']}: no matching request-log line")
        else:
            notes.append(f"{record['task_id']}: {len(candidates)} ambiguous request-log lines")
    expected_records = sum(1 for r in records if not r.get("expected_rejected"))
    attributed = len(per_task) == expected_records and len(used) == len(log_lines)
    return {
        "per_task": per_task,
        "attributed": attributed,
        "lines": len(log_lines),
        "unused_lines": len(log_lines) - len(used),
        "notes": notes,
    }


#: Derived-vs-server queue-wait disagreement beyond this is reported as a
#: real disagreement (the server request-log value stays authoritative).
QUEUE_WAIT_AUDIT_TOLERANCE_MS = 500.0


def resolve_queue_waits(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick each task's authoritative queue wait and audit derived vs server.

    `queue_wait_ms_server` (the engine's own request log) is exact when the
    line-to-task match was unique; `queue_wait_ms_derived` is an estimate
    from observed slot-release transitions. This sets, per record:

      * `queue_wait_ms` — the authoritative value: the exact server value
        when the match is unique, otherwise the (labeled) estimate;
      * `queue_wait_ms_source` — "server-request-log" | "slot-release-estimate"
        | "unattributed";
      * `queue_wait_ms_exact` — whether the presented value is the engine's
        own measurement.

    and returns the audit block comparing both sources. When they disagree,
    the block names the sources of error (client-send -> server-enqueue
    latency is not observable client-side; watcher sampling adds up to one
    poll interval and can miss a release transition under load) so an
    estimate is never presented as exact.
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        derived = record.get("queue_wait_ms_derived")
        server = record.get("queue_wait_ms_server")
        if server is not None:
            record["queue_wait_ms"] = server
            record["queue_wait_ms_source"] = "server-request-log"
            record["queue_wait_ms_exact"] = True
        elif derived is not None:
            record["queue_wait_ms"] = derived
            record["queue_wait_ms_source"] = "slot-release-estimate"
            record["queue_wait_ms_exact"] = False
        else:
            record["queue_wait_ms"] = None
            record["queue_wait_ms_source"] = "unattributed"
            record["queue_wait_ms_exact"] = False
        if derived is not None and server is not None:
            rows.append({
                "task_id": record.get("task_id"),
                "derived_ms": derived,
                "server_ms": server,
                "delta_ms": float(server) - float(derived),
            })
    max_abs = max((abs(row["delta_ms"]) for row in rows), default=None)
    disagreement = max_abs is not None and max_abs > QUEUE_WAIT_AUDIT_TOLERANCE_MS
    notes: List[str] = []
    if rows:
        notes.append(
            "queue_wait_ms_server is the engine's exact queue wait (request log, unique "
            "line-to-task match); queue_wait_ms_derived is an estimate from observed "
            "slot-release transitions (+/- one watcher poll interval; it cannot see the "
            "client-send -> server-enqueue latency, and a transition can be missed under load)")
    if disagreement:
        notes.append(
            f"derived and server queue waits disagree by up to {max_abs:.0f} ms "
            f"(tolerance {QUEUE_WAIT_AUDIT_TOLERANCE_MS:.0f} ms): the expected attribution gap "
            "between an observation-based estimate and the engine's own admission timing, not a "
            "gate failure — the server request-log values are authoritative wherever the match "
            "is unique")
    return {
        "compared": len(rows),
        "max_abs_delta_ms": max_abs,
        "tolerance_ms": QUEUE_WAIT_AUDIT_TOLERANCE_MS,
        "disagreement": disagreement,
        "rows": rows,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Gate evaluation (pure; exercised by --self-check)
# ---------------------------------------------------------------------------

#: Which gates count toward acceptance for each arm. The serial arm is the
#: comparison baseline: its task completion/mapping and its slot-cap
#: invariant must hold, but its queue waits / TTFT are expected to be worse
#: by construction and are recorded as measurements.
ACCEPTANCE_GATES: Dict[str, List[str]] = {
    "overlap": [
        "all_tasks_complete",
        "task_mapping_correct",
        "peak_active_exactly_max_and_never_above",
        "overlap_observed",
        "queueing_above_capacity",
        "cancellation_peer_isolation",
        "interactive_ttft_p95_within_ceiling",
        "memory_envelope",
    ],
    "serial": [
        "all_tasks_complete",
        "task_mapping_correct",
        "peak_active_exactly_max_and_never_above",
    ],
}


def evaluate_arm_gates(arm: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluate the peix-like gates for one arm's evidence."""
    arm_name = arm.get("arm", "overlap")
    gates: Dict[str, Dict[str, Any]] = {}

    def gate(name: str, passed: bool, value: Any = None, detail: str = "") -> None:
        gates[name] = {
            "passed": bool(passed),
            "value": value,
            "detail": detail,
            "acceptance": name in ACCEPTANCE_GATES.get(arm_name, []),
        }

    # 1. Every task reached a healthy terminal state (or the expected
    #    cancellation / refusal).
    problems: List[str] = []
    for scenario in arm["scenarios"]:
        for task in scenario["tasks"]:
            label = f"{scenario['name']}/{task['task_id']}"
            if task.get("expected_cancelled"):
                if (
                    not task.get("cancelled_observed")
                    or task.get("finish_reason") is not None
                    or task.get("error")
                ):
                    problems.append(f"{label}: cancellation not observed cleanly")
            elif task.get("expected_rejected"):
                if not task.get("mapping_ok"):
                    problems.append(f"{label}: refusal not as expected ({task.get('mapping_detail')})")
            elif (
                task.get("error")
                or task.get("stream_error")
                or task.get("finish_reason") not in ("stop", "length", "tool_calls")
            ):
                problems.append(f"{label}: {task.get('error') or task.get('stream_error') or task.get('finish_reason')}")
    gate("all_tasks_complete", not problems, value=len(problems), detail="; ".join(problems))

    # 2. Every response belongs to its own task (nonce/tool/structured match,
    #    no foreign token).
    mapping_problems = [
        f"{scenario['name']}/{task['task_id']}: {task.get('mapping_detail')}"
        for scenario in arm["scenarios"]
        for task in scenario["tasks"]
        if not task.get("mapping_ok")
    ]
    gate("task_mapping_correct", not mapping_problems, value=len(mapping_problems),
         detail="; ".join(mapping_problems))

    # 3. Observed peak == the configured slot count, never above two.
    peak_problems: List[str] = []
    for scenario in arm["scenarios"]:
        observed = scenario["observed"]
        expected_peak = scenario["expected"]["peak_active"]
        if observed["peak_active"] != expected_peak:
            peak_problems.append(f"{scenario['name']}: peak {observed['peak_active']} != {expected_peak}")
        if observed["saw_above_two"]:
            peak_problems.append(f"{scenario['name']}: active count exceeded 2")
        if observed.get("peak_counter", 0) > 2:
            peak_problems.append(f"{scenario['name']}: peakActiveGenerations {observed['peak_counter']} > 2")
    gate("peak_active_exactly_max_and_never_above", not peak_problems,
         value={s["name"]: s["observed"]["peak_active"] for s in arm["scenarios"]},
         detail="; ".join(peak_problems))

    # 4. Real overlap where the workload requires it (overlap arm only):
    #    time actually observed with two generations active, corroborated by
    #    the all-tasks-in-flight window.
    required = [s for s in arm["scenarios"] if s["expected"].get("require_overlap")]
    if not required:
        gate("overlap_observed", True, value=None,
             detail="no scenario in this arm requires overlap (serial baseline)")
    else:
        overlap_problems: List[str] = []
        overlap_values: Dict[str, Any] = {}
        for scenario in required:
            two_active = float(scenario["observed"].get("two_active_seconds") or 0.0)
            all_in_flight = float(scenario.get("all_in_flight_seconds") or 0.0)
            overlap_values[scenario["name"]] = {
                "two_active_seconds": two_active,
                "all_in_flight_seconds": all_in_flight,
            }
            if two_active <= OVERLAP_MIN_SECONDS:
                overlap_problems.append(
                    f"{scenario['name']}: two-active window {two_active:.3f}s "
                    f"(all tasks in flight {all_in_flight:.3f}s)")
        gate("overlap_observed", not overlap_problems, value=overlap_values,
             detail="; ".join(overlap_problems))

    # 5. Queueing above capacity actually happened.
    queue_scenarios = [s for s in arm["scenarios"] if s["expected"].get("queue_depth")]
    if not queue_scenarios:
        gate("queueing_above_capacity", False,
             detail="no scenario in the workload expects queueing (misconfigured)")
    else:
        queue_problems = [
            f"{s['name']}: max queue depth {s['observed']['max_queue_depth']} < {s['expected']['queue_depth']}"
            for s in queue_scenarios
            if s["observed"]["max_queue_depth"] < s["expected"]["queue_depth"]
        ]
        gate("queueing_above_capacity", not queue_problems,
             value={s["name"]: s["observed"]["max_queue_depth"] for s in queue_scenarios},
             detail="; ".join(queue_problems))

    # 6. Cancelling one task leaves every other task healthy. The cancellation
    #    scenario is identified by its `cancel` evidence block, which carries
    #    the victim task ID and the observed cancel-to-release latency.
    cancel_scenarios = [s for s in arm["scenarios"] if s.get("cancel")]
    if not cancel_scenarios:
        gate("cancellation_peer_isolation", False, detail="no cancellation scenario in the workload")
    else:
        cancel_problems: List[str] = []
        cancel_values: Dict[str, Any] = {}
        for scenario in cancel_scenarios:
            cancel_info = scenario.get("cancel") or {}
            victim_id = cancel_info.get("victim")
            victim = next(
                (t for t in scenario["tasks"] if t["task_id"] == victim_id), None)
            if victim is None:
                cancel_problems.append(f"{scenario['name']}: cancel task {victim_id!r} not found")
                continue
            if not victim.get("cancelled_observed") or victim.get("finish_reason") is not None:
                cancel_problems.append(f"{scenario['name']}: cancellation not cleanly observed")
            peers = [t for t in scenario["tasks"] if t["task_id"] != victim_id]
            broken = [
                t["task_id"] for t in peers
                if (t.get("error") or t.get("stream_error")
                    or (not t.get("expected_rejected")
                        and t.get("finish_reason") not in ("stop", "length", "tool_calls"))
                    or not t.get("mapping_ok"))
            ]
            if broken:
                cancel_problems.append(f"{scenario['name']}: peer tasks unhealthy: {', '.join(broken)}")
            milliseconds = cancel_info.get("cancel_to_release_ms")
            cancel_values[scenario["name"]] = {
                "cancel_to_release_ms": milliseconds,
                "peers": len(peers),
            }
            if milliseconds is None:
                cancel_problems.append(f"{scenario['name']}: slot release was not observed")
            elif milliseconds > CANCEL_TO_RELEASE_GATE_MS:
                cancel_problems.append(
                    f"{scenario['name']}: cancel-to-release {milliseconds:.0f} ms > "
                    f"{CANCEL_TO_RELEASE_GATE_MS:.0f} ms")
        gate("cancellation_peer_isolation", not cancel_problems,
             value=cancel_values, detail="; ".join(cancel_problems))

    # 7. Interactive TTFT ceiling (absolute; the ratio is a measurement only).
    ttfts = list(arm.get("interactive_ttft_gated_seconds") or [])
    if not ttfts:
        gate("interactive_ttft_p95_within_ceiling", False,
             detail="no gated interactive TTFT samples were collected")
    else:
        p95 = percentile(ttfts, 95)
        gate("interactive_ttft_p95_within_ceiling", p95 <= INTERACTIVE_TTFT_GATE_SECONDS,
             value={"p95_seconds": p95, "samples": len(ttfts), "ttfts_seconds": ttfts},
             detail=f"p95 {p95:.3f}s (gate <= {INTERACTIVE_TTFT_GATE_SECONDS}s)")

    # 8. Memory envelope (floor + pressure + swap), evaluated by the client's
    #    memory gate; `evaluated: false` only happens in offline stub runs.
    memory = arm.get("memory_gate")
    if memory is None:
        gate("memory_envelope", True, value=None, detail="not evaluated (offline stub run)")
        gates["memory_envelope"]["evaluated"] = False
    else:
        gate("memory_envelope", bool(memory.get("passed")),
             value={
                 "min_reclaimable_gib": memory.get("min_reclaimable_gib"),
                 "sustained_pressure": memory.get("sustained_pressure"),
                 "swap_growth_bytes": memory.get("swap_growth_bytes"),
             },
             detail=(f"min reclaimable {memory.get('min_reclaimable_gib', 0.0):.2f} GiB, "
                     f"swap growth {memory.get('swap_growth_bytes', 0)} B"))
    return gates


def _scenario_by_name(arm: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for scenario in arm.get("scenarios", []):
        if scenario["name"] == name:
            return scenario
    return None


def _peix_measurements(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Recorded measurements (never acceptance gates in this mode)."""
    measurements: Dict[str, Any] = {}
    overlap = arms.get("overlap") or {}
    serial = arms.get("serial") or {}

    overlap_burst = _scenario_by_name(overlap, "burst")
    serial_burst = _scenario_by_name(serial, "burst")
    if overlap_burst and serial_burst:
        o_rate = overlap_burst.get("aggregate_tokens_per_second")
        s_rate = serial_burst.get("aggregate_tokens_per_second")
        if o_rate and s_rate:
            measurements["burst_throughput"] = {
                "overlap_tokens_per_second": o_rate,
                "serial_tokens_per_second": s_rate,
                "ratio_overlap_over_serial": o_rate / s_rate,
                "note": ("measurement only: the legacy >= 1.20x aggregate-throughput gate is "
                         "not applied in peix-like mode (plan stage 7)"),
            }
        o_make = overlap_burst.get("makespan_seconds")
        s_make = serial_burst.get("makespan_seconds")
        if o_make and s_make:
            measurements["burst_makespan"] = {
                "overlap_seconds": o_make,
                "serial_seconds": s_make,
                "serial_over_overlap_ratio": s_make / o_make,
            }

    # Interactive TTFT per arm (absolute ceiling is gated on the overlap arm;
    # the serial baseline is reported for context — no ratio gate).
    ttft_summary: Dict[str, Any] = {}
    for arm_name, arm in (("overlap", overlap), ("serial", serial)):
        samples = list(arm.get("interactive_ttft_gated_seconds") or [])
        if samples:
            ttft_summary[arm_name] = {
                "p95_seconds": percentile(samples, 95),
                "max_seconds": max(samples),
                "samples": len(samples),
                "ttfts_seconds": samples,
            }
    if ttft_summary:
        measurements["interactive_ttft"] = ttft_summary

    # Per-task side-by-side (same task set in both arms).
    rows: List[Dict[str, Any]] = []
    for scenario in overlap.get("scenarios", []):
        serial_scenario = _scenario_by_name(serial, scenario["name"])
        for task in scenario["tasks"]:
            counterpart = next(
                (t for t in (serial_scenario or {}).get("tasks", [])
                 if t["task_id"] == task["task_id"]), None)
            row = {
                "scenario": scenario["name"],
                "task_id": task["task_id"],
                "kind": task.get("kind"),
                "priority": task.get("priority"),
                "overlap": {
                    "ttft_seconds": task.get("ttft_seconds"),
                    "total_seconds": task.get("total_seconds"),
                    "queue_wait_ms": task.get("queue_wait_ms"),
                    "queue_wait_ms_source": task.get("queue_wait_ms_source"),
                    "queue_wait_ms_exact": task.get("queue_wait_ms_exact"),
                    "queue_wait_ms_derived": task.get("queue_wait_ms_derived"),
                    "queue_wait_ms_server": task.get("queue_wait_ms_server"),
                    "completion_tokens": task.get("completion_tokens"),
                    "finish_reason": task.get("finish_reason"),
                },
                "serial": None,
                "fingerprints_equal": None,
            }
            if counterpart is not None:
                row["serial"] = {
                    "ttft_seconds": counterpart.get("ttft_seconds"),
                    "total_seconds": counterpart.get("total_seconds"),
                    "queue_wait_ms": counterpart.get("queue_wait_ms"),
                    "queue_wait_ms_source": counterpart.get("queue_wait_ms_source"),
                    "queue_wait_ms_exact": counterpart.get("queue_wait_ms_exact"),
                    "queue_wait_ms_derived": counterpart.get("queue_wait_ms_derived"),
                    "queue_wait_ms_server": counterpart.get("queue_wait_ms_server"),
                    "completion_tokens": counterpart.get("completion_tokens"),
                    "finish_reason": counterpart.get("finish_reason"),
                }
                if (task.get("fingerprint") is not None
                        and counterpart.get("fingerprint") is not None
                        and not task.get("expected_cancelled")):
                    row["fingerprints_equal"] = task["fingerprint"] == counterpart["fingerprint"]
            rows.append(row)
    measurements["per_task"] = rows
    return measurements


def peix_verdict(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate both arms and return the mode verdict + measurements."""
    gates_by_arm = {name: evaluate_arm_gates(arm) for name, arm in arms.items()}
    failed: List[str] = []
    for arm_name, gates in gates_by_arm.items():
        for gate_name, gate in gates.items():
            if gate["acceptance"] and not gate["passed"]:
                failed.append(f"{arm_name}/{gate_name}")
    return {
        "passed": not failed,
        "failed": sorted(failed),
        "gates": gates_by_arm,
        "measurements": _peix_measurements(arms),
        "notes": [
            "measurement-only: the legacy >= 1.20x aggregate-throughput and <= 1.20x-alone "
            "TTFT-ratio gates are not acceptance gates in peix-like mode (Kiem plan 820814e1 "
            "stage 7); they are reported for context.",
            "memory/pressure/swap are sampled at arm and scenario level (one shared model); "
            "TTFT/completion/queue-wait are per task.",
            "queue_wait_ms_derived is an ESTIMATE from observed slot-release transitions "
            "(+/- one watcher poll interval; the client-send -> server-enqueue latency is not "
            "observable and a release transition can be missed under load); "
            "queue_wait_ms_server is the engine's exact request-log value when the line-to-task "
            "match is unique. queue_wait_ms carries the authoritative value (server when exact, "
            "else the labeled estimate) and each scenario's queue_wait_audit reports any "
            "disagreement between the two sources.",
            "this mode validates Mei-facing request behavior for Peix-like workloads; it is "
            "not Peix end-to-end verification (the Peix harness does not exist yet).",
        ],
    }


# ---------------------------------------------------------------------------
# Stub Mei server (offline self-check only)
# ---------------------------------------------------------------------------

_LOOPBACK = "127.0.0.1"


def _chunk(payload: bytes) -> bytes:
    return f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n"


def _sleep(seconds: float) -> None:
    """A sleep that keeps its promise where the host distorts `time.sleep`.

    Some macOS hosts coalesce short sleeps badly (a 20 ms sleep can take
    150 ms), which would make the stub's scheduling windows unpredictable; a
    short spin keeps the stub's timings deterministic. Offline self-check
    only — the client's own sleeps are deliberately left untouched.
    """
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.002))


def _peer_closed(conn: socket.socket) -> bool:
    """Best-effort check whether the client closed its end of the stream.

    Uses a zero-timeout `select` first: a socket with a timeout set (as the
    stub's connections have) blocks inside `recv` up to its timeout even with
    MSG_DONTWAIT, so readability must be checked separately.
    """
    try:
        readable, _, _ = select.select([conn], [], [], 0.0)
        if not readable:
            return False
        return conn.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class _StubWaiter:
    __slots__ = ("priority", "enqueued_at", "admitted", "admitted_at", "event", "record", "last_send_at")

    def __init__(self, priority: str) -> None:
        self.priority = priority
        self.enqueued_at = time.monotonic()
        self.admitted = False
        self.admitted_at: Optional[float] = None
        self.event = threading.Event()
        self.record: Dict[str, Any] = {"priority": priority}
        self.last_send_at: Optional[float] = None


class StubMeiServer:
    """A tiny in-process Mei stand-in for offline self-check only.

    Emulates the scheduling surface the peix-like client relies on — bounded
    slots, a bounded interactive-first/FIFO queue, slot-release counting,
    cancellation on client disconnect, and `/v1/mei/status` — and answers
    the fixture prompts (nonce echo, counting, the add_numbers tool call,
    structured JSON) so the client's scenario machinery, mapping checks,
    watcher attribution, and gate evaluation run end to end without a model
    or a real server process. It is not a Mei implementation.
    """

    def __init__(
        self,
        max_active: int = 2,
        max_queue_depth: int = 8,
        prefill_delay: float = 0.2,
        per_token_delay: float = 0.02,
        count_cap: int = 50,
        min_stream_seconds: float = 0.5,
        model_id: str = "stub-model",
    ) -> None:
        if max_active not in (1, 2):
            raise ValueError("stub max_active must be 1 or 2")
        self.max_active = max_active
        self.max_queue_depth = max_queue_depth
        self.prefill_delay = prefill_delay
        self.per_token_delay = per_token_delay
        self.count_cap = count_cap
        self.min_stream_seconds = min_stream_seconds
        self.model_id = model_id
        self.port: Optional[int] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._active = 0
        self._waiters: List[_StubWaiter] = []
        self._peak = 0
        self._counters: Dict[str, Any] = {
            "admitted": 0, "completed": 0, "failed": 0,
            "cancelledQueued": 0, "cancelledActive": 0,
            "rejectedQueueFull": 0, "rejectedTimeout": 0, "slotReleases": 0,
            "lastQueueWaitMs": 0.0, "maxQueueWaitMs": 0.0,
            "lastCancelToReleaseMs": None, "maxCancelToReleaseMs": None,
        }
        self._records: List[Dict[str, Any]] = []
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._closing = False

    # -- lifecycle ----------------------------------------------------------

    def start(self, port: int = 0) -> int:
        """Bind the stub (port 0 = ephemeral) and start serving; returns the port."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((_LOOPBACK, port))
        listener.listen(64)
        listener.settimeout(0.5)
        self._listener = listener
        bound = int(listener.getsockname()[1])
        self.port = bound
        self._accept_thread = threading.Thread(target=self._accept_loop, name="stub-mei-accept", daemon=True)
        self._accept_thread.start()
        return bound

    def stop(self) -> None:
        self._closing = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3)

    # -- state --------------------------------------------------------------

    def scheduler_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._active > 0,
                "activeGenerations": self._active,
                "maxActiveGenerations": self.max_active,
                "peakActiveGenerations": self._peak,
                "queueDepth": len(self._waiters),
                "queuedInteractive": sum(1 for w in self._waiters if w.priority == INTERACTIVE),
                "queuedBackground": sum(1 for w in self._waiters if w.priority == BACKGROUND),
                "maxQueueDepth": self.max_queue_depth,
                "queueWaitTimeoutMs": 60000,
                "admitted": self._counters["admitted"],
                "completed": self._counters["completed"],
                "failed": self._counters["failed"],
                "cancelledQueued": self._counters["cancelledQueued"],
                "cancelledActive": self._counters["cancelledActive"],
                "rejectedQueueFull": self._counters["rejectedQueueFull"],
                "rejectedTimeout": self._counters["rejectedTimeout"],
                "slotReleases": self._counters["slotReleases"],
                "lastQueueWaitMs": self._counters["lastQueueWaitMs"],
                "maxQueueWaitMs": self._counters["maxQueueWaitMs"],
                "lastCancelToReleaseMs": self._counters["lastCancelToReleaseMs"],
                "maxCancelToReleaseMs": self._counters["maxCancelToReleaseMs"],
            }

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]

    # -- admission ----------------------------------------------------------

    def _pump_locked(self) -> None:
        while self._active < self.max_active and self._waiters:
            best = min(
                range(len(self._waiters)),
                key=lambda i: (0 if self._waiters[i].priority == INTERACTIVE else 1, i),
            )
            waiter = self._waiters.pop(best)
            waiter.admitted = True
            waiter.admitted_at = time.monotonic()
            self._active += 1
            self._peak = max(self._peak, self._active)
            wait_ms = (waiter.admitted_at - waiter.enqueued_at) * 1000.0
            self._counters["admitted"] += 1
            self._counters["lastQueueWaitMs"] = wait_ms
            self._counters["maxQueueWaitMs"] = max(self._counters["maxQueueWaitMs"], wait_ms)
            waiter.record["queue_wait_ms"] = wait_ms
            waiter.event.set()

    def _release(self, waiter: _StubWaiter, cancelled: bool) -> None:
        with self._cond:
            self._active = max(0, self._active - 1)
            self._counters["slotReleases"] += 1
            if cancelled:
                self._counters["cancelledActive"] += 1
                if waiter.last_send_at is not None:
                    ms = (time.monotonic() - waiter.last_send_at) * 1000.0
                    self._counters["lastCancelToReleaseMs"] = ms
                    current = self._counters["maxCancelToReleaseMs"]
                    self._counters["maxCancelToReleaseMs"] = ms if current is None else max(current, ms)
                waiter.record["finish"] = "cancelled"
            else:
                self._counters["completed"] += 1
                waiter.record.setdefault("finish", "stop")
            self._records.append(dict(waiter.record))
            self._pump_locked()
            self._cond.notify_all()

    # -- serving ------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._closing:
            try:
                conn, _ = self._listener.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(120.0)
            head = b""
            while b"\r\n\r\n" not in head:
                piece = conn.recv(4096)
                if not piece:
                    return
                head += piece
            head, _, rest = head.partition(b"\r\n\r\n")
            lines = head.split(b"\r\n")
            request_line = lines[0].split(b" ")
            method = request_line[0].decode("ascii", "replace")
            path = request_line[1].decode("ascii", "replace") if len(request_line) > 1 else "/"
            content_length = 0
            for line in lines[1:]:
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1])
            body = rest
            while len(body) < content_length:
                piece = conn.recv(4096)
                if not piece:
                    return
                body += piece
            if method == "GET" and path == "/healthz":
                self._send_json(conn, 200, {"status": "ok"})
            elif method == "GET" and path == "/v1/models":
                self._send_json(conn, 200, {"object": "list", "data": [{"id": self.model_id, "object": "model"}]})
            elif method == "GET" and path == "/v1/mei/status":
                self._send_json(conn, 200, {
                    "status": "ok", "model": self.model_id, "scheduler": self.scheduler_snapshot(),
                    "memory": {},
                })
            elif method == "POST" and path == "/v1/chat/completions":
                self._serve_chat(conn, body)
            else:
                self._send_json(conn, 404, {"error": {"message": f"no route {path}", "type": "invalid_request_error"}})
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _send_json(self, conn: socket.socket, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 429: "Too Many Requests",
                  503: "Service Unavailable"}.get(status, "OK")
        conn.sendall(
            f"HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n"
            f"content-length: {len(body)}\r\nconnection: close\r\n\r\n".encode("ascii") + body)

    def _serve_chat(self, conn: socket.socket, body: bytes) -> None:
        try:
            request = json.loads(body.decode("utf-8"))
        except ValueError:
            self._send_json(conn, 400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            return
        priority = request.get("mei_priority", INTERACTIVE)
        if priority is None:
            priority = INTERACTIVE
        if priority not in VALID_PRIORITIES:
            self._send_json(conn, 400, {"error": {
                "message": f"invalid value for 'mei_priority': {priority!r}; expected \"interactive\" or \"background\"",
                "type": "invalid_request_error", "code": "invalid_priority", "param": "mei_priority"}})
            return
        waiter = _StubWaiter(priority)
        with self._cond:
            if len(self._waiters) >= self.max_queue_depth:
                self._counters["rejectedQueueFull"] += 1
                rejected = True
            else:
                rejected = False
                self._waiters.append(waiter)
                self._pump_locked()
                self._cond.notify_all()
        if rejected:
            self._send_json(conn, 429, {"error": {
                "message": f"admission queue is full ({self.max_queue_depth} waiting)",
                "type": "rate_limit_error", "code": "queue_full"}})
            return
        if not waiter.event.wait(timeout=60.0):
            with self._cond:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._counters["rejectedTimeout"] += 1
            self._send_json(conn, 503, {"error": {
                "message": "request waited for the inference slot without being admitted",
                "type": "server_error", "code": "queue_timeout"}})
            return
        try:
            if request.get("stream"):
                self._serve_stream(conn, request, waiter)
            else:
                self._serve_buffered(conn, request, waiter)
        except OSError:
            self._release(waiter, cancelled=True)

    def _frames_for(self, request: Dict[str, Any]) -> Tuple[List[bytes], str, int]:
        messages = request.get("messages") or []
        text = " ".join(
            str(m.get("content") or "") for m in messages if isinstance(m, dict))
        if request.get("tools"):
            fragments = [
                b'{"index":0,"id":"call_stub_1","type":"function","function":{"name":"add_numbers","arguments":""}}',
                b'{"index":0,"function":{"arguments":"{\\"a\\":"}}',
                b'{"index":0,"function":{"arguments":"17"}}',
                b'{"index":0,"function":{"arguments":",\\"b\\":"}}',
                b'{"index":0,"function":{"arguments":"25}"}}',
            ]
            frames = [
                b'{"choices":[{"delta":{"tool_calls":[' + fragment + b']}}]}' for fragment in fragments
            ]
            return frames, "tool_calls", len(fragments)
        if request.get("response_format"):
            payload = json.dumps(EXPECTED_STRUCTURED).encode("utf-8")
            return [b'{"choices":[{"delta":{"content":' + json.dumps(payload.decode()).encode() + b'}}]}'], "stop", 7
        match = re.search(r"exactly this token and nothing else:\s*(\S+)", text)
        if match:
            nonce = match.group(1)
            return [b'{"choices":[{"delta":{"content":"' + nonce.encode() + b'"}}]}'], "stop", 2
        if "Count upward" in text:
            count = min(int(request.get("max_tokens") or self.count_cap), self.count_cap)
            count = max(1, count)
            frames = [
                b'{"choices":[{"delta":{"content":"%d\\n"}}]}' % number for number in range(1, count + 1)
            ]
            return frames, "stop", count
        return [b'{"choices":[{"delta":{"content":"ok"}}]}'], "stop", 1

    def _serve_stream(self, conn: socket.socket, request: Dict[str, Any], waiter: _StubWaiter) -> None:
        frames, finish_reason, completion_tokens = self._frames_for(request)
        prompt_tokens = 16 + sum(len(str(m.get("content") or "").split()) for m in request.get("messages") or [])
        started = time.monotonic()
        _sleep(self.prefill_delay)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"connection: keep-alive\r\ntransfer-encoding: chunked\r\n\r\n")
        for frame in frames:
            _sleep(self.per_token_delay)
            if _peer_closed(conn):
                self._release(waiter, cancelled=True)
                return
            waiter.last_send_at = time.monotonic()
            conn.sendall(_chunk(b"data: " + frame + b"\n\n"))
        # Keep the stream open for at least `min_stream_seconds` (padding with
        # contentless deltas) so the client's scenario orchestration has a
        # stable window; the client's TTFT is unaffected (no content).
        while time.monotonic() - started < self.min_stream_seconds:
            _sleep(self.per_token_delay)
            if _peer_closed(conn):
                self._release(waiter, cancelled=True)
                return
            waiter.last_send_at = time.monotonic()
            conn.sendall(_chunk(b'data: {"choices":[{"delta":{}}]}\n\n'))
        conn.sendall(_chunk(
            b'data: {"choices":[{"delta":{},"finish_reason":"' + finish_reason.encode() + b'"}]}\n\n'))
        conn.sendall(_chunk(
            b'data: {"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}\n\n'
            % (prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)))
        conn.sendall(_chunk(b"data: [DONE]\n\n"))
        conn.sendall(b"0\r\n\r\n")
        waiter.record["finish"] = finish_reason
        waiter.record["completion_tokens"] = completion_tokens
        self._release(waiter, cancelled=False)

    def _serve_buffered(self, conn: socket.socket, request: Dict[str, Any], waiter: _StubWaiter) -> None:
        if request.get("response_format"):
            content = json.dumps(EXPECTED_STRUCTURED)
            completion_tokens = 7
        else:
            match = re.search(
                r"exactly this token and nothing else:\s*(\S+)",
                " ".join(str(m.get("content") or "") for m in request.get("messages") or [] if isinstance(m, dict)))
            content = match.group(1) if match else "ok"
            completion_tokens = 2
        _sleep(self.prefill_delay)
        prompt_tokens = 16 + sum(len(str(m.get("content") or "").split()) for m in request.get("messages") or [])
        body = json.dumps({
            "id": "chatcmpl-stub", "object": "chat.completion", "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }).encode("utf-8")
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
            b"content-length: %d\r\nconnection: close\r\n\r\n" % len(body) + body)
        waiter.record["finish"] = "stop"
        waiter.record["completion_tokens"] = completion_tokens
        self._release(waiter, cancelled=False)
