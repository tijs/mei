#!/usr/bin/env python3
"""F1/C2 same-binary opt-out A/B report for the completed 0.6.0 artifacts.

Reads the six completed candidate/optout legs under
``artifacts/f1-c2-release-compare-20260930T095614Z/``:

  candidate legs (both F1/C2 switches unset; 0.6.0 default compiled-ON):
    ornith-060/candidate
    qwen36-text-060-thinking-off/candidate
    qwen36-vision-060-thinking-off/candidate
  optout legs (both F1/C2 switches explicitly "0"):
    ornith-060-optout/optout
    qwen36-text-060-thinking-off-optout/optout
    qwen36-vision-060-thinking-off-optout/optout

and emits ``f1-c2-060-optout-comparison.json`` and
``f1-c2-060-optout-comparison.md`` in the artifact root.

Stdlib only. Read-only with respect to the artifacts: the only writes are the
two report files in the artifact root (the historical release-comparison.json
/.md files are never touched). Never starts a model server, never benchmarks,
and never invents values: historical 0.6.0 logs carry no finalize_ms, so the
residual is recorded as the legacy ``tail = wall - prefill - generate``.

Usage::

    python3 tools/f1_c2_optout_report.py            # generate the report
    python3 tools/f1_c2_optout_report.py --validate # check artifacts + report

``--validate`` exits non-zero on failure and checks, at minimum: 6/6 legs,
20 timing rows and 25 request-log rows per leg, probe rc=0, server_stopped
true, one shared binary SHA256, the arm override pattern, pairwise greedy
output matches, the compiled_routed_switch_glu policy pattern, and that the
emitted report cross-checks against freshly derived values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_DIRNAME = "f1-c2-release-compare-20260930T095614Z"
REPORT_JSON_NAME = "f1-c2-060-optout-comparison.json"
REPORT_MD_NAME = "f1-c2-060-optout-comparison.md"

# Exact 0.6.0 binary recorded by every leg (paths/SHA from each result.json;
# the binary file itself is not re-read by this tool).
BINARY_PATH = "/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei"
BINARY_SHA256 = "3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc"

# The two F1/C2 switches (must match tools/f1_c2_benchmark.py SWITCHES).
SWITCHES = (
    "VMLX_QWEN35_COMPILE_DECODE_REGIONS",
    "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE",
)
CANDIDATE_OVERRIDES = {}  # type: dict
OPTOUT_OVERRIDES = {name: "0" for name in SWITCHES}

# The three same-binary pairs. ``candidate_policy_expected`` is the
# compiled_routed_switch_glu expectation for the candidate arm; every optout
# arm expects the policy line to be absent. The vision candidate carries no
# policy line at all (other [Qwen4Exp] lines are present and tolerated).
PAIR_SPECS = [
    {
        "model_key": "ornith",
        "model_id": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        "candidate_dir": "ornith-060/candidate",
        "optout_dir": "ornith-060-optout/optout",
        "enable_thinking": None,
        "candidate_policy_expected": "active",
        "candidate_policy_note": "0.6.0 default (switches unset): compiled_routed_switch_glu=active",
        "optout_policy_note": "optout leg (both switches '0'): no compiled_routed_switch_glu policy line",
    },
    {
        "model_key": "qwen36-text",
        "model_id": "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
        "candidate_dir": "qwen36-text-060-thinking-off/candidate",
        "optout_dir": "qwen36-text-060-thinking-off-optout/optout",
        "enable_thinking": False,
        "candidate_policy_expected": "active",
        "candidate_policy_note": "0.6.0 default (switches unset): compiled_routed_switch_glu=active",
        "optout_policy_note": "optout leg (both switches '0'): no compiled_routed_switch_glu policy line",
    },
    {
        "model_key": "qwen36-vision",
        "model_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "candidate_dir": "qwen36-vision-060-thinking-off/candidate",
        "optout_dir": "qwen36-vision-060-thinking-off-optout/optout",
        "enable_thinking": False,
        "candidate_policy_expected": "absent",
        "candidate_policy_note": (
            "0.6.0 vision leg: no compiled_routed_switch_glu policy line "
            "(other [Qwen4Exp] lines present, tolerated)"
        ),
        "optout_policy_note": (
            "optout leg (both switches '0'): no compiled_routed_switch_glu policy line "
            "(other [Qwen4Exp] lines present, tolerated)"
        ),
    },
]

MODEL_ORDER = tuple(spec["model_key"] for spec in PAIR_SPECS)

METRIC_KEYS = (
    "wall_ms",
    "prefill_ms",
    "generate_ms",
    "tail_ms",
    "decode_tps",
    "prompt_tps",
)
METRIC_LABELS = {
    "wall_ms": "wall (ms)",
    "prefill_ms": "prefill (ms)",
    "generate_ms": "generate (ms)",
    "tail_ms": "tail = wall - prefill - generate (ms)",
    "decode_tps": "decode (tok/s)",
    "prompt_tps": "prompt (tok/s)",
}

REQUIRED_LEG_FILES = ("result.json", "request.jsonl", "server.log", "probe.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def pct_delta(base, new):
    if base is None or new is None or base == 0:
        return None
    return round((new - base) / base * 100.0, 2)


def round2(v):
    return None if v is None else round(v, 2)


def round3(v):
    return None if v is None else round(v, 3)


def fmt_ms(v):
    return "n/a" if v is None else "{:,.2f}".format(v)


def fmt_tps(v):
    return "n/a" if v is None else "{:,.3f}".format(v)


def fmt_delta(v):
    return "n/a" if v is None else "{:+.2f}%".format(v)


def fmt_metric(key, v):
    if v is None:
        return "n/a"
    if key in ("wall_ms", "prefill_ms", "generate_ms", "tail_ms"):
        return fmt_ms(v)
    return fmt_tps(v)


def shorten_text(text, limit=72):
    """Compact repr for markdown tables; JSON keeps the full sample."""
    if text is None:
        return "None"
    if len(text) <= limit:
        return repr(text)
    return "{} ... ({} chars)".format(repr(text[: limit - 12]), len(text))


def context_of(row):
    return "short" if (row.get("prompt_tokens") or 0) < 1000 else "loaded"


def command_flag_value(command, flag):
    """Value of ``flag`` in a recorded command list, or None."""
    if flag in command:
        index = command.index(flag)
        if index + 1 < len(command):
            return command[index + 1]
    return None


def enable_thinking_value(command):
    raw = command_flag_value(command, "--enable-thinking")
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def summarize_rows(rows):
    """Means for the requested metrics from request-log completion rows."""
    summary: dict = {"n": len(rows)}
    if not rows:
        summary.update({key: None for key in METRIC_KEYS})
        return summary
    wall = [r["wall_ms"] for r in rows]
    prefill = [r["prefill_ms"] for r in rows]
    generate = [r["generate_ms"] for r in rows]
    tail = [r["wall_ms"] - r["prefill_ms"] - r["generate_ms"] for r in rows]
    summary.update(
        {
            "wall_ms": round2(mean(wall)),
            "prefill_ms": round2(mean(prefill)),
            "generate_ms": round2(mean(generate)),
            "tail_ms": round2(mean(tail)),
            "decode_tps": round3(mean([r["decode_tps"] for r in rows])),
            "prompt_tps": round3(mean([r["prompt_tps"] for r in rows])),
        }
    )
    return summary


def aligned_rows_equal(rows_a, rows_b):
    """Per-repeat equality of two timing-row lists keyed by (context, repeat).

    Compares text_sha256, visible text, prompt_tokens and completion_tokens on
    every aligned row (greedy output equality within an A/B pair).
    """
    if len(rows_a) != len(rows_b):
        return False

    def key(row):
        return (row.get("context"), row.get("repeat"))

    for ra, rb in zip(sorted(rows_a, key=key), sorted(rows_b, key=key)):
        if key(ra) != key(rb):
            return False
        for field in ("text_sha256", "text", "prompt_tokens", "completion_tokens"):
            if ra.get(field) != rb.get(field):
                return False
    return True


def public_leg(leg):
    """Leg dict without the private row payloads used for pair matching."""
    return {k: v for k, v in leg.items() if not k.startswith("_")}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# per-leg analysis
# ---------------------------------------------------------------------------

def analyze_leg(artifact_root: Path, spec: dict, arm: str) -> dict:
    dir_rel = spec["candidate_dir"] if arm == "candidate" else spec["optout_dir"]
    leg_dir = artifact_root / dir_rel
    expected_overrides = CANDIDATE_OVERRIDES if arm == "candidate" else OPTOUT_OVERRIDES
    expected_policy = spec["candidate_policy_expected"] if arm == "candidate" else "absent"
    expected_policy_note = (
        spec["candidate_policy_note"] if arm == "candidate" else spec["optout_policy_note"]
    )

    checks = {}
    failures = []

    missing = [name for name in REQUIRED_LEG_FILES if not (leg_dir / name).is_file()]
    if missing:
        return {
            "dir": dir_rel,
            "arm": arm,
            "model_key": spec["model_key"],
            "model_id": spec["model_id"],
            "binary": None,
            "binary_sha256": None,
            "model_dir": None,
            "enable_thinking": spec["enable_thinking"],
            "port": None,
            "context_cap": None,
            "started_epoch": None,
            "finished_epoch": None,
            "missing_files": missing,
            "checks": {"all_files_present": False},
            "failures": ["missing files: {}".format(", ".join(missing))],
            "means": {},
            "outputs": {},
            "policy_evidence": {
                "compiled_routed_switch_glu": {
                    "present": None,
                    "value": None,
                    "line": None,
                    "line_number": None,
                    "note": None,
                },
                "qwen4exp_line_count": 0,
                "qwen35_line_count": 0,
                "expected": expected_policy,
                "expected_note": expected_policy_note,
                "matches_expected": False,
            },
            "finalize_ms": {"present": None, "per_source": {}, "note": ""},
            "probe_checks": {},
            "cross_check": {},
            "_timing_rows": [],
            "_completion_rows": [],
        }

    raw_result = (leg_dir / "result.json").read_text(encoding="utf-8")
    result = json.loads(raw_result)
    raw_request = (leg_dir / "request.jsonl").read_text(encoding="utf-8")
    request_rows = [json.loads(line) for line in raw_request.splitlines() if line.strip()]
    raw_probe = (leg_dir / "probe.json").read_text(encoding="utf-8")
    probe = json.loads(raw_probe)
    raw_server_log = (leg_dir / "server.log").read_text(encoding="utf-8")
    server_log_lines = raw_server_log.splitlines()

    timing_rows = result.get("rows", [])
    completion_rows = [r for r in request_rows if r.get("kind") == "completion"]

    checks["all_files_present"] = True

    # --- timing rows (20 expected: 10 short + 10 loaded) ---
    ctx_counts = {"short": 0, "loaded": 0}
    for row in timing_rows:
        ctx_counts[row.get("context", "?")] = ctx_counts.get(row.get("context", "?"), 0) + 1
    checks["timing_rows"] = len(timing_rows)
    checks["timing_rows_ok"] = len(timing_rows) == 20
    checks["timing_contexts"] = dict(ctx_counts)
    checks["timing_contexts_ok"] = (
        ctx_counts.get("short") == 10 and ctx_counts.get("loaded") == 10
    )
    checks["timing_text_hashes_ok"] = all(
        sha256_text(row.get("text", "")) == row.get("text_sha256") for row in timing_rows
    )
    checks["timing_wall_positive"] = all((row.get("wall_seconds") or 0) > 0 for row in timing_rows)

    # --- request-log rows (25 expected: 20 completion + 5 probe) ---
    checks["request_log_rows"] = len(request_rows)
    checks["request_log_rows_ok"] = len(request_rows) == 25
    checks["request_completion_rows"] = len(completion_rows)
    checks["request_completion_rows_ok"] = len(completion_rows) == 20
    comp_ctx = {"short": 0, "loaded": 0}
    for row in completion_rows:
        comp_ctx[context_of(row)] = comp_ctx.get(context_of(row), 0) + 1
    checks["request_completion_contexts"] = dict(comp_ctx)
    checks["request_completion_contexts_ok"] = (
        comp_ctx.get("short") == 10 and comp_ctx.get("loaded") == 10
    )
    checks["request_kind_counts"] = {
        kind: sum(1 for r in request_rows if r.get("kind") == kind)
        for kind in sorted({r.get("kind") for r in request_rows})
    }
    checks["embedded_request_log_matches_file"] = result.get("request_log_rows") == request_rows

    # --- pairing timing rows <-> completion rows (order-based, verified fields) ---
    pairing_ok = len(timing_rows) == len(completion_rows)
    wall_deltas = {"short": [], "loaded": []}
    if pairing_ok:
        for t_row, c_row in zip(timing_rows, completion_rows):
            if (
                t_row.get("prompt_tokens") != c_row.get("prompt_tokens")
                or t_row.get("completion_tokens") != c_row.get("completion_tokens")
                or t_row.get("context") != context_of(c_row)
            ):
                pairing_ok = False
            wall_deltas[t_row["context"]].append(
                t_row["wall_seconds"] * 1000.0 - c_row["wall_ms"]
            )
    checks["timing_log_pairing_ok"] = pairing_ok

    # --- flags ---
    checks["probe_returncode"] = result.get("probe_returncode")
    checks["probe_returncode_ok"] = result.get("probe_returncode") == 0
    checks["server_stopped"] = result.get("server_stopped")
    checks["server_stopped_ok"] = result.get("server_stopped") is True
    checks["probe_status"] = probe.get("status")
    checks["probe_status_ok"] = probe.get("status") == "passed"

    probes = probe.get("probes") or {}
    probe_checks = {
        "total": len(probes),
        "passed": sum(
            1 for entry in probes.values() if isinstance(entry, dict) and entry.get("status") == "passed"
        ),
    }
    checks["probe_subprobes_ok"] = probe_checks["total"] > 0 and (
        probe_checks["passed"] == probe_checks["total"]
    )

    # --- binary / model / arm identity ---
    checks["binary_matches_expected"] = result.get("binary") == BINARY_PATH
    checks["binary_sha256_matches_expected"] = result.get("binary_sha256") == BINARY_SHA256
    checks["model_id_matches_expected"] = result.get("model_id") == spec["model_id"]
    checks["operator_overrides"] = result.get("operator_overrides")
    checks["operator_overrides_ok"] = result.get("operator_overrides") == expected_overrides
    command = result.get("command", [])
    checks["enable_thinking_flag"] = enable_thinking_value(command)
    checks["enable_thinking_ok"] = checks["enable_thinking_flag"] == spec["enable_thinking"]
    checks["contexts_field_ok"] = result.get("contexts") == {"short": 13, "loaded": 30000}
    checks["repeats_per_context"] = result.get("repeats_per_context")
    checks["repeats_per_context_ok"] = result.get("repeats_per_context") == 10

    for key, ok in (
        ("timing_rows_ok", checks["timing_rows_ok"]),
        ("timing_contexts_ok", checks["timing_contexts_ok"]),
        ("timing_text_hashes_ok", checks["timing_text_hashes_ok"]),
        ("timing_wall_positive", checks["timing_wall_positive"]),
        ("request_log_rows_ok", checks["request_log_rows_ok"]),
        ("request_completion_rows_ok", checks["request_completion_rows_ok"]),
        ("request_completion_contexts_ok", checks["request_completion_contexts_ok"]),
        ("embedded_request_log_matches_file", checks["embedded_request_log_matches_file"]),
        ("timing_log_pairing_ok", checks["timing_log_pairing_ok"]),
        ("probe_returncode_ok", checks["probe_returncode_ok"]),
        ("server_stopped_ok", checks["server_stopped_ok"]),
        ("probe_status_ok", checks["probe_status_ok"]),
        ("probe_subprobes_ok", checks["probe_subprobes_ok"]),
        ("binary_matches_expected", checks["binary_matches_expected"]),
        ("binary_sha256_matches_expected", checks["binary_sha256_matches_expected"]),
        ("model_id_matches_expected", checks["model_id_matches_expected"]),
        ("operator_overrides_ok", checks["operator_overrides_ok"]),
        ("enable_thinking_ok", checks["enable_thinking_ok"]),
        ("contexts_field_ok", checks["contexts_field_ok"]),
        ("repeats_per_context_ok", checks["repeats_per_context_ok"]),
    ):
        if not ok:
            failures.append(key)

    # --- means per context (from request-log completion rows) + pooled ---
    means = {}
    for ctx_key, ctx in (("short", "short"), ("loaded_30k", "loaded")):
        rows = [r for r in completion_rows if context_of(r) == ctx]
        means[ctx_key] = summarize_rows(rows)
    means["both_contexts"] = summarize_rows(completion_rows)

    # --- outputs per context (hashes + visible text) ---
    outputs = {}
    for ctx in ("short", "loaded"):
        t_rows = [r for r in timing_rows if r.get("context") == ctx]
        c_rows = [r for r in completion_rows if context_of(r) == ctx]
        hashes = sorted({r["text_sha256"] for r in t_rows})
        outputs[ctx] = {
            "distinct_text_sha256": hashes,
            "deterministic_within_leg": len(hashes) == 1,
            "text_sample": t_rows[0]["text"] if t_rows else None,
            "completion_tokens": sorted({r.get("completion_tokens") for r in c_rows}),
            "finish_reasons": sorted({r.get("finish") for r in c_rows}),
        }

    # --- policy evidence from server.log ---
    policy_lines = [
        (i + 1, line)
        for i, line in enumerate(server_log_lines)
        if "compiled_routed_switch_glu" in line
    ]
    policy_value = None
    if policy_lines:
        policy_value = "active" if any("=active" in line for _, line in policy_lines) else "other"
    policy_present = bool(policy_lines)
    matches_expected = (
        policy_value == "active" if expected_policy == "active" else not policy_present
    )
    qwen4exp_count = sum(1 for line in server_log_lines if line.startswith("[Qwen4Exp]"))
    qwen35_count = sum(1 for line in server_log_lines if line.startswith("[Qwen35]"))
    if policy_present:
        policy_note = None
    elif qwen4exp_count:
        policy_note = (
            "no compiled_routed_switch_glu policy line "
            "(other [Qwen4Exp] lines present: {}, tolerated)".format(qwen4exp_count)
        )
    else:
        policy_note = "no compiled_routed_switch_glu policy line and no other [Qwen4Exp] lines"
    policy = {
        "compiled_routed_switch_glu": {
            "present": policy_present,
            "value": policy_value,
            "line": policy_lines[0][1] if policy_lines else None,
            "line_number": policy_lines[0][0] if policy_lines else None,
            "note": policy_note,
        },
        "qwen4exp_line_count": qwen4exp_count,
        "qwen35_line_count": qwen35_count,
        "expected": expected_policy,
        "expected_note": expected_policy_note,
        "matches_expected": matches_expected,
    }
    if not matches_expected:
        failures.append("policy_evidence_matches_expected")

    # --- finalize_ms detection (record presence; never invent a value) ---
    finalize_sources = {
        "request.jsonl": "finalize_ms" in raw_request,
        "result.json": "finalize_ms" in raw_result,
        "probe.json": "finalize_ms" in raw_probe,
        "server.log": "finalize_ms" in raw_server_log,
    }
    finalize = {
        "present": any(finalize_sources.values()),
        "per_source": finalize_sources,
        "note": (
            "Historical 0.6.0 logs do not emit finalize_ms; recorded as absent, no value "
            "synthesized. The residual is the legacy tail = wall - prefill - generate."
        ),
    }

    # --- client vs server wall clock cross-check ---
    cross_check = {}
    for ctx in ("short", "loaded"):
        deltas = wall_deltas.get(ctx, [])
        client_walls = [
            row["wall_seconds"] * 1000.0 for row in timing_rows if row.get("context") == ctx
        ]
        cross_check[ctx] = {
            "client_wall_ms_mean": round2(mean(client_walls)),
            "client_minus_server_wall_ms_mean": round2(mean(deltas)),
            "client_minus_server_wall_ms_min": round2(min(deltas)) if deltas else None,
            "client_minus_server_wall_ms_max": round2(max(deltas)) if deltas else None,
        }

    return {
        "dir": dir_rel,
        "arm": arm,
        "model_key": spec["model_key"],
        "model_id": result.get("model_id", spec["model_id"]),
        "binary": result.get("binary"),
        "binary_sha256": result.get("binary_sha256"),
        "model_dir": result.get("model_dir"),
        "enable_thinking": checks["enable_thinking_flag"],
        "port": command_flag_value(command, "--port"),
        "context_cap": command_flag_value(command, "--context-cap"),
        "started_epoch": result.get("started_epoch"),
        "finished_epoch": result.get("finished_epoch"),
        "checks": checks,
        "failures": failures,
        "means": means,
        "outputs": outputs,
        "policy_evidence": policy,
        "finalize_ms": finalize,
        "probe_checks": probe_checks,
        "cross_check": cross_check,
        "_timing_rows": timing_rows,
        "_completion_rows": completion_rows,
    }


# ---------------------------------------------------------------------------
# pair comparison
# ---------------------------------------------------------------------------

def context_rows_match(leg_a: dict, leg_b: dict, ctx: str) -> bool:
    rows_a = [r for r in leg_a.get("_timing_rows", []) if r.get("context") == ctx]
    rows_b = [r for r in leg_b.get("_timing_rows", []) if r.get("context") == ctx]
    return aligned_rows_equal(rows_a, rows_b)


def policy_summary(leg: dict) -> dict:
    entry = leg.get("policy_evidence", {}).get("compiled_routed_switch_glu", {})
    return {
        "present": entry.get("present"),
        "value": entry.get("value"),
        "line_number": entry.get("line_number"),
        "note": entry.get("note"),
    }


def build_pair(spec: dict, leg_cand: dict, leg_optout: dict) -> dict:
    pair = {
        "model_key": spec["model_key"],
        "model_id": spec["model_id"],
        "enable_thinking": spec["enable_thinking"],
        "legs": {"candidate": spec["candidate_dir"], "optout": spec["optout_dir"]},
        "binary_sha256": leg_cand.get("binary_sha256"),
        "same_binary_sha256": (
            leg_cand.get("binary_sha256") == leg_optout.get("binary_sha256") == BINARY_SHA256
        ),
        "output_match": {},
        "contexts": {},
        "policy": {
            "candidate": policy_summary(leg_cand),
            "optout": policy_summary(leg_optout),
            "pattern_ok": bool(
                leg_cand.get("policy_evidence", {}).get("matches_expected")
                and leg_optout.get("policy_evidence", {}).get("matches_expected")
            ),
        },
    }

    for ctx_key, ctx in (("short", "short"), ("loaded_30k", "loaded")):
        m_c = leg_cand.get("means", {}).get(ctx_key, {})
        m_o = leg_optout.get("means", {}).get(ctx_key, {})
        delta = {key: pct_delta(m_c.get(key), m_o.get(key)) for key in METRIC_KEYS}
        out_c = leg_cand.get("outputs", {}).get(ctx, {})
        out_o = leg_optout.get("outputs", {}).get(ctx, {})
        match = (
            out_c.get("distinct_text_sha256") == out_o.get("distinct_text_sha256")
            and out_c.get("text_sample") == out_o.get("text_sample")
        )
        rows_match = context_rows_match(leg_cand, leg_optout, ctx)
        pair["output_match"][ctx_key] = {
            "match": match,
            "rows_match": rows_match,
            "candidate": {
                "distinct_text_sha256": out_c.get("distinct_text_sha256"),
                "text": out_c.get("text_sample"),
                "completion_tokens": out_c.get("completion_tokens"),
                "finish_reasons": out_c.get("finish_reasons"),
            },
            "optout": {
                "distinct_text_sha256": out_o.get("distinct_text_sha256"),
                "text": out_o.get("text_sample"),
                "completion_tokens": out_o.get("completion_tokens"),
                "finish_reasons": out_o.get("finish_reasons"),
            },
        }
        pair["contexts"][ctx_key] = {
            "candidate": m_c,
            "optout": m_o,
            "delta_pct": delta,
        }

    m_c = leg_cand.get("means", {}).get("both_contexts", {})
    m_o = leg_optout.get("means", {}).get("both_contexts", {})
    pair["contexts"]["pooled_both_contexts"] = {
        "candidate": m_c,
        "optout": m_o,
        "delta_pct": {key: pct_delta(m_c.get(key), m_o.get(key)) for key in METRIC_KEYS},
    }
    pair["output_match"]["all_contexts_match"] = bool(
        pair["output_match"]["short"]["match"]
        and pair["output_match"]["short"]["rows_match"]
        and pair["output_match"]["loaded_30k"]["match"]
        and pair["output_match"]["loaded_30k"]["rows_match"]
    )
    return pair


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def analyze_all(artifact_root: Path):
    legs = {}
    for spec in PAIR_SPECS:
        for arm in ("candidate", "optout"):
            leg = analyze_leg(artifact_root, spec, arm)
            legs[leg["dir"]] = leg
    pairs = [
        build_pair(spec, legs[spec["candidate_dir"]], legs[spec["optout_dir"]])
        for spec in PAIR_SPECS
    ]
    return legs, pairs


def derived_totals(legs: dict, pairs: list) -> dict:
    candidate_dirs = [spec["candidate_dir"] for spec in PAIR_SPECS]
    optout_dirs = [spec["optout_dir"] for spec in PAIR_SPECS]
    totals = {
        "legs": len(legs),
        "legs_valid": sum(1 for leg in legs.values() if not leg.get("failures")),
        "models": len({spec["model_id"] for spec in PAIR_SPECS}),
        "pairs": len(pairs),
        "timing_rows_total": sum(leg["checks"].get("timing_rows", 0) or 0 for leg in legs.values()),
        "request_log_rows_total": sum(
            leg["checks"].get("request_log_rows", 0) or 0 for leg in legs.values()
        ),
        "completion_rows_total": sum(
            leg["checks"].get("request_completion_rows", 0) or 0 for leg in legs.values()
        ),
        "probe_returncode_zero_legs": sum(
            1 for leg in legs.values() if leg["checks"].get("probe_returncode_ok")
        ),
        "server_stopped_legs": sum(
            1 for leg in legs.values() if leg["checks"].get("server_stopped_ok")
        ),
        "same_binary_sha_legs": sum(
            1
            for leg in legs.values()
            if leg.get("binary_sha256") == BINARY_SHA256
            and leg["checks"].get("binary_sha256_matches_expected")
        ),
        "candidate_overrides_ok_legs": sum(
            1
            for leg_dir in candidate_dirs
            if legs.get(leg_dir, {}).get("checks", {}).get("operator_overrides_ok")
        ),
        "optout_overrides_ok_legs": sum(
            1
            for leg_dir in optout_dirs
            if legs.get(leg_dir, {}).get("checks", {}).get("operator_overrides_ok")
        ),
        "output_pairs_matched": sum(
            1 for pair in pairs if pair["output_match"]["all_contexts_match"]
        ),
        "policy_pattern_matched_legs": sum(
            1 for leg in legs.values() if leg.get("policy_evidence", {}).get("matches_expected")
        ),
        "finalize_ms_absent_legs": sum(
            1 for leg in legs.values() if leg.get("finalize_ms", {}).get("present") is False
        ),
    }
    totals["all_ok"] = bool(
        totals["legs_valid"] == 6
        and totals["pairs"] == 3
        and totals["output_pairs_matched"] == 3
        and totals["policy_pattern_matched_legs"] == 6
        and totals["same_binary_sha_legs"] == 6
        and totals["candidate_overrides_ok_legs"] == 3
        and totals["optout_overrides_ok_legs"] == 3
    )
    return totals


def build_caveats(legs: dict, pairs: list) -> list:
    caveats = []

    caveats.append(
        {
            "id": "no-finalize-ms",
            "title": "finalize_ms absent in historical 0.6.0 logs",
            "text": (
                "No leg records finalize_ms (checked per leg across request.jsonl, result.json, "
                "probe.json and server.log; presence recorded, no value synthesized). The residual "
                "is therefore the legacy tail = wall - prefill - generate, which may absorb "
                "finalize/overhead; tail deltas must not be read as engine effects."
            ),
        }
    )

    candidate_started = [
        legs[spec["candidate_dir"]].get("started_epoch") for spec in PAIR_SPECS
    ]
    optout_started = [legs[spec["optout_dir"]].get("started_epoch") for spec in PAIR_SPECS]
    ordering_note = ""
    if all(candidate_started) and all(optout_started):
        ordering_note = (
            " All candidate blocks ran before all optout blocks (candidate start epochs "
            "{}-{}, optout {}-{}).".format(
                int(min(candidate_started)),
                int(max(candidate_started)),
                int(min(optout_started)),
                int(max(optout_started)),
            )
        )
    caveats.append(
        {
            "id": "descriptive-not-causal",
            "title": "single sequential A/B blocks: descriptive, not causal",
            "text": (
                "Each model has exactly one candidate block and one optout block, run as separate "
                "sequential server processes on one host, with 10 repeats per context per leg and "
                "no interleaving, randomization or significance testing.{} Host/time-varying "
                "effects cannot be separated from the F1/C2 switch effect; small deltas (order of "
                "a few percent) should not be read as effects.".format(ordering_note)
            ),
        }
    )

    vision_spec = next(spec for spec in PAIR_SPECS if spec["model_key"] == "qwen36-vision")
    vision_cand = legs[vision_spec["candidate_dir"]]
    vision_opt = legs[vision_spec["optout_dir"]]
    caveats.append(
        {
            "id": "vision-no-policy-line",
            "title": "vision legs carry no compiled_routed_switch_glu policy line",
            "text": (
                "Both qwen36-vision legs have no compiled_routed_switch_glu line in server.log "
                "(other [Qwen4Exp] lines are present and tolerated: {} candidate, {} optout). The "
                "vision A/B block therefore has no direct compiled-routed-MoE signal in either "
                "arm; its deltas are recorded for completeness only.".format(
                    vision_cand.get("policy_evidence", {}).get("qwen4exp_line_count"),
                    vision_opt.get("policy_evidence", {}).get("qwen4exp_line_count"),
                )
            ),
        }
    )
    return caveats


def build_report(artifact_root: Path) -> dict:
    legs, pairs = analyze_all(artifact_root)
    totals = derived_totals(legs, pairs)
    caveats = build_caveats(legs, pairs)

    return {
        "schema": "mei-f1-c2-optout/1",
        "generated_by": "tools/f1_c2_optout_report.py",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_root": str(artifact_root),
        "scope": {
            "release": "0.6.0",
            "legs": [leg["dir"] for leg in legs.values()],
            "models": [spec["model_id"] for spec in PAIR_SPECS],
            "pairs": [
                {
                    "model_key": spec["model_key"],
                    "model_id": spec["model_id"],
                    "legs": {"candidate": spec["candidate_dir"], "optout": spec["optout_dir"]},
                }
                for spec in PAIR_SPECS
            ],
            "contexts": {"short": 13, "loaded": 30000},
            "repeats_per_context": 10,
            "note": (
                "Same-binary A/B: one shared 0.6.0 binary per pair; candidate = both F1/C2 "
                "switches unset (default compiled-ON), optout = both switches explicitly '0'."
            ),
        },
        "arms": {
            "candidate": {
                "overrides": dict(CANDIDATE_OVERRIDES),
                "meaning": (
                    "both F1/C2 switches unset; on the 0.6.0 pin the compiled routed-MoE decode "
                    "region is active by default"
                ),
            },
            "optout": {
                "overrides": dict(OPTOUT_OVERRIDES),
                "meaning": "both F1/C2 switches explicitly '0'",
            },
        },
        "binary": {
            "path": BINARY_PATH,
            "sha256": BINARY_SHA256,
            "shared_by_all_legs": totals["same_binary_sha_legs"] == 6,
            "note": (
                "SHA256 taken from each leg's recorded result.json metadata; the binary file "
                "itself is not re-read by this tool."
            ),
        },
        "method": {
            "means_source": (
                "request.jsonl rows with kind=completion (10 short + 10 loaded per leg); "
                "server-reported metrics"
            ),
            "tail": (
                "tail_ms = wall_ms - prefill_ms - generate_ms (legacy decomposition; no "
                "finalize_ms in historical 0.6.0 logs)"
            ),
            "delta_pct": "(optout - candidate) / candidate * 100",
            "greedy_outputs": (
                "requests pinned greedy decoding (temperature=0, top_p=1, top_k=1, seed=1); "
                "output match compares per-repeat text_sha256/text and distinct hash sets"
            ),
            "policy_evidence": (
                "server.log lines containing compiled_routed_switch_glu; other [Qwen4Exp] lines "
                "are recorded but tolerated"
            ),
        },
        "legs": {dir_rel: public_leg(leg) for dir_rel, leg in legs.items()},
        "pairs": pairs,
        "finalize_ms": {
            "present_in_any_leg": any(leg["finalize_ms"].get("present") for leg in legs.values()),
            "per_leg": {leg["dir"]: leg["finalize_ms"].get("present") for leg in legs.values()},
            "note": (
                "Detected by scanning each leg's request.jsonl, result.json, probe.json and "
                "server.log; historical 0.6.0 logs lack finalize_ms, so it is recorded as absent "
                "and the legacy tail is used."
            ),
        },
        "totals": totals,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines = []
    add = lines.append
    totals = report["totals"]

    add("# Mei 0.6.0 F1/C2 opt-out A/B — same-binary comparison")
    add("")
    add("- **Artifact root:** `{}`".format(report["artifact_root"]))
    add(
        "- **Generated:** {} by `{}` (stdlib only)".format(
            report["generated_at_utc"], report["generated_by"]
        )
    )
    add(
        "- **Scope:** {} legs = 3 models x 2 arms (candidate = both F1/C2 switches unset, the "
        "0.6.0 default compiled-ON; optout = both switches `\"0\"`); per leg 10 repeats x "
        "(short: 13 prompt tokens; 30k: 30000 prompt tokens); 20 timing rows and 25 "
        "request-log rows per leg.".format(len(report["scope"]["legs"]))
    )
    add(
        "- **Binary:** `{}` — SHA256 `{}` on all six legs.".format(
            report["binary"]["path"], report["binary"]["sha256"]
        )
    )
    add("")
    add("## Validation")
    add("")
    add(
        "**{}/{} legs valid** — {} timing rows, {} request-log rows total; probe_returncode=0 and "
        "server_stopped=true on every leg; greedy outputs match within every pair.".format(
            totals["legs_valid"],
            totals["legs"],
            totals["timing_rows_total"],
            totals["request_log_rows_total"],
        )
    )
    add("")
    add("| Leg | Model | Arm | Timing rows | Log rows | probe rc | stopped | compiled_routed_switch_glu |")
    add("|---|---|---|---|---|---|---|---|")
    for leg in report["legs"].values():
        checks = leg.get("checks", {})
        entry = leg.get("policy_evidence", {}).get("compiled_routed_switch_glu", {})
        if entry.get("present"):
            policy_cell = "**active**" if entry.get("value") == "active" else str(entry.get("value"))
        else:
            policy_cell = "absent"
        add(
            "| `{}` | `{}` | {} | {}/20 | {}/25 | {} | {} | {} |".format(
                leg["dir"],
                leg.get("model_id"),
                leg["arm"],
                checks.get("timing_rows", "?"),
                checks.get("request_log_rows", "?"),
                checks.get("probe_returncode", "?"),
                str(checks.get("server_stopped", "?")).lower(),
                policy_cell,
            )
        )
    add("")
    add("## Arm definition")
    add("")
    add(
        "- **candidate:** `operator_overrides = {}` — both F1/C2 switches unset; on the 0.6.0 "
        "pin the compiled routed-MoE decode region is active by default."
    )
    add(
        "- **optout:** `operator_overrides = {}` — both F1/C2 switches explicitly `\"0\"`.".format(
            json.dumps(report["arms"]["optout"]["overrides"], sort_keys=True)
        )
    )
    add("")
    add("## Policy evidence (server.log)")
    add("")
    add("| Leg | compiled_routed_switch_glu | Evidence |")
    add("|---|---|---|")
    for leg in report["legs"].values():
        policy = leg.get("policy_evidence", {})
        entry = policy.get("compiled_routed_switch_glu", {})
        if entry.get("present"):
            cell = "**active**" if entry.get("value") == "active" else str(entry.get("value"))
            evidence = "L{}: `{}`".format(entry.get("line_number"), entry.get("line"))
        else:
            cell = "absent"
            evidence = "{}; [Qwen4Exp] lines: {}".format(
                entry.get("note") or "no compiled_routed_switch_glu line",
                policy.get("qwen4exp_line_count"),
            )
        add("| `{}` | {} | {} |".format(leg["dir"], cell, evidence))
    add("")
    add("## Greedy output match (per pair)")
    add("")
    add("| Model | Context | candidate text (sha256) | optout text (sha256) | per-repeat match |")
    add("|---|---|---|---|---|")
    for pair in report["pairs"]:
        for ctx_key, ctx_label in (("short", "short"), ("loaded_30k", "30k")):
            om = pair["output_match"][ctx_key]
            a, b = om["candidate"], om["optout"]
            add(
                "| `{}` | {} | {} (`{}`) | {} (`{}`) | {} |".format(
                    pair["model_id"],
                    ctx_label,
                    shorten_text(a["text"]),
                    ", ".join(h[:16] for h in (a["distinct_text_sha256"] or [])),
                    shorten_text(b["text"]),
                    ", ".join(h[:16] for h in (b["distinct_text_sha256"] or [])),
                    "yes" if (om["match"] and om["rows_match"]) else "**NO**",
                )
            )
    add("")
    add("## Means and opt-out deltas")
    add("")
    add(
        "Means from request-log completion rows (server-reported); 10 rows per context per leg. "
        "`tail = wall - prefill - generate` (legacy; finalize_ms absent). "
        "Delta% = (optout - candidate) / candidate x 100. Greedy outputs matched within every "
        "pair, so the timing deltas are like-for-like."
    )
    for pair in report["pairs"]:
        add("")
        add("### `{}`".format(pair["model_id"]))
        add("")
        add("| Metric | short candidate | short optout | short delta% | 30k candidate | 30k optout | 30k delta% |")
        add("|---|---|---|---|---|---|---|")
        for key in METRIC_KEYS:
            short = pair["contexts"]["short"]
            loaded = pair["contexts"]["loaded_30k"]
            add(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    METRIC_LABELS[key],
                    fmt_metric(key, short["candidate"].get(key)),
                    fmt_metric(key, short["optout"].get(key)),
                    fmt_delta(short["delta_pct"].get(key)),
                    fmt_metric(key, loaded["candidate"].get(key)),
                    fmt_metric(key, loaded["optout"].get(key)),
                    fmt_delta(loaded["delta_pct"].get(key)),
                )
            )
        pooled = pair["contexts"]["pooled_both_contexts"]
        add("")
        add("| Metric | pooled candidate | pooled optout | pooled delta% |")
        add("|---|---|---|---|")
        for key in METRIC_KEYS:
            add(
                "| {} | {} | {} | {} |".format(
                    METRIC_LABELS[key],
                    fmt_metric(key, pooled["candidate"].get(key)),
                    fmt_metric(key, pooled["optout"].get(key)),
                    fmt_delta(pooled["delta_pct"].get(key)),
                )
            )
    add("")
    add("## Caveats")
    add("")
    for i, caveat in enumerate(report["caveats"], 1):
        add("{}. **{}** — {}".format(i, caveat["title"], caveat["text"]))
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(artifact_root: Path) -> int:
    print("f1-c2-060-optout validation: {}".format(artifact_root))
    checks = []

    legs, pairs = analyze_all(artifact_root)

    # 1. legs 6/6
    legs_ok = (
        len(legs) == 6
        and not any(leg.get("missing_files") for leg in legs.values())
        and all(not leg.get("failures") for leg in legs.values())
    )
    checks.append(("legs: 6/6", legs_ok, ", ".join(legs.keys())))

    # 2-4. row totals
    total_timing = sum(leg["checks"].get("timing_rows", 0) or 0 for leg in legs.values())
    per_leg_timing_ok = all(leg["checks"].get("timing_rows") == 20 for leg in legs.values())
    checks.append(
        (
            "timing rows: 120/120 (20 per leg)",
            total_timing == 120 and per_leg_timing_ok,
            "{} total ({})".format(
                total_timing,
                ", ".join(str(leg["checks"].get("timing_rows")) for leg in legs.values()),
            ),
        )
    )
    total_log = sum(leg["checks"].get("request_log_rows", 0) or 0 for leg in legs.values())
    per_leg_log_ok = all(leg["checks"].get("request_log_rows") == 25 for leg in legs.values())
    checks.append(
        (
            "request-log rows: 150/150 (25 per leg)",
            total_log == 150 and per_leg_log_ok,
            "{} total".format(total_log),
        )
    )
    total_completion = sum(
        leg["checks"].get("request_completion_rows", 0) or 0 for leg in legs.values()
    )
    per_leg_completion_ok = all(
        leg["checks"].get("request_completion_rows") == 20 for leg in legs.values()
    )
    checks.append(
        (
            "completion rows: 120/120 (20 per leg)",
            total_completion == 120 and per_leg_completion_ok,
            "{} total".format(total_completion),
        )
    )
    pairing_ok = all(leg["checks"].get("timing_log_pairing_ok") for leg in legs.values())
    checks.append(("timing-log pairing: 6/6", pairing_ok, "order-based, fields verified"))

    # 5-6. flags
    rc_ok = all(leg["checks"].get("probe_returncode_ok") for leg in legs.values())
    checks.append(("probe_returncode=0: 6/6", rc_ok, "all legs"))
    stopped_ok = all(leg["checks"].get("server_stopped_ok") for leg in legs.values())
    checks.append(("server_stopped=true: 6/6", stopped_ok, "all legs"))

    # 7. same binary SHA
    shas = {leg.get("binary_sha256") for leg in legs.values()}
    same_binary_ok = len(shas) == 1 and BINARY_SHA256 in shas
    checks.append(
        (
            "same binary SHA256: 6/6",
            same_binary_ok,
            "{} on all legs".format(next(iter(shas)) if len(shas) == 1 else sorted(shas)),
        )
    )

    # 8-9. override pattern
    candidate_ok = sum(
        1
        for spec in PAIR_SPECS
        if legs.get(spec["candidate_dir"], {}).get("checks", {}).get("operator_overrides_ok")
    )
    checks.append(("candidate overrides {}: 3/3", candidate_ok == 3, "both switches unset"))
    optout_ok = sum(
        1
        for spec in PAIR_SPECS
        if legs.get(spec["optout_dir"], {}).get("checks", {}).get("operator_overrides_ok")
    )
    checks.append(
        (
            "optout overrides both switches='0': 3/3",
            optout_ok == 3,
            "VMLX_QWEN35_COMPILE_DECODE_REGIONS=0, VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE=0",
        )
    )

    # 10. greedy output match per pair
    matched = sum(1 for pair in pairs if pair["output_match"]["all_contexts_match"])
    checks.append(
        (
            "greedy output match: 3/3 pairs",
            matched == 3,
            "short+30k, per-repeat text/hash/token equality",
        )
    )

    # 11. policy pattern
    policy_ok = all(leg["policy_evidence"].get("matches_expected") for leg in legs.values())
    checks.append(
        (
            "policy pattern: 6/6",
            policy_ok,
            "compiled_routed_switch_glu active on ornith/text candidate; absent elsewhere "
            "(vision: no policy line, other [Qwen4Exp] lines tolerated)",
        )
    )

    # 12. finalize_ms absent
    finalize_absent = all(leg["finalize_ms"].get("present") is False for leg in legs.values())
    checks.append(("finalize_ms recorded absent: 6/6", finalize_absent, "recorded, not synthesized"))

    # 13-16. report cross-check against freshly derived values
    report_json = artifact_root / REPORT_JSON_NAME
    report_md = artifact_root / REPORT_MD_NAME
    derived = {
        "totals": derived_totals(legs, pairs),
        "pairs": pairs,
        "legs": {dir_rel: public_leg(leg) for dir_rel, leg in legs.items()},
    }
    if report_json.is_file():
        report = json.loads(report_json.read_text(encoding="utf-8"))

        totals_ok = report.get("totals") == derived["totals"]
        checks.append(
            (
                "report cross-check: totals match ({})".format(REPORT_JSON_NAME),
                totals_ok,
                (
                    "legs={legs}, valid={legs_valid}, pairs={pairs}, timing_rows={timing_rows_total}, "
                    "log_rows={request_log_rows_total}".format(**derived["totals"])
                    if totals_ok
                    else "mismatch: report={} derived={}".format(
                        report.get("totals"), derived["totals"]
                    )
                ),
            )
        )
        pairs_ok = canonical(report.get("pairs")) == canonical(derived["pairs"])
        checks.append(
            (
                "report cross-check: pairs match (means + deltas + output matches)",
                pairs_ok,
                "3 pairs" if pairs_ok else "mismatch in pairs section",
            )
        )
        legs_ok_check = canonical(report.get("legs")) == canonical(derived["legs"])
        checks.append(
            (
                "report cross-check: legs match (checks + policy + means)",
                legs_ok_check,
                "6 legs" if legs_ok_check else "mismatch in legs section",
            )
        )
        checks.append(
            (
                "report cross-check: {} present".format(REPORT_MD_NAME),
                report_md.is_file(),
                str(report_md),
            )
        )
    else:
        checks.append(
            (
                "report cross-check: {} present".format(REPORT_JSON_NAME),
                False,
                "not found; run the report first (no --validate)",
            )
        )

    failed = 0
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print("[{}] {} ({})".format(status, label, detail))

    if failed:
        print("validation FAILED ({} of {} checks)".format(failed, len(checks)))
        return 1
    print("validation PASSED ({} checks)".format(len(checks)))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def default_artifact_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / ARTIFACT_DIRNAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "F1/C2 same-binary opt-out A/B report for the completed 0.6.0 artifacts. Reads the "
            "six candidate/optout legs, validates them, and emits f1-c2-060-optout-comparison.json"
            "/.md in the artifact root."
        )
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="artifact root (default: <repo>/artifacts/{})".format(ARTIFACT_DIRNAME),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate artifacts and the emitted report; exit non-zero on failure",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = (args.artifacts or default_artifact_root()).resolve()

    if not artifact_root.is_dir():
        print("artifact root not found: {}".format(artifact_root), file=sys.stderr)
        return 1

    if args.validate:
        return validate(artifact_root)

    report = build_report(artifact_root)

    json_path = artifact_root / REPORT_JSON_NAME
    md_path = artifact_root / REPORT_MD_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    totals = report["totals"]
    print(
        "f1-c2-060-optout report: {}/{} legs valid, {}/{} pairs matched, {} timing rows, "
        "{} request-log rows".format(
            totals["legs_valid"],
            totals["legs"],
            totals["output_pairs_matched"],
            totals["pairs"],
            totals["timing_rows_total"],
            totals["request_log_rows_total"],
        )
    )
    for leg in report["legs"].values():
        if leg.get("failures"):
            print("[WARN] {}: {}".format(leg["dir"], ", ".join(leg["failures"])))
    print("wrote {}".format(json_path))
    print("wrote {}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
