#!/usr/bin/env python3
"""Release-attribution report for the completed F1/C2 historical benchmark artifacts.

Reads the six completed final legs under
``artifacts/f1-c2-release-compare-20260930T095614Z/`` and emits
``release-comparison.json`` and ``release-comparison.md`` in the artifact root.

Stdlib only. Read-only with respect to the artifacts: the only writes are the
two report files in the artifact root. Never starts a model server, never
benchmarks, and never invents values -- e.g. ``finalize_ms`` is recorded as
absent when the historical tags do not emit it, and the residual is computed
as the legacy ``tail = wall - prefill - generate``.

Usage::

    python3 tools/release_attribution_report.py            # generate the report
    python3 tools/release_attribution_report.py --validate # check artifacts + report

``--validate`` exits non-zero on failure and checks, at minimum: the three
expected models, three release pairs, 6/6 legs, and 120 timing rows total.
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
REPORT_JSON_NAME = "release-comparison.json"
REPORT_MD_NAME = "release-comparison.md"

# The six completed final legs. Binary paths and SHA256 values are the exact
# recorded values from the run (also present in each leg's result.json); the
# constants make validation fail loudly if artifacts were swapped.
LEG_SPECS = [
    {
        "dir": "ornith-050",
        "model_key": "ornith",
        "model_id": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        "release": "0.5.0",
        "binary": "/opt/homebrew/Cellar/mei/0.5.0/bin/mei",
        "binary_sha256": "580e3d31af2e1ce32c4decf31f85e3fd64506c0c9c05c1d4ed1bb7a16741e2bb",
        "enable_thinking": None,
        "policy_expected": "absent",
        "policy_note": "0.5.0 leg: no compiled_routed_switch_glu policy line in server.log",
    },
    {
        "dir": "ornith-060",
        "model_key": "ornith",
        "model_id": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        "release": "0.6.0",
        "binary": "/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei",
        "binary_sha256": "3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc",
        "enable_thinking": None,
        "policy_expected": "active",
        "policy_note": "0.6.0 text/Ornith leg: compiled_routed_switch_glu=active",
    },
    {
        "dir": "qwen36-text-050-thinking-off",
        "model_key": "qwen36-text",
        "model_id": "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
        "release": "0.5.0",
        "binary": "/opt/homebrew/Cellar/mei/0.5.0/bin/mei",
        "binary_sha256": "580e3d31af2e1ce32c4decf31f85e3fd64506c0c9c05c1d4ed1bb7a16741e2bb",
        "enable_thinking": False,
        "policy_expected": "absent",
        "policy_note": "0.5.0 leg: no compiled_routed_switch_glu policy line in server.log",
    },
    {
        "dir": "qwen36-text-060-thinking-off",
        "model_key": "qwen36-text",
        "model_id": "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
        "release": "0.6.0",
        "binary": "/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei",
        "binary_sha256": "3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc",
        "enable_thinking": False,
        "policy_expected": "active",
        "policy_note": "0.6.0 text leg: compiled_routed_switch_glu=active",
    },
    {
        "dir": "qwen36-vision-050-thinking-off",
        "model_key": "qwen36-vision",
        "model_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "release": "0.5.0",
        "binary": "/opt/homebrew/Cellar/mei/0.5.0/bin/mei",
        "binary_sha256": "580e3d31af2e1ce32c4decf31f85e3fd64506c0c9c05c1d4ed1bb7a16741e2bb",
        "enable_thinking": False,
        "policy_expected": "absent",
        "policy_note": "0.5.0 leg: no compiled_routed_switch_glu policy line in server.log",
    },
    {
        "dir": "qwen36-vision-060-thinking-off",
        "model_key": "qwen36-vision",
        "model_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "release": "0.6.0",
        "binary": "/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei",
        "binary_sha256": "3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc",
        "enable_thinking": False,
        "policy_expected": "absent",
        "policy_note": "0.6.0 vision leg: compiled_routed_switch_glu absent (other Qwen4Exp lines present)",
    },
]

MODEL_ORDER = ("ornith", "qwen36-text", "qwen36-vision")

METRIC_KEYS = (
    "wall_ms",
    "prefill_ms",
    "generate_ms",
    "tail_ms",
    "decode_tps",
    "prompt_tps",
    "cached_tokens",
    "mem_peak_bytes",
)
METRIC_LABELS = {
    "wall_ms": "wall (ms)",
    "prefill_ms": "prefill (ms)",
    "generate_ms": "generate (ms)",
    "tail_ms": "tail = wall - prefill - generate (ms)",
    "decode_tps": "decode (tok/s)",
    "prompt_tps": "prompt (tok/s)",
    "cached_tokens": "cached_tokens",
    "mem_peak_bytes": "memory peak (bytes)",
}

# Verified 0.6.0 metallib provenance facts (build dir re-checked read-only by
# this tool when the paths are present; the facts themselves are recorded).
METALLIB_PROVENANCE = {
    "build_dir": "/Users/tijs/.local/share/local-model-bench/mei-build-060",
    "vendored_mlx": "0.32.2",
    "metallib_relpath": "out/Products/Release/mlx.metallib",
    "metallib_bytes": 182351120,
    "metallib_sha256": "dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed",
    "mlx_version_source": "checkouts/vmlx-swift/Package.swift (MLX_VERSION define)",
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


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt_ms(v):
    return "n/a" if v is None else "{:,.2f}".format(v)


def fmt_tps(v):
    return "n/a" if v is None else "{:,.3f}".format(v)


def fmt_bytes(v):
    return "n/a" if v is None else "{:,.0f}".format(v)


def fmt_delta(v):
    return "n/a" if v is None else "{:+.2f}%".format(v)


def fmt_metric(key, v):
    if v is None:
        return "n/a"
    if key in ("wall_ms", "prefill_ms", "generate_ms", "tail_ms"):
        return fmt_ms(v)
    if key in ("decode_tps", "prompt_tps"):
        return fmt_tps(v)
    if key == "mem_peak_bytes":
        return fmt_bytes(v)
    return "{:,.3f}".format(v)


# ---------------------------------------------------------------------------
# per-leg analysis
# ---------------------------------------------------------------------------

def analyze_leg(artifact_root: Path, spec: dict) -> dict:
    leg_dir = artifact_root / spec["dir"] / "candidate"
    checks = {}
    failures = []

    missing = [name for name in REQUIRED_LEG_FILES if not (leg_dir / name).is_file()]
    if missing:
        return {
            "dir": spec["dir"],
            "model_key": spec["model_key"],
            "model_id": spec["model_id"],
            "release": spec["release"],
            "binary": spec["binary"],
            "binary_sha256": spec["binary_sha256"],
            "enable_thinking": spec["enable_thinking"],
            "missing_files": missing,
            "checks": {"all_files_present": False},
            "failures": ["missing files: {}".format(", ".join(missing))],
            "means": {},
            "outputs": {},
            "policy_evidence": {},
            "finalize_ms": {"present": None},
            "cross_check": {},
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

    def context_of(row):
        return "short" if row.get("prompt_tokens", 0) < 1000 else "loaded"

    # --- timing rows (20 expected: 10 short + 10 loaded) ---
    ctx_counts = {"short": 0, "loaded": 0}
    for row in timing_rows:
        ctx_counts[row.get("context", "?")] = ctx_counts.get(row.get("context", "?"), 0) + 1
    checks["timing_rows"] = len(timing_rows)
    checks["timing_rows_ok"] = len(timing_rows) == 20
    checks["timing_contexts"] = dict(ctx_counts)
    checks["timing_contexts_ok"] = ctx_counts.get("short") == 10 and ctx_counts.get("loaded") == 10
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
    checks["request_completion_contexts_ok"] = comp_ctx.get("short") == 10 and comp_ctx.get("loaded") == 10
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
    checks["binary_matches_expected"] = result.get("binary") == spec["binary"]
    checks["binary_sha256_matches_expected"] = result.get("binary_sha256") == spec["binary_sha256"]
    checks["model_id_matches_expected"] = result.get("model_id") == spec["model_id"]

    for key, ok in (
        ("timing_rows_ok", checks["timing_rows_ok"]),
        ("timing_contexts_ok", checks["timing_contexts_ok"]),
        ("timing_text_hashes_ok", checks["timing_text_hashes_ok"]),
        ("request_log_rows_ok", checks["request_log_rows_ok"]),
        ("request_completion_rows_ok", checks["request_completion_rows_ok"]),
        ("request_completion_contexts_ok", checks["request_completion_contexts_ok"]),
        ("timing_log_pairing_ok", checks["timing_log_pairing_ok"]),
        ("probe_returncode_ok", checks["probe_returncode_ok"]),
        ("server_stopped_ok", checks["server_stopped_ok"]),
        ("probe_status_ok", checks["probe_status_ok"]),
        ("binary_matches_expected", checks["binary_matches_expected"]),
        ("binary_sha256_matches_expected", checks["binary_sha256_matches_expected"]),
        ("model_id_matches_expected", checks["model_id_matches_expected"]),
    ):
        if not ok:
            failures.append(key)

    # --- means per context (from request-log completion rows) ---
    means = {}
    for ctx_key, is_short in (("short", True), ("loaded_30k", False)):
        rows = [r for r in completion_rows if (context_of(r) == "short") == is_short]
        wall = [r["wall_ms"] for r in rows]
        prefill = [r["prefill_ms"] for r in rows]
        generate = [r["generate_ms"] for r in rows]
        tail = [r["wall_ms"] - r["prefill_ms"] - r["generate_ms"] for r in rows]
        mem = mean([r["mem_peak_bytes"] for r in rows])
        means[ctx_key] = {
            "n": len(rows),
            "wall_ms": round2(mean(wall)),
            "prefill_ms": round2(mean(prefill)),
            "generate_ms": round2(mean(generate)),
            "tail_ms": round2(mean(tail)),
            "decode_tps": round3(mean([r["decode_tps"] for r in rows])),
            "prompt_tps": round3(mean([r["prompt_tps"] for r in rows])),
            "cached_tokens": round3(mean([r["cached_tokens"] for r in rows])),
            "mem_peak_bytes": int(round(mem)) if mem is not None else None,
        }

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
        policy_value == "active" if spec["policy_expected"] == "active" else not policy_present
    )
    policy = {
        "compiled_routed_switch_glu": {
            "present": policy_present,
            "value": policy_value,
            "line": policy_lines[0][1] if policy_lines else None,
            "line_number": policy_lines[0][0] if policy_lines else None,
        },
        "qwen4exp_line_count": sum(1 for line in server_log_lines if line.startswith("[Qwen4Exp]")),
        "qwen35_line_count": sum(1 for line in server_log_lines if line.startswith("[Qwen35]")),
        "expected": spec["policy_expected"],
        "expected_note": spec["policy_note"],
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
            "Historical 0.5.0/0.6.0 tags do not emit finalize_ms; recorded as absent. "
            "The residual is the legacy tail = wall - prefill - generate. "
            "Newer 0.6.1 F1 artifacts (outside this report) do carry finalize_ms."
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
        "dir": spec["dir"],
        "model_key": spec["model_key"],
        "model_id": result.get("model_id", spec["model_id"]),
        "release": spec["release"],
        "binary": result.get("binary", spec["binary"]),
        "binary_sha256": result.get("binary_sha256", spec["binary_sha256"]),
        "model_dir": result.get("model_dir"),
        "enable_thinking": spec["enable_thinking"],
        "contexts": result.get("contexts"),
        "checks": checks,
        "failures": failures,
        "means": means,
        "outputs": outputs,
        "policy_evidence": policy,
        "finalize_ms": finalize,
        "cross_check": cross_check,
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

ARM_SUBDIRS = ("candidate", "control", "optout")


def discover_excluded_dirs(artifact_root: Path, expected_dirs) -> list:
    """Directories present in the artifact root that hold completed runs for
    some arm but are not among the six final legs of this report. They are
    listed for transparency and excluded from every aggregate."""
    excluded = []
    if not artifact_root.is_dir():
        return excluded
    for child in sorted(artifact_root.iterdir()):
        if child.is_dir() and child.name not in expected_dirs:
            arms = [arm for arm in ARM_SUBDIRS if (child / arm / "result.json").is_file()]
            if arms:
                excluded.append(
                    {
                        "dir": child.name,
                        "arms": arms,
                        "reason": "present in artifact root but not among the six final legs; excluded from aggregates",
                    }
                )
    return excluded


def build_comparisons(legs: dict) -> list:
    comparisons = []
    for model_key in MODEL_ORDER:
        specs = [s for s in LEG_SPECS if s["model_key"] == model_key]
        by_release = {s["release"]: s for s in specs}
        spec_050, spec_060 = by_release["0.5.0"], by_release["0.6.0"]
        leg_050, leg_060 = legs[spec_050["dir"]], legs[spec_060["dir"]]

        comparison = {
            "model_key": model_key,
            "model_id": leg_050["model_id"],
            "legs": {"0.5.0": spec_050["dir"], "0.6.0": spec_060["dir"]},
        }
        for ctx_key, ctx in (("short", "short"), ("loaded_30k", "loaded")):
            m_050 = leg_050["means"].get(ctx_key, {})
            m_060 = leg_060["means"].get(ctx_key, {})
            delta = {key: pct_delta(m_050.get(key), m_060.get(key)) for key in METRIC_KEYS}
            out_050 = leg_050["outputs"].get(ctx, {})
            out_060 = leg_060["outputs"].get(ctx, {})
            match = (
                out_050.get("distinct_text_sha256") == out_060.get("distinct_text_sha256")
                and out_050.get("text_sample") == out_060.get("text_sample")
            )
            comparison[ctx_key] = {
                "0.5.0": {key: m_050.get(key) for key in METRIC_KEYS},
                "0.6.0": {key: m_060.get(key) for key in METRIC_KEYS},
                "delta_pct": delta,
                "output_match": {
                    "match": match,
                    "0.5.0": {
                        "text_sha256": out_050.get("distinct_text_sha256"),
                        "text": out_050.get("text_sample"),
                        "completion_tokens": out_050.get("completion_tokens"),
                        "finish_reasons": out_050.get("finish_reasons"),
                    },
                    "0.6.0": {
                        "text_sha256": out_060.get("distinct_text_sha256"),
                        "text": out_060.get("text_sample"),
                        "completion_tokens": out_060.get("completion_tokens"),
                        "finish_reasons": out_060.get("finish_reasons"),
                    },
                },
            }
        comparisons.append(comparison)
    return comparisons


def build_caveats(comparisons: list, legs: dict) -> list:
    caveats = []

    for comparison in comparisons:
        for ctx_key, ctx_label in (("short", "short"), ("loaded_30k", "30k")):
            om = comparison[ctx_key]["output_match"]
            if not om["match"]:
                a, b = om["0.5.0"], om["0.6.0"]
                caveats.append(
                    {
                        "id": "{}-{}-output-mismatch".format(comparison["model_key"], ctx_key),
                        "title": "{} {} output mismatch between releases (protocol caveat)".format(
                            comparison["model_id"], ctx_label
                        ),
                        "text": (
                            "0.5.0 produced visible text {a_text!r} ({a_tok} completion tokens, finish={a_finish}); "
                            "0.6.0 produced {b_text!r} ({b_tok} completion tokens, finish={b_finish}). "
                            "The {ctx}-context outputs are not matched between releases, so {ctx}-context timing "
                            "deltas for this model are not a like-for-like comparison and must not be read as an "
                            "engine effect without noting the output difference."
                        ).format(
                            a_text=a["text"],
                            a_tok=a["completion_tokens"],
                            a_finish=a["finish_reasons"],
                            b_text=b["text"],
                            b_tok=b["completion_tokens"],
                            b_finish=b["finish_reasons"],
                            ctx=ctx_label,
                        ),
                    }
                )

    any_finalize = any(leg.get("finalize_ms", {}).get("present") for leg in legs.values())
    caveats.append(
        {
            "id": "no-finalize-ms",
            "title": "finalize_ms absent in historical 0.5.0/0.6.0 tags",
            "text": (
                "No leg in this artifact set records finalize_ms (checked request.jsonl, result.json, "
                "probe.json, server.log per leg). The residual is therefore the legacy "
                "tail = wall - prefill - generate, which may absorb finalize/overhead. Presence is "
                "recorded per leg; no value is synthesized. Newer 0.6.1 F1 artifacts carry finalize_ms "
                "but are outside this report."
            ),
        }
    )

    caveats.append(
        {
            "id": "descriptive-not-causal",
            "title": "deltas are descriptive, not causal",
            "text": (
                "Means are single-run comparisons (10 repeats per context per leg); no significance "
                "testing is applied. Policy-line presence in server.log is a configuration difference, "
                "not proof of cause. Small deltas (order of a few percent, and any short-context delta "
                "for qwen36-text) should not be read as effects."
            ),
        }
    )

    wall_parts = []
    for leg in legs.values():
        cc = leg.get("cross_check", {})
        for ctx in ("short", "loaded"):
            if cc.get(ctx, {}).get("client_minus_server_wall_ms_mean") is not None:
                wall_parts.append((leg["dir"], ctx, cc[ctx]))
    if wall_parts:
        shorts = [p[2]["client_minus_server_wall_ms_min"] for p in wall_parts if p[1] == "short"]
        shortx = [p[2]["client_minus_server_wall_ms_max"] for p in wall_parts if p[1] == "short"]
        loadmin = [p[2]["client_minus_server_wall_ms_min"] for p in wall_parts if p[1] == "loaded"]
        loadmax = [p[2]["client_minus_server_wall_ms_max"] for p in wall_parts if p[1] == "loaded"]
        caveats.append(
            {
                "id": "wall-clock-cross-check",
                "title": "wall metric is server-reported wall_ms",
                "text": (
                    "All means use the server-reported wall_ms from the request log (same source as "
                    "prefill/generate/tail). Client-observed wall from the timing rows runs higher by "
                    "~{smin}-{smax} ms at short and ~{lmin}-{lmax} ms at 30k on every leg; the source of "
                    "that difference is not determined here."
                ).format(
                    smin=fmt_ms(min(shorts)),
                    smax=fmt_ms(max(shortx)),
                    lmin=fmt_ms(min(loadmin)),
                    lmax=fmt_ms(max(loadmax)),
                ),
            }
        )

    return caveats


def build_report(artifact_root: Path) -> dict:
    legs = {}
    for spec in LEG_SPECS:
        legs[spec["dir"]] = analyze_leg(artifact_root, spec)

    excluded = discover_excluded_dirs(artifact_root, [s["dir"] for s in LEG_SPECS])
    comparisons = build_comparisons(legs)
    caveats = build_caveats(comparisons, legs)

    total_timing = sum(leg["checks"].get("timing_rows", 0) or 0 for leg in legs.values())
    total_log = sum(leg["checks"].get("request_log_rows", 0) or 0 for leg in legs.values())
    legs_valid = sum(1 for leg in legs.values() if not leg.get("failures"))
    all_ok = legs_valid == len(LEG_SPECS)

    report = {
        "schema": "mei-release-attribution/1",
        "generated_by": "tools/release_attribution_report.py",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_root": str(artifact_root),
        "scope": {
            "legs": [s["dir"] for s in LEG_SPECS],
            "models": [s["model_id"] for s in LEG_SPECS[::2]],
            "release_pairs": [
                {
                    "model_key": model_key,
                    "model_id": next(s["model_id"] for s in LEG_SPECS if s["model_key"] == model_key),
                    "legs": {
                        s["release"]: s["dir"]
                        for s in LEG_SPECS
                        if s["model_key"] == model_key
                    },
                }
                for model_key in MODEL_ORDER
            ],
            "contexts": {"short": 13, "loaded": 30000},
            "excluded_dirs": excluded,
        },
        "method": {
            "means_source": "request.jsonl rows with kind=completion (10 short + 10 loaded per leg)",
            "tail": "tail_ms = wall_ms - prefill_ms - generate_ms (legacy decomposition; no finalize_ms in historical tags)",
            "delta_pct": "(0.6.0 - 0.5.0) / 0.5.0 * 100",
            "wall_clock": "wall_ms is server-reported; client-observed wall is recorded under cross_check",
        },
        "settings": {
            "enable_thinking_false_legs": [
                s["dir"] for s in LEG_SPECS if s["enable_thinking"] is False
            ],
            "legs_without_enable_thinking_flag": [
                s["dir"] for s in LEG_SPECS if s["enable_thinking"] is None
            ],
            "note": (
                "The four Qwen3.6 legs were run with --enable-thinking false; the two Ornith legs "
                "carry no --enable-thinking flag. Recorded from each leg's command in result.json."
            ),
        },
        "binaries": {
            release: {
                "path": next(s["binary"] for s in LEG_SPECS if s["release"] == release),
                "sha256": next(s["binary_sha256"] for s in LEG_SPECS if s["release"] == release),
                "legs": [s["dir"] for s in LEG_SPECS if s["release"] == release],
            }
            for release in ("0.5.0", "0.6.0")
        },
        "finalize_ms": {
            "present_in_any_leg": any(leg["finalize_ms"].get("present") for leg in legs.values()),
            "per_leg": {leg["dir"]: leg["finalize_ms"].get("present") for leg in legs.values()},
            "note": (
                "Detected by scanning each leg's request.jsonl, result.json, probe.json and server.log; "
                "historical 0.5.0/0.6.0 tags lack finalize_ms, so it is recorded as absent and the "
                "legacy tail is used."
            ),
        },
        "provenance_0_6_0": dict(METALLIB_PROVENANCE),
        "legs": legs,
        "comparisons": comparisons,
        "totals": {
            "legs": len(LEG_SPECS),
            "legs_valid": legs_valid,
            "models": len({s["model_id"] for s in LEG_SPECS}),
            "release_pairs": len(MODEL_ORDER),
            "timing_rows_total": total_timing,
            "request_log_rows_total": total_log,
            "probe_returncode_zero_legs": sum(
                1 for leg in legs.values() if leg["checks"].get("probe_returncode_ok")
            ),
            "server_stopped_legs": sum(
                1 for leg in legs.values() if leg["checks"].get("server_stopped_ok")
            ),
            "all_legs_valid": all_ok,
        },
        "caveats": caveats,
    }
    return report


def recheck_metallib() -> dict:
    """Read-only re-check of the recorded 0.6.0 metallib provenance facts."""
    result = {"checked": False}  # type: dict
    build_dir = Path(METALLIB_PROVENANCE["build_dir"])
    metallib = build_dir / METALLIB_PROVENANCE["metallib_relpath"]
    if metallib.is_file():
        size = metallib.stat().st_size
        digest = hashlib.sha256(metallib.read_bytes()).hexdigest()
        result = {
            "checked": True,
            "metallib_path": str(metallib),
            "size_match": size == METALLIB_PROVENANCE["metallib_bytes"],
            "sha256_match": digest == METALLIB_PROVENANCE["metallib_sha256"],
            "observed_bytes": size,
            "observed_sha256": digest,
        }
    else:
        result["reason"] = "metallib path not present; recorded facts left as-is"

    package_swift = build_dir / "checkouts" / "vmlx-swift" / "Package.swift"
    if package_swift.is_file():
        text = package_swift.read_text(encoding="utf-8")
        needle = 'MLX_VERSION", to: "\\"{}\\"'.format(METALLIB_PROVENANCE["vendored_mlx"])
        result["mlx_version_source_checked"] = True
        result["mlx_version_match"] = needle in text
    else:
        result["mlx_version_source_checked"] = False
    return result


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines = []
    add = lines.append
    totals = report["totals"]

    add("# Mei release comparison — F1/C2 historical artifacts")
    add("")
    add("- **Artifact root:** `{}`".format(report["artifact_root"]))
    add("- **Generated:** {} by `{}` (stdlib only)".format(report["generated_at_utc"], report["generated_by"]))
    add(
        "- **Scope:** {} final legs = 3 models x releases 0.5.0/0.6.0; per leg 10 repeats x "
        "(short: 13 prompt tokens; 30k: 30000 prompt tokens); 20 timing rows and 25 request-log rows per leg.".format(
            len(report["scope"]["legs"])
        )
    )
    if report["scope"]["excluded_dirs"]:
        add(
            "- **Excluded from aggregates:** "
            + ", ".join("`{}`".format(e["dir"]) for e in report["scope"]["excluded_dirs"])
            + " (present in artifact root but not among the six final legs)."
        )
    add("")
    add("## Validation")
    add("")
    add(
        "**{}/{} legs valid** — {} timing rows, {} request-log rows total; probe_returncode=0 and "
        "server_stopped=true on every leg.".format(
            totals["legs_valid"],
            totals["legs"],
            totals["timing_rows_total"],
            totals["request_log_rows_total"],
        )
    )
    add("")
    add("| Leg | Model | Release | Timing rows | Log rows | probe rc | stopped | compiled_routed_switch_glu |")
    add("|---|---|---|---|---|---|---|---|")
    for spec in LEG_SPECS:
        leg = report["legs"][spec["dir"]]
        checks = leg.get("checks", {})
        policy = leg.get("policy_evidence", {}).get("compiled_routed_switch_glu", {})
        policy_cell = "active" if policy.get("value") == "active" else "absent"
        add(
            "| `{}` | `{}` | {} | {}/20 | {}/25 | {} | {} | {} |".format(
                spec["dir"],
                leg.get("model_id", spec["model_id"]),
                spec["release"],
                checks.get("timing_rows", "?"),
                checks.get("request_log_rows", "?"),
                checks.get("probe_returncode", "?"),
                str(checks.get("server_stopped", "?")).lower(),
                policy_cell,
            )
        )
    add("")
    add("## Binaries")
    add("")
    add("| Release | Path | SHA256 |")
    add("|---|---|---|")
    for release in ("0.5.0", "0.6.0"):
        binary = report["binaries"][release]
        add("| {} | `{}` | `{}` |".format(release, binary["path"], binary["sha256"]))
    add("")
    add("## Settings")
    add("")
    add(
        "- `--enable-thinking false`: "
        + ", ".join("`{}`".format(d) for d in report["settings"]["enable_thinking_false_legs"])
    )
    add(
        "- No `--enable-thinking` flag: "
        + ", ".join("`{}`".format(d) for d in report["settings"]["legs_without_enable_thinking_flag"])
    )
    add("")
    add("## 0.6.0 metallib provenance")
    add("")
    prov = report["provenance_0_6_0"]
    add("- Build dir: `{}`".format(prov["build_dir"]))
    add("- Vendored MLX: {} ({})".format(prov["vendored_mlx"], prov["mlx_version_source"]))
    add(
        "- metallib: {} bytes, SHA256 `{}` (`{}`)".format(
            fmt_bytes(prov["metallib_bytes"]), prov["metallib_sha256"], prov["metallib_relpath"]
        )
    )
    recheck = report.get("provenance_recheck", {})
    if recheck.get("checked"):
        add(
            "- Read-only re-check: size match = {}, sha256 match = {} (observed {} bytes, `{}`)".format(
                recheck.get("size_match"),
                recheck.get("sha256_match"),
                recheck.get("observed_bytes"),
                recheck.get("observed_sha256"),
            )
        )
        if recheck.get("mlx_version_source_checked"):
            add("- MLX_VERSION source check: match = {}".format(recheck.get("mlx_version_match")))
    else:
        add("- Read-only re-check: not performed ({})".format(recheck.get("reason", "paths unavailable")))
    add("")
    add("## Policy evidence (server.log)")
    add("")
    add("| Leg | compiled_routed_switch_glu | Evidence line |")
    add("|---|---|---|")
    for spec in LEG_SPECS:
        policy = report["legs"][spec["dir"]].get("policy_evidence", {})
        entry = policy.get("compiled_routed_switch_glu", {})
        if entry.get("present"):
            cell = "**active**" if entry.get("value") == "active" else str(entry.get("value"))
            evidence = "L{}: `{}`".format(entry.get("line_number"), entry.get("line"))
        else:
            cell = "absent"
            evidence = "no compiled_routed_switch_glu line ({})".format(policy.get("expected_note", ""))
        add("| `{}` | {} | {} |".format(spec["dir"], cell, evidence))
    add("")
    add("## Visible output comparison")
    add("")
    add("| Model | Context | 0.5.0 visible text (sha256) | 0.6.0 visible text (sha256) | Match |")
    add("|---|---|---|---|---|")
    for comparison in report["comparisons"]:
        for ctx_key, ctx_label in (("short", "short"), ("loaded_30k", "30k")):
            om = comparison[ctx_key]["output_match"]
            a, b = om["0.5.0"], om["0.6.0"]
            add(
                "| `{}` | {} | {!r} (`{}`) | {!r} (`{}`) | {} |".format(
                    comparison["model_id"],
                    ctx_label,
                    a["text"],
                    ", ".join(h[:16] for h in (a["text_sha256"] or [])),
                    b["text"],
                    ", ".join(h[:16] for h in (b["text_sha256"] or [])),
                    "yes" if om["match"] else "**NO**",
                )
            )
    add("")
    add("## Means and 0.6.0-minus-0.5.0 deltas")
    add("")
    add(
        "Means from request-log completion rows (server-reported); 10 rows per context per leg. "
        "`tail = wall - prefill - generate` (legacy; finalize_ms absent). Delta% = (0.6.0 - 0.5.0) / 0.5.0 x 100."
    )
    for comparison in report["comparisons"]:
        add("")
        add("### `{}`".format(comparison["model_id"]))
        add("")
        add("| Metric | short 0.5.0 | short 0.6.0 | delta% | 30k 0.5.0 | 30k 0.6.0 | delta% |")
        add("|---|---|---|---|---|---|---|")
        for key in METRIC_KEYS:
            short = comparison["short"]
            loaded = comparison["loaded_30k"]
            add(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    METRIC_LABELS[key],
                    fmt_metric(key, short["0.5.0"][key]),
                    fmt_metric(key, short["0.6.0"][key]),
                    fmt_delta(short["delta_pct"][key]),
                    fmt_metric(key, loaded["0.5.0"][key]),
                    fmt_metric(key, loaded["0.6.0"][key]),
                    fmt_delta(loaded["delta_pct"][key]),
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
    expected_models = set(s["model_id"] for s in LEG_SPECS)
    print("release-attribution validation: {}".format(artifact_root))
    checks = []

    legs = {}
    missing_dirs = []
    for spec in LEG_SPECS:
        leg = analyze_leg(artifact_root, spec)
        legs[spec["dir"]] = leg
        if leg.get("missing_files"):
            missing_dirs.append(spec["dir"])

    # 1. expected models
    actual_models = {leg.get("model_id") for leg in legs.values() if leg.get("model_id")}
    ok = actual_models == expected_models and len(actual_models) == 3
    checks.append(("expected models: 3/3", ok, ", ".join(sorted(actual_models))))

    # 2. release pairs
    pairs = {}
    for spec in LEG_SPECS:
        pairs.setdefault(spec["model_key"], {})[spec["release"]] = spec["dir"]
    pairs_ok = (
        len(pairs) == 3
        and all(set(releases) == {"0.5.0", "0.6.0"} for releases in pairs.values())
    )
    checks.append(
        (
            "release pairs: 3/3",
            pairs_ok,
            "; ".join(
                "{} {}->{}".format(key, releases["0.5.0"], releases["0.6.0"])
                for key, releases in pairs.items()
            ),
        )
    )

    # 3. legs 6/6
    legs_ok = len(legs) == 6 and not missing_dirs and all(not leg.get("failures") for leg in legs.values())
    checks.append(
        (
            "legs: 6/6",
            legs_ok,
            ", ".join(legs.keys()),
        )
    )

    # 4. timing rows total
    total_timing = sum(leg["checks"].get("timing_rows", 0) or 0 for leg in legs.values())
    per_leg_ok = all(leg["checks"].get("timing_rows") == 20 for leg in legs.values())
    checks.append(
        (
            "timing rows total: 120/120",
            total_timing == 120 and per_leg_ok,
            "{} total ({} per leg)".format(total_timing, ", ".join(str(leg["checks"].get("timing_rows")) for leg in legs.values())),
        )
    )

    # extra: request-log rows, flags
    total_log = sum(leg["checks"].get("request_log_rows", 0) or 0 for leg in legs.values())
    checks.append(
        (
            "request-log rows total: 150/150",
            total_log == 150,
            "{} total".format(total_log),
        )
    )
    rc_ok = all(leg["checks"].get("probe_returncode_ok") for leg in legs.values())
    checks.append(("probe_returncode=0: 6/6", rc_ok, "all legs"))
    stopped_ok = all(leg["checks"].get("server_stopped_ok") for leg in legs.values())
    checks.append(("server_stopped=true: 6/6", stopped_ok, "all legs"))
    policy_ok = all(leg.get("policy_evidence", {}).get("matches_expected") for leg in legs.values())
    checks.append(
        (
            "policy evidence pattern: 6/6",
            policy_ok,
            "active on 0.6.0 ornith/text; absent on vision and all 0.5.0 legs",
        )
    )
    finalize_absent = all(leg.get("finalize_ms", {}).get("present") is False for leg in legs.values())
    checks.append(("finalize_ms recorded absent: 6/6", finalize_absent, "recorded, not synthesized"))

    # report cross-check: emitted report must match the freshly derived values
    report_json = artifact_root / REPORT_JSON_NAME
    report_md = artifact_root / REPORT_MD_NAME
    derived = {
        "legs": len(LEG_SPECS),
        "legs_valid": sum(1 for leg in legs.values() if not leg.get("failures")),
        "models": len(actual_models),
        "release_pairs": len(pairs),
        "timing_rows_total": total_timing,
        "request_log_rows_total": total_log,
    }
    if report_json.is_file():
        report = json.loads(report_json.read_text(encoding="utf-8"))
        totals = report.get("totals", {})
        mismatches = [
            "{}: report={} derived={}".format(key, totals.get(key), value)
            for key, value in derived.items()
            if totals.get(key) != value
        ]
        consistency = not mismatches
        detail = (
            "legs={legs}, valid={legs_valid}, pairs={release_pairs}, models={models}, "
            "timing_rows={timing_rows_total}".format(**derived)
            if consistency
            else "; ".join(mismatches)
        )
        checks.append(
            (
                "report cross-check: {} totals match".format(REPORT_JSON_NAME),
                consistency,
                detail,
            )
        )
        md_ok = report_md.is_file()
        checks.append(("report cross-check: {} present".format(REPORT_MD_NAME), md_ok, str(report_md)))
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
            "Release-attribution report for the completed F1/C2 historical benchmark artifacts. "
            "Reads the six final legs, validates them, and emits release-comparison.json/.md in the artifact root."
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

    if args.validate:
        return validate(artifact_root)

    report = build_report(artifact_root)
    report["provenance_recheck"] = recheck_metallib()

    json_path = artifact_root / REPORT_JSON_NAME
    md_path = artifact_root / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")

    totals = report["totals"]
    print(
        "release-attribution report: {}/{} legs valid, {} pairs, {} timing rows, {} request-log rows".format(
            totals["legs_valid"],
            totals["legs"],
            totals["release_pairs"],
            totals["timing_rows_total"],
            totals["request_log_rows_total"],
        )
    )
    for spec in LEG_SPECS:
        leg = report["legs"][spec["dir"]]
        if leg.get("failures"):
            print("[WARN] {}: {}".format(spec["dir"], ", ".join(leg["failures"])))
    print("wrote {}".format(json_path))
    print("wrote {}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
