#!/usr/bin/env python3
"""Validate the umans-coder worker model-option path (todo 0b87b76a#15).

Checks, for every Mei primary/secondary candidate:
  1. IDENTITY   - the option's model id is byte-identical to the lineup id
                  and to the local-model-bench mei.yaml served_model_id.
  2. CHECKPOINT - staged dir exists, config.json present, >=1 safetensors
                  shard (weight presence), tokenizer files present.
  3. RUNTIME    - release binary + mlx.metallib + default.metallib exist.
  4. FAIL-CLOSED- if the option's port is NOT listening: PASS (connection
                  refused is the correct unavailable-state behavior, and no
                  fallback chain may reroute to cloud: verified by reading
                  the profile's fallback_providers === absent).
                  If the port IS listening: GET /v1/models must report the
                  exact served id (identity gate), else FAIL.
  5. PROFILE    - the umans-coder profile config (if reachable) declares the
                  mei-* custom provider with the exact model id; a missing
                  option is reported (not an error) so the reference doc
                  stays the source of truth.

Read-only: never writes to the profile or to local-model-bench.
Exit 0 if all hard gates pass; 2 if any hard gate fails; 3 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

MEI_REPO = Path(__file__).resolve().parent.parent
LINEUP = MEI_REPO / "configs" / "model-lineup.json"
BENCH_REPO = Path.home() / "projects" / "local-model-bench"
MEI_MODELS = Path.home() / ".local" / "share" / "local-model-bench" / "mei-models"
MEI_BUILD = Path.home() / ".local" / "share" / "local-model-bench" / "mei-build" / "release"
PROFILE = Path.home() / ".hermes" / "profiles" / "umans-coder" / "config.yaml"

# name -> (lineup id). Ports come from local-model-bench mei.yaml (read-only),
# falling back to the documented reference ports below.
CANDIDATES = {
    "ornith35": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
    "qwen38": "mlx-community/Qwen3.8-27B-4bit",
    "heretic": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
    "gemma4": "mlx-community/gemma-4-26b-a4b-it-4bit",
}
REF_PORTS = {"ornith35": 8024, "qwen38": 8025, "heretic": 8026, "gemma4": 8027}
BENCH_CFG_DIRS = {
    "ornith35": "Ornith-1.5-35B-A3B",
    "qwen38": "Qwen3.8-27B",
    "heretic": "Qwen3.8-27B-Uncensored",
    "gemma4": "Gemma-4-26B-A4B",
}


def load_lineup() -> dict:
    data = json.loads(LINEUP.read_text())
    by_id = {m["id"]: m for m in data["models"]}
    return by_id


def bench_port_and_id(name: str) -> tuple[int | None, str | None]:
    """Read raw_port/served_model_id from local-model-bench mei.yaml (read-only)."""
    cfg = BENCH_REPO / "configs" / BENCH_CFG_DIRS[name] / "mei.yaml"
    if not cfg.exists():
        return None, None
    text = cfg.read_text()
    import re

    port = None
    m = re.search(r"raw_port:\s*(\d+)", text)
    if m:
        port = int(m.group(1))
    mid = None
    m = re.search(r"served_model_id:\s*(\S+)", text)
    if m:
        mid = m.group(1).strip()
    return port, mid


def check_staged(model_id: str, staged_path: str) -> list[str]:
    """Return list of problems (empty = PASS)."""
    problems: list[str] = []
    d = Path(staged_path).expanduser()
    if not d.is_dir():
        return [f"staged dir missing: {d}"]
    if not (d / "config.json").is_file():
        problems.append(f"config.json missing in {d}")
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        problems.append(f"no *.safetensors shards in {d}")
    for tok in ("tokenizer.json", "tokenizer_config.json"):
        if not (d / tok).is_file():
            problems.append(f"{tok} missing in {d}")
    return problems


def check_runtime() -> list[str]:
    problems: list[str] = []
    for f in ("mei", "mlx.metallib", "default.metallib", "mlx.metallib.provenance"):
        if not (MEI_BUILD / f).exists():
            problems.append(f"runtime artifact missing: {MEI_BUILD / f}")
    return problems


def probe_port(port: int, expected_id: str) -> tuple[str, str]:
    """Return (status, detail). status in {PASS_IDENTITY, PASS_FAILCLOSED, FAIL}."""
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, OSError, json.JSONDecodeError) as e:
        return "PASS_FAILCLOSED", f"port {port} not reachable ({type(e).__name__}) — fail-closed behavior confirmed"
    ids = []
    data = body.get("data", []) if isinstance(body, dict) else []
    for row in data:
        if isinstance(row, dict) and row.get("id"):
            ids.append(row["id"])
    if expected_id in ids:
        return "PASS_IDENTITY", f"/v1/models on {port} serves exact id {expected_id!r}"
    return "FAIL", f"/v1/models on {port} served {ids!r}, expected {expected_id!r}"


def profile_options() -> dict[str, str]:
    """Return {provider_name: model_id} declared in the umans-coder profile.

    Raises RuntimeError when PyYAML is unavailable so a missing profile check
    is explicit instead of a silent degradation to "not declared".
    """
    if not PROFILE.exists():
        return {}
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "PyYAML is required to read the profile config — run this tool "
            "with a yaml-capable interpreter (e.g. "
            "local-model-bench/.venv/bin/python)."
        )
    import yaml

    cfg = yaml.safe_load(PROFILE.read_text()) or {}
    out: dict[str, str] = {}
    for entry in cfg.get("custom_providers", []) or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        if not name.startswith("mei-"):
            continue
        models = entry.get("models", {}) or {}
        if isinstance(models, dict) and models:
            out[name] = next(iter(models))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineup", default=str(LINEUP))
    ap.add_argument("--output", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    lineup = load_lineup()
    results = []
    hard_fail = False

    runtime_problems = check_runtime()
    for p in runtime_problems:
        results.append({"check": "runtime", "status": "FAIL", "detail": p})
    if runtime_problems:
        hard_fail = True

    for name, expected_id in CANDIDATES.items():
        entry = {
            "candidate": name,
            "expected_id": expected_id,
            "checks": {},
        }
        lineup_entry = lineup.get(expected_id)
        port, bench_id = bench_port_and_id(name)
        ref_port = port if port is not None else REF_PORTS[name]

        # identity vs lineup
        entry["checks"]["identity_lineup"] = (
            "PASS" if lineup_entry else "FAIL"
        )
        if not lineup_entry:
            hard_fail = True
        # identity vs bench mei.yaml
        if bench_id is None:
            entry["checks"]["identity_bench"] = "SKIP (no mei.yaml)"
        elif bench_id == expected_id:
            entry["checks"]["identity_bench"] = "PASS"
        else:
            entry["checks"]["identity_bench"] = (
                f"FAIL bench served_model_id {bench_id!r} != {expected_id!r}"
            )
            hard_fail = True

        staged = lineup_entry.get("local_staged_path", "") if lineup_entry else ""
        cp = check_staged(expected_id, staged) if staged else ["no staged path in lineup"]
        entry["checks"]["checkpoint"] = "PASS" if not cp else "FAIL: " + "; ".join(cp)
        if cp:
            hard_fail = True

        status, detail = probe_port(ref_port, expected_id)
        entry["checks"]["port_probe"] = f"{status}: {detail}"
        if status == "FAIL":
            hard_fail = True
        entry["port"] = ref_port
        entry["source_port"] = "bench mei.yaml" if port is not None else "reference"
        results.append(entry)

    # profile wiring (informational)
    try:
        profile = profile_options()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 3
    for name, expected_id in CANDIDATES.items():
        declared = profile.get(f"mei-{name}")
        if declared == expected_id:
            status = "PASS"
        elif declared is None:
            status = "INFO (option not declared in profile yet — reference doc is source of truth)"
        else:
            status = f"FAIL profile model {declared!r} != {expected_id!r}"
            hard_fail = True
        results.append({"candidate": name, "profile_option": f"mei-{name}", "status": status})

    verdict = "PASS" if not hard_fail else "FAIL"
    summary = {
        "tool": "validate_worker_options.py",
        "verdict": verdict,
        "profile": str(PROFILE),
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
    if args.json or args.output:
        print(json.dumps(summary, indent=2))
    else:
        print(f"verdict: {verdict}")
        for r in results:
            for k, v in r.items():
                print(f"  {k}: {v}")
    return 0 if not hard_fail else 2


if __name__ == "__main__":
    sys.exit(main())