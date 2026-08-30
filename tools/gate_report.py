#!/usr/bin/env python3
"""Deterministic gate-verdict helper for the Mei 40-tok/s-at-45K goal.

Reads sweep + acceptance + survival artifacts and prints, per variant:
  - short-context decode (ctx_512_fresh family): median/min/max
  - 45K loaded fresh (ctx_45000_fresh) and reuse (ctx_45000_reuse_*):
    median/min/max (the gate rows)
  - 40K chat decode
  - prefill ms, cached tokens, allocator active/peak for the gate rows
  - any row with contended_during_row=true is labeled CONTAMINATED
  - variant verdict: MET (>=40), CEILING (25-39), PIVOT (<25)

Usage:
  python3 tools/gate_report.py artifacts/sweep-*.json \
      [--acceptance artifacts/acceptance-variant-*.json] \
      [--survival artifacts/survival-variant-*.json] \
      [--ceiling artifacts/llama-ceiling-*.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

GATE = 40.0
CEILING_MIN = 25.0


def family(name: str) -> str:
    if "reuse" in name:
        return "reuse45"
    if "chat" in name:
        return "chat40k"
    if "ctx_512" in name:
        return "short"
    if "ctx_16384" in name:
        return "ctx16k"
    if "ctx_33175" in name:
        return "ctx33k"
    if "ctx_45000" in name:
        return "fresh45"
    return name


def summarize_sweep(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {"file": path.name, "model": data.get("model"), "cells": []}
    for cell in data.get("cells", []):
        # Sweep cells carry their configuration at the top level (no
        # nested "config" dict); tolerate either shape.
        cfg = cell.get("config", cell)
        fam: dict[str, list] = {}
        contaminated = False
        for r in cell.get("rows", []):
            d = r.get("decode_tps_engine")
            if not d:
                continue
            if r.get("contended_during_row"):
                contaminated = True
            f = family(r.get("name", ""))
            fam.setdefault(f, []).append({
                "decode": float(d),
                "prefill_ms": r.get("prefill_ms"),
                "cached": r.get("cached_tokens") or 0,
                "status": r.get("status"),
            })
        row: dict = {}
        for f, rows in fam.items():
            vals = [x["decode"] for x in rows]
            gate_row = f == "reuse45"
            row[f] = {
                "n": len(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "prefill_ms_median": statistics.median(
                    [x["prefill_ms"] for x in rows if x["prefill_ms"] is not None]) or None,
                "cached_median": statistics.median(x["cached"] for x in rows),
                "contaminated": contaminated,
            }
        verdict = "no-reuse45-row"
        if "reuse45" in row:
            med = row["reuse45"]["median"]
            verdict = "MET" if med >= GATE else ("CEILING" if med >= CEILING_MIN else "PIVOT")
        out["cells"].append({
            "tag": cfg.get("tag"),
            "kv": cfg.get("kv_bits"),
            "compiled": cfg.get("compiled"),
            "threshold": cfg.get("compiled_threshold") or None,
            "window": cfg.get("max_kv_window") or 0,
            "families": row,
            "verdict": verdict,
        })
    return out


def summarize_ceiling(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = {}
    for r in data.get("rows", []):
        d = r.get("decode_tps_engine")
        if not d:
            continue
        f = "reuse45" if "reuse" in r.get("name", "") else "fresh45"
        rows.setdefault(f, []).append(float(d))
    return {
        "file": path.name, "alias": data.get("alias"),
        "engine": data.get("engine"),
        "families": {f: {"n": len(v), "median": statistics.median(v),
                         "min": min(v), "max": max(v)} for f, v in rows.items()},
    }


def summarize_probe(path: Path, kind: str) -> dict:
    data = json.loads(path.read_text())
    probes = data.get("probes") or {}
    passed = all(p.get("status") == "passed" for p in probes.values())
    return {"file": path.name, "kind": kind, "passed": bool(passed),
            "failed": [k for k, p in probes.items() if p.get("status") != "passed"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", nargs="+", type=Path)
    parser.add_argument("--acceptance", nargs="*", type=Path, default=[])
    parser.add_argument("--survival", nargs="*", type=Path, default=[])
    parser.add_argument("--ceiling", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    for path in args.sweeps:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        s = summarize_sweep(path)
        print(f"\n== {s['file']}  (model {s['model']}) ==")
        for cell in s["cells"]:
            komp = f"compiled={cell['compiled']} thresh={cell['threshold']} window={cell['window']} kv={cell['kv']}"
            print(f"  cell {cell['tag']} [{komp}] verdict={cell['verdict']}")
            for f, st in sorted(cell["families"].items()):
                cont = " CONTAMINATED" if st["contaminated"] else ""
                print(f"    {f:10s} n={st['n']} median={st['median']:6.2f} "
                      f"min={st['min']:6.2f} max={st['max']:6.2f} "
                      f"prefill_ms_med={st['prefill_ms_median']} cached_med={st['cached_median']}{cont}")
    for path in args.ceiling:
        c = summarize_ceiling(path)
        print(f"\n== ceiling {c['file']}  ({c['alias']}) ==")
        for f, st in c["families"].items():
            print(f"    {f:10s} n={st['n']} median={st['median']:6.2f} min={st['min']:6.2f} max={st['max']:6.2f}")
    for path in args.acceptance + args.survival:
        p = summarize_probe(path, "acceptance" if "acceptance" in path.name else "survival")
        print(f"  {p['kind']:10s} {p['file']}: {'PASS' if p['passed'] else 'FAIL'} {p['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())