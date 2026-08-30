#!/usr/bin/env python3
"""Summarize Mei sweep / ceiling artifacts into compact tables.

Reads one or more sweep_mei.py / llama_ceiling.py JSON artifacts and
prints, per cell / variant:
  - decode tok/s per row (with cached_tokens and prefill ms)
  - median/min/max decode per named row family (ctx_N_fresh, reuse, chat)
  - allocator active/cache/peak for the loaded rows
  - any error/failure rows

Usage:
  python3 tools/summarize_rows.py artifacts/sweep-*.json [more.json...]

Output is plain text for terminal review; the raw JSON artifacts remain
the authoritative record.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


_CTX_RE = re.compile(r"(?:ctx_|ctx)(\d+)|(\d+)k?_")


def family(name: str) -> str:
    # Group medians per context length, not across lengths: lumping
    # ctx_512_fresh and ctx_45000_fresh into one "fresh" family hides the
    # cliff between 16K and 33K.
    m = _CTX_RE.search(name)
    length = (m.group(1) or m.group(2)) if m else ""
    if "reuse" in name:
        return f"reuse@{length}" if length else "reuse"
    if "chat" in name:
        return "chat"
    if "fresh" in name or name.startswith("ctx_"):
        return f"fresh@{length}" if length else "fresh"
    return name


def summarize_sweep(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n== {path.name}  (model {data.get('model')}) ==")
    print(f"   contention: {data.get('contention_boundary')}")
    for cell in data.get("cells", []):
        cfg = cell.get("config", {})
        print(f"  cell {cfg.get('tag', '?')}: ps={cfg.get('prefill_step')} "
              f"ssm={cfg.get('ssm_rederive')} cl={cfg.get('cache_limit_bytes', 0) // 1_000_000_000}g "
              f"kv={cfg.get('kv_bits')} compiled={cfg.get('compiled')}")
        if cell.get("error"):
            print(f"    ERROR: {cell['error']}")
            continue
        rows = cell.get("rows", [])
        if not rows:
            print("    (no rows)")
            continue
        by_family: dict[str, list[float]] = {}
        for r in rows:
            name = r.get("name", "?")
            status = r.get("status")
            decode = r.get("decode_tps_engine") or r.get("decode_tokens_per_second_engine")
            cached = r.get("cached_tokens") or 0
            prefill_ms = r.get("prefill_ms")
            if status != "passed" and status != "error":
                mark = " [FAIL]"
            elif status == "error":
                mark = " [ERR: %s]" % (r.get("error", "?")[:60])
            else:
                mark = ""
            print(f"    {name:26s} {str(decode)[:7]:>8s} t/s cached={cached:>6d} "
                  f"prefill_ms={str(prefill_ms)[:8]:>8s}{mark}")
            if decode:
                fam = family(name)
                by_family.setdefault(fam, []).append(float(decode))
        for fam, vals in sorted(by_family.items()):
            if len(vals) >= 2:
                print(f"    -- {fam}: n={len(vals)} median={statistics.median(vals):.2f} "
                      f"min={min(vals):.2f} max={max(vals):.2f}")


def summarize_ceiling(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n== {path.name}  (llama.cpp ceiling, {data.get('alias')}) ==")
    print(f"   contention: {data.get('contention_boundary')}")
    rows = data.get("rows", [])
    by_family: dict[str, list[float]] = {}
    for r in rows:
        name = r.get("name", "?")
        status = r.get("status")
        decode = r.get("decode_tps_engine")
        cached = r.get("cached_tokens") or 0
        mem = (r.get("mem_after") or {}).get("rss_bytes", 0) / 1e9
        mark = "" if status == "passed" else f" [{status}] {r.get('error', '')[:60]}"
        print(f"    {name:24s} {str(decode)[:7]:>8s} t/s cached={cached:>6d} rss={mem:.1f}GB{mark}")
        if decode:
            by_family.setdefault(family(name.replace("45k", "ctx")), []).append(float(decode))
    for fam, vals in sorted(by_family.items()):
        if len(vals) >= 2:
            print(f"    -- {fam}: n={len(vals)} median={statistics.median(vals):.2f} "
                  f"min={min(vals):.2f} max={max(vals):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.artifacts:
        if not path.exists():
            print(f"missing artifact: {path}", file=sys.stderr)
            continue
        if "ceiling" in path.name:
            summarize_ceiling(path)
        else:
            summarize_sweep(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())