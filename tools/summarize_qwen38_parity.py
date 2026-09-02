#!/usr/bin/env python3
"""Report the canonical compact summary for a Qwen3.8-27B-5bit-affine-g64
parity run from its artifact JSONs (probe-load r1..r3, probe-mei, probe-longctx,
probe-coding). Prints provenance JSON (model id, bin, config fingerprint) for
the handoff note -- total 'produce' command + loadability + parity rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-load", nargs="+", required=True)
    ap.add_argument("--probe-mei", required=True)
    ap.add_argument("--probe-longctx", required=True)
    ap.add_argument("--probe-coding", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    for f in args.probe_load:
        d = json.loads(Path(f).read_text())
        rows.append({
            "artifact": Path(f).name,
            "hello_tps": d["probes"]["hello"]["usage"].get("tokens_per_second"),
            "hello_prefill_pps": d["probes"]["hello"]["usage"].get("prompt_tokens_per_second"),
            "short_decode_tps": d["probes"]["short_decode"]["usage"].get("tokens_per_second"),
            "active_GB": round(d["probes"]["hello"]["usage"].get("mei_memory_active_bytes", 0) / 1e9, 2),
            "peak_GB": round(d["probes"]["hello"]["usage"].get("mei_memory_peak_bytes", 0) / 1e9, 2),
        })

    mei = json.loads(Path(args.probe_mei).read_text())
    mei_probes = {
        k: v.get("status") for k, v in mei["probes"].items()
    }
    lc = json.loads(Path(args.probe_longctx).read_text())
    lc_rows = {
        k: {
            "status": v.get("status"),
            "cached": v.get("cached_tokens"),
            "tps": v.get("decode_tokens_per_second"),
            "prefill_pps": v.get("prompt_tokens_per_second"),
            "mem_active_GB": round(v.get("memory_active_bytes", 0) / 1e9, 2),
            "mem_peak_GB": round(v.get("memory_peak_bytes", 0) / 1e9, 2),
        }
        for k, v in lc["probes"].items()
    }
    coding = json.loads(Path(args.probe_coding).read_text())
    coding_rows = {
        k: {"status": v.get("status"), "tps": v.get("decode_tokens_per_second_engine"),
            "failed": v.get("failed_checks")}
        for k, v in coding["probes"].items()
    }

    bin_path = Path(os.path.expanduser(
        "~/.local/share/local-model-bench/mei-build/release/mei"))
    bin_sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()[:16] if bin_path.exists() else None

    print(json.dumps({
        "engine": "mei",
        "model": "Qwen/Qwen3.8-27B-5bit-affine-g64",
        "config": {
            "optimization_profile": "generic",
            "context_cap": 65536,
            "prefill_step_size": 64,
            "kv_bits": None,
            "kv_cache_dir": "disposable default (4119ec5 model-aware)",
            "cache_reuse": True,
            "compiled_decode": False,
            "fork": "91fed8be",
            "binary_sha256_16": bin_sha,
            "config_sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
            "source": "Qwen/Qwen3.8-27B@1d4bf0f2 (5bit affine g64, bf16)",
        },
        "load_rows": rows,
        "probe_mei_overall": mei.get("status"),
        "probe_mei": mei_probes,
        "probe_longctx": lc_rows,
        "probe_coding_overall": coding.get("status"),
        "probe_coding": coding_rows,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())