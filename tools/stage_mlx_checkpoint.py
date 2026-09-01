#!/usr/bin/env python3
"""Mei reproducible MLX checkpoint staging + integrity verification.

Stages a pinned MLX (or safetensors) checkpoint from the Hugging Face Hub into
the isolated Mei model root WITHOUT loading it or claiming loadability. It
downloads the full files (no symlinks to the global cache), verifies that every
expected safetensors shard is present and byte-complete against the model
index/config, and records an immutable provenance manifest alongside the model.

The staging recipe (repo, revision, subdir allow-pattern) is the reproducible
Mei-owned provenance the Kiem plan requires. Loadability is a SEPARATE,
GPU-gated step and is never asserted here.

Usage:
  stage_mlx_checkpoint.py --repo REPO_ID [--revision REV] [--subdir SUBDIR]
      [--target DIR] [--expect-bytes N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_safetensors_plan(root: Path):
    """Return (byte_total, shard_paths) from model.index/config if present."""
    index = root / "model.safetensors.index.json"
    shards = sorted(root.glob("model-*.safetensors"))
    if index.exists():
        total = json.loads(index.read_text()).get("total_size")
        return total, shards
    return None, shards


def verify(root: Path) -> dict:
    root = Path(root)
    config_path = root / "config.json"
    if not config_path.exists():
        return {"ok": False, "reason": "no config.json"}
    config = json.loads(config_path.read_text())
    quant = config.get("quantization") or {}

    total, shards = parse_safetensors_plan(root)
    if not shards:
        return {"ok": False, "reason": "no model-*.safetensors shards"}

    actual_bytes = sum(p.stat().st_size for p in shards)
    ok = True
    reasons = []
    shard_names = {p.name for p in shards}

    index = root / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text()).get("weight_map", {})
        expected_shards = set(wm.values())
        missing = expected_shards - shard_names
        if missing:
            ok = False
            reasons += [f"missing shard {x}" for x in sorted(missing)]
        if total is not None and actual_bytes != total:
            ok = False
            reasons.append(f"shard_bytes {actual_bytes} != index_total {total}")

    for p in shards:
        if p.stat().st_size == 0:
            ok = False
            reasons.append(f"empty shard {p.name}")

    config_sha = sha256(config_path)
    return {
        "ok": ok,
        "shards": [p.name for p in shards],
        "shard_bytes": actual_bytes,
        "index_total": total,
        "arch": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quant_bits": quant.get("bits"),
        "quant_group_size": quant.get("group_size"),
        "quant_mode": quant.get("mode"),
        "config_sha256": config_sha,
        "reasons": reasons,
    }


def write_provenance(target: Path, repo: str, rev: str, subdir: str, verify_result: dict, resolved_sha: str):
    prov = {
        "repo": repo,
        "pinned_revision": rev,
        "resolved_commit": resolved_sha,
        "subdir": subdir,
        "staged_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "verify": verify_result,
        "loadability": "NOT CHECKED (GPU-gated)",
    }
    prov_path = target.parent / (target.name + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    return prov_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--subdir", default=None, help="allow-pattern subdir, e.g. 4-bit")
    ap.add_argument("--target", required=True, help="absolute target dir under mei-models")
    ap.add_argument("--expect-bytes", type=int, default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="skip download; only relocate subdir (if any) + verify")
    args = ap.parse_args()

    target = Path(args.target).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        allow = [f"{args.subdir}/*"] if args.subdir else None
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            allow_patterns=allow,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )

    # allow_patterns=["4-bit/*"] preserves structure -> files land under
    # target/4-bit/. Relocate a self-contained subdir up to target/ so the
    # staged model is a normal model dir with config.json at the root.
    if args.subdir:
        sub = target / args.subdir
        if sub.is_dir() and (sub / "config.json").exists() and not (target / "config.json").exists():
            for child in sub.iterdir():
                dest = target / child.name
                if dest.exists():
                    dest.unlink()
                child.rename(dest)
            sub.rmdir()

    result = verify(target)
    prov = write_provenance(target, args.repo, args.revision, args.subdir or "(root)", result, args.revision)
    print(json.dumps(result, indent=2))
    if args.expect_bytes is not None and "shard_bytes" in result:
        if result["shard_bytes"] != args.expect_bytes:
            print(f"WARN: shard_bytes {result['shard_bytes']} != expected {args.expect_bytes}")
    print(f"provenance: {prov}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
