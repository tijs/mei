#!/usr/bin/env python3
"""Mei reproducible MLX checkpoint staging + integrity verification.

Stages a pinned MLX (or safetensors) checkpoint from the Hugging Face Hub into
the isolated Mei model root WITHOUT loading it or claiming loadability. It
downloads the full files (no symlinks to the global cache), verifies that every
expected safetensors weight file is present, byte-complete, and format-valid
against the model index/config, and records an immutable provenance manifest
alongside the model.

The staging recipe (repo, revision, subdir allow-pattern) is the reproducible
Mei-owned provenance the Kiem plan requires. Loadability is a SEPARATE,
GPU-gated step and is never asserted here.

Layout support
--------------
* Sharded checkpoints use ``model-<NNN>-of-<NNN>.safetensors`` plus a
  ``model.safetensors.index.json``.
* Single-file checkpoints use one ``model.safetensors`` (optionally still with
  an index whose weight_map points only at that file, e.g. Ornith 9B).
* A root may have no index at all (backward compatible): the verifier then
  discovers ``model-*.safetensors`` / ``model.safetensors`` directly.

Index / byte-accounting model
-----------------------------
Hugging Face weight indexes carry keys ``['metadata', 'weight_map']``; the
payload (tensor-data) byte count lives under ``index['metadata']['total_size']``
NOT a top-level ``total_size``. In Mei-produced indexes the top-level key is
absent, so we fall back to ``metadata.total_size`` (and accept either).

``metadata.total_size`` is a PAYLOAD total: it sums tensor data bytes and
excludes each file's ~8-byte header-length prefix, its JSON header and padding.
Bytes on disk (``file_bytes``) therefore always exceed payload by the total
header overhead. We must never compare raw file bytes equal to payload bytes;
instead we report both and only fail when ``file_bytes < payload`` (a truncated
or missing-data file), never when it is merely larger.

Usage:
  stage_mlx_checkpoint.py --repo REPO_ID [--revision REV] [--subdir SUBDIR]
      [--target DIR] [--expect-bytes N]
  stage_mlx_checkpoint.py --verify-only --target DIR
  stage_mlx_checkpoint.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

HEADER_LEN_BYTES = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_files(root: Path):
    """Discover weight files when no index is present (backward compatible).

    Returns sorted name list matching sharded ``model-NNN-of-NNN.safetensors``
    plus a single-file ``model.safetensors``. Only regular files are returned.
    """
    names = set()
    for p in root.glob("model-*.safetensors"):
        if p.is_file():
            names.add(p.name)
    single = root / "model.safetensors"
    if single.is_file():
        names.add(single.name)
    return sorted(names)


def parse_safetensors_plan(root: Path):
    """Return weight-file plan from the model index if present.

    Returns ``(weight_files, payload_total, has_index)`` where ``weight_files``
    is the exact ordered file set the model should be made of, ``payload_total``
    is the tensor-data byte total (``metadata.total_size`` with top-level
    fallback, else ``None``), and ``has_index`` tells whether the set is
    authoritative (from weight_map) or merely discovered.
    """
    index = root / "model.safetensors.index.json"
    if index.exists():
        plan = json.loads(index.read_text())
        weight_map = plan.get("weight_map", {}) or {}
        names = sorted({v for v in weight_map.values() if isinstance(v, str)})
        meta = plan.get("metadata") or {}
        payload = meta.get("total_size")
        if payload is None:
            payload = plan.get("total_size")
        return names, payload, True
    return safetensors_files(root), None, False


def verify_safetensors_file(path: Path):
    """Validate the safetensors container format of a single file.

    Returns ``(ok, reason)``. Checks the 8-byte little-endian header length,
    that the JSON header is parseable and byte-exact, and that every tensor's
    ``data_offsets`` range lies within the file's data region (catches empty,
    truncated, and corrupt files).
    """
    size = path.stat().st_size
    if size == 0:
        return False, "empty file"
    if size < HEADER_LEN_BYTES:
        return False, f"file smaller than {HEADER_LEN_BYTES}-byte header-length field"
    with path.open("rb") as fh:
        raw = fh.read(HEADER_LEN_BYTES)
    header_len = struct.unpack("<Q", raw)[0]
    if header_len > size - HEADER_LEN_BYTES:
        return False, f"header length {header_len} exceeds file size {size}"
    if header_len < 2:
        return False, f"implausible header length {header_len}"
    with path.open("rb") as fh:
        fh.seek(HEADER_LEN_BYTES)
        header_bytes = fh.read(header_len)
    if len(header_bytes) != header_len:
        return False, "truncated JSON header"
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        return False, f"header is not valid JSON: {exc}"
    if not isinstance(header, dict):
        return False, "header is not a JSON object"
    data_len = size - HEADER_LEN_BYTES - header_len
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            return False, f"tensor {name!r} entry is not an object"
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return False, f"tensor {name!r} has malformed data_offsets"
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int):
            return False, f"tensor {name!r} data_offsets not integers"
        if start < 0 or end < start or end > data_len:
            return False, (
                f"tensor {name!r} data range [{start},{end}) "
                f"exceeds data region {data_len}"
            )
    return True, "ok"


def verify(root: Path) -> dict:
    root = Path(root)
    config_path = root / "config.json"
    if not config_path.exists():
        return {"ok": False, "reason": "no config.json"}
    config = json.loads(config_path.read_text())
    quant = config.get("quantization") or {}

    weight_files, payload_total, has_index = parse_safetensors_plan(root)
    if not weight_files:
        return {"ok": False, "reason": "no safetensors weight files found"}

    ok = True
    reasons = []

    # 1. Every expected weight file must exist on disk.
    present = {}
    missing = []
    for name in weight_files:
        p = root / name
        if p.is_file():
            present[name] = p
        else:
            missing.append(name)
    if missing:
        ok = False
        reasons += [f"missing weight file {x}" for x in missing]

    # 2. No stray weight files beyond the authoritative weight_map set.
    #    Scan ALL .safetensors files (not just the model glob) so any extra
    #    weight file -- e.g. a stray.safetensors -- is flagged.
    if has_index:
        on_disk = {p.name for p in root.glob("*.safetensors") if p.is_file()}
        expected = set(weight_files)
        extra = sorted(on_disk - expected)
        if extra:
            ok = False
            reasons += [f"unexpected weight file {x}" for x in extra]

    # 3. Format integrity of every present file (empty/truncated/corrupt).
    file_bytes = 0
    for name in weight_files:
        p = present.get(name)
        if p is None:
            continue
        file_bytes += p.stat().st_size
        valid, why = verify_safetensors_file(p)
        if not valid:
            ok = False
            reasons.append(f"{name}: {why}")

    # 4. Byte accounting. Payload < file_bytes by header overhead; never the
    #    reverse on a complete model, so only fail when file_bytes under-runs
    #    the payload (truncated / missing tensor data).
    header_overhead = None
    if payload_total is not None:
        header_overhead = file_bytes - payload_total
        if header_overhead < 0:
            ok = False
            reasons.append(
                f"file_bytes {file_bytes} < payload total {payload_total} "
                f"(model incomplete / truncated)"
            )

    config_sha = sha256(config_path)
    return {
        "ok": ok,
        "shards": list(present.keys()) or weight_files,
        "weight_files": weight_files,
        "shard_bytes": file_bytes,
        "file_bytes": file_bytes,
        "payload_total": payload_total,
        "index_total": payload_total,
        "header_overhead_bytes": header_overhead,
        "has_index": has_index,
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


def _make_safetensors(tensors, metadata=None, bytes_per=1024):
    """Build a valid safetensors container.

    Returns ``(container_bytes, payload_bytes)`` where ``payload_bytes`` is the
    tensor-data region length (what ``metadata.total_size`` should report) --
    i.e. the file size minus the 8-byte header-length prefix and JSON header.
    tensors: list[(name, numel)].
    """
    header = {}
    offsets = []
    cursor = 0
    for name, numel in tensors:
        nbytes = numel * bytes_per
        offsets.append((name, cursor, nbytes))
        header[name] = {"dtype": "I8", "shape": [nbytes], "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    if metadata:
        header["__metadata__"] = metadata
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data = bytearray(cursor)
    for name, start, nbytes in offsets:
        data[start:start + nbytes] = (name.encode() * (nbytes // max(1, len(name.encode()))))[:nbytes]
    return len(hbytes).to_bytes(HEADER_LEN_BYTES, "little") + hbytes + bytes(data), cursor


def _write_index(root, weight_map, payload_total):
    plan = {"metadata": {"total_size": payload_total}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(json.dumps(plan))
    (root / "config.json").write_text(
        json.dumps({"model_type": "test", "architectures": ["TestModel"], "quantization": {"bits": 4}})
    )


def _mkroot(base, name):
    root = Path(base) / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_self_test() -> int:
    """Deterministic, download-free verification of this verifier itself."""
    failures = []
    lines = []

    def check(label, ok, detail=""):
        if not ok:
            failures.append(label)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))

    base = tempfile.mkdtemp(prefix="mei-stage-selftest-")

    # (a) Sharded with index; payload under metadata.total_size (top-level absent).
    root = _mkroot(base, "a_sharded")
    shard1, p1 = _make_safetensors([("a", 4), ("b", 8)])
    shard2, p2 = _make_safetensors([("c", 16)])
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    (root / "model-00001-of-00002.safetensors").write_bytes(shard1)
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    r = verify(root)
    check("(a) sharded + metadata.total_size ok", r["ok"] is True
          and r["header_overhead_bytes"] is not None and r["header_overhead_bytes"] > 0,
          f"overhead={r['header_overhead_bytes']}")
    check("(a) payload/file bytes reported", r["payload_total"] == p1 + p2
          and r["file_bytes"] == len(shard1) + len(shard2)
          and r["file_bytes"] > r["payload_total"])

    # (b) Single-file model.safetensors with index (Ornith-9B-style).
    root = _mkroot(base, "b_single")
    single, ps = _make_safetensors([("x", 5), ("y", 7)])
    (root / "model.safetensors").write_bytes(single)
    _write_index(root, {"x": "model.safetensors", "y": "model.safetensors"}, payload_total=ps)
    r = verify(root)
    check("(b) single-file model.safetensors ok", r["ok"] is True
          and r["shards"] == ["model.safetensors"] and r["weight_files"] == ["model.safetensors"])

    # (c) Missing shard.
    root = _mkroot(base, "c_missing")
    (root / "model-00001-of-00002.safetensors").write_bytes(shard1)
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    (root / "model-00002-of-00002.safetensors").unlink()
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    r = verify(root)
    check("(c) missing shard rejected", r["ok"] is False
          and any("missing weight file" in x for x in r["reasons"]))

    # (d) Unexpected extra weight file (stray.safetensors).
    root = _mkroot(base, "d_extra")
    (root / "model-00001-of-00002.safetensors").write_bytes(shard1)
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    (root / "stray.safetensors").write_bytes(_make_safetensors([("z", 3)])[0])
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    r = verify(root)
    check("(d) unexpected weight file rejected", r["ok"] is False
          and any("unexpected weight file" in x for x in r["reasons"]))

    # (e) Empty shard.
    root = _mkroot(base, "e_empty")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"")
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    r = verify(root)
    check("(e) empty shard rejected", r["ok"] is False
          and any("empty file" in x for x in r["reasons"]))

    # (f) Truncated / corrupt shards.
    root = _mkroot(base, "f_trunc")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 40)  # implausible header len
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    r = verify(root)
    check("(f) implausible header length rejected", r["ok"] is False
          and any("header length" in x for x in r["reasons"]))
    # file whose header length overruns the file size (truncated shard)
    (root / "model-00001-of-00002.safetensors").write_bytes((99999999).to_bytes(8, "little") + b"{}")
    r = verify(root)
    check("(f2) oversized header length (truncated) rejected", r["ok"] is False
          and any("header length" in x for x in r["reasons"]))
    # valid header length but tensor data range beyond data region (corrupt)
    hj = b'{"t":{"dtype":"I8","shape":[4096],"data_offsets":[0,4096]}}'
    (root / "model-00001-of-00002.safetensors").write_bytes(
        len(hj).to_bytes(8, "little") + hj)
    r = verify(root)
    check("(f3) out-of-range tensor rejected", r["ok"] is False
          and any("exceeds data region" in x for x in r["reasons"]))

    # (g) payload under-report (file_bytes < payload) -> truncated detection.
    root = _mkroot(base, "g_underreport")
    (root / "model-00001-of-00002.safetensors").write_bytes(shard1)
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"},
                 payload_total=p1 + p2 + 1000000)
    r = verify(root)
    check("(g) file_bytes < payload rejected", r["ok"] is False
          and any("incomplete / truncated" in x for x in r["reasons"]))

    # (h) No index present -> discovered sharded files (backward compatible).
    root = _mkroot(base, "h_noidx_shards")
    (root / "model-00001-of-00002.safetensors").write_bytes(shard1)
    (root / "model-00002-of-00002.safetensors").write_bytes(shard2)
    _write_index(root, {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors"}, payload_total=p1 + p2)
    (root / "model.safetensors.index.json").unlink()
    r = verify(root)
    check("(h) no index, discovered shards ok", r["ok"] is True
          and r["has_index"] is False and r["payload_total"] is None
          and r["weight_files"] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])

    # (i) No index, single model.safetensors discovered.
    root = _mkroot(base, "i_noidx_single")
    (root / "model.safetensors").write_bytes(single)
    (root / "config.json").write_text(
        json.dumps({"model_type": "test", "architectures": ["TestModel"]}))
    r = verify(root)
    check("(i) no index, single file discovered ok", r["ok"] is True
          and r["shards"] == ["model.safetensors"])

    # (j) No config.json -> clear failure.
    root = _mkroot(base, "j_noconfig")
    r = verify(root)
    check("(j) no config.json rejected", r["ok"] is False and r["reason"] == "no config.json")

    print("-- stage_mlx_checkpoint self-test --")
    for ln in lines:
        print("  " + ln)
    print(f"RESULT: {'FAIL' if failures else 'PASS'} ({len(failures)} failure(s))")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--subdir", default=None, help="allow-pattern subdir, e.g. 4-bit")
    ap.add_argument("--target", default=None, help="absolute target dir under mei-models")
    ap.add_argument("--expect-bytes", type=int, default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="skip download; only relocate subdir (if any) + verify")
    ap.add_argument("--self-test", action="store_true", help="run deterministic self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.target:
        ap.error("--target is required (unless --self-test)")
    target = Path(args.target).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        if not args.repo:
            ap.error("--repo is required unless --verify-only")
        # Lazy import so verify/self-test paths need no hub dependency.
        from huggingface_hub import snapshot_download
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
    prov = write_provenance(target, args.repo or "(verify-only)", args.revision,
                            args.subdir or "(root)", result, args.revision)
    print(json.dumps(result, indent=2))
    if args.expect_bytes is not None and "file_bytes" in result:
        if result["file_bytes"] != args.expect_bytes:
            print(f"WARN: file_bytes {result['file_bytes']} != expected {args.expect_bytes}")
    print(f"provenance: {prov}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
