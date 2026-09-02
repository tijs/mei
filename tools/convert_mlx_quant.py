#!/usr/bin/env python3
"""Mei-owned reproducible MLX quant conversion wrapper.

Converts a pinned HF source model (hub repo id or local source tree) to a
regular affine (or mxfp4/nvfp4/mxfp8) MLX quant via ``mlx_lm.convert`` and
records an immutable provenance JSON next to the output ONLY after a
successful conversion.

The wrapper is deliberately narrow and honest:

* It NEVER claims UD (Unsloth Dynamic) equivalence and NEVER labels the
  produced artifact as GGUF-derived. The produced quant is a plain affine
  quant of the ORIGINAL model source, not a GGUF class-map port. The
  disclaimer is embedded in the provenance JSON and printed on every plan.
* It distinguishes, in one provenance document: source repo, source revision,
  source tree digest (structural: sorted name+size listing -- documented as
  such, NOT a content hash; use ``stage_mlx_checkpoint.py`` for content-level
  integrity), quant recipe (bits/group/mode/dtype), converter version
  (mlx_lm + mlx), output path, and the exact command.
* A ``--dry-run`` planning mode validates inputs and the disk requirement
  WITHOUT downloading, converting, or deleting anything.
* The disk guard refuses to start when the estimated source + output +
  >= ``--min-free-gib`` free floor cannot fit. Source cache is NEVER
  auto-deleted; ``--delete-source-cache`` is an explicit opt-in that runs
  only after the provenance file is safely written.
* Refuses to write into a non-empty output directory and refuses to overwrite
  an existing provenance file unless ``--force``.

Exit codes:
  0  success (or dry-run plan that would proceed)
  2  usage/validation error (bad quant args, bad source identity, unresolved
     converter, non-empty output, existing provenance without --force)
  3  safety refusal -- the disk guard cannot be proven safe (missing byte
     estimates for hub plans, or the requirement exceeds free space)
  *  non-zero converter exit code is passed through unchanged

Provenance schema: ``mei.convert-mlx-quant/provenance-v1`` (see
``PROVENANCE_SCHEMA``).

Usage:
  convert_mlx_quant.py --source SRC [--revision REV] --output DIR
      [--q-bits N] [--q-group-size N] [--q-mode MODE] [--dtype DTYPE]
      [--converter PATH] [--dry-run] [--min-free-gib N]
      [--source-bytes N] [--expect-output-bytes N]
      [--force] [--delete-source-cache]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MIN_FREE_GIB = 20  # matches the Mei disk-safety policy (note 9beee290)
Q_BITS_MIN, Q_BITS_MAX = 2, 8
VALID_GROUP_SIZES = (32, 64, 128, 256, 512, 1024)
VALID_Q_MODES = ("affine", "mxfp4", "nvfp4", "mxfp8")
VALID_DTYPES = ("float16", "bfloat16", "float32")
OUTPUT_OVERHEAD_BYTES = 256 * 1024 * 1024  # sidecars + non-quantized params
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PROVENANCE_SCHEMA = "mei.convert-mlx-quant/provenance-v1"
UD_DISCLAIMER = (
    "regular MLX affine quant of the original model source - NOT UD "
    "(Unsloth Dynamic), NOT GGUF-derived, no equivalence claim"
)


class UsageError(RuntimeError):
    """A validation or safety check failed before any conversion started."""


def validate_quant_args(q_bits: int, q_group_size: int, q_mode: str, q_dtype: str) -> None:
    if not isinstance(q_bits, int) or isinstance(q_bits, bool) or not (Q_BITS_MIN <= q_bits <= Q_BITS_MAX):
        raise UsageError(f"--q-bits must be an int in [{Q_BITS_MIN}, {Q_BITS_MAX}], got {q_bits!r}")
    if not isinstance(q_group_size, int) or isinstance(q_group_size, bool) or q_group_size not in VALID_GROUP_SIZES:
        raise UsageError(f"--q-group-size must be one of {VALID_GROUP_SIZES}, got {q_group_size!r}")
    if q_mode not in VALID_Q_MODES:
        raise UsageError(f"--q-mode must be one of {VALID_Q_MODES}, got {q_mode!r}")
    if q_dtype not in VALID_DTYPES:
        raise UsageError(f"--dtype must be one of {VALID_DTYPES}, got {q_dtype!r}")


def dir_digest(path: Path):
    """Structural tree digest (sorted ``relpath:size`` sha256) + total bytes.

    The digest covers file NAMES and SIZES only -- it is a reproducible
    identity for the exact file set, NOT a content hash. Content-level
    integrity is the job of ``stage_mlx_checkpoint.py`` / ``sha256``.
    Returns ``(None, 0)`` for a missing path.
    """
    path = Path(path)
    if not path.is_dir():
        return None, 0
    lines = []
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            full = Path(root) / name
            rel = str(full.relative_to(path))
            try:
                size = full.stat().st_size  # follows file symlinks (HF snapshots)
            except OSError:
                size = 0
            total += size
            lines.append(f"{rel}:{size}")
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return digest, total


def validate_source(source: str, revision: str | None) -> dict:
    """Validate source identity. Returns a plan dict.

    Local directory: must exist and contain ``config.json``; revision, when
    given, must be a 40-hex pinned commit. Hub repo id: revision is REQUIRED
    and must be a 40-hex pinned commit.
    """
    if revision is not None and not REVISION_PATTERN.match(revision):
        raise UsageError(
            f"revision must be a pinned 40-hex commit hash, got {revision!r}"
        )
    local = Path(source).expanduser()
    if local.is_dir():
        if not (local / "config.json").is_file():
            raise UsageError(f"local source has no config.json at its root: {local}")
        digest, total = dir_digest(local)
        config_sha = _sha256_file(local / "config.json")
        return {
            "kind": "local",
            "source": str(local.resolve()),
            "local_path": local.resolve(),
            "revision": revision,
            "tree_digest": digest,
            "source_bytes": total,
            "config_sha256": config_sha,
        }
    if local.exists():
        raise UsageError(f"source exists but is not a directory: {source}")
    path_like = source.startswith(("/", "./", "../", "~")) or "/" not in source
    if path_like:
        raise UsageError(
            f"source path does not exist: {source} "
            "(for a hub conversion pass an owner/repo id such as Qwen/Qwen3.8-27B)"
        )
    if revision is None:
        raise UsageError(
            f"hub source requires a pinned --revision (40-hex commit): {source}"
        )
    return {
        "kind": "hub",
        "source": source,
        "local_path": None,
        "revision": revision,
        "tree_digest": None,
        "source_bytes": None,
        "config_sha256": None,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_output_bytes(source_bytes: int, q_bits: int) -> int:
    """Empirical estimate: weights scale by bits/16 plus a fixed overhead.

    Anchors (Qwen3.8-27B, 55,563,006,776 B bf16 source):
    4-bit ~16.05 GB staged; 5-bit expected ~17.5-18.5 GB.
    """
    return int(source_bytes * q_bits / 16) + OUTPUT_OVERHEAD_BYTES


def fits_requirement(
    source_bytes: int,
    output_bytes: int,
    free_bytes: int,
    min_free_gib: int,
    *,
    source_present: bool = False,
) -> bool:
    if min_free_gib < 0:
        raise ValueError("minimum free space cannot be negative")
    needed = output_bytes + min_free_gib * 1024**3
    if not source_present:
        # Hub plan: the source is downloaded inside the run, so the whole
        # source+output+floor must be able to coexist. Local plan: the source
        # already occupies its bytes on disk (sunk cost), so only
        # output+floor must fit on top of the current free space.
        needed += source_bytes
    return free_bytes >= needed


def free_bytes(path) -> int:
    root = Path(path).expanduser()
    if not root.exists():
        root = root.parent
    return shutil.disk_usage(root).free


def build_command(converter: str, source: str, output: str, *, q_bits: int, q_group_size: int, q_mode: str, q_dtype: str) -> list[str]:
    return [
        converter,
        "--hf-path", source,
        "--mlx-path", output,
        "-q",
        "--q-bits", str(q_bits),
        "--q-group-size", str(q_group_size),
        "--q-mode", q_mode,
        "--dtype", q_dtype,
    ]


def resolve_converter(converter: str | None) -> str:
    if converter:
        return converter
    found = shutil.which("mlx_lm.convert")
    if found:
        return found
    raise UsageError(
        "mlx_lm.convert not found on PATH; pass --converter "
        "(e.g. /Users/tijs/projects/local-model-bench/.venv/bin/mlx_lm.convert)"
    )


def converter_versions(converter: str) -> dict:
    """mlx_lm / mlx versions from the interpreter running the converter.

    The converter binary is a script whose shebang names its venv python; the
    wrapper's own interpreter may differ (e.g. system python3 wrapping a
    local-model-bench venv converter). Resolve the shebang interpreter and
    query ITS package metadata; fall back to the wrapper interpreter, then to
    "unknown" if neither has the packages.
    """
    probe = None
    try:
        script = Path(converter).resolve()
        if script.is_file():
            first = script.read_text(encoding="utf-8", errors="replace").splitlines()
            if first and first[0].startswith("#!"):
                shebang = first[0][2:].strip()
                candidate = shebang.split()[0].strip()
                if candidate and Path(candidate).exists():
                    probe = candidate
    except OSError:
        probe = None
    candidates = ([probe] if probe else []) + [sys.executable]
    for interp in candidates:
        try:
            out = subprocess.run(
                [interp, "-c",
                 "import importlib.metadata as m;"
                 "print(m.version('mlx_lm'));print(m.version('mlx'))"],
                check=False, capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if len(lines) == 2:
                return {"mlx_lm": lines[0].strip(), "mlx": lines[1].strip()}
    return {"mlx_lm": "unknown", "mlx": "unknown"}


def provenance_path(output: Path) -> Path:
    return output.parent / (output.name + ".provenance.json")


def provenance_payload(
    plan: dict,
    *,
    versions: dict,
    converter: str,
    output_path: str,
    command: list,
    converted_at_utc: str,
    output_tree_digest: str | None = None,
    output_bytes: int | None = None,
    source_tree_digest: str | None = None,
    source_bytes: int | None = None,
    config_sha256: str | None = None,
    q_bits: int = 5,
    q_group_size: int = 64,
    q_mode: str = "affine",
    q_dtype: str = "bfloat16",
    source_cache_deleted: bool = False,
    source_cache_delete_error: str | None = None,
    source_cache_delete_requested: bool = False,
) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA,
        "label": UD_DISCLAIMER,
        "source": {
            "kind": plan["kind"],
            "repo": plan["source"],
            "revision": plan["revision"],
            "tree_digest": source_tree_digest or plan.get("tree_digest"),
            "bytes": source_bytes if source_bytes is not None else plan.get("source_bytes"),
            "bytes_are_estimate": plan["kind"] == "hub",
            "config_sha256": config_sha256 if config_sha256 is not None else plan.get("config_sha256"),
        },
        "quant_recipe": {
            "bits": q_bits,
            "group_size": q_group_size,
            "mode": q_mode,
            "dtype": q_dtype,
        },
        "converter": {"path": converter, "version": dict(versions)},
        "output": {"path": output_path, "tree_digest": output_tree_digest, "bytes": output_bytes},
        "command": list(command),
        "converted_at_utc": converted_at_utc,
        "claims": {
            "ud_equivalence": False,
            "gguf_derived": False,
            "label": UD_DISCLAIMER,
        },
        "post_conversion": {
            "source_cache_deleted_requested": source_cache_delete_requested,
            "source_cache_deleted": source_cache_deleted,
            "source_cache_delete_error": source_cache_delete_error,
        },
    }


def write_provenance_json(payload: dict, path: Path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise UsageError(f"refusing to overwrite existing provenance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_output_valid(output: Path) -> None:
    if not (output / "config.json").is_file():
        raise UsageError(f"conversion succeeded but output lacks config.json: {output}")
    if not any(output.glob("model-*.safetensors")) and not (output / "model.safetensors").is_file():
        raise UsageError(f"conversion succeeded but output has no weight files: {output}")


def _delete_local_source(local_path: Path) -> tuple[bool, str | None]:
    try:
        shutil.rmtree(local_path)
        return True, None
    except OSError as exc:
        return False, str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="hub repo id or local source tree")
    parser.add_argument("--revision", help="pinned 40-hex commit (required for hub)")
    parser.add_argument("--output", required=True, help="output MLX model directory")
    parser.add_argument("--q-bits", type=int, default=5)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-mode", default="affine", choices=VALID_Q_MODES)
    parser.add_argument("--dtype", default="bfloat16", choices=VALID_DTYPES)
    parser.add_argument("--converter", help="path to mlx_lm.convert (default: PATH lookup)")
    parser.add_argument("--dry-run", action="store_true", help="plan only: no download, no convert, no delete")
    parser.add_argument("--min-free-gib", type=int, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--source-bytes", type=int, help="source tree bytes (required for hub plans)")
    parser.add_argument("--expect-output-bytes", type=int, help="override the output size estimate")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing provenance file")
    parser.add_argument("--delete-source-cache", action="store_true", help="EXPLICIT opt-in: delete a local source tree only after provenance is written")
    return parser


def run(argv: list[str] | None = None, env: dict | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_quant_args(args.q_bits, args.q_group_size, args.q_mode, args.dtype)
        plan = validate_source(args.source, args.revision)
        plan.update(
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
            q_mode=args.q_mode,
            q_dtype=args.dtype,
        )
        converter = resolve_converter(args.converter)
        command = build_command(
            converter, plan["source"], args.output,
            q_bits=args.q_bits, q_group_size=args.q_group_size,
            q_mode=args.q_mode, q_dtype=args.dtype,
        )

        source_bytes = plan.get("source_bytes")
        if args.source_bytes is not None:
            source_bytes = args.source_bytes
        if args.expect_output_bytes is not None:
            output_bytes = args.expect_output_bytes
        elif plan["kind"] == "local":
            assert source_bytes is not None  # measured for local trees
            output_bytes = estimate_output_bytes(source_bytes, args.q_bits)
        else:
            output_bytes = None

        output_path = Path(args.output).expanduser()
        prov_path = provenance_path(output_path)

        print("mei-convert-qlm: PLAN")
        print(f"  source             = {plan['source']} ({plan['kind']})")
        print(f"  source_revision    = {plan['revision']}")
        if plan.get("tree_digest"):
            print(f"  source_tree_digest = {plan['tree_digest']}")
        print(f"  source_bytes       = {source_bytes}")
        print(f"  output             = {output_path}")
        print(f"  output_bytes_est   = {output_bytes}")
        print(f"  recipe             = {args.q_bits}-bit {args.q_mode} g{args.q_group_size} dtype={args.dtype}")
        print(f"  converter          = {converter}")
        print(f"  command            = {' '.join(command)}")
        print(f"  disclaimer         = {UD_DISCLAIMER}")
        print(f"  provenance         = {prov_path}")

        if source_bytes is None or output_bytes is None:
            print(
                "REFUSE: disk requirement cannot be proven for a hub plan; "
                "pass --source-bytes and --expect-output-bytes (see Kiem note "
                "e0eebd31: source 55,563,006,776 B, output ~17.5-19 GB).",
                file=sys.stderr,
            )
            return 3

        available = free_bytes(output_path.parent)
        # A local source is already on disk: its bytes are a sunk cost, so the
        # guard requires output + floor fit on top of the current free space.
        # A hub source is downloaded during the run: source + output + floor
        # must all be able to coexist.
        source_present = plan["kind"] == "local"
        if not fits_requirement(
            source_bytes, output_bytes, available, args.min_free_gib,
            source_present=source_present,
        ):
            needed = output_bytes + args.min_free_gib * 1024**3 + (
                0 if source_present else source_bytes
            )
            print(
                f"REFUSE: disk guard not safe: free={available} B, "
                f"need >= {needed} B ({'output + floor (local source already on disk)' if source_present else 'source + output + floor (hub source will be downloaded)'}). "
                "Free the disk "
                "(mei_disk_guard cleanup, note 9beee290) or pass "
                "--delete-source-cache after the disk is freed, then retry.",
                file=sys.stderr,
            )
            return 3

        if output_path.exists() and any(output_path.iterdir()):
            raise UsageError(f"refusing to convert into non-empty output dir: {output_path}")
        if prov_path.exists() and not args.force:
            raise UsageError(f"refusing to overwrite existing provenance (use --force): {prov_path}")

        if args.dry_run:
            print("DRY-RUN: would convert; nothing was downloaded, converted, or deleted.")
            return 0

        print(f"mei-convert-qlm: running conversion")
        result = subprocess.run(command, env=env)
        if result.returncode != 0:
            print(
                f"mei-convert-qlm: conversion FAILED (rc={result.returncode}); "
                "no provenance written.",
                file=sys.stderr,
            )
            return result.returncode

        _assert_output_valid(output_path)
        out_digest, out_bytes = dir_digest(output_path)

        deleted, delete_error = False, None
        if args.delete_source_cache:
            if plan["kind"] != "local" or plan.get("local_path") is None:
                print("WARNING: --delete-source-cache only applies to a local source tree; nothing deleted.", file=sys.stderr)
            elif plan["local_path"] == output_path.resolve() or output_path.resolve() in plan["local_path"].parents:
                delete_error = "refusing: source contains the output dir"
            else:
                deleted, delete_error = _delete_local_source(plan["local_path"])

        payload = provenance_payload(
            plan,
            versions=converter_versions(converter),
            converter=converter,
            output_path=str(output_path.resolve()) if output_path.exists() else str(output_path),
            command=command,
            converted_at_utc=datetime.now(timezone.utc).isoformat(),
            output_tree_digest=out_digest,
            output_bytes=out_bytes,
            source_tree_digest=plan.get("tree_digest"),
            source_bytes=source_bytes,
            config_sha256=plan.get("config_sha256"),
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
            q_mode=args.q_mode,
            q_dtype=args.dtype,
            source_cache_deleted=deleted,
            source_cache_delete_error=delete_error,
            source_cache_delete_requested=args.delete_source_cache,
        )
        write_provenance_json(payload, prov_path, force=args.force)

        print(f"mei-convert-qlm: DONE rc=0")
        print(f"  output             = {output_path} ({out_bytes} B)")
        print(f"  provenance         = {prov_path}")
        if args.delete_source_cache:
            print(f"  source_cache       = deleted={deleted}" + (f" error={delete_error}" if delete_error else ""))
        return 0

    except UsageError as exc:
        print(f"mei-convert-qlm: refusal: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"mei-convert-qlm: error: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())