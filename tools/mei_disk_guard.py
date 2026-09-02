#!/usr/bin/env python3
"""Guard Mei experiments against filling the filesystem with disposable KV caches."""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MIN_FREE_GIB = 20
DISPOSABLE_PREFIXES = ("kv-cache-cell-", "kv-cache-anchors-", "kv-cache-exp-")
DISPOSABLE_EXACT_NAMES = {"kv-cache-sweep"}


class GuardError(RuntimeError):
    """A safety check prevented an unsafe cache operation."""


def _resolved_root(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path.resolve()
    return path.parent.resolve()


def _log_path(root: Path, requested: str | None) -> Path:
    path = Path(requested).expanduser() if requested else root / "mei-disk-guard.log"
    path = path.resolve()
    if path.parent != root.resolve():
        raise GuardError(f"log path must be directly under runtime root: {path}")
    return path


def log_event(root: Path, event: str, status: str, **fields: object) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "mei-disk-guard.log"
    rendered = " ".join(
        f"{key}={str(value).replace(chr(10), ' ')}" for key, value in fields.items()
    )
    line = (
        f"{datetime.now(timezone.utc).isoformat()} event={event} status={status}"
        f"{(' ' + rendered) if rendered else ''}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(_resolved_root(path)).free


def free_space_ok(path: Path, min_free_gib: int = DEFAULT_MIN_FREE_GIB, *, available: int | None = None) -> bool:
    if min_free_gib < 0:
        raise ValueError("minimum free space cannot be negative")
    available = free_bytes(path) if available is None else available
    return available >= min_free_gib * 1024**3


def check_disk(root: Path, min_free_gib: int = DEFAULT_MIN_FREE_GIB, log_file: str | None = None) -> int:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    available = free_bytes(root)
    minimum = min_free_gib * 1024**3
    status = "pass" if available >= minimum else "refuse"
    log_path = _log_path(root, log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} event=disk-check status={status} "
            f"free_bytes={available} min_free_bytes={minimum}\n"
        )
    print(f"mei-disk-guard: status={status} free_bytes={available} min_free_bytes={minimum}")
    return 0 if status == "pass" else 3


def _is_disposable(path: Path) -> bool:
    return path.name in DISPOSABLE_EXACT_NAMES or any(
        path.name.startswith(prefix) for prefix in DISPOSABLE_PREFIXES
    )


def _safe_cache_path(root: Path, cache_dir: Path) -> Path:
    root = root.expanduser().resolve()
    cache_dir = cache_dir.expanduser()
    if cache_dir.is_symlink():
        raise GuardError(f"refusing symlink cache path: {cache_dir}")
    resolved = cache_dir.resolve()
    if resolved.parent != root or resolved == root:
        raise GuardError(f"cache path must be a direct child of runtime root: {resolved}")
    if not _is_disposable(resolved):
        raise GuardError(f"cache path is protected; use an explicit retained experiment policy: {resolved.name}")
    return resolved


def _mei_server_active(root: Path) -> bool:
    pid_file = root / "server.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
        if pid is not None:
            try:
                import subprocess

                command = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except OSError:
                return True
            if command and "mei" in command.lower():
                return True
    try:
        import subprocess

        commands = subprocess.run(
            ["ps", "-axo", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except OSError:
        return True
    for command in commands:
        tokens = command.split()
        if "--kv-cache-dir" not in tokens:
            continue
        if any(token == "mei" or token.endswith("/mei") for token in tokens):
            return True
    return False


def cleanup_cache(
    root: Path,
    cache_dir: Path,
    *,
    retain: bool = False,
    log_file: str | None = None,
    active: bool | None = None,
) -> bool:
    root = root.expanduser().resolve()
    path = _safe_cache_path(root, cache_dir)
    log_path = _log_path(root, log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if retain:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} event=cache-cleanup status=retained path={path.name}\n")
        return False
    if active if active is not None else _mei_server_active(root):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} event=cache-cleanup status=active-skip path={path.name}\n")
        return False
    if not path.exists():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} event=cache-cleanup status=absent path={path.name}\n")
        return False
    if not path.is_dir():
        raise GuardError(f"refusing to remove non-directory cache path: {path}")
    shutil.rmtree(path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} event=cache-cleanup status=removed path={path.name}\n")
    print(f"mei-disk-guard: removed {path}")
    return True


def cleanup_all(root: Path, *, log_file: str | None = None, active: bool | None = None) -> int:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if active if active is not None else _mei_server_active(root):
        log_event(root, "cache-cleanup", "active-skip", scope="all-disposable")
        return 3
    failures = 0
    for child in list(root.iterdir()):
        if not child.is_dir() or not _is_disposable(child):
            continue
        try:
            cleanup_cache(root, child, log_file=log_file, active=False)
        except (GuardError, OSError) as exc:
            failures += 1
            log_event(root, "cache-cleanup", "error", path=child.name, reason=exc)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--runtime-root", required=True, type=Path)
    check.add_argument("--min-free-gib", type=int, default=DEFAULT_MIN_FREE_GIB)
    check.add_argument("--log-file")
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--runtime-root", required=True, type=Path)
    cleanup.add_argument("--cache-dir", type=Path)
    cleanup.add_argument("--all-disposable", action="store_true")
    cleanup.add_argument("--retain", action="store_true")
    cleanup.add_argument("--log-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return check_disk(args.runtime_root, args.min_free_gib, args.log_file)
        if args.cache_dir is not None and args.all_disposable:
            raise GuardError("choose --cache-dir or --all-disposable, not both")
        if args.cache_dir is not None:
            cleanup_cache(args.runtime_root, args.cache_dir, retain=args.retain, log_file=args.log_file)
            return 0
        if args.all_disposable:
            if args.retain:
                raise GuardError("--retain is only valid with --cache-dir")
            return cleanup_all(args.runtime_root, log_file=args.log_file)
        raise GuardError("cleanup requires --cache-dir or --all-disposable")
    except (GuardError, OSError, ValueError) as exc:
        print(f"mei-disk-guard: refusal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
