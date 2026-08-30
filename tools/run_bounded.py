#!/usr/bin/env python3
"""Bounded foreground supervisor for Mei measurement phases.

Enforces an explicit wall-clock cap on a child command, killing the child's
entire process group (new session) on expiry. macOS has no `timeout`/
`setsid` in stock bash, so the Mei cycle orchestrator supervises phases
through this instead of relying on a detached waiter.

Exit codes (GNU-timeout compatible):
  0      child finished within the cap (exit status propagated)
  124    child killed after the deadline expired
  125    child could not be spawned

Usage:
  python3 tools/run_bounded.py MINUTES -- command args...
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 3 or args[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 125
    try:
        minutes = float(args[0])
    except ValueError:
        print(f"bad minutes: {args[0]}", file=sys.stderr)
        return 125
    if args[1] != "--":
        print("expected '--' separator", file=sys.stderr)
        return 125
    cmd = args[2:]
    deadline = time.monotonic() + minutes * 60.0

    proc = subprocess.Popen(cmd, start_new_session=True)
    while True:
        try:
            code = proc.wait(timeout=min(15.0, max(0.1, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                print(
                    f"[run_bounded] deadline ({minutes:.1f}min) expired; "
                    f"killing '{cmd[0]}' process group {proc.pid}",
                    file=sys.stderr,
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=10)
                return 124
            continue
        return code if code is not None else 124


if __name__ == "__main__":
    raise SystemExit(main())