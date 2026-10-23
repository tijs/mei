#!/usr/bin/env python3
"""Model/server-free checks of the real launcher's dry-run argv."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

LAUNCHER = Path(__file__).resolve().with_name("start_mei_server.sh")


class FlintLauncherTests(unittest.TestCase):
    def run_launcher(self, **values):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEI_")}
        env.update(values)
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=env,
                              text=True, capture_output=True, check=False)

    def argv(self, **values):
        result = self.run_launcher(**values)
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(result.stdout.strip())

    def test_gate_default_off_and_explicit_on_off_are_forwarded(self):
        for value in (None, "true", "false"):
            args = self.argv(**({} if value is None else {"MEI_ENABLE_FLINT": value}))
            self.assertEqual(args[args.index("--enable-flint") + 1], value or "false")

    def test_no_unrequested_sampler_or_reasoning_overrides(self):
        args = self.argv(MEI_ENABLE_FLINT="true")
        for flag in ("--temperature", "--top-p", "--top-k", "--enable-thinking",
                     "--reasoning-effort", "--repetition-penalty"):
            self.assertNotIn(flag, args)

    def test_operator_sampler_values_are_forwarded_including_zero(self):
        args = self.argv(MEI_ENABLE_FLINT="true", MEI_TEMPERATURE="0",
                         MEI_TOP_P="0.8", MEI_TOP_K="4")
        for flag, value in (("--temperature", "0"), ("--top-p", "0.8"), ("--top-k", "4")):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_dry_run_does_not_touch_runtime_build_or_model_paths(self):
        with tempfile.TemporaryDirectory(prefix="mei-flint-launcher-") as root:
            args = self.argv(MEI_RUNTIME_BASE=root + "/runtime", MEI_CACHE_ROOT=root + "/cache",
                             MEI_BUILD_DIR=root + "/build", MEI_MODEL_DIR=root + "/missing-model")
            self.assertEqual(list(Path(root).iterdir()), [])
            self.assertEqual(args[args.index("--model-dir") + 1], root + "/missing-model")

    def test_obsolete_profile_still_refuses_before_dry_run(self):
        result = self.run_launcher(MEI_OPTIMIZATION_PROFILE="auto")
        self.assertEqual(result.returncode, 2)
        self.assertIn("MEI_MODEL_PROFILE", result.stderr)


if __name__ == "__main__":
    unittest.main()
