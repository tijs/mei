#!/usr/bin/env python3
"""Model-free unit tests for tools/f1_c2_benchmark.py arm handling.

No Mei server, no network, no benchmark runs: only the arm -> switch-override
mapping, the argparse surface, and the environment construction are exercised.
Run with the system Python (stdlib only):

    python3 tools/test_f1_c2_benchmark.py
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_c2_benchmark import (  # noqa: E402
    ARM_OVERRIDES,
    SWITCHES,
    arm_overrides,
    build_env,
    build_parser,
)


class ArmOverrideTests(unittest.TestCase):
    def test_candidate_leaves_both_switches_unset(self):
        self.assertEqual(arm_overrides("candidate"), {})

    def test_control_forces_both_switches_on(self):
        self.assertEqual(arm_overrides("control"), {name: "1" for name in SWITCHES})

    def test_optout_forces_both_switches_off(self):
        self.assertEqual(arm_overrides("optout"), {name: "0" for name in SWITCHES})

    def test_every_arm_maps_onto_both_switches_or_none(self):
        for arm, overrides in ARM_OVERRIDES.items():
            self.assertIn(overrides, ({}, {name: "1" for name in SWITCHES}, {name: "0" for name in SWITCHES}), arm)

    def test_overrides_are_fresh_copies(self):
        first = arm_overrides("control")
        first["mutated"] = "1"
        self.assertNotIn("mutated", arm_overrides("control"))
        self.assertNotIn("mutated", ARM_OVERRIDES["control"])

    def test_env_clears_switches_then_applies_arm(self):
        base = {SWITCHES[0]: "1", SWITCHES[1]: "1", "PATH": "/usr/bin"}

        candidate = build_env(arm_overrides("candidate"), base=base)
        for name in SWITCHES:
            self.assertNotIn(name, candidate)
        self.assertEqual(candidate["VMLX_ENABLE_UNSAFE_COMPILE"], "1")
        self.assertEqual(candidate["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"], "0")
        self.assertEqual(candidate["PATH"], "/usr/bin")

        control = build_env(arm_overrides("control"), base={name: "0" for name in SWITCHES})
        self.assertEqual({name: control[name] for name in SWITCHES}, {name: "1" for name in SWITCHES})

        optout = build_env(arm_overrides("optout"), base=base)
        self.assertEqual({name: optout[name] for name in SWITCHES}, {name: "0" for name in SWITCHES})

    def test_base_env_is_not_mutated(self):
        base = {SWITCHES[0]: "1", "KEEP": "yes"}
        build_env(arm_overrides("optout"), base=base)
        self.assertEqual(base, {SWITCHES[0]: "1", "KEEP": "yes"})


class ParserTests(unittest.TestCase):
    REQUIRED = [
        "--binary", "mei",
        "--model-dir", "model",
        "--model-id", "id",
        "--probe", "probe.py",
        "--output", "out",
        "--port", "8080",
    ]

    def parse(self, *extra):
        return build_parser().parse_args(self.REQUIRED + list(extra))

    def test_arm_choices_cover_exactly_the_override_table(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for arm in ARM_OVERRIDES:
                self.assertEqual(self.parse("--arm", arm).arm, arm)

    def test_optout_arm_is_accepted(self):
        args = self.parse("--arm", "optout")
        self.assertEqual(arm_overrides(args.arm), {name: "0" for name in SWITCHES})

    def test_unknown_arm_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--arm", "compiled")

    def test_missing_arm_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse()


if __name__ == "__main__":
    unittest.main()
