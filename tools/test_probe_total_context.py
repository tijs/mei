"""Model-free negative checks for live boundary evidence classification."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import probe_total_context as probe


class TotalContextTests(unittest.TestCase):
    def run_probe(self, replies):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(probe, "exact_prompt", side_effect=lambda _, n: "x" * n),
                patch.object(probe, "request_json", side_effect=replies) as request,
            ):
                result = probe.check_boundary(
                    "http://localhost/v1",
                    "test",
                    Path(directory),
                    16,
                    10,
                    Path(directory) / "result.json",
                )
            return result, request.call_args_list

    def valid(self):
        return (
            {
                "choices": [{"text": "a"}],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 1,
                    "total_tokens": 16,
                },
            },
            0.5,
        )

    def rejection(self, size):
        return RuntimeError(
            f'HTTP 400: {{"error":"request exceeded context cap: {size} prompt tokens > 15 allowed"}}'
        )

    def test_exact_total_boundary_and_budget_clamp(self):
        result, calls = self.run_probe(
            [self.valid(), self.rejection(16), self.rejection(17)]
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual([len(call.args[1]["prompt"]) for call in calls], [15, 16, 17])
        self.assertEqual(calls[0].args[1]["max_tokens"], 8)

    def test_unrelated_bad_request_cannot_pass(self):
        result, _ = self.run_probe(
            [self.valid(), RuntimeError("HTTP 400: unknown model"), self.rejection(17)]
        )
        self.assertEqual(result["status"], "failed")

    def test_overflow_and_missing_usage_cannot_pass(self):
        for usage in (
            {},
            {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        ):
            with self.subTest(usage=usage):
                result, _ = self.run_probe(
                    [
                        ({"choices": [{}], "usage": usage}, 1),
                        self.rejection(16),
                        self.rejection(17),
                    ]
                )
                self.assertEqual(result["status"], "failed")

    def test_over_cap_success_cannot_pass(self):
        result, _ = self.run_probe([self.valid(), self.valid(), self.rejection(17)])
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
