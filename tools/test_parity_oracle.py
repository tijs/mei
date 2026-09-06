#!/usr/bin/env python3
"""No-network unit/self tests for tools/parity_oracle.py (C1 greedy parity oracle).

Run under the local-model-bench venv (has `tokenizers`):
    /Users/tijs/projects/local-model-bench/.venv/bin/python tools/test_parity_oracle.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_oracle import (  # noqa: E402
    FUSED_GDN_MARKER,
    STRICT_LIMITATION,
    byte_identical,
    canonicalize_tool_calls,
    check_fused_gdn_log,
    compare_legs,
    extract_snapshot,
    message_fields,
    tokenize_ids,
)


def make_tokenizer_json(tmp: str) -> Path:
    """Build a tiny word-level tokenizer.json on disk, no network."""
    from tokenizers import Tokenizer  # type: ignore[import-not-found]
    from tokenizers.models import WordLevel  # type: ignore[import-not-found]
    from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-not-found]

    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = Path(tmp) / "tokenizer.json"
    tok.save(str(path))
    return path


class FieldExtractionTests(unittest.TestCase):
    def test_message_fields_picks_content_reasoning_tools(self):
        fields = message_fields({
            "content": "visible",
            "reasoning_content": "think",
            "tool_calls": [
                {"function": {"name": "f", "arguments": '{"b":2,"a":1}'}}
            ],
        })
        self.assertEqual(fields["content"], "visible")
        self.assertEqual(fields["reasoning_content"], "think")
        self.assertEqual(fields["combined_text"], "think\nvisible")
        self.assertEqual(fields["tool_calls"], [{"name": "f", "arguments": '{"a":1,"b":2}'}])

    def test_message_fields_tolerates_no_reasoning(self):
        fields = message_fields({"content": "only visible"})
        self.assertEqual(fields["reasoning_content"], "")
        self.assertEqual(fields["combined_text"], "only visible")

    def test_message_fields_empty_message(self):
        self.assertEqual(message_fields({})["content"], "")

    def test_snapshot_pulls_usage_and_finish(self):
        snap = extract_snapshot({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 7},
        })
        self.assertEqual(snap["completion_tokens"], 3)
        self.assertEqual(snap["finish_reason"], "stop")

    def test_tool_call_canonicalization_sorts_and_parses(self):
        got = canonicalize_tool_calls([
            {"id": "1", "type": "function", "function": {"name": "g", "arguments": '{"z":0,"a":1}'}},
            {"function": {"name": "f", "arguments": "not-json"}},
        ])
        self.assertEqual(got, [
            {"name": "f", "arguments": "not-json"},
            {"name": "g", "arguments": '{"a":1,"z":0}'},
        ])


class ByteCompareTests(unittest.TestCase):
    def test_byte_identical_true(self):
        ok, diff = byte_identical("abc", "abc")
        self.assertTrue(ok)
        self.assertIsNone(diff)

    def test_byte_identical_false(self):
        ok, diff = byte_identical("abc", "abd")
        self.assertFalse(ok)
        self.assertIn("abc", diff)

    def test_byte_identical_respects_whitespace(self):
        self.assertFalse(byte_identical("a b", "a  b")[0])


class TokenizeReencodeTests(unittest.TestCase):
    def test_reencode_produces_known_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tj = make_tokenizer_json(tmp)
            self.assertEqual(tokenize_ids(tj, "a b c"), [1, 2, 3])
            self.assertEqual(tokenize_ids(tj, "c b a"), [3, 2, 1])


class FusedGdnLogGateTests(unittest.TestCase):
    def test_no_log_is_skipped(self):
        self.assertEqual(check_fused_gdn_log(None)["status"], "skipped")

    def test_missing_log_fails(self):
        self.assertEqual(check_fused_gdn_log(Path("/nonexistent/run.log"))["status"], "failed")

    def test_marker_present_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("mei start\n" + FUSED_GDN_MARKER + "\nloaded\n")
            res = check_fused_gdn_log(log)
            self.assertEqual(res["status"], "passed")
            self.assertTrue(res["found"])

    def test_marker_absent_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("mei start\nno marker here\n")
            self.assertEqual(check_fused_gdn_log(log)["status"], "failed")


class CompareLegsTests(unittest.TestCase):
    @staticmethod
    def snapshot(content="", reasoning="", completion_tokens=0, prompt_tokens=5):
        snap = message_fields({"content": content, "reasoning_content": reasoning})
        snap["completion_tokens"] = completion_tokens
        snap["prompt_tokens"] = prompt_tokens
        return snap

    def test_identical_legs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tj = make_tokenizer_json(tmp)
            base = self.snapshot(content="hello world", reasoning="think here", completion_tokens=3)
            cand = self.snapshot(content="hello world", reasoning="think here", completion_tokens=3)
            res = compare_legs(base, cand, tj)
            self.assertTrue(res["passed"])
            self.assertTrue(res["checks"]["content_byte_identical"])
            self.assertTrue(res["checks"]["reasoning_content_byte_identical"])
            self.assertTrue(res["checks"]["completion_tokens_equal"])
            self.assertTrue(res["checks"]["reencoded_token_ids_equal"])

    def test_content_divergence_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tj = make_tokenizer_json(tmp)
            base = self.snapshot(content="a b", completion_tokens=2)
            cand = self.snapshot(content="a c", completion_tokens=2)
            res = compare_legs(base, cand, tj)
            self.assertFalse(res["passed"])
            self.assertFalse(res["checks"]["content_byte_identical"])
            self.assertFalse(res["checks"]["reencoded_token_ids_equal"])
            self.assertTrue(res["diff"])
            self.assertIn("diverge", " ".join(res["diff"]))

    def test_identical_reencode_can_mask_unknown_word_difference(self):
        # Documents the documented proxy: two *different* strings whose words
        # are both OOV tokenize to identical UNK id sequences => the re-encoded
        # token-ID check alone cannot see the difference (byte compare does).
        with tempfile.TemporaryDirectory() as tmp:
            tj = make_tokenizer_json(tmp)
            base = self.snapshot(content="hello world", completion_tokens=2)
            cand = self.snapshot(content="hello earth", completion_tokens=2)
            res = compare_legs(base, cand, tj)
            self.assertFalse(res["checks"]["content_byte_identical"])
            self.assertTrue(res["checks"]["reencoded_token_ids_equal"])

    def test_completion_token_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tj = make_tokenizer_json(tmp)
            base = self.snapshot(content="hello world", completion_tokens=2)
            cand = self.snapshot(content="hello world", completion_tokens=9)
            res = compare_legs(base, cand, tj)
            self.assertFalse(res["passed"])
            self.assertFalse(res["checks"]["completion_tokens_equal"])


class ArtifactContentTests(unittest.TestCase):
    def test_strict_limitation_makes_no_token_id_claim(self):
        self.assertIn("no token IDs", STRICT_LIMITATION)
        self.assertIn("NOT claimed", STRICT_LIMITATION)
        self.assertIn("client-side re-encoding", STRICT_LIMITATION)

    def test_marker_constant_wired(self):
        self.assertIn("fused_gdn_decode_input_projections=active", FUSED_GDN_MARKER)


if __name__ == "__main__":
    unittest.main(verbosity=2)