#!/usr/bin/env python3
"""No-network unit tests for tools/probe_mei.py result schema.

Run: python3 tools/test_probe_mei.py
   (or: python3 -m unittest tools.test_probe_mei -v)

Covers the probe() wrapper's collision-safe result schema and the
aggregate_status() verdict, plus the exact_prompt() tokenizer load. No
server, model, or network is touched: probe() is exercised directly with
synthetic detail dicts and raising functions, exactly the probe shapes
used by the P0 expected-rejection gates (max_tokens conflict, deferred
fields, legacy stream rejection, chat over-cap chats/streams), whose
detail carries the numeric HTTP status of the rejection that was
*expected* and therefore must not be mistaken for the per-probe verdict.
exact_prompt() is exercised with a fake transformers module injected into
sys.modules (no transformers install required) so the tokenizer load
keyword — trust_remote_code=False + fix_mistral_regex=True — and the
token-count arithmetic are pinned deterministically.
"""
import sys
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_mei  # noqa: E402


class _FakeAutoTokenizer:
    """Records the exact from_pretrained call exact_prompt() makes."""

    calls: list[tuple[Path, dict]] = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return _FakeCountTokenizer()


class _FakeCountTokenizer:
    """Deterministic count-based tokenizer.

    Mirrors the staged P0 tokenizers' arithmetic: the unit " hello" is
    exactly one token, and add_special_tokens=True adds one BOS token
    (the Laguna-XS overhead case exact_prompt() measures).
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        count = text.count(" hello")
        return [0] * (count + (1 if add_special_tokens else 0))


def _patched_transformers():
    """Inject a fake `transformers` module so no real install is needed."""
    fake = types.ModuleType("transformers")
    setattr(fake, "AutoTokenizer", _FakeAutoTokenizer)
    return unittest.mock.patch.dict(sys.modules, {"transformers": fake})


class ExactPromptTokenizerTests(unittest.TestCase):
    """exact_prompt() loads the tokenizer warning-free and counts exactly."""

    def test_exact_prompt_loads_with_fix_mistral_regex_and_trust_remote_code_off(self):
        # The central regression: the tokenizer load must pass
        # fix_mistral_regex=True (silences transformers' incorrect-regex
        # warning for the Mistral-derived pre-tokenizer the staged
        # Qwen-lineage tokenizers ship) alongside trust_remote_code=False —
        # and nothing else.
        _FakeAutoTokenizer.calls = []
        with _patched_transformers():
            prompt = probe_mei.exact_prompt(Path("/fake/tokenizer-dir"), 100)
        self.assertEqual(prompt, " hello" * 99)
        (path, kwargs), = _FakeAutoTokenizer.calls
        self.assertEqual(path, Path("/fake/tokenizer-dir"))
        self.assertEqual(kwargs, {"trust_remote_code": False, "fix_mistral_regex": True})

    def test_exact_prompt_rejects_multi_token_unit(self):
        # A tokenizer whose unit is not one token must fail loudly rather
        # than build a mis-measured prompt.
        original = _FakeCountTokenizer.encode

        def broken_encode(self, text, add_special_tokens=False):
            return [0, 0]

        _FakeCountTokenizer.encode = broken_encode
        try:
            with _patched_transformers():
                with self.assertRaises(RuntimeError) as raised:
                    probe_mei.exact_prompt(Path("/fake/tokenizer-dir"), 100)
        finally:
            _FakeCountTokenizer.encode = original
        self.assertIn("not one token", str(raised.exception))

    def test_exact_prompt_rejects_measured_mismatch(self):
        # The final count must equal the target; a tokenizer whose counts
        # drift (e.g. an unpatchable broken regex) fails the gate.
        original = _FakeCountTokenizer.encode

        def drifting_encode(self, text, add_special_tokens=False):
            count = text.count(" hello")
            if count <= 1:
                return [0] * (count + (1 if add_special_tokens else 0))
            return [0] * (len(text) + 1)

        _FakeCountTokenizer.encode = drifting_encode
        try:
            with _patched_transformers():
                with self.assertRaises(RuntimeError) as raised:
                    probe_mei.exact_prompt(Path("/fake/tokenizer-dir"), 100)
        finally:
            _FakeCountTokenizer.encode = original
        self.assertIn("expected 100", str(raised.exception))


class WrapperStatusTests(unittest.TestCase):
    """Per-probe verdict vs returned HTTP metadata stays collision-free."""

    def test_expected_http_400_detail_keeps_probe_passed(self):
        # A probe that anticipates a rejection carries the numeric HTTP
        # status in its detail; the wrapper must keep the per-probe verdict
        # 'passed' and preserve the code as http_status, not clobber status.
        result = {"probes": {}}
        probe_mei.probe(
            "max_tokens_conflict_rejected",
            result,
            lambda: {
                "status": 400,
                "error": {"type": "invalid_request_error", "message": "conflicting max_tokens forms"},
            },
        )
        entry = result["probes"]["max_tokens_conflict_rejected"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["http_status"], 400)
        self.assertEqual(entry["error"]["message"], "conflicting max_tokens forms")
        self.assertIsInstance(entry["elapsed_seconds"], float)

    def test_chat_over_cap_rejection_metadata_preserved(self):
        # The chat (non-streaming and streaming) over-cap probes return
        # {"status": status, "error": ...}; the wrapper renames the numeric
        # status to http_status and keeps the error envelope.
        result = {"probes": {}}
        probe_mei.probe(
            "context_chat_over_cap_rejected",
            result,
            lambda: {"status": 400, "error": {"message": "context cap exceeded"}},
        )
        entry = result["probes"]["context_chat_over_cap_rejected"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["http_status"], 400)
        self.assertEqual(entry["error"]["message"], "context cap exceeded")

    def test_nested_field_statuses_untouched(self):
        # deferred_fields_rejected returns per-field nested entries that
        # carry their own HTTP statuses; only the top-level detail key is
        # reserved, so the nested metadata must survive verbatim.
        result = {"probes": {}}
        checked = {
            "response_format": {"status": 400, "error_message": "not supported"},
            "n": {"status": 400, "error_message": "not supported"},
        }
        probe_mei.probe("deferred_fields_rejected", result, lambda: {"checked": checked})
        entry = result["probes"]["deferred_fields_rejected"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["checked"], checked)
        self.assertNotIn("http_status", entry)

    def test_explicit_http_status_detail_passes_through(self):
        # context_over_cap_rejected records its numeric code explicitly as
        # http_status (no `status` key in detail); the wrapper keeps it.
        result = {"probes": {}}
        probe_mei.probe(
            "context_over_cap_rejected",
            result,
            lambda: {"rejected_as_expected": True, "http_status": 400, "error": "HTTP 400: context length exceeded"},
        )
        entry = result["probes"]["context_over_cap_rejected"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["http_status"], 400)
        self.assertTrue(entry["rejected_as_expected"])
        self.assertIn("HTTP 400", entry["error"])

    def test_reserved_status_key_is_always_owned_by_wrapper(self):
        # Non-numeric or odd `status` values in detail must not clobber the
        # verdict either; the wrapper's reserved keys always win.
        result = {"probes": {}}
        probe_mei.probe("plain_completion", result, lambda: {"status": "ok", "content": "ready"})
        entry = result["probes"]["plain_completion"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["content"], "ready")
        self.assertNotIn("http_status", entry)

    def test_non_dict_detail_is_wrapped(self):
        result = {"probes": {}}
        probe_mei.probe("odd_detail", result, lambda: "not-a-dict")
        entry = result["probes"]["odd_detail"]
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["detail"], "not-a-dict")

    def test_real_failure_records_error_and_failed_verdict(self):
        result = {"probes": {}}

        def failing() -> dict:
            raise AssertionError("server returned empty content")

        probe_mei.probe("plain_completion", result, failing)
        entry = result["probes"]["plain_completion"]
        self.assertEqual(entry["status"], "failed")
        self.assertIn("AssertionError: server returned empty content", entry["error"])
        self.assertIsInstance(entry["elapsed_seconds"], float)


class AggregateStatusTests(unittest.TestCase):
    """Top-level verdict reflects every per-probe verdict."""

    def test_aggregate_passed_when_all_cases_pass(self):
        result = {"probes": {}}
        probe_mei.probe(
            "max_tokens_conflict_rejected",
            result,
            lambda: {"status": 400, "rejected_as_expected": True},
        )
        probe_mei.probe("context_chat_stream_over_cap_rejected", result, lambda: {"status": 400})
        probe_mei.probe("plain_completion", result, lambda: {"content": "ready", "usage": {}})
        self.assertEqual(probe_mei.aggregate_status(result["probes"]), "passed")

    def test_aggregate_failed_when_a_real_failure_occurs(self):
        result = {"probes": {}}
        probe_mei.probe(
            "max_tokens_conflict_rejected",
            result,
            lambda: {"status": 400, "rejected_as_expected": True},
        )

        def failing() -> dict:
            raise AssertionError("server returned empty content")

        probe_mei.probe("plain_completion", result, failing)
        self.assertEqual(
            result["probes"]["plain_completion"]["status"], "failed"
        )
        self.assertEqual(probe_mei.aggregate_status(result["probes"]), "failed")

    def test_aggregate_failed_on_empty_probes(self):
        # Matches the historical main() rule: an empty probe set is not a
        # pass (exit code stays 1).
        self.assertEqual(probe_mei.aggregate_status({}), "failed")

    def test_aggregate_failed_on_skipped_probe(self):
        # A probe explicitly recorded as skipped (e.g. context probes run
        # without --tokenizer) keeps the aggregate from passing.
        result = {"probes": {
            "context_exact_cap": {
                "status": "skipped",
                "elapsed_seconds": 0,
                "error": "no --tokenizer path provided",
            },
        }}
        self.assertEqual(probe_mei.aggregate_status(result["probes"]), "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)