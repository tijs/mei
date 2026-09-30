#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmlx_gate_benchmark as gate


class CmlxGateBenchmarkTests(unittest.TestCase):
    def test_prepare_stage_records_fixed_binary_and_library_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "mei"
            source = root / "source.metallib"
            stock = root / "stock.metallib"
            source_prov = root / "source.provenance"
            stock_prov = root / "stock.provenance"
            binary.write_bytes(b"fixed-binary")
            source.write_bytes(b"source-library")
            stock.write_bytes(b"stock-library")
            source_prov.write_text("sha256: " + gate.sha256(source) + "\n")
            stock_prov.write_text("label: stock\n")
            args = argparse.Namespace(
                output=root / "out",
                binary=binary,
                source_metallib=source,
                source_provenance=source_prov,
                stock_metallib=stock,
                stock_provenance=stock_prov,
            )
            staged, identity = gate.prepare_stage(args, "source")
            self.assertEqual(staged.read_bytes(), binary.read_bytes())
            self.assertEqual(identity["binary_sha256"], gate.sha256(binary))
            self.assertEqual(identity["metallib_sha256"], gate.sha256(source))
            self.assertEqual((root / "out/stage/source/bin/default.metallib").read_bytes(), source.read_bytes())

    def test_load_results_rehydrates_server_metrics_from_request_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leg = root / "source-performance-1"
            leg.mkdir()
            result = {
                "kind": "performance",
                "status": "passed",
                "server_stopped": True,
                "identity": {"arm": "source"},
                "rows": [{"context": "short", "repeat": 1, "wall_seconds": 1.0}],
            }
            (leg / "result.json").write_text(json.dumps(result))
            (leg / "request.jsonl").write_text(json.dumps({"kind": "completion", "prompt_tokens": 5, "completion_tokens": 8}) + "\n" + json.dumps({"kind": "completion", "prompt_tokens": 13, "prefill_ms": 12.5, "generate_ms": 34.5, "decode_tps": 55.5, "prompt_tps": 100.0, "mem_peak_bytes": 123}) + "\n")
            loaded = gate.load_results(root)
            row = loaded[0]["rows"][0]
            self.assertEqual(row["prefill_ms"], 12.5)
            self.assertEqual(row["generate_ms"], 34.5)
            self.assertEqual(row["decode_tps"], 55.5)
            self.assertEqual(row["mei_memory_peak_bytes"], 123)

    def test_report_requires_stopped_passed_behavior_and_context_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, arm in (("source-behavior-1", "source"), ("source-behavior-2", "source"), ("stock-behavior-1", "stock")):
                leg = root / name
                leg.mkdir()
                probe = {"status": "passed", "probe_returncode": 0, "probes": {
                    "models_identity": {"status": "passed"}, "mei_status": {"status": "passed"},
                    "plain_completion": {"status": "passed", "content": "ready"},
                    "parity_stream_vs_nonstream": {"status": "passed", "content": "parity-ok"},
                    "tool_nonstreaming": {"status": "passed", "validated_call": {"name": "add_numbers", "arguments": {"a": 15, "b": 27}, "finish_reason": "tool_calls"}},
                    "tool_streaming": {"status": "passed", "validated_call": {"name": "add_numbers", "arguments": {"a": 15, "b": 27}, "finish_reason": "tool_calls"}},
                    "cache_growing_turn1": {"status": "passed", "usage": {"prompt_tokens_details": {"cached_tokens": 0}}},
                    "cache_growing_turn2_reuses_slot": {"status": "passed", "usage": {"prompt_tokens_details": {"cached_tokens": 785}}},
                    "context_exact_cap": {"status": "passed", "usage": {"prompt_tokens": 65536}},
                    "context_over_cap_rejected": {"status": "passed", "http_status": 400},
                }}
                (leg / "result.json").write_text(json.dumps({"kind": "behavior", "status": "passed", "server_stopped": True, "identity": {"arm": arm}, "probe": probe}))
            summary, _ = gate.report(root)
            self.assertEqual(summary["gate_verdict"], "pass")
            self.assertTrue(summary["checks"]["source_behavior_determinism"])
            self.assertTrue(summary["checks"]["cross_library_tool_valid"])
            self.assertTrue(summary["checks"]["cross_library_context_boundary_equal"])


if __name__ == "__main__":
    unittest.main()
