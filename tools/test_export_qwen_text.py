"""Payload identity and fail-closed metadata checks, with no model/MLX load."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

from export_qwen_text import config_for_text, export


def shard(path, entries):
    header, data = {}, bytearray()
    for key, dtype, shape, payload in entries:
        start = len(data)
        data.extend(payload)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class ExportTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        prefix = "language_model.model.proj"
        shard(
            source / "mixed.safetensors",
            [
                (prefix + ".weight", "U32", [2, 8], bytes(range(64))),
                (prefix + ".scales", "F16", [2, 1], b"abcd"),
                (prefix + ".biases", "F16", [2, 1], b"efgh"),
                ("model.visual.weight", "BF16", [2], b"visn"),
            ],
        )
        shard(
            source / "language.safetensors",
            [("lm_head.weight", "BF16", [2, 2], b"language")],
        )
        shard(source / "mtp.safetensors", [("mtp.fc.weight", "BF16", [2], b"mtp!")])
        config = {
            "model_type": "qwen3_5_moe",
            "vision_config": {"hidden_size": 2},
            "text_config": {"hidden_size": 64, "mtp_num_hidden_layers": 1},
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        }
        (source / "config.json").write_text(json.dumps(config))
        (source / "preprocessor_config.json").write_text("{}")
        (source / "tokenizer.json").write_text("{}")
        return source

    def test_export_preserves_source_and_payload_and_links_unchanged_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            before = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in source.iterdir()
            }
            target = root / "text"
            result = export(source, target)
            self.assertEqual(
                before,
                {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in source.iterdir()
                },
            )
            self.assertFalse((target / "mtp.safetensors").exists())
            self.assertFalse((target / "preprocessor_config.json").exists())
            self.assertTrue((target / "tokenizer.json").is_file())
            self.assertEqual(
                (source / "language.safetensors").stat().st_ino,
                (target / "language.safetensors").stat().st_ino,
            )
            # Read the exported format independently, not through the exporter.
            blob = (target / "mixed.safetensors").read_bytes()
            length = struct.unpack("<Q", blob[:8])[0]
            header = json.loads(blob[8 : 8 + length])
            self.assertNotIn("model.visual.weight", header)
            self.assertEqual(blob[8 + length :], bytes(range(64)) + b"abcdefgh")
            self.assertEqual(len(result["tensors"]), 4)
            config = json.loads((target / "config.json").read_text())
            self.assertNotIn("vision_config", config)
            self.assertEqual(config["text_config"]["mtp_num_hidden_layers"], 0)
            self.assertEqual(config["quantization"], config["quantization_config"])
            with self.assertRaisesRegex(ValueError, "destination must be new"):
                export(source, target)

    def test_omitted_six_bit_entry_is_derived_only_at_declared_group_and_known_width(
        self,
    ):
        prefix = "language_model.model.proj"
        tensors = {
            prefix + ".weight": {"dtype": "U32", "shape": [2, 12]},
            prefix + ".scales": {"dtype": "F16", "shape": [2, 1]},
            prefix + ".biases": {"dtype": "F16", "shape": [2, 1]},
        }
        quant_metadata: dict[str, Any] = {"bits": 4, "group_size": 64, "mode": "affine"}
        root: dict[str, Any] = {
            "model_type": "qwen3_5_moe",
            "hidden_size": 64,
            "quantization": quant_metadata,
        }
        self.assertEqual(
            config_for_text(root, tensors)["quantization"][prefix]["bits"], 6
        )
        self.assertNotIn(prefix, root["quantization"])
        root["quantization"][prefix] = {"bits": 4, "group_size": 64}
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            config_for_text(root, tensors)
        del root["quantization"][prefix]
        root["hidden_size"] = 128
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            config_for_text(root, tensors)

    def test_conflicting_maps_and_unknown_tensor_names_fail_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            config = json.loads((source / "config.json").read_text())
            config["quantization_config"] = {"bits": 8}
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                export(source, root / "text")
            self.assertFalse((root / "text").exists())
            self.assertFalse((root / "text.incomplete").exists())
            shard(
                source / "unexpected.safetensors",
                [("unknown.weight", "BF16", [2], b"oops")],
            )
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                export(source, root / "other")
            self.assertFalse((root / "other").exists())


if __name__ == "__main__":
    unittest.main()
