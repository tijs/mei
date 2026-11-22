#!/usr/bin/env python3
"""Build the Mei Inkling vocabulary Q4 candidate from the two pinned checkpoints."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

SOURCE_SHA = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"
SOURCE_REV = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
LAYOUT_REV = "26f71eef259a1dd7fc476778d67020351e0dfb8c"
VOCAB = {
    "language_model.model.embed_tokens": "model.llm.embed.weight",
    "language_model.lm_head": "model.llm.unembed.weight",
}
RECIPE = {"bits": 4, "group_size": 64, "mode": "affine"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_header(path):
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(size)), 8 + size


def convert_table(source, key, scratch):
    header, start = read_header(source)
    info = header[VOCAB[key]]
    if info["dtype"] != "BF16" or info["shape"] != [201024, 2304]:
        raise ValueError(f"unexpected source tensor: {key}: {info}")
    rows, width = info["shape"]
    arrays = {}
    for suffix, dtype, cols, itemsize in [
        ("weight", "U32", width // 8, 4),
        ("scales", "BF16", width // 64, 2),
        ("biases", "BF16", width // 64, 2),
    ]:
        name = key + "." + suffix
        arrays[name] = (scratch / name, 0, rows * cols * itemsize, dtype, [rows, cols])
    matrix = np.memmap(
        source,
        dtype="<u2",
        mode="r",
        shape=(rows, width),
        offset=start + info["data_offsets"][0],
    )
    with (
        (scratch / (key + ".weight")).open("xb") as weights,
        (scratch / (key + ".scales")).open("xb") as scales,
        (scratch / (key + ".biases")).open("xb") as biases,
    ):
        for row in range(0, rows, 1024):
            x = mx.array(matrix[row : row + 1024]).view(mx.bfloat16)
            w, s, b = mx.quantize(x, group_size=64, bits=4)
            weights.write(np.asarray(w).tobytes())
            scales.write(np.asarray(s.view(mx.uint16)).tobytes())
            biases.write(np.asarray(b.view(mx.uint16)).tobytes())
    return arrays


def write_shard(path, arrays, metadata):
    header = {}
    offset = 0
    for name, (_, _, size, dtype, shape) in arrays.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header["__metadata__"] = metadata
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(data)) + data)
        for source, position, size, _, _ in arrays.values():
            with source.open("rb") as stream:
                stream.seek(position)
                while size:
                    chunk = stream.read(min(size, 8 << 20))
                    if not chunk:
                        raise EOFError(source)
                    output.write(chunk)
                    size -= len(chunk)
    return offset


def build(source, layout, output):
    if output.exists():
        raise FileExistsError(output)
    if digest(source) != SOURCE_SHA:
        raise ValueError("publisher shard SHA-256 mismatch")
    mx.set_default_device(mx.cpu)
    manifest = layout / "checksums.json"
    if (
        digest(manifest)
        != "aff6c2f840814ad743c7c0648021c6bfa69c58f4c434609aa47d4ed916411ba3"
    ):
        raise ValueError("layout manifest SHA-256 mismatch")
    checked = json.loads(manifest.read_text())
    config = json.loads((layout / "config.json").read_text())
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        scratch = Path(temp)
        bundle = scratch / "bundle"
        bundle.mkdir()
        for name in [
            "LICENSE",
            "chat_template.jinja",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "config.json",
        ]:
            if digest(layout / name) != checked[name]["sha256"]:
                raise ValueError(name)
            if name != "config.json":
                shutil.copyfile(layout / name, bundle / name)
        converted = set()
        for shard in sorted(layout.glob("model-*.safetensors")):
            if digest(shard) != checked[shard.name]["sha256"]:
                raise ValueError(shard.name)
            header, start = read_header(shard)
            metadata = header.pop("__metadata__", {})
            arrays = {}
            for name, info in header.items():
                module = name.removesuffix(".weight")
                if name.endswith(".weight") and module in VOCAB:
                    arrays.update(convert_table(source, module, scratch))
                    converted.add(module)
                else:
                    begin, end = info["data_offsets"]
                    arrays[name] = (
                        shard,
                        start + begin,
                        end - begin,
                        info["dtype"],
                        info["shape"],
                    )
            index["metadata"]["total_size"] += write_shard(
                bundle / shard.name, arrays, metadata
            )
            index["weight_map"].update(dict.fromkeys(arrays, shard.name))
        if converted != set(VOCAB):
            raise ValueError(f"incomplete vocabulary conversion: {converted}")
        for section in ["quantization", "quantization_config"]:
            config[section].update(dict.fromkeys(VOCAB, RECIPE))
        write_json(bundle / "config.json", config)
        write_json(bundle / "model.safetensors.index.json", index)
        write_json(
            bundle / "conversion.json",
            {
                "source": "thinkingmachines/Inkling-2-Mini",
                "source_revision": SOURCE_REV,
                "source_sha256": SOURCE_SHA,
                "layout": "nativ-community/Inkling-2-Mini-nvfp4",
                "layout_revision": LAYOUT_REV,
                "original_bf16_tensors": VOCAB,
                "vocabulary_recipe": RECIPE,
                "expert_policy": "retain original packed payloads and scales",
                "calibration": "MLX per-group affine range, no calibration or evaluation examples",
                "mlx": importlib.metadata.version("mlx"),
                "converter_sha256": digest(Path(__file__)),
            },
        )
        write_json(
            bundle / "checksums.json",
            {
                p.name: {"sha256": digest(p), "size": p.stat().st_size}
                for p in sorted(bundle.iterdir())
            },
        )
        bundle.rename(output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "layout", "output"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    build(args.source, args.layout, args.output)
