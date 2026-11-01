"""Export a Qwen3.5-family text bundle without changing language tensor bytes.

No MLX import or model load. Unchanged shards are hardlinked; mixed shards are
streamed into new files. Source files remain untouched. The manifest records
every retained tensor hash, removed namespace, and quantization metadata change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any

SIZES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1}
SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "README.md",
    "LICENSE",
    "ATTRIBUTIONS.md",
)


def header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= 16 * 1024**2:
            raise ValueError("invalid header length")
        result = json.loads(stream.read(length))
    end = 0
    for key, tensor in sorted(
        ((k, v) for k, v in result.items() if k != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        start, stop = tensor["data_offsets"]
        size = SIZES[tensor["dtype"]]
        if start != end or stop - start != math.prod(tensor["shape"]) * size:
            raise ValueError(f"non-contiguous or invalid tensor: {key}")
        end = stop
    if path.stat().st_size != 8 + length + end:
        raise ValueError("payload size mismatch")
    return 8 + length, result


def retained(key: str) -> bool:
    if key.startswith(("model.visual.", "vision_tower.", "mtp.")):
        return False
    if key.startswith(("language_model.model.", "language_model.lm_head.", "lm_head.")):
        return True
    raise ValueError(f"unrecognized tensor namespace: {key}")


def config_for_text(root: dict[str, Any], tensors: dict[str, Any]) -> dict[str, Any]:
    if root.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("only audited Qwen3.5-family layouts are supported")
    result = copy.deepcopy(root)
    q = result.get("quantization", {})
    if q.get("mode", "affine") != "affine":
        raise ValueError("only ordinary affine quantization is supported")
    modules = {k.removesuffix(".scales") for k in tensors if k.endswith(".scales")}
    text = result.get("text_config", result)
    widths = {
        text[k]
        for k in (
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        )
        if k in text
    }
    if "linear_num_value_heads" in text and "linear_value_head_dim" in text:
        widths.add(text["linear_num_value_heads"] * text["linear_value_head_dim"])
    if "num_attention_heads" in text and "head_dim" in text:
        widths.add(text["num_attention_heads"] * text["head_dim"])
    for name in modules:
        options = q.get(name, q)
        bits, group = options["bits"], options["group_size"]
        if bits not in (2, 3, 4, 6, 8) or type(group) is not int or group <= 0:
            raise ValueError(f"unsupported quantization: {name}")
        weight, scales, biases = (
            tensors[name + suffix] for suffix in (".weight", ".scales", ".biases")
        )
        width = scales["shape"][-1] * group
        if name not in q and weight["shape"][-1] * 32 != width * bits:
            inferred, remainder = divmod(weight["shape"][-1] * 32, width)
            if remainder or inferred not in (2, 3, 4, 6, 8) or width not in widths:
                raise ValueError(f"ambiguous omitted quantization entry: {name}")
            bits = inferred
            q[name] = {"bits": bits, "group_size": group, "mode": "affine"}
        if (
            weight["dtype"] != "U32"
            or scales["shape"] != biases["shape"]
            or weight["shape"][:-1] != scales["shape"][:-1]
            or weight["shape"][-1] * 32 != scales["shape"][-1] * group * bits
        ):
            raise ValueError(f"quantization geometry mismatch: {name}")
    quant = {k: v for k, v in q.items() if not isinstance(v, dict) or k in modules}
    # Converted bundles can retain pre-conversion names in quantization_config.
    # Never merge those stale names into actual tensor metadata. Shared entries
    # must agree; conflicting metadata is refused rather than guessed.
    for name, value in root.get("quantization_config", {}).items():
        canonical = (
            name.replace("model.language_model.", "language_model.model.", 1)
            if name.startswith("model.language_model.")
            else name
        )
        if canonical in quant and quant[canonical] != value:
            raise ValueError(f"conflicting quantization metadata: {name}")
    result["quantization"] = quant
    result["quantization_config"] = copy.deepcopy(quant)
    for key in list(result):
        if key == "vision_config" or key.startswith(("image_", "video_", "vision_")):
            del result[key]
    text = result.get("text_config", result)
    text["mtp_num_hidden_layers"] = 0
    result["architectures"] = [
        "Qwen3_5MoeForCausalLM"
        if root["model_type"].endswith("moe")
        else "Qwen3_5ForCausalLM"
    ]
    return result


def tensor_hash(path: Path, start: int, size: int, destination=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = size
        while remaining:
            block = stream.read(min(8 * 1024**2, remaining))
            if not block:
                raise ValueError("truncated payload")
            digest.update(block)
            if destination is not None:
                destination.write(block)
            remaining -= len(block)
    return digest.hexdigest()


def export(source: Path, destination: Path) -> dict[str, Any]:
    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("destination must be new and outside source")
    inventories = [
        (path, *header(path)) for path in sorted(source.glob("*.safetensors"))
    ]
    tensors = {}
    for _, _, items in inventories:
        for key, value in items.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"duplicate tensor: {key}")
            if retained(key):
                tensors[key] = value
    if not tensors:
        raise ValueError("no language tensors")
    config = config_for_text(json.loads((source / "config.json").read_text()), tensors)
    stage = destination.with_name(destination.name + ".incomplete")
    stage.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "source": str(source),
        "format": 1,
        "tensors": [],
        "removed": [],
        "shards": [],
    }
    try:
        for path, base, original in inventories:
            kept = [
                (k, v)
                for k, v in original.items()
                if k != "__metadata__" and retained(k)
            ]
            manifest["removed"].extend(
                k for k in original if k != "__metadata__" and not retained(k)
            )
            if not kept:
                continue
            target = stage / path.name
            unchanged = len(kept) == len(original) - ("__metadata__" in original)
            if unchanged:
                os.link(path, target)
            else:
                metadata = (
                    {"__metadata__": original["__metadata__"]}
                    if "__metadata__" in original
                    else {}
                )
                offset = 0
                for key, tensor in kept:
                    item = copy.deepcopy(tensor)
                    size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
                    item["data_offsets"] = [offset, offset + size]
                    metadata[key] = item
                    offset += size
                encoded = json.dumps(metadata, separators=(",", ":")).encode()
                encoded += b" " * ((-len(encoded)) % 8)
                with target.open("xb") as stream:
                    stream.write(struct.pack("<Q", len(encoded)))
                    stream.write(encoded)
                    for _, tensor in kept:
                        a, b = tensor["data_offsets"]
                        tensor_hash(path, base + a, b - a, stream)
            new_base, written = header(target)
            for key, tensor in kept:
                a, b = tensor["data_offsets"]
                digest = tensor_hash(path, base + a, b - a)
                new = written[key]
                if (
                    tensor_hash(target, new_base + new["data_offsets"][0], b - a)
                    != digest
                ):
                    raise ValueError(f"payload changed: {key}")
                if (new_base + new["data_offsets"][0]) % SIZES[new["dtype"]]:
                    raise ValueError(f"unaligned export: {key}")
                manifest["tensors"].append(
                    {
                        "key": key,
                        "source_shard": path.name,
                        "shape": tensor["shape"],
                        "dtype": tensor["dtype"],
                        "bytes": b - a,
                        "sha256": digest,
                    }
                )
            manifest["shards"].append(
                {
                    "name": path.name,
                    "bytes": target.stat().st_size,
                    "hardlinked": unchanged,
                }
            )
        for name in SIDECARS:
            if (source / name).is_file():
                shutil.copy2(source / name, stage / name)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest["source_config_sha256"] = hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest()
        manifest["export_config_sha256"] = hashlib.sha256(
            (stage / "config.json").read_bytes()
        ).hexdigest()
        manifest["weight_bytes"] = sum(s["bytes"] for s in manifest["shards"])
        (stage / "MEI_TEXT_EXPORT.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = export(args.source, args.destination)
    print(
        f"Exported {len(result['tensors'])} tensors, {result['weight_bytes']} bytes; source preserved"
    )
