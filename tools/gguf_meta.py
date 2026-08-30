#!/usr/bin/env python3
"""Read GGUF header metadata (arch, tensor count, key values) without
loading the model — validates llama.cpp compatibility before any GPU use.

Usage: python3 tools/gguf_meta.py path.gguf [--keys general.architecture,general.name]
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def read_gguf_meta(path: Path, keys: list or None = None) -> dict:
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {magic!r}"
        (version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (metadata_kv_count,) = struct.unpack("<Q", f.read(8))

        def read_str():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def read_val():
            (t,) = struct.unpack("<I", f.read(4))
            return read_typed(t)

        def read_typed(t):
            if t == 0:  # uint8
                return f.read(1)[0]
            if t == 1:  # int8
                return struct.unpack("<b", f.read(1))[0]
            if t == 2:  # uint16
                return struct.unpack("<H", f.read(2))[0]
            if t == 3:  # int16
                return struct.unpack("<h", f.read(2))[0]
            if t == 4:  # uint32
                return struct.unpack("<I", f.read(4))[0]
            if t == 5:  # int32
                return struct.unpack("<i", f.read(4))[0]
            if t == 6:  # float32
                return struct.unpack("<f", f.read(4))[0]
            if t == 7:  # bool
                return bool(f.read(1)[0])
            if t == 8:  # string
                return read_str()
            if t == 9:  # array
                (at,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                return [read_typed(at) for _ in range(n)]
            if t == 10:  # uint64
                return struct.unpack("<Q", f.read(8))[0]
            if t == 11:  # int64
                return struct.unpack("<q", f.read(8))[0]
            if t == 12:  # float64
                return struct.unpack("<d", f.read(8))[0]
            if t == 13:  # uint64 (gguf v3)
                return struct.unpack("<Q", f.read(8))[0]
            if t == 14:  # int64 (gguf v3)
                return struct.unpack("<q", f.read(8))[0]
            if t == 15:  # float64 (gguf v3)
                return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"unknown GGUF value type {t}")

        meta = {}
        for _ in range(metadata_kv_count):
            key = read_str()
            if keys is None or any(k in key for k in keys):
                meta[key] = read_val()
            else:
                read_val()  # skip
        return {"magic": "GGUF", "version": version, "tensor_count": tensor_count,
                "metadata_kv_count": metadata_kv_count, "metadata": meta}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--keys", default="general.architecture,general.name,general.file_type,general.size_label,general.alignment,general.quantization_version,llama.attention.head_count_kv")
    args = parser.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    meta = read_gguf_meta(args.path, keys)
    print(json.dumps(meta, indent=1))
    if "general.architecture" in meta["metadata"]:
        arch = meta["metadata"]["general.architecture"]
        if arch not in ("qwen3", "qwen3moe", "qwen3next", "qwen3_hybrid", "qwen3moe_hybrid", "mamba"):
            print(f"NOTE: arch '{arch}' may need a recent llama.cpp build (10470+); "
                  "llama-server will fail loudly at load if unsupported.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())