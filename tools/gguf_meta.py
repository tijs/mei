#!/usr/bin/env python3
"""Read GGUF header/tensor metadata (arch, quant, MTP head, tensor names)
without loading the model — validates llama.cpp compatibility and
comparator hygiene (MTP/nextn heads) before any GPU use.

Usage:
  python3 tools/gguf_meta.py path.gguf
  python3 tools/gguf_meta.py path.gguf --tensors [substring-filter]
  python3 tools/gguf_meta.py path.gguf --check-mtp
"""
import argparse
import json
import struct
import sys
from pathlib import Path

# Keys reported by default; also matched as substrings. nextn_predict_layers
# is the llama.cpp metadata flag for a Next-N/MTP head (arch qwen35).
DEFAULT_KEYS = (
    "general.architecture,general.name,general.file_type,general.size_label,"
    "general.alignment,general.quantization_version,llama.attention.head_count_kv,"
    "nextn_predict_layers,full_attention_interval,block_count,"
    "attention.head_count,attention.key_length,attention.value_length,"
)
SUPPORTED_ARCHS = ("qwen3", "qwen3moe", "qwen3next", "qwen3_hybrid",
                   "qwen3moe_hybrid", "mamba", "qwen35")


def _open_meta(path: Path):
    """Yield (f, version, tensor_count, metadata_kv_count, read_str, read_val)
    positioned right after the metadata section; caller consumes remaining
    keys via read_str/read_val."""
    f = open(path, "rb")
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
        if t in (10, 11, 12, 13, 14, 15):  # u64 / i64 / f64 (v2 and v3 spellings)
            return struct.unpack("<Q" if t in (10, 13) else "<q" if t in (11, 14) else "<d", f.read(8))[0]
        raise ValueError(f"unknown GGUF value type {t}")

    return f, version, tensor_count, metadata_kv_count, read_str, read_val


def read_gguf_meta(path: Path, keys=None) -> dict:
    f, version, tensor_count, metadata_kv_count, read_str, read_val = _open_meta(path)
    with f:
        meta = {}
        for _ in range(metadata_kv_count):
            key = read_str()
            if keys is None or any(k in key for k in keys):
                meta[key] = read_val()
            else:
                read_val()  # skip
        return {"magic": "GGUF", "version": version, "tensor_count": tensor_count,
                "metadata_kv_count": metadata_kv_count, "metadata": meta}


def read_tensor_names(path: Path) -> list:
    """Tensor names only, consuming ~11MB of a multi-GB file (header +
    metadata + tensor-info section); never touches tensor payload data.

    Layout note: GGUF v2 carries a redundant tensor_info_count u64 between
    the metadata section and the tensor-info records; GGUF v3 removed it
    (the header tensor_count is authoritative). Both handled here.
    """
    f, version, tensor_count, metadata_kv_count, read_str, read_val = _open_meta(path)
    with f:
        for _ in range(metadata_kv_count):
            read_str()
            read_val()
        count = tensor_count
        if version < 3:
            (count,) = struct.unpack("<Q", f.read(8))
        # Some v3 writers leave a small (<=16B) pad/alignment between the
        # metadata section and the tensor-info records. Locate the tensor
        # section start by searching for the first plausible string-length +
        # ASCII-name pair instead of trusting the exact boundary.
        pos = f.tell()
        head = f.read(64)
        start = None
        for off in range(len(head) - 16):
            (n,) = struct.unpack("<Q", head[off:off + 8])
            if not 1 <= n <= 240:
                continue
            name_bytes = head[off + 8:off + 8 + n]
            if name_bytes and all(32 <= b < 127 and b not in (b'"', b"'") for b in name_bytes):
                start = pos + off
                break
        if start is None:
            raise ValueError("could not locate the tensor-info section after metadata")
        f.seek(start)
        # GGUF v3 stores tensor dims as u64; v2 used u32.
        dim_width = 8 if version >= 3 else 4
        names = []
        for _ in range(count):
            name = read_str()
            names.append(name)
            (n_dims,) = struct.unpack("<I", f.read(4))
            f.read(dim_width * n_dims)  # dims
            f.read(4)  # tensor type
            f.read(8)  # data offset
        return names


def mtp_report(path: Path, meta: dict) -> None:
    names = read_tensor_names(path)
    arch = str(meta.get("general.architecture", "?"))
    npred = meta.get("qwen35.nextn_predict_layers") or meta.get("nextn_predict_layers")
    nextn = [n for n in names if ".nextn." in n or n.endswith(".nextn") or ".mtp." in n]
    drafter = [n for n in names if ".dft" in n or ".draft" in n or ".spec" in n]
    if npred or nextn or drafter:
        print(f"MTP/Next-N head PRESENT for arch '{arch}': nextn_predict_layers={npred}, "
              f"nextn tensors={len(nextn)}, draft tensors={len(drafter)}", file=sys.stderr)
        for n in sorted(nextn + drafter)[:8]:
            print(f"    {n}", file=sys.stderr)
        print("NOTE: an MTP/Next-N variant is NOT comparable to a no-MTP baseline "
              "(e.g. ornith-ai/Ornith-1.5-9B-MLX-4bit). Run llama.cpp WITHOUT "
              "--spec-type draft-mtp for a single-token decode path.", file=sys.stderr)
    else:
        print(f"MTP/Next-N head: none found for arch '{arch}' (nextn_predict_layers={npred})", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--keys", default=DEFAULT_KEYS)
    parser.add_argument("--tensors", nargs="?", const="", default=None,
                        help="list tensor names (optional substring filter)")
    parser.add_argument("--check-mtp", action="store_true",
                        help="report MTP/Next-N head presence (metadata + tensor names)")
    args = parser.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    meta = read_gguf_meta(args.path, keys)["metadata"]
    print(json.dumps(read_gguf_meta(args.path, keys), indent=1))
    arch = meta.get("general.architecture")
    if arch and arch not in SUPPORTED_ARCHS:
        print(f"NOTE: arch '{arch}' may need a recent llama.cpp build (10470+); "
              "llama-server will fail loudly at load if unsupported.", file=sys.stderr)
    if args.tensors is not None:
        names = read_tensor_names(args.path)
        filt = [n for n in names if args.tensors in n] if args.tensors else names
        print(f"tensors: {len(filt)}/{len(names)} matching {args.tensors or '<all>'}")
        for n in filt:
            print("  ", n)
    if args.check_mtp:
        mtp_report(args.path, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())