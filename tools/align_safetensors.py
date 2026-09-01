#!/usr/bin/env python3
"""align_safetensors.py — repack safetensors shards so every tensor starts at a
naturally-aligned absolute file offset (data segment start aligned to 8 bytes).

Why: vmlx-swift's C++ mmap loader (MLX_SAFETENSORS_MMAP=1) zero-copy maps
aligned tensors but memcpy's every tensor whose absolute offset is not a
multiple of its dtype size. The official Ornith MLX checkpoints pack
U32 4-bit weights and BF16 scales back-to-back after an unaligned JSON header,
so ~100% of the payload gets "realigned" into anonymous RAM (measured:
9B = 5,038,040,064 B; 35B = ~20.1 GB of 21.5 GB). Since safetensors
data_offsets are RELATIVE to the data segment start, padding only the JSON
header (valid JSON whitespace) shifts every absolute tensor offset by the same
amount; aligning the data segment start to 8 bytes makes every U32/BF16/F16
tensor naturally aligned with zero payload rewrite.

Usage:
  align_safetensors.py <src_model_dir> <dst_model_dir>

Outputs:
  - dst shards with header padding (payload bytes bit-identical to src)
  - all non-safetensors sidecar files copied
  - dst/MEI_ALIGN_MANIFEST.json with per-shard provenance + verification

Verification (fails loudly):
  - padded header still parses as JSON
  - payload region sha256 == src payload region sha256
  - alignment audit: 0 unaligned tensors per shard
  - index.json weight_map files all present
"""
import hashlib
import json
import os
import shutil
import struct
import sys
import time


def parse_header(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: too short")
        hlen = struct.unpack("<Q", head)[0]
        header = f.read(hlen)
        if len(header) != hlen:
            raise ValueError(f"{path}: header truncated")
        json.loads(header.decode("utf-8"))  # validity check
    return hlen, header


def sized(dtype):
    return {"F16": 2, "BF16": 2, "F32": 4, "U8": 1, "U16": 2, "U32": 4,
            "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U64": 8, "F64": 8,
            "BOOL": 1}.get(dtype.upper())


def audit(path):
    """Return (total_tensors, unaligned_tensors, unaligned_bytes)."""
    hlen, header = parse_header(path)
    meta = json.loads(header.decode("utf-8"))
    data_start = 8 + hlen
    un_t = 0
    un_b = 0
    n = 0
    for key, v in meta.items():
        if key == "__metadata__":
            continue
        n += 1
        sz = sized(v["dtype"])
        abs0 = data_start + v["data_offsets"][0]
        if abs0 % sz != 0:
            un_t += 1
            un_b += v["data_offsets"][1] - v["data_offsets"][0]
    return n, un_t, un_b


def sha256_region(path, start, length, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        rem = length
        while rem > 0:
            b = f.read(min(chunk, rem))
            if not b:
                raise ValueError("unexpected EOF")
            h.update(b)
            rem -= len(b)
    return h.hexdigest()


def repack_shard(src_path, dst_path):
    hlen, header = parse_header(src_path)
    P = (8 - ((8 + hlen) % 8)) % 8
    if P:
        # Insert P spaces immediately before the final '}' of the JSON header.
        idx = header.rfind(b"}")
        padded = header[:idx] + b" " * P + header[idx:]
    else:
        padded = header
    new_hlen = hlen + P
    src_size = os.path.getsize(src_path)
    payload_len = src_size - 8 - hlen
    # Verify padded header still parses identically.
    meta_a = json.loads(header.decode("utf-8"))
    meta_b = json.loads(padded.decode("utf-8"))
    if meta_a != meta_b:
        raise ValueError(f"{src_path}: padding corrupted header JSON")
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        fout.write(struct.pack("<Q", new_hlen))
        fout.write(padded)
        # Seek past header into a fresh (sparse-zero) region, then stream the
        # payload verbatim. Total = 8 + new_hlen + payload == src_size + P.
        fout.seek(8 + new_hlen)
        fin.seek(8 + hlen)
        shutil.copyfileobj(fin, fout, 1 << 20)
    dst_size = os.path.getsize(dst_path)
    if dst_size != src_size + P:
        raise ValueError(f"{dst_path}: size mismatch {dst_size} != {src_size}+{P}")
    return P, payload_len


def main():
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)
    manifest = {
        "tool": "align_safetensors.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": os.path.abspath(src_dir),
        "target_alignment": 8,
        "shards": [],
    }
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    has_index = os.path.exists(index_path)
    if has_index:
        index_json = json.load(open(index_path))
        shard_names = sorted(set(index_json["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]

    for name in shard_names:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f"missing shard {src}")
        dst = os.path.join(dst_dir, name)
        t0 = time.time()
        P, payload_len = repack_shard(src, dst)
        # verify payload identity + alignment
        src_payload_sha = sha256_region(src, 8 + parse_header(src)[0], payload_len)
        dst_payload_sha = sha256_region(dst, 8 + parse_header(dst)[0], payload_len)
        n, un_t, un_b = audit(dst)
        if src_payload_sha != dst_payload_sha:
            raise SystemExit(f"{name}: payload sha mismatch")
        if un_t != 0:
            raise SystemExit(f"{name}: {un_t} tensors still unaligned ({un_b} B)")
        manifest["shards"].append({
            "file": name,
            "src_header_len": parse_header(src)[0],
            "padding_bytes": P,
            "payload_bytes": payload_len,
            "payload_sha256": dst_payload_sha,
            "tensors": n,
            "unaligned_after": un_t,
            "repack_seconds": round(time.time() - t0, 2),
        })
        print(f"  {name}: padded {P}B, payload {payload_len}B sha={dst_payload_sha[:16]}.. audit unaligned={un_t} ({time.time()-t0:.1f}s)")

    # copy sidecars
    for entry in sorted(os.listdir(src_dir)):
        sp = os.path.join(src_dir, entry)
        if entry.endswith(".safetensors") or entry.endswith(".safetensors.index.json"):
            if entry == "model.safetensors.index.json":
                shutil.copy2(sp, os.path.join(dst_dir, entry))
            continue
        if os.path.isdir(sp):
            shutil.copytree(sp, os.path.join(dst_dir, entry), dirs_exist_ok=True)
        else:
            shutil.copy2(sp, os.path.join(dst_dir, entry))
    with open(os.path.join(dst_dir, "MEI_ALIGN_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {dst_dir}/MEI_ALIGN_MANIFEST.json  ({len(manifest['shards'])} shards)")
    if has_index:
        # index still consistent: all weight_map files exist in dst
        missing = [f for f in shard_names if not os.path.exists(os.path.join(dst_dir, f))]
        if missing:
            raise SystemExit(f"index references missing dst shards: {missing}")
        print("index weight_map: consistent")


if __name__ == "__main__":
    main()