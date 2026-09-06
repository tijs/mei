#!/usr/bin/env python3
"""strip_vision_tower.py — produce a text-only MLX checkpoint from a
qwen3_5_moe VLM bundle by removing the vision tower.

Why: Qwen3.6-35B-A3B-4bit ships a 0.83 GiB vision tower that a text-only
coding agent never uses. Its `config.json` differs from the text-only Ornith
sibling by exactly ONE key (`vision_config`), which is what routes it to
vmlx's VLM factory instead of the LLM factory. Dropping the vision tensors AND
that key makes the bundle structurally identical to the known-good Ornith
shape.

Contract:
  - retained tensor PAYLOAD bytes are bit-identical to the source (verified by
    per-tensor sha256, source vs destination)
  - tensors stay back-to-back with no gaps (the safetensors reference
    implementation rejects non-contiguous data_offsets)
  - the data segment start is padded to 8 bytes, matching the convention in
    mei/tools/align_safetensors.py, so vmlx's mmap loader zero-copy maps
    instead of realigning into anonymous RAM
  - shards containing no vision tensors are APFS-cloned (`cp -c`), not rewritten
  - fails loudly on any digest mismatch or residual unaligned tensor

Usage:
  strip_vision_tower.py <src_model_dir> <dst_model_dir> [--dry-run]
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, time

DROP_PREFIXES = ("vision_tower.", "model.visual.")
DROP_SIDECARS = ("preprocessor_config.json", "processor_config.json",
                 "video_preprocessor_config.json")
DTYPE_SIZE = {"F16": 2, "BF16": 2, "F32": 4, "U8": 1, "U16": 2, "U32": 4,
              "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U64": 8, "F64": 8, "BOOL": 1}

def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(hlen)
    return hlen, json.loads(raw.decode("utf-8"))

def sha_region(path, start, length, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError(f"{path}: short read at {start}+{length-left}")
            h.update(b); left -= len(b)
    return h.hexdigest()

def keep(key):
    return not key.startswith(DROP_PREFIXES)

def rewrite_shard(src, dst, dry=False):
    hlen, meta = read_header(src)
    src_data = 8 + hlen
    tensors = [(k, v) for k, v in meta.items() if k != "__metadata__"]
    tensors.sort(key=lambda kv: kv[1]["data_offsets"][0])
    kept = [(k, v) for k, v in tensors if keep(k)]
    dropped = [(k, v) for k, v in tensors if not keep(k)]
    drop_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for _, v in dropped)

    # every retained tensor must be a multiple of 4 bytes, else back-to-back
    # packing cannot keep U32 tensors 4-aligned and we must not insert gaps
    for k, v in kept:
        n = v["data_offsets"][1] - v["data_offsets"][0]
        if n % 4:
            raise SystemExit(f"{src}: {k} is {n} bytes (not a multiple of 4); "
                             "back-to-back packing cannot stay aligned")

    new_meta, cursor = {}, 0
    if "__metadata__" in meta:
        new_meta["__metadata__"] = meta["__metadata__"]
    plan = []
    for k, v in kept:
        n = v["data_offsets"][1] - v["data_offsets"][0]
        new_meta[k] = {"dtype": v["dtype"], "shape": v["shape"],
                       "data_offsets": [cursor, cursor + n]}
        plan.append((k, src_data + v["data_offsets"][0], cursor, n))
        cursor += n

    body = json.dumps(new_meta, separators=(",", ":")).encode("utf-8")
    pad = (-(8 + len(body))) % 8          # data segment start -> multiple of 8
    body += b" " * pad
    if dry:
        return len(kept), len(dropped), drop_bytes, cursor, None

    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(body))); fo.write(body)
        for _, s0, _, n in plan:
            fi.seek(s0); left = n
            while left:
                b = fi.read(min(1 << 22, left))
                if not b: raise IOError(f"{src}: short read")
                fo.write(b); left -= len(b)

    # verify: per-tensor digest src vs dst, and alignment audit
    dst_hlen, dst_meta = read_header(dst)
    dst_data = 8 + dst_hlen
    if dst_data % 8: raise SystemExit(f"{dst}: data start {dst_data} not 8-aligned")
    bad = 0
    for k, s0, d0, n in plan:
        if sha_region(src, s0, n) != sha_region(dst, dst_data + d0, n):
            raise SystemExit(f"{dst}: DIGEST MISMATCH for {k}")
        if (dst_data + d0) % DTYPE_SIZE[dst_meta[k]["dtype"]]:
            bad += 1
    if bad: raise SystemExit(f"{dst}: {bad} unaligned tensors after repack")
    return len(kept), len(dropped), drop_bytes, cursor, dst_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    if os.path.exists(dst) and not a.dry_run:
        raise SystemExit(f"refusing to overwrite existing {dst}")
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    vision_shards = {v for k, v in idx["weight_map"].items() if not keep(k)}
    print(f"src {src}\ndst {dst}")
    print(f"shards {len(shards)}; carrying vision tensors: {sorted(vision_shards)}\n")
    if not a.dry_run:
        os.makedirs(dst)
    t0 = time.time(); total_dropped = 0; total_bytes = 0
    new_map = {}
    for s in shards:
        sp, dp = os.path.join(src, s), os.path.join(dst, s)
        if s in vision_shards:
            k, d, db, size, meta = rewrite_shard(sp, dp, a.dry_run)
            total_dropped += d; total_bytes += db
            print(f"  {s}: rewrote, kept {k}, dropped {d} ({db/2**20:.1f} MiB) -> {size/2**30:.2f} GiB payload")
            if meta:
                for key in meta:
                    if key != "__metadata__": new_map[key] = s
        else:
            if not a.dry_run:
                subprocess.run(["cp", "-c", sp, dp], check=True)   # APFS clone
            print(f"  {s}: cloned unchanged")
            for key, sh in idx["weight_map"].items():
                if sh == s: new_map[key] = s
    if a.dry_run:
        print(f"\nDRY RUN — would drop {total_dropped} tensors / {total_bytes/2**20:.1f} MiB")
        return
    # index.json
    total = sum(os.path.getsize(os.path.join(dst, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    # config.json minus vision_config
    cfg = json.load(open(os.path.join(src, "config.json")))
    removed_keys = [k for k in ("vision_config",) if k in cfg]
    for k in removed_keys: cfg.pop(k)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    # sidecars
    skipped = []
    for f in sorted(os.listdir(src)):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json"):
            continue
        if f in DROP_SIDECARS or f.startswith("MEI_"):
            skipped.append(f); continue
        p = os.path.join(src, f)
        if os.path.isfile(p): shutil.copy2(p, os.path.join(dst, f))
    prov = {
        "tool": "strip_vision_tower.py",
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": src,
        "source_provenance": (json.load(open(src + ".provenance.json"))
                              if os.path.exists(src + ".provenance.json") else None),
        "removed_tensor_prefixes": list(DROP_PREFIXES),
        "removed_tensors": total_dropped,
        "removed_tensor_bytes": total_bytes,
        "removed_config_keys": removed_keys,
        "removed_sidecars": skipped,
        "payload_equivalence": "per-tensor sha256 verified src vs dst for every retained tensor",
        "alignment": "data segment start padded to 8 bytes; 0 unaligned tensors (audited)",
        "claim_limits": ("Text-only derivative. Vision/VLM capability is REMOVED, not "
                         "merely disabled. Not an upstream artifact; do not relabel as "
                         "mlx-community/Qwen3.6-35B-A3B-4bit."),
    }
    json.dump(prov, open(os.path.join(dst, "conversion-provenance.json"), "w"), indent=2)
    print(f"\nOK in {time.time()-t0:.1f}s -> {dst}")
    print(f"  dropped {total_dropped} tensors ({total_bytes/2**20:.1f} MiB)")
    print(f"  removed config keys: {removed_keys}; sidecars: {skipped}")
    print(f"  total size {total/2**30:.2f} GiB")

if __name__ == "__main__":
    main()
