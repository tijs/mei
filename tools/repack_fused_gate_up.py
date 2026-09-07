#!/usr/bin/env python3
"""repack_fused_gate_up.py — pre-concatenate MoE gate_proj+up_proj on disk.

WHY
---
vmlx's `SwitchGLU.ensureFusedGateUp()` concatenates gate+up at runtime to issue
ONE `gatherQuantizedMM` instead of two at decode. On Ornith that allocated a
permanent +12.2 GiB duplicate (dirty anonymous memory, measured 19.55 -> 31.77 GB
active) and had to be disabled via VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0, which
also gave up the decode fusion.

The duplication is avoidable. Concatenating [E, H, P] + [E, H, P] along the
output axis gives [E, 2H, P] -- exactly the same bytes. The runtime cost exists
only because vmlx builds the concatenation while ALSO keeping the originals
resident. Storing it pre-fused costs ZERO extra disk and ZERO extra RAM.

Verified in MLX before writing this (tools/ notes): gather_qmm accepts a SLICE
of the fused weight directly, so the prefill path can recover gate/up as
zero-copy views, and fused-then-split reproduces the separate calls exactly.

WHAT IT DOES
------------
For every `...switch_mlp.gate_proj.{weight,scales,biases}` / `up_proj.*` pair,
writes `...switch_mlp.gate_up_proj.*` = concat(gate, up) along the output axis,
and drops the two originals.

The concatenation is done as a pure BYTE operation, per expert: for a
[E, out, packed] tensor each expert's slab is contiguous, so
fused[e] = gate[e] || up[e]. No dtype interpretation, so it is exact for U32
weights and BF16 scales/biases alike, and needs no bf16 support in numpy.

Usage:
  repack_fused_gate_up.py <src_model_dir> <dst_model_dir> [--dry-run]
"""
import argparse, hashlib, json, os, shutil, struct, sys, time

DTYPE_SIZE = {"F16": 2, "BF16": 2, "F32": 4, "U8": 1, "U16": 2, "U32": 4,
              "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U64": 8, "F64": 8, "BOOL": 1}
PARTS = ("weight", "scales", "biases")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return 8 + n, json.loads(raw.decode("utf-8"))


def nbytes(v):
    n = 1
    for d in v["shape"]:
        n *= d
    return n * DTYPE_SIZE[v["dtype"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-originals", action="store_true",
                    help="also retain gate_proj/up_proj (default). Required unless the "
                         "loader can serve the prefill path from the fused tensor; "
                         "SwitchGLU's gateProj/upProj are non-optional and referenced "
                         "from 200+ sites, so dropping them is not a bounded change.")
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    if os.path.exists(dst) and not a.dry_run:
        raise SystemExit(f"refusing to overwrite existing {dst}")

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))

    # global tensor table
    table = {}   # key -> (shard, data_start, offset, length, dtype, shape)
    for s in shards:
        ds, meta = read_header(os.path.join(src, s))
        for k, v in meta.items():
            if k == "__metadata__":
                continue
            o0, o1 = v["data_offsets"]
            table[k] = (s, ds, o0, o1 - o0, v["dtype"], v["shape"])

    pairs = {}   # prefix -> {part: (gate_key, up_key)}
    for k in table:
        if ".switch_mlp.gate_proj." not in k:
            continue
        prefix, part = k.split(".switch_mlp.gate_proj.")
        up = f"{prefix}.switch_mlp.up_proj.{part}"
        if part in PARTS and up in table:
            pairs.setdefault(prefix, {})[part] = (k, up)
    complete = {p: d for p, d in pairs.items() if set(d) == set(PARTS)}
    print(f"src {src}\ndst {dst}")
    print(f"shards {len(shards)}, tensors {len(table)}, "
          f"fusable MoE layers {len(complete)} (of {len(pairs)} with a gate_proj)")
    if not complete:
        raise SystemExit("no fusable gate_proj/up_proj pairs found")

    # cross-shard check: a pair must live in the same shard to stream cheaply
    cross = [p for p, d in complete.items()
             if len({table[k][0] for kk in d.values() for k in kk}) > 1]
    print(f"pairs spanning multiple shards: {len(cross)}"
          + (" (handled: read from each source shard)" if cross else ""))

    if a.dry_run:
        saved = sum(nbytes({"shape": table[g][5], "dtype": table[g][4]})
                    for d in complete.values() for g, _ in [d["weight"]])
        print(f"\nDRY RUN — would fuse {len(complete)} layers; "
              f"gate weight bytes per model {saved/2**30:.2f} GiB "
              f"(fused tensor is the SAME total size as the two originals)")
        return

    os.makedirs(dst)
    t0 = time.time()
    handles = {s: open(os.path.join(src, s), "rb") for s in shards}
    new_map = {}
    fused_keys = {k for d in complete.values() for kk in d.values() for k in kk}

    for s in shards:
        ds, meta = read_header(os.path.join(src, s))
        keys = [(k, v) for k, v in meta.items() if k != "__metadata__"]
        keys.sort(key=lambda kv: kv[1]["data_offsets"][0])
        plan, new_meta, cursor = [], {}, 0
        if "__metadata__" in meta:
            new_meta["__metadata__"] = meta["__metadata__"]
        for k, v in keys:
            if k in fused_keys and not a.keep_originals:
                continue                                  # emitted as gate_up below
            n = nbytes(v)
            new_meta[k] = {"dtype": v["dtype"], "shape": v["shape"],
                           "data_offsets": [cursor, cursor + n]}
            plan.append(("copy", (s, ds + v["data_offsets"][0], n)))
            cursor += n
        # append this shard's fused tensors, in stable layer order
        for prefix in sorted(complete, key=lambda p: [int(x) if x.isdigit() else x
                                                      for x in p.replace(".", " ").split()]):
            gsh = table[complete[prefix]["weight"][0]][0]
            if gsh != s:
                continue
            for part in PARTS:
                gk, uk = complete[prefix][part]
                gs_, gds, go, gl, gdt, gshape = table[gk]
                us_, uds, uo, ul, udt, ushape = table[uk]
                assert gdt == udt and gshape[0] == ushape[0] and gshape[2:] == ushape[2:], \
                    f"{prefix}/{part}: shape or dtype mismatch {gshape}{gdt} vs {ushape}{udt}"
                E, GH = gshape[0], gshape[1]
                UH = ushape[1]
                per_g, per_u = gl // E, ul // E
                fshape = [E, GH + UH] + list(gshape[2:])
                key = f"{prefix}.switch_mlp.gate_up_proj.{part}"
                new_meta[key] = {"dtype": gdt, "shape": fshape,
                                 "data_offsets": [cursor, cursor + gl + ul]}
                for e in range(E):                        # interleave per expert
                    plan.append(("copy", (gs_, gds + go + e * per_g, per_g)))
                    plan.append(("copy", (us_, uds + uo + e * per_u, per_u)))
                cursor += gl + ul
                new_map[key] = s
        for k in new_meta:
            if k != "__metadata__" and k not in new_map:
                new_map[k] = s

        body = json.dumps(new_meta, separators=(",", ":")).encode("utf-8")
        body += b" " * ((-(8 + len(body))) % 8)           # 8-align the data segment
        outp = os.path.join(dst, s)
        with open(outp, "wb") as fo:
            fo.write(struct.pack("<Q", len(body))); fo.write(body)
            for _, (sh, off, ln) in plan:
                fh = handles[sh]; fh.seek(off); left = ln
                while left:
                    b = fh.read(min(1 << 22, left))
                    if not b:
                        raise IOError(f"{sh}: short read")
                    fo.write(b); left -= len(b)
        print(f"  {s}: {len(new_meta)-('__metadata__' in new_meta)} tensors, "
              f"{cursor/2**30:.2f} GiB payload")
    for fh in handles.values():
        fh.close()

    total = sum(os.path.getsize(os.path.join(dst, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    for f in sorted(os.listdir(src)):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    json.dump({
        "tool": "repack_fused_gate_up.py",
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": src,
        "fused_layers": len(complete),
        "transform": ("switch_mlp.gate_proj.{weight,scales,biases} and up_proj.* "
                      "concatenated along the output axis into "
                      "switch_mlp.gate_up_proj.*"
                      + ("; originals RETAINED" if a.keep_originals else "; originals removed")),
        "size_claim": ("byte-neutral: concat([E,H,P],[E,H,P]) along the output axis "
                       "is [E,2H,P], the same total bytes"),
        "claim_limits": ("Requires a loader that understands gate_up_proj. Without it "
                         "this checkpoint will NOT load. Not an upstream artifact."),
    }, open(os.path.join(dst, "conversion-provenance.json"), "w"), indent=2)
    print(f"\nOK in {time.time()-t0:.1f}s -> {dst}  total {total/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
