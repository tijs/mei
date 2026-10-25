#!/usr/bin/env python3
"""Build a payload-preserving Mei adapter for the pinned Flint JANGH2 contract.

The original bundle is retained. Only executable quantization metadata is
partitioned: ordinary affine modules remain ordinary; custom banks become explicit
skip entries and are consumed by Mei's separately gated JANGH modules.
"""

import argparse, copy, hashlib, json, math, pathlib, struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def partition(root):
    require(root.get("model_type") == "flint", "not Flint")
    q = root.get("quantization", {})
    require(q == root.get("quantization_config"), "conflicting quantization maps")
    require(
        q.get("bits") == 8 and q.get("group_size") == 64 and q.get("mode") == "affine",
        "ordinary default must be affine8/group64",
    )
    h = root.get("jangtq", {})
    require(
        all(
            h.get(k) == v
            for k, v in {
                "version": 2,
                "packing": "lsb-bitstream",
                "scale_dtype": "float16",
                "codebook_family": "odd-cubic",
                "rotation": "hadamard32",
            }.items()
        ),
        "unsupported JANGH contract",
    )
    books = h.get("codebooks", {})
    for bits in (2, 3, 4):
        b = books.get(str(bits), {})
        a = b.get("alpha", math.nan)
        beta = b.get("beta", math.nan)
        levels = b.get("levels", [])
        require(
            math.isfinite(a) and math.isfinite(beta) and len(levels) == 1 << bits,
            "invalid codebook",
        )
        for i, l in enumerate(levels):
            u = i - ((1 << bits) - 1) / 2
            expected = u * (a + beta * u * u)
            require(
                math.isfinite(l)
                and abs(l - expected) <= max(1, abs(expected)) * 2e-6
                and (i == 0 or l > levels[i - 1]),
                "codebook coefficients disagree",
            )
    expected = {
        f"model.layers.{layer}.mlp.switch_mlp.{role}"
        for layer in range(root["num_hidden_layers"])
        if layer not in root["mlp_only_layers"]
        for role in ("gate_proj", "up_proj", "down_proj")
    }
    custom = {
        k for k, v in q.items() if isinstance(v, dict) and v.get("mode") == "jangtq2"
    }
    require(custom == expected, "incomplete or unexpected packed projection map")
    for name in custom:
        v = q[name]
        require(
            v.get("bits") in (2, 3, 4)
            and v.get("rotation") == "hadamard32"
            and set(v) == {"bits", "rotation", "mode"},
            "invalid packed width/rotation/fields",
        )
    for name, v in q.items():
        if name in custom or name in ("bits", "group_size", "mode"):
            continue
        require(
            name == "model.embed_tokens"
            and v == {"mode": "affine", "bits": 8, "group_size": 64},
            "unexpected ordinary override",
        )
    ordinary = copy.deepcopy(root)
    ordinary.pop("jangtq")
    ordinary["mei_jangh_adapter_version"] = 1
    ordinary["quantization"] = {k: (False if k in custom else v) for k, v in q.items()}
    ordinary["quantization_config"] = copy.deepcopy(ordinary["quantization"])
    return ordinary


def read_headers(source):
    idx = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {}
    files = {}
    widths = {"U32": 4, "F16": 2, "BF16": 2, "F32": 4}
    for filename in sorted(set(idx.values())):
        require(
            pathlib.PurePath(filename).name == filename
            and filename.endswith(".safetensors"),
            "unsafe shard path",
        )
        p = source / filename
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            require(0 < n < 4000000, "invalid header size")
            h = json.loads(f.read(n))
        spans = []
        for k, v in h.items():
            if k == "__metadata__":
                continue
            require(
                k not in result and idx.get(k) == filename, "duplicate/unindexed tensor"
            )
            width = widths.get(v["dtype"])
            shape = v["shape"]
            start, end = v["data_offsets"]
            require(
                width
                and all(type(d) == int and d > 0 for d in shape)
                and 0 <= start < end <= p.stat().st_size - 8 - n
                and end - start == math.prod(shape) * width,
                "invalid tensor span/shape",
            )
            require(
                (8 + n + start) % width == 0,
                "unaligned tensor; preserve original and align a derivative first",
            )
            result[k] = v
            spans.append((start, end))
        spans.sort()
        require(
            all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "overlapping tensors"
        )
        files[filename] = {"bytes": p.stat().st_size, "header_bytes": n}
    require(set(result) == set(idx), "missing indexed tensors")
    return result, files


def validate_headers(config, tensors):
    # Full architecture key accounting: affine8 linears + BF16 exceptions + custom banks.
    expected = {}
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv = config["num_key_value_heads"]
    dim = config["head_dim"]
    experts = config["num_experts"]
    inter = config["moe_intermediate_size"]

    def add(name, shape, dtype):
        expected[name] = (shape, dtype)

    def affine(name, out, inp):
        require(inp % 64 == 0, "affine group mismatch")
        add(name + ".weight", [out, inp // 4], "U32")
        add(name + ".scales", [out, inp // 64], "BF16")
        add(name + ".biases", [out, inp // 64], "BF16")

    affine("lm_head", config["vocab_size"], hidden)
    affine("model.embed_tokens", config["vocab_size"], hidden)
    add("model.norm.weight", [hidden], "BF16")
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        add(p + ".input_layernorm.weight", [hidden], "BF16")
        add(p + ".post_attention_layernorm.weight", [hidden], "BF16")
        for role, out, inp in [
            ("q_proj", heads * dim, hidden),
            ("k_proj", kv * dim, hidden),
            ("v_proj", kv * dim, hidden),
            ("o_proj", hidden, heads * dim),
        ]:
            affine(p + ".self_attn." + role, out, inp)
        for role in ("q_norm", "k_norm"):
            add(p + ".self_attn." + role + ".weight", [dim], "BF16")
        add(p + ".self_attn.softmax_offset", [heads], "BF16")
        dense = (
            config["intermediate_size"]
            if layer in config["mlp_only_layers"]
            else config["shared_expert_intermediate_size"]
        )
        parent = (
            p + ".mlp"
            if layer in config["mlp_only_layers"]
            else p + ".mlp.shared_expert"
        )
        for role, out, inp in [
            ("gate_proj", dense, hidden),
            ("up_proj", dense, hidden),
            ("down_proj", hidden, dense),
        ]:
            affine(parent + "." + role, out, inp)
        if layer not in config["mlp_only_layers"]:
            add(p + ".mlp.gate.weight", [experts, hidden], "BF16")
            for role, out, inp in [
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ]:
                name = p + ".mlp.switch_mlp." + role
                bits = config["quantization"][name]["bits"]
                require(inp % 32 == 0, "packed input alignment")
                add(name + ".tq2_packed", [experts, out, inp * bits // 32], "U32")
                add(name + ".tq2_scales", [experts, out], "F16")
    require(
        set(tensors) == set(expected),
        "architecture key mismatch: " + str(set(tensors) ^ set(expected)),
    )
    for k, (shape, dtype) in expected.items():
        require(
            tensors[k]["shape"] == shape and tensors[k]["dtype"] == dtype,
            "tensor geometry/dtype mismatch: " + k,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("destination", type=pathlib.Path)
    args = ap.parse_args()
    source = args.source.resolve()
    dest = args.destination
    raw = (source / "config.json").read_bytes()
    config = json.loads(raw)
    ordinary = partition(config)
    side = json.loads((source / "jang_config.json").read_text())
    require(
        side.get("format") == "jangtq2"
        and side.get("format_version") == 2
        and side.get("profile") == "JANGH2",
        "sidecar contract mismatch",
    )
    tensors, files = read_headers(source)
    validate_headers(config, tensors)
    require(not dest.exists(), "destination exists; never overwrite an adapter")
    dest.mkdir(parents=True)
    # No executable old-JANG sidecar is exposed to the generic affine loader.
    for path in source.iterdir():
        if (
            path.name not in ("config.json", "jang_config.json", "osaurus.json")
            and not path.name.startswith(".")
            and path.is_file()
        ):
            (dest / path.name).symlink_to(path)
    (dest / "config.json").write_text(json.dumps(ordinary, indent=2) + "\n")
    (dest / "mei_jangh_config.json").write_bytes(raw)
    (dest / "mei_jangh_bundle.json").write_bytes(
        (source / "jang_config.json").read_bytes()
    )
    (dest / "mei_jangh_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "bundle_sha256": hashlib.sha256(
                    (source / "jang_config.json").read_bytes()
                ).hexdigest(),
                "tensor_count": len(tensors),
                "files": files,
                "tensor_payloads_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"destination": str(dest), "tensor_count": len(tensors), "files": files},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
