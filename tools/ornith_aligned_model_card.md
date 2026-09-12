---
language: en
library_name: mlx
pipeline_tag: text-generation
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
tags:
- mlx
- safetensors
- qwen3_5_moe
- 4-bit
- mmap-aligned
---

# Ornith-1.5-35B-A3B-MLX-4bit-aligned

The [ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit)
weights, repacked so every tensor starts at a naturally-aligned file offset.
**The weights themselves are bit-identical** — only whitespace padding in each
shard's JSON header changed.

## Why this exists

safetensors stores `data_offsets` relative to the start of the data segment,
which sits at `8 + header_length`. In the published checkpoint that start is not
8-byte aligned, so almost nothing in the file is naturally aligned either. MLX
mmaps what it can and memcpy's the rest into anonymous RAM at load — for this
model, **1,421 of 1,757 tensors** (~97%).

Padding the JSON header with a few bytes of valid whitespace shifts every
absolute tensor offset by the same amount. Align the data segment start and the
whole payload becomes naturally aligned, with **zero payload rewrite**.

## What it buys

Measured on an M-series Mac, same binary, same settings, only the backing store
differing:

| | published checkpoint | this repack |
|---|---|---|
| post-load memory | 24.28 GB | **19.55 GB** |
| fresh prefill @ 80k ctx | 117.6 s | **36.7 s** |
| unaligned tensors | 1,421 / 1,757 | **0** |

That is 4.7 GB of memory back and a 3.2× faster long-context prefill, for a
change that touches no weights.

## What changed, exactly

Per shard, only the JSON header was padded:

| shard | header padding | payload bytes | payload sha256 |
|---|---|---|---|
| `model-00001-of-00004` | 6 bytes | 5,343,636,096 | `ae982af457a3c4cb…` |
| `model-00002-of-00004` | 3 bytes | 5,368,405,504 | `4570d482fbc3ddb5…` |
| `model-00003-of-00004` | 3 bytes | 5,368,256,256 | `411ffa6f850b8f88…` |
| `model-00004-of-00004` | 3 bytes | 3,428,489,600 | `32a9b3e550f03bb3…` |

`MEI_ALIGN_MANIFEST.json` ships with the model and records the source directory,
the target alignment, and each shard's payload hash, so the bit-identical claim
is checkable:

```python
import json, hashlib, os
man = json.load(open("MEI_ALIGN_MANIFEST.json"))
for s in man["shards"]:
    with open(s["file"], "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        f.seek(8 + hlen)                      # skip to the data segment
        h = hashlib.sha256()
        while (b := f.read(1 << 24)):
            h.update(b)
    assert h.hexdigest() == s["payload_sha256"], s["file"]
    assert (8 + hlen) % 8 == 0, "data segment not aligned"
print("payload bit-identical, all shards aligned")
```

## Usage

```bash
mei --model-dir /path/to/Ornith-1.5-35B-A3B-MLX-4bit-aligned \
    --model-profile ornith-1.5-35b-a3b
```

Any MLX-based runtime benefits; nothing here is Mei-specific. The repack was
produced by `tools/align_safetensors.py` in the [Mei](https://github.com/tijs/mei)
repository.

## License and attribution

MIT, following the base model
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
All credit for the model belongs to the Ornith authors; this repository changes
only the file layout. Quantisation to MLX 4-bit was done by
[ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit).
