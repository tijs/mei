# Mei 0.12.0

Buffered and streaming chat now honor the OpenAI-compatible `stop` field.
Previously, the chat paths parsed the requested strings and failed to pass
them to the detokenizer.

## Experimental SenseNova Loop

Mei can load a text-only export of
[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B/tree/329a65f701ba8c48b73c30d0cbaa6f0449733673)
with explicit opt-in. The supported export uses 6-bit or 8-bit group-64
affine weights with FP32 activations. The experimental profile allows 4,096
total tokens, 64-token prefill chunks or smaller, one active generation and
unquantized attention KV. Prefix reuse uses the disk tier or can be disabled.

The Q6 prototype passed 9 of 12 held-out synthetic household tool tasks.
It reached 19.57 decode tokens/s median at 8.91 GiB peak on a 32 GiB M1 Max.
The routine-scheduling fixture recorded two unauthorized changes in synthetic
state. Use the profile for text experiments. Deployment as an autonomous
tool model falls outside this qualification.

## Prepare and run the Q6 export

Use the versioned converter from the
[Mei vMLX pin](https://github.com/tijs/vmlx-swift/tree/3c2823b74a917b9c03ee9e46b9c84fb9953606e6/tools/sensenova_loop).
The example assumes `hf`, `uv`, and a local checkout of that pin. The input
and output reside in separate directories. Allow space for both checkpoints.

```sh
revision=329a65f701ba8c48b73c30d0cbaa6f0449733673
model=sensenova/SenseNova-Loop-9B
hf download "$model" --revision "$revision" --local-dir ./SenseNova-Loop-9B-source
curl -fL "https://huggingface.co/api/models/$model/revision/$revision?blobs=true" \
  -o ./SenseNova-Loop-9B-manifest.json
uv run --no-project --python 3.12 --with mlx==0.32.4 \
  /path/to/vmlx-swift/tools/sensenova_loop/convert.py \
  --source ./SenseNova-Loop-9B-source --out ./SenseNova-Loop-9B-Q6 \
  --hub-manifest ./SenseNova-Loop-9B-manifest.json --bits 6 --float32-activation
VMLX_SENSENOVA_FUSED_SINKHORN=1 mei \
  --model-dir ./SenseNova-Loop-9B-Q6 --enable-sensenova-loop true \
  --served-model-id sensenova-loop-9b-q6 --port 8025
```

The converter verifies the original shard digests and retains the source
license, tokenizer, and chat template. The new bundle's FP32 routing and
separate per-visit cache state preserve both passes of the published model.
The Metal opt-in selects the tested faster router implementation.
