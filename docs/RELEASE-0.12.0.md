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
curl -fL https://raw.githubusercontent.com/OpenSenseNova/SenseNova-Loop/7e5f7cd97fc975fef27e9696be9128e386c0dff3/LICENSE \
  -o ./SenseNova-Loop-9B-Q6/LICENSE
printf 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  ./SenseNova-Loop-9B-Q6/LICENSE\n' \
  | shasum -a 256 -c -
VMLX_SENSENOVA_FUSED_SINKHORN=1 mei \
  --model-dir ./SenseNova-Loop-9B-Q6 --enable-sensenova-loop true \
  --served-model-id sensenova-loop-9b-q6 --port 8025
```

The converter verifies the original shard digests and retains the tokenizer
and chat template. The separate download preserves the pinned upstream license. The new bundle's FP32 routing and
separate per-visit cache state preserve both passes of the published model.
The Metal opt-in selects the tested faster router implementation.

## Release verification

On a 32 GiB Mac Studio M1 Max, the 0.12.0 build matched 0.11.2 on all
80 paired greedy observations for Groot, Ornith, Qwen3.6 text-only, and
the text route of the Qwen3.6 vision bundle. Each profile had ten short
and ten approximately 2K-token loaded observations, with cache off and
identical model and sampler settings. Checks matched output, token counts,
and finish reasons. The largest median decode slowdown was 1.12%, and the
largest median end-to-end increase was 0.90%. The comparison did not exercise
vision input.

The live Groot checks reproduced the missing stop behavior on 0.11.2 and
passed after the fix, including buffered and streaming chat. SenseNova Q6
passed the 4K text boundary, disk-cache restore, cancellation, and synthetic
tool and structured-output checks. A separate Q8 smoke passed.
