# Mei 0.15.0

Download and run Inkling 2 Mini with two commands:

```sh
mei pull inkling-2-mini
mei --model inkling-2-mini
```

Mei picks the tested checkpoint revision, skips unused speculative decoding
weights, and applies the same serving settings as 0.14.0. Use
`inkling-2-mini` as the model ID at `http://127.0.0.1:8024/v1`.
See the [Inkling guide](https://github.com/tijs/mei/blob/v0.15.0/docs/INKLING.md)
for memory and context limits.

For any curated download, use the same name with `mei pull` and `mei --model`.
If you already have a model in another directory, use `mei --model-dir`.
