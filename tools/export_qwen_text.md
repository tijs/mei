# Prepare a text-only Qwen3.5-family bundle

```sh
uv run python tools/export_qwen_text.py /path/to/source /path/to/new-text-bundle
```

This opt-in preparation tool removes vision and MTP tensors for ordinary text
serving. It supports the audited `qwen3_5` and `qwen3_5_moe` namespace layouts;
unknown tensor namespaces, duplicate keys, malformed payloads and contradictory
precision declarations fail before publication. It does not change Mei defaults
or claim multimodal acceptance.

Retained tensor bytes, shapes and dtypes are preserved and independently hashed.
Unchanged shards are hardlinked on the same filesystem, while mixed shards are
streamed into a new file. The source directory is never modified. New output is
published only after verification. Do not modify either linked copy afterward;
verify the manifest again before serving. Keep both source and export provenance.
The export omits processor/vision metadata and sets MTP depth to zero explicitly.

Some converted packs leave stale pre-conversion names in `quantization_config`,
or omit entries for split six-bit projections under a four-bit default. The tool
validates the current `quantization` map against actual packed/scales/biases
geometry. An omitted precision may be made explicit only at the declared group
size, with a supported integral bit count and architecture-valid input width.
Explicit conflicting precision is rejected. The exported quantization maps agree;
no payload is re-quantized. The source/export config hashes are in the manifest.

`MEI_TEXT_EXPORT.json` records retained tensor hashes, removed keys and exported
shards. This proves artifact preparation, not successful loading, numerical
parity with BF16, tool reliability or benchmark quality. Run pinned native
admission and cache checks, followed by the benchmark, before promoting a model.
