# Contributing to Mei

Mei is a focused Apple Silicon inference server. Changes should preserve the
OpenAI-compatible API and should include a focused test or an artifact-backed
reason for changes that cannot be unit-tested.

## Development

```bash
swift package resolve
bash scripts/apply_vmlx_patches.sh --reset
swift test --filter ServerConfigParsingTests
swift test --filter CacheRestoreTrackerTests
python3 -m py_compile tools/mei_disk_guard.py tools/test_mei_disk_guard.py
bash -n scripts/*.sh
```

The local vMLX patch queue must be applied to the exact revision declared in
`Package.swift`. Re-pinning vMLX requires rerunning the focused tests and the
model acceptance/performance gates before changing `Package.resolved`.

Do not commit model weights, runtime KV caches, credentials, or benchmark
scratch data. Keep experimental optimizations default-off until correctness,
memory, and performance evidence supports enabling them.

## Pull requests

Describe the model family, hardware, runtime settings, and verification
performed. Separate fresh prefill/TTFT results from reused-prefix decode
results, and include at least three clean repeats for performance claims.
