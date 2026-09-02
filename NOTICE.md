# Notices and provenance

## Mei

The Mei source in this repository is Copyright (c) 2026 Tijs Teulings and is
licensed under the MIT License. See `LICENSE`.

## vmlx-swift

Mei depends on [`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift),
which is fetched at the immutable revision:

```text
aeb5e21c195d8519609488ef75a25ce7e48d8f88
```

The upstream vmlx-swift project is MIT-licensed. Mei's `patches/` directory
contains local diff files applied to that exact upstream revision by
`scripts/apply_vmlx_patches.sh`; those files are not claims of upstream
ownership or upstream acceptance. The patch queue is maintained and reviewed
in this repository.

Mei also depends on Apple's MLX components and SwiftNIO. Their respective
licenses and notices are obtained with their Swift package checkouts and are
not replaced by Mei's license.

## Model checkpoints

Model weights and tokenizer files are not included in this repository. Users
must obtain each checkpoint directly from its publisher and comply with that
checkpoint's license and usage terms. A repacked aligned checkpoint is a
separate model artifact, not a Mei source distribution.
