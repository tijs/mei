# Notices and provenance

## Mei

The Mei source in this repository is Copyright (c) 2026 Tijs Teulings and is
licensed under the MIT License. See `LICENSE`.

## vmlx-swift

Mei depends on the public Mei-maintained fork
[`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift), whose parent is
[`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift). Mei pins
fork revision:

```text
91fed8be21319f92ce5220622c6dcde0b851bdae
```

The fork's `main` contains five separate Mei-maintained commits ported from
the former local patch queue. They remain clearly attributable to Mei and
have not been represented as upstream-accepted changes. See
[`docs/VMLX-FORK.md`](docs/VMLX-FORK.md) for the commit mapping and upstream
PR workflow.

The upstream vmlx-swift project is MIT-licensed; the fork retains that license
and provenance. Mei also depends on Apple's MLX components and SwiftNIO. Their
respective licenses and notices are obtained with their Swift package
checkouts and are not replaced by Mei's license.

## Model checkpoints

Model weights and tokenizer files are not included in this repository. Users
must obtain each checkpoint directly from its publisher and comply with that
checkpoint's license and usage terms. A repacked aligned checkpoint is a
separate model artifact, not a Mei source distribution.
