# Mei 0.14.0

This release adds native text support for Inkling 2 Mini. Start Mei with
`mei --model-dir ./inkling-2-mini` after downloading the bundle. Mei automatically
selects the tested context, prefill, and Metal settings. The
[Inkling guide](INKLING.md) has the exact download command and API example.

The local candidate completed all 25 unchanged benchmark cases on a 32 GB
M1 Max. It passed 8 of 10 model-only cases, 11 of 12 household cases, and all
three Python automation cases, with no harness errors. Two tasks exceeded the
tool turn limit, and one household task repeated a code from a deleted source.
Measured decode averaged 23.86 tokens/s on a short prompt and 18.35 at 2K
context. Coding-specific superiority has not been measured.

Inkling uses single-request text inference at up to 8,192 total tokens.
It supports buffered and streaming tool calls. Prefix persistence, vision,
speculative decoding, and constrained output are outside this release profile.
See the [vMLX source notes](https://github.com/tijs/vmlx-swift/blob/23fa28dd26aad9222ec2efe9bfdb19c3b75cb5d3/docs/INKLING2_SOURCE.md)
for port attribution and checkpoint provenance.
