# Mei 0.11.2

This release corrects the finish reason when a configured stop string is
discovered during the final detokenizer flush at the token cap. The response
now reports `stop`; actual cancellation and decoding failures keep their
original finish reasons.

It also recognises supported model artifacts and automatically loads their
validated profiles, including precise router/GDN and eligible stock-order
blocked-prefill kernels for Groot. Explicit settings remain authoritative;
unknown artifacts retain safe architecture defaults.

## Automatic model settings

Starting Mei with a supported model directory selects that artifact's validated
settings automatically. Recognition uses exact metadata fingerprints tied to
the curated revisions, not an architecture or filename guess. Modified or
unknown metadata does not inherit another model's tuning. Startup names the
selected profile. `--model-profile NAME` remains an explicit selection, and
`--model-profile none` opts out. This recognition does not verify weight payloads.

## Groot profile

The tested artifact is `Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly`,
revision `53f52e0eb8d67d2941010688b2a6b8807d9a3bea`, available through Mei's
existing `groot-19b-a3b-text` model profile. Other Groot packs are unqualified.
No environment variables are needed:

```sh
mei --model-dir /path/to/groot
```

The Groot profile selects eligible kernels as a group, while preserving
explicit operator overrides. Other models do not receive these kernels:
tested Qwen/Ornith combinations regressed and Gemma uses a different architecture.
Groot keeps its existing 256-token default prefill. The blocked kernel uses the
stock fallback for small chunks; an explicit 4096-token chunk can use the
larger blocked path, at a greater memory cost. No chunk default is raised.

On an M1 Max with 32 GB, the preceding qualified campaign build improved loaded
whole-request speed by 5.25% against released 0.11.1 at the same 4096-token chunk
size. Six counterbalanced process pairs preserved all 72 full 256-token
responses. This is a scoped internal Mei result, not a universal engine or
task-quality ranking. The release build has separate qualification records.

## Scope and provenance

The pinned vmlx revision is
`3aa7b086b22c67f5e36db29128f9f251b7227c63`, based directly on the released
0.11.1 pin. Its inherited MLX core/C bridge graph is unchanged. The release
does not incorporate or enable the campaign's prompt-token caching, DSpark,
prompt lookup, MTP, whole-step compilation, inexact B2 route or rejected
cache/fairness experiments. Existing supported runtime capabilities are unchanged.

Retained kernels credit Rapid-MLX, oMLX, mlx-vlm and vmlx-swift. Full notices
and copied-code license texts ship in `THIRD-PARTY-NOTICES.md`.

## Release qualification

Qualification is in progress. Publication requires the offline suite, live
native/API checks, greedy correctness and throughput checks on the actual
remotely pinned build, followed by checksummed packaging and Homebrew install
verification. This paragraph will be replaced with the completed results
before publication.
