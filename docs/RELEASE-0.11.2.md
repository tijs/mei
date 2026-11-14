# Mei 0.11.2

This release corrects the finish reason when a configured stop string is
discovered during the final detokenizer flush at the token cap. The response
now reports `stop`; actual cancellation and decoding failures keep their
original finish reasons.

It also includes precise router/GDN and stock-order blocked-prefill kernels
as an explicit opt-in for the tested Groot profile. Existing model defaults,
prefill policy and cache settings remain unchanged.

## Opt-in Groot profile

The tested artifact is `Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly`,
revision `53f52e0eb8d67d2941010688b2a6b8807d9a3bea`, available through Mei's
existing `groot-19b-a3b-text` model profile. Other Groot packs are unqualified.
Set these switches when starting Mei and pass the explicit prefill size:

```sh
MEI_EXPERIMENTAL_ROUTER_TAIL=1 \
MEI_EXPERIMENTAL_GDN=1 \
MEI_EXPERIMENTAL_BLOCKED_GDN=1 \
mei --model-dir /path/to/groot --model-profile groot-19b-a3b-text \
    --prefill-step-size 4096 --compiled-decode false
```

The switches are off by default. They do not select a checkpoint, add model
weights or enable speculative decoding. Do not apply this profile to Qwen or
Ornith: the tested compositions regressed on those models. Gemma does not use
these Qwen-family kernels. A 4096-token prefill requires more memory than the
ordinary smaller chunks; this release does not raise the default chunk size.

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
