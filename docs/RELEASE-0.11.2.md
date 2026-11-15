# Mei 0.11.2

This release corrects the finish reason when a configured stop string is
discovered during the final detokenizer flush at the token cap. The response
now reports `stop`; actual cancellation and decoding failures keep their
original finish reasons.

It also recognises supported model artifacts and automatically loads their
validated profiles, including eligible stock-order blocked prefill for Groot
with stock router and decode kernels. Explicit settings remain authoritative;
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

The Groot profile selects eligible blocked prefill while preserving
explicit operator overrides. Other models do not receive this kernel:
tested Qwen/Ornith combinations regressed and Gemma uses a different architecture.
Groot keeps its existing 256-token default prefill. The blocked kernel uses the
stock fallback for small chunks; an explicit 4096-token chunk can use the
larger blocked path, at a greater memory cost. No chunk default is raised.

Normal release-policy measurements rejected the broader precise router/GDN
decode group because it slowed short requests. An isolated blocked-prefill
screen preserved short-request performance and improved loaded requests at
the same 4096-token chunk. The final automatic policy has separate release
qualification records; these are scoped internal Mei observations, not a
universal engine or task-quality ranking.

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

The actual remotely pinned release build passed 502 XCTest cases (two skips),
12 Swift Testing cases, and a separate GDN-enabled malformed-state regression.
Both 256- and 4096-token prefill passed all 13 native admission probes, including
the exact 65,536-token capacity boundary, and all nine live Swift API tests.
The blocked kernel also passed its real-layer byte verifier against stock
output and recurrent state on all 30 linear-attention layers.

On an M1 Max with 32 GB, six mirrored fresh processes per chunk compared
released 0.11.1, the new binary with kernels off, and automatic settings. Each
role has ten measured full 256-token responses per prompt length. All 168
cold, warmup and measured responses exactly matched independent, deterministic
released-build references for content, reasoning, token counts and finish reason.

| Prefill chunk | Prompt | Released median request | Automatic median request | Latency change |
| --- | --- | ---: | ---: | ---: |
| 256 (default) | Short | 4.464 s | 4.462 s | -0.04% |
| 256 (default) | Loaded | 19.756 s | 19.936 s | +0.91% |
| 4096 (explicit) | Short | 4.466 s | 4.464 s | -0.05% |
| 4096 (explicit) | Loaded | 18.813 s | 17.919 s | -4.75% |

The loaded prompt is approximately 8,192 tokens before chat rendering. These
are medians, not confidence intervals or task-quality results. Default-prefill
changes are treated as neutral, not a speedup. All median latency, first-token,
decode and sampled whole-job footprint comparisons stayed within the 3%
material-regression gate against both released and same-binary-off controls.
API-attached jobs were excluded from footprint comparison; no isolated
request-memory reduction is claimed. The explicit 4096-token improvement does
not change the default chunk size or eliminate its greater memory cost.

An incomplete 4096-token timing set hit the cumulative host-swap guard during
an optimization-off control. It was preserved and excluded. The reported 4096
results come from a complete fresh six-process set after seven stable,
normal-pressure samples, with the same resource limits and full workloads.
The two chunk sizes use separate resource windows and are not compared as a
chunk-size experiment.

The production binary SHA-256 is
`584e4deb1149f8ef0c9fbdda74640ebc69536e5db7a3c99b6ed35c72375cabc0`.
Its source inputs are commit `32af276c071e87db625af74641c7d4852dce0160`;
subsequent release-documentation changes do not alter those inputs. The Metal
library is unchanged from 0.11.1:
`56289ad54445d717bce56d7b964927dc092b5face4428db7881ab8f1ad5269cc`.
