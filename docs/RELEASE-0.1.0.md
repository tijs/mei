# Mei 0.1.0

Initial public source release of Mei, a narrow native Swift/MLX
OpenAI-compatible inference server for Apple Silicon.

## Scope

This release is optimized around the validated Ornith 1.5 35B A3B MLX
checkpoint on a 32 GB Apple Silicon Mac. It is intentionally not a claim that
every optimization is safe or faster for every model family.

The release includes:

- one-model-per-process OpenAI-compatible chat/completions serving;
- automatic model-aware profile selection from `config.json`;
- conservative `generic` defaults and an explicit `ornith` override;
- optional in-process and disk-tier KV/prefix reuse;
- a path-scoped disposable-cache cleanup policy and 20 GiB free-space guard;
- an immutable upstream vMLX revision plus a locally maintained patch queue.

No model weights are included. The aligned Ornith checkpoint is a separate
model artifact prepared with Mei's model tooling.

## Profiles

| Profile | Selection | Prefill | Fused gate/up cache | Experimental features |
|---|---|---:|---|---|
| `auto` | Detect `qwen3_5_moe` or nested `qwen3_5_moe_text`; otherwise generic | 512 for detected Ornith, 64 otherwise | Disabled only for detected Ornith | Off |
| `generic` | Explicit | 64 | Unchanged | Off |
| `ornith` | Explicit operator override | 512 unless explicitly set | Disabled before model load | Off |

Usage:

```bash
MEI_OPTIMIZATION_PROFILE=auto scripts/start_mei_server.sh
# or
swift run mei --model-dir /path/to/model \
  --served-model-id model-id --optimization-profile ornith \
  --prefill-step-size 512
```

Automatic selection fails closed. Missing, malformed, or unknown model
metadata never activates the Ornith profile. An explicit `ornith` profile is
provided for controlled experiments and reproducibility. Automatic selection
respects an explicitly supplied fused-cache environment override; explicit
`ornith` forces the validated setting.

Compiled decode, rotating-KV quantization, bounded KV windows, and semantic
SSM anchors remain default-off. Their plumbing is present for isolated
experiments, not production claims.

## Compatibility matrix

| Area | Status in 0.1.0 |
|---|---|
| macOS | 15 or newer |
| Hardware | Apple Silicon; validated on a 32 GB Mac Studio |
| Primary model | Ornith 1.5 35B A3B MLX 4-bit, aligned safetensors repack |
| Other model families | Generic profile only unless separately validated |
| Context | Ornith acceptance validated through exactly 90,000 tokens; the default server cap is 65,536 |
| API | OpenAI-compatible chat/completions, streaming, tool calls, and status endpoint |
| Metal runtime | Requires a working MLX Metal kernel library; startup fails loudly when absent |

## Evidence behind the Ornith profile

The profile incorporates only the validated defaults:

- fused gate/up cache disabled: removes the observed approximately 11.25 GiB
  retained MoE cache;
- aligned safetensors repack: avoids the observed load-time realignment copies;
- prefill step 512: selected by the dated step-size experiment;
- eager decode: compiled decode was correct and memory-safe but slower;
- disk-tier KV reuse: required for hybrid SSM cache reuse across turns.

Recorded evidence is under `artifacts/`, including the long-context, memory,
prefill-step, and compiled-decode gate reports. Throughput claims in those
reports use clean repeated measurements; reused-prefix decode and fresh
prefill are reported separately.

## Ownership and provenance

| Component | Location | Ownership/status |
|---|---|---|
| Server, API, profile resolver, cache policy, launch scripts | Mei source and `scripts/` | Mei code in this repository |
| Rotating-KV quantization and disk serialization | `tijs/vmlx-swift` commits `ae1783be` and `1326d803` | Mei-maintained commits in the public fork; not upstream claims |
| Compiled-decode threshold | `tijs/vmlx-swift` commit `ab09d363` | Mei-maintained commit; experimental/default-off |
| Bounded KV window | `tijs/vmlx-swift` commit `9b8e93b1` | Mei-maintained Qwen35-specific experiment/default-off |
| SSM anchor boundaries | `tijs/vmlx-swift` commit `91fed8be` | Mei-maintained experiment/default-off |
| Fused gate/up cache control | forked vMLX plus runtime environment | Existing vMLX capability, enabled only by the Ornith profile |
| Aligned Ornith safetensors | separate model directory | Model/checkpoint preparation artifact; not bundled |

The vMLX dependency is fetched from the Mei-maintained fork
[`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift) at
`91fed8be21319f92ce5220622c6dcde0b851bdae`. A fresh build retrieves the fork
through SwiftPM; no local patch application is required. The fork's `main`
contains the five separate Mei-maintained commits documented in
[`docs/VMLX-FORK.md`](VMLX-FORK.md). The parent upstream remains visible as
`upstream` in the local fork checkout so each commit can be evaluated for a
future upstream PR.

## Rollback

To return to the upstream-style conservative runtime:

```bash
MEI_OPTIMIZATION_PROFILE=generic \
MEI_PREFILL_STEP_SIZE=64 \
MEI_COMPILED_DECODE=false \
MEI_MAX_KV_WINDOW=0 \
MEI_SSM_ANCHOR_BOUNDARIES=0 \
scripts/start_mei_server.sh
```

For source rollback, update `Package.swift` and `Package.resolved` to a
previous verified commit from `tijs/vmlx-swift`; do not reset the Mei repository
or delete model/artifact evidence. The disk cleanup tool supports
`MEI_RETAIN_KV_CACHE=true` when a test needs cache state preserved across runs.

## Publication status

This is the initial public release candidate. It is published as source and a
locally verifiable macOS artifact when the release build succeeds. No model
weights, credentials, or public benchmark repository data are included.
