# umans-coder worker model-option routing validated (todo 0b87b76a#15) — 2026-09-04

Unit: validate the `umans-coder` Hermes profile model-option path for the
primary (Ornith-35B) and secondary candidates (Qwen3.8-27B-4bit, Heretic,
Gemma4-26B), preserving exact model identity, each option failing closed when
its checkpoint or required runtime artifact is unavailable.

## Current profile state (before this tick)

`~/.hermes/profiles/umans-coder/config.yaml` had a single custom provider
(`umans` cloud) and NO Mei model options; `model.default` = 
`umans-deepseek-v4-flash-0731` via `custom:umans`.

## Changes made

1. `docs/WORKER-MODEL-OPTIONS.md` (Mei repo) — reference spec: candidate
   table (exact served ids, staged dirs, ports), the additive profile wiring
   block, and the two-layer fail-closed contract.
2. `tools/validate_worker_options.py` (Mei repo) — validator: identity vs
   lineup + local-model-bench mei.yaml, staged checkpoint completeness,
   runtime artifacts (binary + metallib + provenance), per-port probe
   (listening -> /v1/models identity; not listening -> fail-closed PASS),
   profile option cross-check.
3. `~/.hermes/profiles/umans-coder/config.yaml` — ADDITIVE mei-* entries
   (mei-ornith35/qwen38/heretic/gemma4/ornith9) with byte-exact served model
   ids; `model.default`/`provider` untouched; no fallback_providers added.
   Backup: `config.yaml.bak-20260904`.

## Measured results

Validator (venv RC=0, verdict PASS; artifact
`artifacts/worker-options-validation-20260904b.json`):
- identity_lineup PASS ×4, identity_bench PASS ×4 (bench served_model_id ==
  lineup id for all four ports 8024–8027)
- checkpoint PASS ×4 (config.json + >=1 safetensors shard + tokenizer files)
- runtime PASS (release binary + mlx.metallib + default.metallib +
  mlx.metallib.provenance present)
- port probe PASS_FAILCLOSED ×4 (ports 8024–8027 not listening -> connection
  refused is the correct unavailable-state behavior)
- profile option PASS ×4 (mei-ornith35/qwen38/heretic/gemma4 declared with
  the exact expected model id)

Request-layer fail-closed (server down, live):
```
$ hermes -p umans-coder chat -q "say ok" \
    --provider custom:mei-ornith35 -m ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit -Q
session_id: 20260904_062837_a7b797
API call failed after 3 retries: Connection error.
RC=1
```
No silent substitution to cloud umans. `hermes -p umans-coder config check`
clean (no structural warnings; RC=0).

## Remaining uncertainty

- Identity gate when a Mei server IS listening (PASS_IDENTITY branch) was
  not exercised live this tick (no server started; Metal kept free). The
  same exact served-id contract was already proven in every prior
  probe_mei/probe-load acceptance run (`models_identity` exact served id),
  and all four staged checkpoints are the same dirs those runs used, so the
  config-level identity match is considered sufficient for the routing todo.
- The umans-coder profile change lives outside the Mei repo (Hermes profile
  dir); backup taken, default route untouched, fully reversible.

## Commands

```
cp ~/.hermes/profiles/umans-coder/config.yaml{,.bak-20260904}
hermes -p umans-coder config check
local-model-bench/.venv/bin/python tools/validate_worker_options.py \
  --output artifacts/worker-options-validation-20260904b.json
hermes -p umans-coder chat --provider custom:mei-ornith35 -m <exact-id> -q "say ok" -Q
```

Next queued distinct unit: todo #17 (manifests/reports) — update Mei
manifests/reports with the four-model exact provenance/quant/speed/memory
evidence and rollback-safe defaults, then continue the #12 optimization loop
per-model.