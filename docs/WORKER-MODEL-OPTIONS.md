# umans-coder worker model options — reference spec (todo 0b87b76a#15)

The `umans-coder` Hermes profile (`~/.hermes/profiles/umans-coder/config.yaml`)
routes Mei's primary and secondary MLX candidates exactly as served, one
custom-provider entry per Mei port, and fails closed when the checkpoint or
required runtime artifact is unavailable.

## Candidates (from configs/model-lineup.json, reconciled 2026-09-04)

| role | exact model id (served) | staged dir (mei-models/) | port | arch |
|---|---|---|---|---|
| primary | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | `Ornith-1.5-35B-A3B-MLX-4bit` | 8024 | qwen3_5_moe |
| secondary | `mlx-community/Qwen3.8-27B-4bit` | `Qwen3.8-27B-4bit` | 8025 | qwen3_5 |
| secondary | `orcarouter/Qwen3.8-27B-Uncensored-MLX` | `Qwen3.8-27B-Uncensored-MLX-4bit` | 8026 | qwen3_5 |
| secondary | `mlx-community/gemma-4-26b-a4b-it-4bit` | `gemma-4-26b-a4b-it-4bit` | 8027 | gemma4 |

The former `mei-ornith9` fallback option (port 8028) was DROPPED on
2026-09-04 when its upstream checkpoint repository
(`ornith-ai/Ornith-1.5-9B-MLX-4bit`) became unavailable and the lineup entry
was removed; an option whose checkpoint can never exist adds no fail-closed
value, and port 8028 is no longer a Mei backend port. A stale
`mei-ornith9` provider left in the umans-coder profile (if present) is
harmless: it fails closed by connection refusal like any other `mei-*`
option with no server on its port.

Ports 8024–8027 are the isolated Mei backend ports registered by
`local-model-bench/configs/*/mei.yaml` (raw_port), never the shared bench
ports 8012/8015/8016/8018/8020. These are the ONLY ports Mei's isolated
backend is allowed to claim; the umans-coder private server reuses them one
at a time via `scripts/start_mei_server.sh`.

## Profile wiring (additive; `model.default`/`model.provider` untouched)

Each entry is a legacy-format `custom_providers` item (the umans-coder
profile uses the legacy list shape):

```yaml
custom_providers:
  - name: umans                      # existing cloud route — unchanged
    base_url: https://api.code.umans.ai/v1
    key_env: UMANS_API_KEY
    api_mode: chat_completions
    models:
      umans-deepseek-v4-flash-0731:
        context_length: 1048576
  - name: mei-ornith35               # primary
    base_url: http://127.0.0.1:8024/v1
    api_mode: chat_completions
    models:
      ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit:
        context_length: 65536
  - name: mei-qwen38                 # secondary
    base_url: http://127.0.0.1:8025/v1
    api_mode: chat_completions
    models:
      mlx-community/Qwen3.8-27B-4bit:
        context_length: 65536
  - name: mei-heretic                # secondary
    base_url: http://127.0.0.1:8026/v1
    api_mode: chat_completions
    models:
      orcarouter/Qwen3.8-27B-Uncensored-MLX:
        context_length: 65536
  - name: mei-gemma4                 # secondary
    base_url: http://127.0.0.1:8027/v1
    api_mode: chat_completions
    models:
      mlx-community/gemma-4-26b-a4b-it-4bit:
        context_length: 65536
```

Identity contract: the `models:` key MUST be byte-identical to the served
model id (probe_mei's `/v1/models` identity gate), and `context_length` must
not exceed the Mei server's `--context-cap` (65536). No alias, no rewrite.

## Fail-closed contract

Each option fails closed at BOTH layers:

1. **Launch layer** — `scripts/start_mei_server.sh` refuses to start when
   the checkpoint is missing (`FATAL: model directory missing`), when the
   disk floor is unmet (disk guard `rc=3 REFUSE`), or when the port is
   already listening. A missing runtime artifact (release binary, metallib)
   also aborts launch. So an unavailable checkpoint can never be silently
   substituted by another model on the same port.
2. **Request layer** — no `fallback_providers` chain maps `mei-*` options
   to the cloud umans route (the profile has no fallback_providers at all).
   When the Mei server is not running, a request through a `mei-*` option
   fails with a connection error (fail closed), never a silent route to
   umans cloud or another model.

Validator: `tools/validate_worker_options.py` (see repo). It re-checks the
identity contract against configs/model-lineup.json, cross-checks the four
`mei.yaml` backend registrations in local-model-bench (read-only), verifies
staged checkpoint completeness, and probes each option's port with a short
timeout: listening -> `/v1/models` must report the exact served id
(PASS identity); not listening -> PASS fail-closed (connection refused is
the unavailable-state behavior).

last_updated: 2026-09-04