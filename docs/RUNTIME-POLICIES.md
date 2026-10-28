# Runtime policies

Mei resolves one prefix-cache mode and one MLXPress policy before loading. The
same resolver runs for the CLI and `Engine.load(config:)`. Parsing does not
create cache directories or load weights.

```sh
mei --model-dir /models/qwen --cache-mode auto
mei --model-dir /models/qwen --cache-mode off --mlxpress off
mei --model-dir /models/qwen --cache-mode disk --kv-cache-dir /cache/qwen
```

`--cache-mode auto` retains model-aware prefix reuse: ordinary models use memory;
models whose hybrid state needs disk restoration use a temporary disk tier.
Startup states the effective mode and why. Temporary directories are unique to
each configuration instance, including two servers with the same served ID.
They live under the OS temporary directory and are OS-reclaimable; they do not
promise cross-restart persistence. Pass an explicit directory for intentional
persistence and ensure only one server owns it.

`off` disables prefix reuse. `memory` explicitly forbids a disk tier and fails
early if the model/SSM anchors cannot support that choice. `disk` uses an explicit
or temporary directory. Contradictory combinations of `--cache-mode` with legacy
`--cache-reuse`/`--kv-cache-dir` flags fail regardless of flag order. Existing configurations without the new
selector retain legacy selection rules, except an explicit empty directory is
rejected when it would break required disk reuse. Use `--cache-mode off` to opt
out. Library users can set `ServerConfig.cacheMode` and `mlxPressPolicy` directly.

MLXPress is experimental weight-page advice. `--mlxpress off` is the default;
`auto` explicitly permits the dependency to select it based on the bundle and
host. An integer `1..95` selects a cold percentage. This is not a promised RAM
reduction. The actual activation and fraction appear in status after loading.

For compatibility, the CLI reads legacy `MLXPRESS` then `JANGPRESS` environment
values only when `--mlxpress` is absent. `off`, `false` and `0` disable it;
`auto` and integer percentages are supported. Malformed values fail instead of
silently becoming auto. An explicit CLI value wins over both variables. Startup
and status identify the source. Programmatic configuration defaults to off and
does not implicitly consult the process environment. This intentionally removes
the old absent-variable/big-model automatic activation.

There are three distinct memory controls:

- `--prefix-cache-limit-bytes`: live retained prefix/companion budget, default
  512 MiB. It is separate from disk storage and active generation allocations.
- `--cache-limit-bytes`: freed MLX allocator buffers. Omitted uses the backend
  default; explicitly passing `0` disables the buffer pool. Nonnegative explicit
  values are passed to model loading and reapplied for inference.
- `--memory-limit-bytes`: MLX allocation budget, with `0`/omitted selecting the
  backend default. A budget below the working set can stall allocations.

None is a total process/GPU/host memory ceiling. Disk usage, physical footprint,
pressure and swap still need independent measurement. Model-specific backend
policies may adjust limits; compare requests with live values rather than assume
requested values are effective throughout every internal operation.

`GET /v1/mei/status` adds `runtimePolicies` with requested/effective cache mode,
selection reason, actual directory and temporary-directory flag, prefix budget,
explicit allocator requests, MLXPress policy/source and actual activation.
Existing `memory.memoryLimitBytes` and `memory.cacheLimitBytes` report live
allocator settings. Existing OpenAI request/response formats are unchanged.

The unit regressions validate policy selection, loader handoff and allocator
application without loading a model. They do not prove lower Qwen residency,
model accuracy or a completed benchmark. A fresh pinned, supervised admission
is required before any new production-memory or performance claim.
