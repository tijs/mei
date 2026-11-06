# Runtime policies

Mei resolves prefix-cache and weight-residency settings before loading. The
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

Weights now use owned storage. The upstream loader removed cold-tier MLXPress
advice and no longer uses mmap weight storage. Remove `--load-mmap true` and
non-off `--mlxpress`/`MLXPRESS`/`JANGPRESS` settings when upgrading; Mei rejects
these requests before loading instead of silently changing their meaning.
Explicit legacy `--load-mmap false` and `--mlxpress off` remain accepted.
An explicit `--mlxpress off` overrides stale legacy environment values.
Programmatic configuration does not read those environment variables.

`--keep-model-in-memory true|false` requests the upstream bounded weight-residency
lease (default `true`, matching the upstream loader). It is independent of prefix
reuse and allocator buffer caching. The loader can reduce or decline the lease
based on device and system headroom; a true request does not prove that all
weights are wired. Setting it to `false` opts out of that lease, while weights
still use owned storage. `MEI_KEEP_MODEL_IN_MEMORY` forwards the same choice in
`scripts/start_mei_server.sh`. Omitting it preserves the server default.

The former mmap/cold-percentage performance and footprint measurements describe
older releases. They do not qualify this owned-storage path. Fresh model loading,
unloading, memory-pressure, context, cache and speed checks are required.

There are three distinct memory controls:

- `--prefix-cache-limit-bytes`: one shared retained budget for paged KV, rotating
  companions, separate recurrent SSM snapshots and optional solo exact-prompt
  copies, default 512 MiB. It is separate from disk storage and the live
  generation cache/workspace. Status's `promptSnapshotBytes` is a component of
  the shared `retainedBytes`, not an additional allowance.
  In-flight boundary snapshots for eligible cache stores are generation
  workspace and are not counted in that retained budget. With no coordinator
  or for auxiliary requests, Mei skips those snapshots; auxiliary requests
  may still restore a previously cached prefix.
- `--cache-limit-bytes`: freed MLX allocator buffers. Omitted uses the backend
  default; explicitly passing `0` disables the buffer pool. Nonnegative explicit
  values are passed to model loading and reapplied for inference. Returned
  buffers that fit displace older cached entries as needed; a single buffer
  larger than the allowance is released. This lets decode-sized buffers replace
  unused prefill shapes. The separate prefix cache stores model state.
- `--memory-limit-bytes`: MLX allocation budget, with `0`/omitted selecting the
  backend default. A budget below the working set can stall allocations.

Cache checkpoints are optional. When the kernel reports warning or critical
host memory pressure, Mei skips nonempty prefix-cache stores before making
their materialization copy and logs the reason. The completed answer stays
successful; a later request may need to prefill again. Normal or unavailable
pressure readings retain the existing byte-budget policy. This snapshot does
not reserve RAM or guarantee that a later prefill working set will fit.
Solo generation skips exact-prompt copies without a usable persistence consumer.
Optional copies reserve shared bytes before materialization and respect that same
pressure check; a decline does not trigger a replacement prefill. Live iterator
copies retain their charge until the final owner releases the snapshot.

None is a total process/GPU/host memory ceiling. Disk usage, physical footprint,
pressure and swap still need independent measurement. Model-specific backend
policies may adjust limits; compare requests with live values rather than assume
requested values are effective throughout every internal operation.

`GET /v1/mei/status` adds `runtimePolicies` with requested/effective cache mode,
selection reason, actual directory and temporary-directory flag, prefix budget,
explicit allocator requests, `weightStorage: "owned"`, and
`keepModelInMemoryRequested`. The latter reports intent, not a per-model grant.
Existing `memory.memoryLimitBytes` and `memory.cacheLimitBytes` report live
allocator settings. Existing OpenAI request/response formats are unchanged.

The unit regressions validate policy selection, loader handoff and allocator
application without loading a model. They do not prove lower Qwen residency,
model accuracy or a completed benchmark. A fresh pinned, supervised admission
is required before any new production-memory or performance claim.

Startup/loadMemory records the measured load peak before resetting the counter
for inference. Live allocator status remains the current inference epoch.
