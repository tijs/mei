# Qwen3.8-27B generic-profile KV safety default — release fix + smoke proof

Run window: 2026-09-02T11:22:38Z – 11:23:50Z. Clean: no foreign 80xx
listeners before launch, port 8024 free, disk guard 93.3 GiB free >= 20 GiB
floor (start.log), single sequential probe, zero pre-existing Mei source
change at launch time (binary = Mei HEAD + this commit's fix, fork
vmlx-swift 91fed8be).

## Fix (this commit)

`ServerConfig.parse` now applies a **model-aware safe default** for the KV
cache directory: when `--kv-cache-dir` was NOT supplied and cache reuse is
on, any bundle whose `config.json` carries a dense qwen3_5 family
`model_type` (`qwen3_5`, `qwen3_5_text` — verified on Qwen3.8-27B-4bit and
Qwen3.8-27B-Uncensored-MLX-4bit) lands on a disposable on-disk cache:

```
$TMPDIR/mei-kv-cache/<sanitized served model id>
```

- Explicit `--kv-cache-dir` (even empty) always wins (`kvCacheDirExplicit`).
- `--cache-reuse false` keeps caching fully disabled — no dir is created.
- Ornith behavior preserved: `qwen3_5_moe`/`qwen3_5_moe_text` are NOT in the
  unsafe set; no implicit default fires for them.
- No compile/env enablement: this does not touch `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES`
  / `BENCH_NO_FUSED_GATE_UP` / compiled decode — the dense model keeps the
  generic profile defaults otherwise (prefill 64, kv-bits none, compiled-decode
  false).
- Rollback-safe: revert this commit and the generic default is byte-identical
  to the released behavior (in-memory-only paged tier, no temp dir).

## Smoke proof (the previously-crashing cell-A config, now passing)

Launch (NO `--kv-cache-dir` anywhere — see launch line):

```
MEI_RUNTIME_BASE/MEI_CACHE_ROOT=…/mei-runtime-kvfix-smoke-20260902T112238Z \
MEI_MODEL_DIR=$HOME/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit \
MEI_SERVED_MODEL_ID=mlx-community/Qwen3.8-27B-4bit \
MEI_OPTIMIZATION_PROFILE=generic MEI_PORT=8024 MEI_CONTEXT_CAP=65536 \
MEI_MAX_TOKENS=32768 bash scripts/start_mei_server.sh
python3 tools/probe_load.py --base-url http://127.0.0.1:8024/v1 \
  --model mlx-community/Qwen3.8-27B-4bit \
  --server-log <runtime>/logs/server.log \
  --output artifacts/load-Qwen3.8-27B-kvfix-smoke-20260902T112238Z.json
```

Server log (server.log line 3):

```
mei: prefix cache enabled (paged in-memory + disk at
/var/folders/…/T/mei-kv-cache/mlx-community-Qwen3.8-27B-4bit);
topology layers=64 kvLayers=16 mambaLayers=48 companion=ssm restore=disk-backed
```

Result: **PASSED** (artifact load-Qwen3.8-27B-kvfix-smoke-20260902T112238Z.json):

- hello: 15.26 t/s (prompt 35.7 pps), peak 18.98 GB
- short decode: 15.68 t/s engine (46.7 pps), peak 18.99 GB
- 0 occurrences of `SmallVector` in server.log
- disposable disk cache 301 MiB under $TMPDIR/mei-kv-cache/… (OS-reclaimed;
  no benchmark-runtime pollution, no cross-restart persistence claim)

The numbers match the 2x2 disk cells B/D (15.14 / 15.66-15.70 t/s, peak
18.98 GB) — i.e. the auto default costs nothing vs the manually configured
disk tier, and the crash config (cell A) is no longer reachable from the
generic profile with cache reuse on.

## Tests (non-Metal suite, all green on this commit)

```
swift test --scratch-path $HOME/.local/share/local-model-bench/mei-build \
  --filter ServerConfigParsingTests     # 23/23 (8 new dense-qwen35 tests)
swift test --scratch-path $HOME/.local/share/local-model-bench/mei-build \
  --skip MeiAcceptanceTests             # 52/52, 0 failures
swift build -c release --scratch-path $HOME/.local/share/local-model-bench/mei-build
  # Build complete (MeiCore + Mei, 22.33s incremental)
```

Note: QuantizedRotatingKVCacheTests needs an mlx.metallib next to the debug
test bundle; provisioned from the already-verified build dir before the
non-Metal run (environmental, not source).

Key commits: HEAD Mei = this fix (63e4105 + 1), fork vmlx-swift pinned
91fed8be. Blocked-by note 1532b3fc (2x2 isolation) is RESOLVED for the
generic profile by this default; dense qwen3_5 still needs its UD-quant
selection/gates (todos 0b87b76a#3-#6 remain open).