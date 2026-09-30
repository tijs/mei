# Mei release comparison — F1/C2 historical artifacts

- **Artifact root:** `/private/tmp/mei-c2-optout-0.6.1/artifacts/f1-c2-release-compare-20260930T095614Z`
- **Generated:** 2026-09-30T12:16:09Z by `tools/release_attribution_report.py` (stdlib only)
- **Scope:** 6 final legs = 3 models x releases 0.5.0/0.6.0; per leg 10 repeats x (short: 13 prompt tokens; 30k: 30000 prompt tokens); 20 timing rows and 25 request-log rows per leg.
- **Excluded from aggregates:** `ornith-060-optout`, `qwen36-text-050`, `qwen36-text-060`, `qwen36-text-060-thinking-off-optout` (present in artifact root but not among the six final legs).

## Validation

**6/6 legs valid** — 120 timing rows, 150 request-log rows total; probe_returncode=0 and server_stopped=true on every leg.

| Leg | Model | Release | Timing rows | Log rows | probe rc | stopped | compiled_routed_switch_glu |
|---|---|---|---|---|---|---|---|
| `ornith-050` | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 0.5.0 | 20/20 | 25/25 | 0 | true | absent |
| `ornith-060` | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 0.6.0 | 20/20 | 25/25 | 0 | true | active |
| `qwen36-text-050-thinking-off` | `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | 0.5.0 | 20/20 | 25/25 | 0 | true | absent |
| `qwen36-text-060-thinking-off` | `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | 0.6.0 | 20/20 | 25/25 | 0 | true | active |
| `qwen36-vision-050-thinking-off` | `mlx-community/Qwen3.6-35B-A3B-4bit` | 0.5.0 | 20/20 | 25/25 | 0 | true | absent |
| `qwen36-vision-060-thinking-off` | `mlx-community/Qwen3.6-35B-A3B-4bit` | 0.6.0 | 20/20 | 25/25 | 0 | true | absent |

## Binaries

| Release | Path | SHA256 |
|---|---|---|
| 0.5.0 | `/opt/homebrew/Cellar/mei/0.5.0/bin/mei` | `580e3d31af2e1ce32c4decf31f85e3fd64506c0c9c05c1d4ed1bb7a16741e2bb` |
| 0.6.0 | `/Users/tijs/projects/mei/dist/mei-0.6.0-macos-arm64/bin/mei` | `3d828371326dd170364f4312b1ef8ffcbffbf235ed67f735d29c5a5214edbffc` |

## Settings

- `--enable-thinking false`: `qwen36-text-050-thinking-off`, `qwen36-text-060-thinking-off`, `qwen36-vision-050-thinking-off`, `qwen36-vision-060-thinking-off`
- No `--enable-thinking` flag: `ornith-050`, `ornith-060`

## 0.6.0 metallib provenance

- Build dir: `/Users/tijs/.local/share/local-model-bench/mei-build-060`
- Vendored MLX: 0.32.2 (checkouts/vmlx-swift/Package.swift (MLX_VERSION define))
- metallib: 182,351,120 bytes, SHA256 `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed` (`out/Products/Release/mlx.metallib`)
- Read-only re-check: size match = True, sha256 match = True (observed 182351120 bytes, `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed`)
- MLX_VERSION source check: match = True

## Policy evidence (server.log)

| Leg | compiled_routed_switch_glu | Evidence line |
|---|---|---|
| `ornith-050` | absent | no compiled_routed_switch_glu line (0.5.0 leg: no compiled_routed_switch_glu policy line in server.log) |
| `ornith-060` | **active** | L17: `[Qwen4Exp] compiled_routed_switch_glu=active stock_gather_qmm=true shared_weight_inputs=true dtype=bfloat16` |
| `qwen36-text-050-thinking-off` | absent | no compiled_routed_switch_glu line (0.5.0 leg: no compiled_routed_switch_glu policy line in server.log) |
| `qwen36-text-060-thinking-off` | **active** | L17: `[Qwen4Exp] compiled_routed_switch_glu=active stock_gather_qmm=true shared_weight_inputs=true dtype=bfloat16` |
| `qwen36-vision-050-thinking-off` | absent | no compiled_routed_switch_glu line (0.5.0 leg: no compiled_routed_switch_glu policy line in server.log) |
| `qwen36-vision-060-thinking-off` | absent | no compiled_routed_switch_glu line (0.6.0 vision leg: compiled_routed_switch_glu absent (other Qwen4Exp lines present)) |

## Visible output comparison

| Model | Context | 0.5.0 visible text (sha256) | 0.6.0 visible text (sha256) | Match |
|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | short | '' (`e3b0c44298fc1c14`) | '' (`e3b0c44298fc1c14`) | yes |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 30k | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | yes |
| `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | short | '1\n2\n3\n4\n5\n6\n7\n8\n9\n10' (`d68dbfb354c79395`) | '' (`e3b0c44298fc1c14`) | **NO** |
| `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly` | 30k | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | yes |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | short | '' (`e3b0c44298fc1c14`) | '' (`e3b0c44298fc1c14`) | yes |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | 30k | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | 'hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello' (`4e0b7037f631632d`) | yes |

## Means and 0.6.0-minus-0.5.0 deltas

Means from request-log completion rows (server-reported); 10 rows per context per leg. `tail = wall - prefill - generate` (legacy; finalize_ms absent). Delta% = (0.6.0 - 0.5.0) / 0.5.0 x 100.

### `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`

| Metric | short 0.5.0 | short 0.6.0 | delta% | 30k 0.5.0 | 30k 0.6.0 | delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 701.17 | 700.78 | -0.06% | 72,852.15 | 58,014.04 | -20.37% |
| prefill (ms) | 112.86 | 98.68 | -12.56% | 71,971.67 | 57,151.36 | -20.59% |
| generate (ms) | 552.57 | 570.55 | +3.25% | 674.01 | 692.60 | +2.76% |
| tail = wall - prefill - generate (ms) | 35.74 | 31.56 | -11.70% | 206.47 | 170.08 | -17.62% |
| decode (tok/s) | 57.936 | 56.092 | -3.18% | 47.528 | 46.203 | -2.79% |
| prompt (tok/s) | 115.300 | 131.799 | +14.31% | 416.831 | 524.922 | +25.93% |
| cached_tokens | 0.000 | 0.000 | n/a | 0.000 | 0.000 | n/a |
| memory peak (bytes) | 20,918,523,164 | 20,803,593,808 | -0.55% | 24,596,373,044 | 22,714,718,478 | -7.65% |

### `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`

| Metric | short 0.5.0 | short 0.6.0 | delta% | 30k 0.5.0 | 30k 0.6.0 | delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 579.90 | 686.75 | +18.43% | 73,348.68 | 58,413.22 | -20.36% |
| prefill (ms) | 107.44 | 99.37 | -7.51% | 72,466.10 | 57,534.21 | -20.61% |
| generate (ms) | 434.86 | 554.26 | +27.46% | 680.01 | 710.27 | +4.45% |
| tail = wall - prefill - generate (ms) | 37.61 | 33.12 | -11.94% | 202.57 | 168.75 | -16.70% |
| decode (tok/s) | 57.491 | 57.739 | +0.43% | 47.137 | 45.055 | -4.42% |
| prompt (tok/s) | 121.169 | 130.946 | +8.07% | 413.987 | 521.429 | +25.95% |
| cached_tokens | 0.000 | 0.000 | n/a | 0.000 | 0.000 | n/a |
| memory peak (bytes) | 20,886,547,066 | 20,827,184,488 | -0.28% | 24,596,380,958 | 22,714,725,768 | -7.65% |

### `mlx-community/Qwen3.6-35B-A3B-4bit`

| Metric | short 0.5.0 | short 0.6.0 | delta% | 30k 0.5.0 | 30k 0.6.0 | delta% |
|---|---|---|---|---|---|---|
| wall (ms) | 3,690.86 | 2,352.57 | -36.26% | 103,054.44 | 83,880.99 | -18.61% |
| prefill (ms) | 389.38 | 290.54 | -25.38% | 99,180.77 | 81,330.26 | -18.00% |
| generate (ms) | 3,264.07 | 2,054.63 | -37.05% | 3,771.88 | 2,501.90 | -33.67% |
| tail = wall - prefill - generate (ms) | 37.40 | 7.39 | -80.24% | 101.79 | 48.84 | -52.02% |
| decode (tok/s) | 9.804 | 15.575 | +58.86% | 8.492 | 12.790 | +50.61% |
| prompt (tok/s) | 33.388 | 44.750 | +34.03% | 302.925 | 369.771 | +22.07% |
| cached_tokens | 0.000 | 0.000 | n/a | 0.000 | 0.000 | n/a |
| memory peak (bytes) | 25,156,670,746 | 25,149,625,926 | -0.03% | 26,980,033,112 | 26,980,033,112 | +0.00% |

## Caveats

1. **Tostibrown/Qwen3.6-35B-A3B-4bit-textonly short output mismatch between releases (protocol caveat)** — 0.5.0 produced visible text '1\n2\n3\n4\n5\n6\n7\n8\n9\n10' ([25] completion tokens, finish=['stop']); 0.6.0 produced '' ([32] completion tokens, finish=['length']). The short-context outputs are not matched between releases, so short-context timing deltas for this model are not a like-for-like comparison and must not be read as an engine effect without noting the output difference.
2. **finalize_ms absent in historical 0.5.0/0.6.0 tags** — No leg in this artifact set records finalize_ms (checked request.jsonl, result.json, probe.json, server.log per leg). The residual is therefore the legacy tail = wall - prefill - generate, which may absorb finalize/overhead. Presence is recorded per leg; no value is synthesized. Newer 0.6.1 F1 artifacts carry finalize_ms but are outside this report.
3. **deltas are descriptive, not causal** — Means are single-run comparisons (10 repeats per context per leg); no significance testing is applied. Policy-line presence in server.log is a configuration difference, not proof of cause. Small deltas (order of a few percent, and any short-context delta for qwen36-text) should not be read as effects.
4. **wall metric is server-reported wall_ms** — All means use the server-reported wall_ms from the request log (same source as prefill/generate/tail). Client-observed wall from the timing rows runs higher by ~1.18-3.15 ms at short and ~303.50-342.57 ms at 30k on every leg; the source of that difference is not determined here.
