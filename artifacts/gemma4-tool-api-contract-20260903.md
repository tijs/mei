# Gemma 4 tool/API contract + Ornith regression (release v0.2.0-alpha.1) — 2026-09-03

Parent-verified evidence against the release binary built from current `main`
(HEAD = `ba1e9df1f1a2416e39361059ca5efcee80441eed`, the
"schema-aware tool-call argument typing + explicit usage contract" commit).
This clears the Gemma tool strict-schema blocker previously recorded as B2
(`{"a":"15","b":"27"}` string-typed args) and records the /v1 usage contract
and the canonical Ornith acceptance regression that this candidate ships.

## Scope

- Engine: Mei, release binary from `swift build -c release` on current `main`;
  binary `mei --version` → `mei 0.2.0-alpha.1`; matching MLX 0.31.1 metallib
  provisioned.
- Model under test (Gemma): `mlx-community/gemma-4-26b-a4b-it-4bit`
  (4-bit affine g64, staged), served at `127.0.0.1:8024`.
- Model under test (Ornith): `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
  (primary), served via the canonical acceptance harness.
- Build/test/profile context unchanged from the candidate notes: generic
  profile, port 8024, cache-reuse on, temperature 0.

## 1. Focused unit suite against the release binary (parent-verified)

The exact suites enumerated by the parent against the binary built from the
release candidate source:

| Test suite | Result |
|---|---|
| `ToolArgumentNormalizerTests` | 9/9 PASS |
| `OpenAITypesTests` | 12/12 PASS |
| `ServerConfigParsingTests` | 28/28 PASS |
| `SSMAnchorBoundariesTests` | 9/9 PASS |
| `CacheRestoreTrackerTests` | 6/6 PASS |
| `QuantizedRotatingKVCacheTests` | 6/6 PASS |

These are the focused suites exercised for this evidence; this is **not** a
claim that the entire non-acceptance suite passed in a single run this tick
(the release doc separately records the broader 52/52 prior baseline plus the
5 live-server-required `MeiAcceptanceTests`). Only the suites above were
re-run and verified here; exact counts are as listed.

## 2. Live Gemma tool/API contract at 127.0.0.1:8024 (`mlx-community/gemma-4-26b-a4b-it-4bit`)

Schema-aware tool-call argument typing (the `ToolArgumentNormalizer` in
`ba1e9df`) is now verified live against Gemma 4:

- **add_numbers tool**, integer JSON-Schema fields, non-streaming AND streaming
  chat requests both returned arguments exactly `{"a":15,"b":27}` with **JSON
  integer** types — no longer the previously-observed string-typed
  `{"a":"15","b":"27"}`. Gemma MLX now matches the GGUF reference coercion
  for number/integer fields. **Blocker B2 CLEARED.**
- **`stream_options.include_usage=true`** → the streaming finish emitted a
  final usage chunk.
- **`stream_options.include_usage=false`** (or absent) → stream omitted the
  final usage chunk.
- **Raw `/v1/completions`** returned the same integer usage fields, with
  `total_tokens = prompt_tokens + completion_tokens`.
- Caching/usage nuance: a **separate streaming run** reported
  `cached_tokens=186` vs **non-streaming `cached_tokens=0`** because the
  streaming run reused the prefix. `prompt/completion/total` counts were
  **187/22/209 in both** — i.e. the new shared usage contract produces
  identical prompt/completion/total regardless of stream/cache path.

## 3. Canonical Ornith acceptance regression (parent-verified)

`MEIAcceptanceTests` run against the **Ornith release binary** with the
primary model `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`:

| Test | Result |
|---|---|
| health | PASS |
| models identity | PASS |
| plain completion | PASS |
| tool non-streaming | PASS |
| tool streaming | PASS |

**5/5 PASS.** This is the live, server-required acceptance set for the
primary model on the release candidate runtime; no regression introduced by
the `ba1e9df` tool-argument/usage-contract change.

## Status / remaining caveats

- **Blocker B2 (Gemma tool string-args) CLEARED** by commit `ba1e9df` + live
  verification. It is removed from the release blocker table.
- Remaining, unchanged caveats:
  - vMLX fork commit `318a4e68` (Gemma4 growing-reuse fix) is still
    **un-pushed / local-only**; the remote pin stays `91fed8be`. B1 stands.
  - Qwen3.8-27B decode is below the 30 t/s primary target (4-bit 15.66 t/s,
    5-bit 13.11 t/s — hardware ceiling recorded); Qwen full-cap 65536 memory
    constraint stands. B3/B5 stand.
  - No weights are bundled; release is source/tooling only.
  - Release is experimental and **not tagged, pushed, or published**.

## Artifacts

- `artifacts/gemma4-tool-api-contract-20260903.md` (this note).
- Parent-verified binary + focused test + live-server runs as described above
  (no auxiliary JSON artifacts newly committed — verification recorded
  inline here).
- Prior art: `artifacts/gemma4-26b-common-matrix-20260903.md`,
  `artifacts/gemma4-growing-reuse-fix-20260903.md`, `artifacts/load-gemma4-26b-20260901T165909Z.json`.