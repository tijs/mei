# Gemma 4 tool/API contract + Ornith regression (release v0.2.0-alpha.1) — 2026-09-03

Parent-verified evidence against the **final** release binary built from the
current `main` (HEAD = `00418a5aae607500b43de2e49dc955b84c913540`, the
"primitive numeric arrays + reject non-finite conversions" hardening commit,
on top of the installer commit `9593126e1272d6d9ed28b0be345a49d95927090b` and
the `ba1e9df1f1a2416e39361059ca5efcee80441eed` tool-argument typing commit).
This clears the Gemma tool strict-schema blocker previously recorded as B2
(`{"a":"15","b":"27"}` string-typed args), records the /v1 usage contract, the
final-build live verification (binary SHA-256, focused suites, installer) and
the canonical Ornith acceptance regression that this candidate ships.

## Scope

- Engine: Mei, **final** release binary from `swift build -c release` on the
  current `main`; binary `mei --version` → `mei 0.2.0-alpha.1`; matching MLX
  0.31.1 metallib provisioned.
- Model under test (Gemma): `mlx-community/gemma-4-26b-a4b-it-4bit`
  (4-bit affine g64, staged), served at `127.0.0.1:8024`.
- Model under test (Ornith): `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
  (primary), served via the canonical acceptance harness.
- Build/test/profile context unchanged from the candidate notes: generic
  profile, port 8024, cache-reuse on, temperature 0.

## 1. Final build identity + focused suites (parent-verified)

Parent-verified against the release binary built from the final
release-candidate source at `main`:

- **Final build:** `swift build -c release` completed with exit 0.
- **Final binary SHA-256:** `e998782c9f2019449a73ccbeb348e91a8cb568a73a392a8ab792be7bb90aa60a`
  (`mei --version` → `mei 0.2.0-alpha.1`).
- **Installer harness** `scripts/test_install_mei.sh`: **31/31 PASS**.

Final model-free suites re-run against the release binary:

| Test suite | Result |
|---|---|
| `ToolArgumentNormalizerTests` | 16/16 PASS |
| `OpenAITypesTests` | 12/12 PASS |
| `ServerConfigParsingTests` | 28/28 PASS |
| `SSMAnchorBoundariesTests` | 9/9 PASS |
| `CacheRestoreTrackerTests` | 6/6 PASS |
| `QuantizedRotatingKVCacheTests` | 6/6 PASS |

> `ToolArgumentNormalizerTests` grew from 9/9 to 16/16 with the hardening
> commit `00418a5` (recursion into primitive numeric arrays per `items`
> schema + rejection of non-finite conversions). These are the focused
> suites exercised for this evidence; this is **not** a claim that the entire
> non-acceptance suite passed in a single run this tick (the release doc
> separately records the broader 70/70 prior baseline plus the 5
> live-server-required `MeiAcceptanceTests`).

## 2. Live Gemma tool/API contract at 127.0.0.1:8024 (`mlx-community/gemma-4-26b-a4b-it-4bit`)

Schema-aware tool-call argument typing (the `ToolArgumentNormalizer`) is
verified live against the final Gemma 4 build:

- **add_numbers tool**, integer JSON-Schema fields, non-streaming AND streaming
  chat requests both returned arguments exactly `{"a":15,"b":27}` with **JSON
  integer** types — no longer the previously-observed string-typed
  `{"a":"15","b":"27"}`. Gemma MLX now matches the GGUF reference coercion
  for number/integer fields. **Blocker B2 CLEARED.**
- **`stream_options.include_usage=true`** → the streaming finish emitted a
  final usage chunk.
- **`stream_options.include_usage=false`** (or absent) → stream omitted the
  final usage chunk.
- **Raw `/v1/completions`** returned integer usage fields with
  `total_tokens = prompt_tokens + completion_tokens`.
- **Final usage tuples** (prompt, completion, total, cached_tokens) — every
  count field an integer and total arithmetic valid:
  - chat **non-streaming**: `(168, 22, 190, 0)`.
  - chat **streaming** with `include_usage=true`: `(168, 22, 190, 167)` — the
    streaming run reused the prefix (cached 167) yet prompt/completion/total
    stay identical to the non-streaming path.
  - raw **`/v1/completions`**: `(5, 16, 21, 4)`.
  - `include_usage=false` produced **no** usage chunk.
- **Hardening (commit `00418a5`)**: the `ToolArgumentNormalizer` now recurses
  every array element against its `items` schema, so primitive numeric arrays
  (`items: {type: integer|number}`) coerce numeric strings while string-typed
  items pass through untouched; and it rejects non-finite conversions
  (NaN / ±infinity / overflow-to-infinity stay strings) so JSON serialization
  can never throw and fall back to `{}` (which would drop every argument).

## 3. Canonical Ornith acceptance regression (parent-verified)

`MEIAcceptanceTests` run against the **final Ornith release binary** with the
primary model `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`:

| Test | Result |
|---|---|
| health | PASS |
| models identity | PASS |
| plain completion | PASS |
| tool non-streaming | PASS |
| tool streaming | PASS |

**5/5 PASS.** This is the live, server-required acceptance set for the
primary model on the final release-candidate runtime; no regression introduced
by `ba1e9df` (tool-argument/usage contract) or the `00418a5` hardening change.

## 4. Final-build user-local install (parent-verified)

Installer `scripts/install_mei.sh` run from `main` at HEAD `00418a5`:

- Installed to `/Users/tijs/.local/bin/mei`; `mei --version` →
  `mei 0.2.0-alpha.1`.
- The installed binary's SHA-256 matches the final release binary
  `e998782c9f2019449a73ccbeb348e91a8cb568a73a392a8ab792be7bb90aa60a`.
- Colocated `mlx.metallib`, `default.metallib`, and the provenance sidecar
  were installed alongside the executable.
- Installer safety contract exercised by `scripts/test_install_mei.sh`
  (31/31): `--help`, unknown-option exit 2, missing-binary failure, `--dry-run`,
  install + executable bit, idempotent rerun, `--force` overwrite, colocated
  Metal companions, and the system-prefix guard.

## Status / remaining caveats

- **Blocker B2 (Gemma tool string-args) CLEARED** by commit `ba1e9df` +
  the `00418a5` hardening, verified live. It is removed from the release
  blocker table.
- Remaining, unchanged caveats:
  - vMLX fork commit `318a4e68` (Gemma4 growing-reuse fix) is still
    **un-pushed / local-only**; the remote pin stays `91fed8be`. B1 stands.
  - Qwen3.8-27B decode is below the 30 t/s primary target (4-bit 15.66 t/s,
    5-bit 13.11 t/s — hardware ceiling recorded); Qwen full-cap 65536 memory
    constraint stands. B3/B5 stand.
  - No weights are bundled; release is source/tooling only.
  - Release is experimental and **not tagged, pushed, or published** — this
    documentation covers pre-publication readiness only. Publication remains
    a user decision.

## Artifacts

- `artifacts/gemma4-tool-api-contract-20260903.md` (this note, refreshed for
  the final build at `00418a5`).
- Parent-verified final binary + focused test + live-server + installer runs
  as described above (verification recorded inline here; no auxiliary JSON
  artifacts newly committed).
- Prior art: `artifacts/gemma4-26b-common-matrix-20260903.md`,
  `artifacts/gemma4-growing-reuse-fix-20260903.md`, `artifacts/load-gemma4-26b-20260901T165909Z.json`.