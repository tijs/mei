# Mei vMLX fork

Mei consumes the public fork [`tijs/vmlx-swift`](https://github.com/tijs/vmlx-swift)
through SwiftPM. The fork keeps [`osaurus-ai/vmlx-swift`](https://github.com/osaurus-ai/vmlx-swift)
as its `upstream` parent; the local checkout lives at
`~/projects/vmlx-swift` and has:

```text
origin   https://github.com/tijs/vmlx-swift.git
upstream https://github.com/osaurus-ai/vmlx-swift.git
```

Mei's `Package.swift` and `Package.resolved` pin the fork's integrated `main`
revision. A fresh Mei build retrieves the fork through SwiftPM. There is no
normal build-time patch application and no dependency on a local SwiftPM
repository cache being pre-populated.

## Integrated commits

The five changes formerly carried as local diff files in Mei were ported onto
the fork's current upstream `main` (`2422cfb8`) as separate commits, plus one
additional Mei-maintained commit (0006) that exists only in the local fork
checkout so far:

| Former patch | Fork commit | Scope | Pushed |
|---|---|---|---|
| `0001-quantized-rotating-kv` | `ae1783be` | Real affine 4/8-bit rotating KV conversion with ring/sink state preservation | yes |
| `0002-quantized-rotating-diskstore` | `1326d803` | Safe disk persistence through the existing fp16 rotating record | yes |
| `0003-compiled-decode-threshold` | `ab09d363` | Long-prompt compiled-decode guard with eager fallback | yes |
| `0004-max-kv-window-probe` | `9b8e93b1` | Explicit, opt-in Qwen3.5 bounded KV-window probe | yes |
| `0005-ssm-anchor-boundaries` | `91fed8be` | Explicit recurrent-cache anchor boundaries, default off | yes |
| `0006-rotating-stripped-boundary` | `318a4e68` | Persist gen-suffix-stripped cross-turn boundary for standalone rotating/companion topologies (Gemma3/4, Mistral SWA) + env-gated cache-fetch diagnostics | **no — local only** |

`main` currently points to `91fed8be` in the remote fork; the local checkout
at `~/projects/vmlx-swift` carries the additional `318a4e68` on top. Each
commit has a focused message and remains independently cherry-pickable.
The commits are Mei-maintained changes; they are not represented as upstream
accepted changes.

> **Publish gate (2026-09-03):** commit `318a4e68` is the Gemma 4
> growing-transcript reuse fix used by the verified `0.2.0-alpha.1` runtime.
> It is consumed via a local SwiftPM edit and must be pushed to
> `tijs/vmlx-swift` before the alpha.1 candidate is fully source-reproducible.
> Pushing is a user action (no autonomous GitHub writes).

## Normal workflow

For Mei work:

```bash
cd ~/projects/mei
git pull --ff-only origin main
swift package resolve
swift test
```

For vMLX work:

```bash
cd ~/projects/vmlx-swift
git fetch origin upstream --prune
git switch main
git pull --ff-only origin main
# Make and test a focused change.
git push origin main
```

Keep `upstream` available for comparison and future synchronization. Do not
rewrite fork history or force-push `main` as a shortcut.

## Preparing upstream PRs

Before proposing a change upstream, compare the fork with the latest parent:

```bash
cd ~/projects/vmlx-swift
git fetch upstream --prune
git log --oneline upstream/main..main
git diff --check upstream/main...main
```

The five commits can be offered separately, in dependency order. Commits
`0001` and `0002` form the rotating-KV/storage pair; `0003` is independent;
`0004` depends on the generation parameter additions from `0003`; and `0005`
adds the SSM boundary path independently of the bounded-window probe. Any
upstream PR must include focused regression tests and re-run Mei's acceptance
matrix before Mei advances its pin.

## Updating Mei's pin

After a verified fork change:

1. Push the fork commit to `tijs/vmlx-swift`.
2. Update the revision in `Package.swift`.
3. Run `swift package resolve` so `Package.resolved` records the fork URL and revision.
4. Run the focused tests, release build, and relevant long-context/agentic acceptance probes.
5. Commit the Mei pin update separately from unrelated source changes.

The old `patches/` directory is intentionally not part of the normal workflow;
the fork commit history is now the source of truth for these engine changes.
