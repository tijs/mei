# Mei experiment folder cleanup — 2026-10-02

## Scope and decision

Removed the five completed experiment folders requested by Tijs. The Bonsai
experiments are documented in Kiem under the Bonsai MLX/Mei-vMLX work, including
the transform slice and 65K admission notes. The OpenAI P0 worktree was clean at
its root and its local and live remote branch tips matched; its plan was already
recorded complete. No source or evidence in the primary Mei checkout was changed
by the cleanup.

## Exact paths removed

| Path | Measured size (KiB) | Approx. GiB | Removal |
|---|---:|---:|---|
| `/Users/tijs/projects/mei-bonsai-prism-12121099` | 815,592 | 0.778 | `shutil.rmtree` on exact path |
| `/Users/tijs/projects/mei-bonsai-prism-8c7df5b-20260924` | 3,356 | 0.003 | `shutil.rmtree` on exact path |
| `/Users/tijs/projects/mei-bonsai-prism-attnchunk-merge` | 3,559,368 | 3.395 | `shutil.rmtree` on exact path |
| `/Users/tijs/projects/mei-bonsai-prism-p0-8c7df5b-20260924` | 3,588 | 0.003 | `shutil.rmtree` on exact path |
| `/Users/tijs/projects/mei-openai-p0` | 5,065,336 | 4.831 | `git worktree remove --force` on exact registered worktree |
| **Total** | **9,447,240** | **9.010** | |

## Pre-delete checks

- All five paths were exact direct children of `/Users/tijs/projects`, real
  directories (not symlinks). The four Bonsai snapshots were not Git worktree
  roots. No in-scope path had open file handles or an active writer.
- The `mei-openai-p0` root was clean at
  `ab2e59bf9c730716792d3156f0751627dc88b2b5`, branch
  `task/mei-openai-p0`; local ahead/behind was `0/0`. A live `git ls-remote`
  readback returned the same SHA for `refs/heads/task/mei-openai-p0`.
- Its ignored SwiftPM dependency checkout `vmlx-swift` was detached at
  `44461ffdf8dca836b624b2d0d268c800dedea05e` and contained a local 4-line
  behavior-preserving rewrite of the cache/store-boundary diagnostic plus a
  zero-byte `.hermes-tmp.Eayqnk`. Those dependency-local changes were discarded
  with the explicitly authorized folder removal. The base commit was verified
  through GitHub; the separate compiler-fix commit `12121099…` is also present
  upstream, though its file content is not byte-identical to the discarded
  local variant.
- `/Users/tijs/projects/mei` was at `41160cf440e56d1d7fdc453acbd33f000f65efc6`.
  Its pre-existing untracked `docs/RESEARCH.md` and `references/research/*.pdf`
  files were preserved exactly; no other worktrees or branch refs were removed.

## Verification and disk readings

- All five exact paths were absent after deletion; no `mei-*` directories remain
  under `/Users/tijs/projects`.
- Local branch `task/mei-openai-p0` remains, and its live remote branch still
  points to `ab2e59bf9c730716792d3156f0751627dc88b2b5`. The registered worktree
  was removed; other pre-existing worktree registrations were not targeted.
- Primary Mei status remained unchanged after cleanup:
  `main...origin/main`, plus the four pre-existing untracked items listed above.
- Filesystem snapshots (`df -h /Users/tijs`): before cleanup, 293 GiB used / 142
  GiB available / 68%; after the four Bonsai snapshots, 275 GiB used / 160 GiB
  available / 64%; after the P0 worktree, 271 GiB used / 165 GiB available /
  63%. The filesystem delta differs from the measured folder total; report the
  directory-size sum and APFS `df` readings separately.
- No tests were run: this was deletion of completed local experiment copies,
  with no code changes in the primary checkout.
