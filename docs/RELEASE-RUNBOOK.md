# Mei release runbook

Every step, in order, for cutting a Mei release. Written 2026-09-08 while
cutting 0.4.0, after discovering that **0.3.0 was tagged and published but the
Homebrew tap was never updated** — `brew install tijs/tap/mei` still served
0.2.0. A release is not finished when the tag is pushed. It is finished when a
user running the documented install command gets the new version.

The version string lives in exactly one place: `Sources/MeiCore/ServerConfig.swift`
(`public static let version`). `scripts/package_release.sh` refuses to package a
version that does not match it.

---

## 0. Decide what ships, and from which vmlx revision

Mei pins `vmlx-swift` to an exact revision in `Package.swift`. A release almost
always moves that pin, so **assemble the vmlx side first**.

- Work on a dedicated vmlx branch (`mei/<version>`), branched from the previous
  release's pin, and cherry-pick only what ships.
- **Cherry-pick, do not merge a research branch.** Research branches carry
  experiments that measured negative and must not ship. 0.4.0's durability fix
  sat on top of the C3 fused-`gate_up` experiment (−2.9% decode); it was
  cherry-picked off that base so C3 stayed out.
- Push the vmlx branch before pinning to it. SwiftPM resolves a `revision:` pin
  from the remote, so an unpushed commit cannot be built from a clean checkout —
  and a local `.package(path:)` dependency must **never** reach a release commit.

Verify the pin actually resolved from the remote, not from a local checkout:

```bash
python3 -c "
import json; s=json.load(open('<scratch>/workspace-state.json'))
print([d['state']['checkoutState']['revision']
       for d in s['object']['dependencies']
       if 'vmlx' in d['packageRef']['identity']])"
```

## 1. Bump the version and write the changelog

- `Sources/MeiCore/ServerConfig.swift` — the version constant.
- `CHANGELOG.md` — a new `## [<version>] - <date>` section above the previous
  one. Say what changed, why it was wrong before, and the measured numbers.
- `README.md` — the "Current stable release", the install heading, the
  `mei --version` sample output, and the release-asset filename. Grep for the
  old version string; it appears in more places than you expect.

## 2. Build from the real pin

```bash
swift build -c release --scratch-path <scratch> --package-path <repo>
bash scripts/prepare_metallib.sh <scratch>/release
<scratch>/release/mei --version   # must print the new version
```

`prepare_metallib.sh` provisions `mlx.metallib` from a version-matched Python
mlx wheel, because the local Xcode lacks the metallib archiver. The bundle is
useless without it.

## 3. Verify before tagging

Do not tag on "it builds". At minimum:

- `tools/probe_mei.py` against a live server on a real staged model — streaming
  and non-streaming tool calls, cache reuse, the exact context-cap boundary.
  Expect 12/12.
- If the release changes cache, prefill, or decode behaviour: a correctness gate
  comparing generated output against a known-good configuration, greedy at
  temperature 0, byte-for-byte. **Run a determinism baseline first** — the same
  prompt through two identical cold servers must produce identical output, or no
  byte comparison between runs means anything.
- Change exactly one variable between the legs of any comparison. A control that
  differs in three ways will produce a confident wrong answer.
- If the vmlx pin moved and any config uses `VMLX_ENABLE_UNSAFE_COMPILE=1`,
  re-verify greedy token equality on the new pin. That flag's documented failure
  mode is silent numerical corruption, and its risk is version-dependent.
- Verify the change actually reached the binary: `strings <binary> | grep <new-symbol>`.
  A SwiftPM `edit`/path setup has silently built the unmodified dependency before.

## 4. Commit, tag, push

```bash
git commit -m "release(<version>): ..."
git tag v<version>
git push origin release/<version> && git push origin v<version>
```

## 5. Package

```bash
scripts/package_release.sh <version>
```

Produces `dist/mei-<version>-macos-arm64/`, the `.tar.gz`, and its `.sha256`.
The script verifies every file and the checksum round-trip, and fails loudly on
mismatch. Model weights are never bundled.

## 6. Publish the GitHub release

```bash
gh release create v<version> --repo tijs/mei --title "Mei <version>" \
  --notes-file <notes> \
  dist/mei-<version>-macos-arm64.tar.gz \
  dist/mei-<version>-macos-arm64.tar.gz.sha256
```

Both assets. The tap needs the tarball URL and its checksum.

## 7. Update the Homebrew tap — THE STEP THAT GETS FORGOTTEN

The tap is a separate repository: `https://github.com/tijs/homebrew-tap`,
cloned at `tap/` inside this repo (untracked). The formula is
`Formula/mei.rb`.

```bash
cd tap && git pull --ff-only
# edit Formula/mei.rb: url -> the new release asset, sha256 -> from the .sha256
git commit -am "chore: update mei to v<version>" && git push
```

The `url` and `sha256` must both change. Take the checksum from the published
`.sha256` asset, not from a local rebuild — the published artifact is what users
download.

Then confirm the release is actually reachable:

```bash
brew update && brew info tijs/tap/mei    # must show the new version
```

## 7b. Merge the release back to `main` — THE OTHER STEP THAT GETS FORGOTTEN

```bash
git checkout main && git merge release/<version> && git push origin main
```

The tag is not the merge. On 2026-09-12 `main` was found 19 commits behind and
0 ahead: 0.4.1 and 0.4.2 were both tagged, published and live in the Homebrew
tap, while anyone cloning the repo got pre-0.4.1 code that reported an old
version. Releases had been cut from a research branch and left there.

## Publishing model artifacts

If a release depends on a prepared artifact — a repack, a requantisation —
publish it rather than documenting the preparation. A step a user must perform
themselves is a step most will skip, and the failure is silent: they get a
working server with worse numbers and nothing to search for.

Before uploading weights:

- **Check the licence on the SOURCE.** The base model's licence governs a
  derivative. Mirror it and attribute both the original and any intermediate
  (e.g. whoever did the quantisation).
- **Verify what you are about to publish**, not what you think you built. Hash
  each shard's payload against its manifest and confirm the claimed property
  actually holds.
- **Verify what landed**, not the tool's success message. Query the repo for
  file count and sizes. "UPLOAD DONE" cannot distinguish a real transfer from a
  metadata-only commit.

### HF token gotcha

`HF_TOKEN` in the environment **shadows** `~/.cache/huggingface/token`, and
`whoami` reports whichever is winning, not which tokens exist. A read-only env
token made a write-scoped file token invisible and produced a convincing
`403 … don't have the rights to create a model` — which reads as "this token
lacks scope" rather than "you are using the wrong token". Enumerate the stores
before concluding anything about permissions.

## 8. Record it

- Kiem `proj/mei`: a note with the version, what shipped, the measured numbers,
  and anything deliberately excluded (and why).
- Check off the corresponding todos.

---

## Post-release sanity checklist

- [ ] `mei --version` in the packaged bundle prints the new version
- [ ] The GitHub release carries both the `.tar.gz` and the `.sha256`
- [ ] `Formula/mei.rb` points at the new asset with the new checksum
- [ ] `brew info tijs/tap/mei` shows the new version
- [ ] `Package.swift` pins a **pushed** vmlx revision, never a local path
- [ ] No experiment that measured negative was swept in by a branch merge
