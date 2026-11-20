# Fork diff: what this fork changes, and keeping that page honest

This repository is a fork of [`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm).
Everything it changes relative to upstream is published as a browsable page:

**https://layr-labs.github.io/mlx-swift-lm/**

The page is rendered by [`protolambda/forkdiff`](https://github.com/protolambda/forkdiff)
from [`fork.yaml`](fork.yaml) at the repo root, in the style of
[op-geth's go-ethereum fork diff](https://op-geth.optimism.io/). `fork.yaml`
groups the changed files into sections with a paragraph each, names the exact
upstream commit the fork branched from (`base.hash`), and lists files that are
not code (`ignore`). Every push to `main` re-renders and redeploys it
(`.github/workflows/forkdiff-pages.yml`).

**One-time setup.** GitHub Pages has to be switched on by a repo admin:
Settings → Pages → Source: **GitHub Actions**. The workflow token cannot do
this itself. Until it's done the deploy workflow still builds the page (kept
as a run artifact, `forkdiff-page`) and exits with a notice instead of failing.
If the first deploy fails before it creates a deployment (the packaged artifact
is not listed yet), the job waits 20 seconds and tries once more.

## Native CI

Fork Tests, PR lint, the fork-diff check and page rendering use Darkbloom
macOS or Linux ARM64 runners. Each compute job queues independently with
the full native runner labels; these jobs have no hosted fallback and do
not reserve all seven Mac jobs before the build starts. The repository's
approval still determines the image and the network policy. Each macOS job
names its VM size ([Job sizes](#job-sizes)).

The shard planner stays hosted. Seven strict audio tests also stay on the
hosted `xcode-27` image while native decoding is investigated. They share
the native build products and remain part of the same exactly-once check
for all 7,718 tests. No assertion, extra test pass or timeout is relaxed.

Queues remain bounded. As reviewed on 2026-11-14 UTC, the initial Linux
qualification lasts one hour and the Mac qualification lasts 90 minutes,
each capped by approval expiry. Queue time consumes that initial lifetime.
The existing watchdog has a default 30-minute wait threshold and at most
one rescue; a rerun of these native-only jobs still requests native
runners. Job timeouts also remain in force. These limits are not a promise
of a 24-hour native queue.

The GitHub-hosted Pages job still packages and publishes the rendered
artifact through the `github-pages` environment. The disabled native
CodeQL workflow has separate activation and rollback gates in
[`docs/ci-codeql-native.md`](docs/ci-codeql-native.md); merging it does not
switch off GitHub's current default setup.

### Job sizes

Each Darkbloom macOS job names its VM size in its runner label, for example
`darkbloom-macos-arm64-4x16` (4 vCPUs, 16 GiB). A job without a size gets the
size that the approval pins (12 vCPUs, 32 GiB). The image pipeline signs
4x16, 4x24, 8x16 and 12x32. A job gets the smallest signed size that has 30%
headroom over its measured memory peak (used memory: app + wired +
compressed). A job that took less than 10 minutes must stay under 10 minutes.

| Job | Size | Data |
|---|---|---|
| Build | 8x16 | A cold build (the plane gives these sandboxes no cache): 363 to 583 s on 12 vCPUs, of which the compile is 261 to 412 s (runs 44505218067 to 44685968782). The same cold build took 1,273 s on the hosted 3-vCPU `xcode-27` Mac (run 43365935094). On 4 vCPUs the job would take more than 10 minutes. |
| Metal library | 4x16 | 153 to 301 s on 12 vCPUs (same runs); 436 s cold on the hosted 3-vCPU, 7 GiB Mac. |
| Test shard 1 to 12 | 4x16 | 206 to 480 s on 12 vCPUs with 12 shards (runs 44682883296, 44685968782). About 150 s of each is the download of the test products. All ten shards of the earlier map passed on the hosted 3-vCPU, 7 GiB Mac (runs 43365935094, 43408681875). |
| Coverage (pushes to `main` only) | 4x16 | 56 to 196 s on 12 vCPUs. |
| PR lint | 4x16 | 35 s on 12 vCPUs. |

Every macOS job runs `.github/scripts/ci-resource-sampler.py`, a copy of the
sampler of Layr-Labs/d-inference, from its checkout on. Its last step,
"Report the resource samples", prints the memory peak of each class, the
three largest processes, the busy vCPUs and the sampled time scaled to 4 and
8 vCPUs. Use these numbers to change a size. The Shard map job runs
`.github/scripts/test-runner-sizes.py`. It fails when a macOS job names no
size, a size that is not signed, or a size other than the one its table
gives.

## The gate

A fork-diff page is only useful while it is true, and two things make it go
stale silently. `scripts/check_forkdiff.py` runs on every PR and on every push
to `main` (`forkdiff-check.yml`) and fails on both:

| Drift | Check | Why it matters |
|-------|-------|----------------|
| **Merge or rebase onto newer upstream** | `base.hash` must equal `git merge-base HEAD upstream/main` and be an ancestor of upstream `main` | After a rebase the old base makes upstream's own commits look like fork changes: hundreds of files, none of them ours. |
| **New fork change nobody described** | every path in `git diff --name-only base.hash HEAD` must match a section glob or a global `ignore` | An undescribed file is a change the page can't explain. |
| **Section describing code we no longer carry** | every glob must match at least one changed path | Stale sections are as misleading as missing ones. |

The check also renders the page, so a `fork.yaml` that forkdiff itself rejects
cannot merge. The deploy workflow runs the same check before publishing, so a
stale analysis is never served.

## Day to day

**Adding or changing fork files in a PR.** If the check lists uncovered files,
add each to the section that explains it in `fork.yaml` (or start a new
section with a short description). Files that are not code go under the
top-level `ignore`. Keep globs specific: a `Tests/**` catch-all would swallow
upstream's test changes after a bad rebase and defeat the gate.

**Upstream sync policy (2026-10-22).** The fork merges upstream `main` when
David directs it. Each merge is a merge commit, not a rebase. After the merge,
set `base.hash` to `git merge-base HEAD upstream/main`. Then re-validate every
section: remove globs that match nothing and describe new fork files. Between
merges, port an upstream fix the fork needs as its own commit. Run the check
before you push:

```
python3 -m pip install pyyaml
python3 scripts/check_forkdiff.py --upstream-ref refs/remotes/upstream/main
```

**Previewing the page locally** (Go 1.21+):

```
go run github.com/protolambda/forkdiff@v0.1.1 -repo . -fork fork.yaml -out tmp/index.html
open tmp/index.html
```

## Design notes

- `base.hash` is a full 40-hex commit id, never a branch name: a symbolic base
  would move underneath the page and the gate alike.
- Section `ignore` lists count as coverage (forkdiff still lists those files,
  grayed out); the top-level `ignore` is for things that aren't code at all.
- The glob semantics are forkdiff's: `*` and `?` stop at `/`, `**` spans
  directories (and may match none), `[!x]` negates a class.
- The check is pure git + PyYAML so the same command runs locally and in CI.
