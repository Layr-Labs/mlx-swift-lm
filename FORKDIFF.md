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
the full native runner labels; these jobs have no hosted fallback. The
repository's approval still determines the image and the network policy.
Each macOS job names its VM size ([Job sizes](#job-sizes)).

The test shards start beside Build and Metal library, as soon as the shard
map is ready, as Layr-Labs/d-inference's test jobs do. Each one queues for
its Mac, checks out and selects Xcode while the build runs, then
`.github/scripts/wait-for-products.py` holds it until both jobs have
succeeded (it reads the run's jobs every 15 s and fails when either did not
succeed). Before, the shards were created only when the build ended, and
their queue and setup (p50 107 s from the build's end to the download,
runs 44750661761 to 44868059797) came after it. In the first run with early
start (44881140704) it took 2 to 13 s. The cost: each shard's Mac idles for
most of the build. In that run the shards waited 289 to 355 s, 356
vCPU-minutes in all (about $0.36 at $0.001 per vCPU-minute). A shard whose
build failed fails at the wait instead of being skipped. The wait counts
against the shard's 45-minute job timeout, so no step gets more time than
before; it ends with an error after 30 minutes.

There are 16 test shards. The run ends with the slowest shard, usually one
on a slow Mac, so fewer seconds of tests per shard end it sooner. #478's
penalty test checks six independent cases. It was one test of 150 to 281 s
(median 207 s over 11 green runs), alone in shard 12, which set the floor:
about 230 s predicted with 14 shards. Each case is now its own suite
(`CBv2MTPPenaltyLongRunTests/<case>`, same requests and assertions), so the
map spreads the cases; they took 18 to 49 s each (run 44937126105). Sixteen
shards hold a predicted median "Run the shard" of 169 to 170 s (14 shards
with the split cases: 189 to 191 s). The longest unit is now
`KernelTests/GLM5NextMTPTests`, about 117 s. Against 14 shards, each run
asks for two more 4x24 Macs, and they also wait for the build: about 12
Mac-minutes, 48 vCPU-minutes (about $0.05 at $0.001 per vCPU-minute), more
per run. The shards hold their Macs while they wait for the build. In an
earlier 16-shard run the 12x32 build was placed after all of its shards
(129 s, against 33 to 69 s with 12 or 14 shards). Since the runner plane
places jobs round robin, the build of a 16-shard run started 39 s after it
was queued, before any of its shards (run 44937126105).

Every native shard ends with a rest run. The rest runs share the tests that
no shard names, which are mostly the new suites of a pull request, until a
new plan adds them to the map. The map has no durations for them, so
`test_shards.py` gives their units one at a time, most tests first, to the
shard with the lowest estimate so far, counting one second per test. Every
job computes the same split from the build's test list, and "Each test ran
once" still checks each test. Before, shard 12 ran all of them. Of the 15
pull request rounds on 16 shards from 2026-11-21 19:40 to 2026-11-22 19:05
UTC, 13 had such tests, and 10 had 105 to 193 of them (91 to 394 s). Shard
12 ended 6 of those 10, its "Run the shard" at 331 to 654 s against 195 to
316 s for the slowest other shard.

A shard runs its share in the same `swift test` process as its main run
with the default settings, where the planner puts units, as a new plan
would place them. A rest run of its own cost a second process, about 3 s
(median "Run the shard" minus test time: 22 s with a share in two processes,
19 s with one). Replayed on those 15 rounds over all 16 ways to number the
shards, that cost was the only way the split made a round slower on average
than when shard 12 ran them all (1 to 3 s on 4 rounds). A shard without such
a run keeps a process for its share. A round that a late shard ends (a long
queue, a slow Mac or a rerun) still pays for that shard's share, as it paid
for all of them when shard 12 was the late one. Every shard partitions the
whole list to find its share: 0.5 s for the 8,076 tests of run 45046267507
on a developer Mac.

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
headroom over its measured memory peak. A job that took less than 10 minutes
must stay under 10 minutes. Build is the exception: every test job waits for
it, and its cold build is CPU-bound, so it takes the largest signed size.

The label also names the job's workflow run:
`darkbloom-macos-arm64-4x16-run-${{ github.run_id }}`, the form of the
run-scoped labels that Layr-Labs/d-inference's route asks Darkbloom for.
Darkbloom starts a runner with the labels of the job it was started for.
Without the run, a runner started for one run's 4x16 job (PR lint) could
take another run's queued 4x16 job (Metal library), and the job it was
started for would wait for the next runner. An idle runner of an older run
could also take this run's job and lose it when a new push cancels the older
run. Darkbloom refuses a job whose run label names another run. Jobs of one
run and size still share their runners.

These jobs need a Darkbloom plane that admits run-scoped labels, and they
have no fallback runner: a plane without that support refuses them. Revert
the suffix before such a plane is rolled out.

The peak of a job is the larger of two sampler figures: used memory (app +
wired + compressed) and the sum of the lifetime peak footprints of the
runner user's processes. The second matters where memory is compressed:
used memory counts compressed pages at their compressed size. The idle VM
uses about 6 GiB. Pull request runs 44750661761, 44754337456 and 44764725460
measured every job.

| Job | Size | Data before the sampler | Measured (size: wall time, peak) |
|---|---|---|---|
| Build | 12x32 | A cold build (the plane gives these sandboxes no cache): 363 to 583 s on 12 vCPUs, of which the compile is 261 to 412 s (runs 44505218067 to 44685968782). The same cold build took 1,273 s on the hosted 3-vCPU `xcode-27` Mac (run 43365935094). | 8x16: 361 and 457 s, 10.04 GiB. On 8x16 (#468) the median of the build step went from 269 s on 12 vCPUs (n=51 runs, 43711096547 to 44767807509) to 362 s (n=16, 44750661761 to 44819969123); it was 301 s after #480 (n=7). Every test job waits for it, so it went back to 12x32: 32 GiB is 3.2 times the peak, and it costs about the same vCPU-minutes because the build is CPU-bound. 12x32 did not queue longer: the build queued for a median 55 s on 12 vCPUs and 88 s on 8x16 (n=51 and 23), and in the same runs the 12x32 test shards queued for 70 s against 78 s for 4x24 (n=39 and 205). |
| Metal library | 4x16 | 153 to 301 s on 12 vCPUs (same runs); 436 s cold on the hosted 3-vCPU, 7 GiB Mac. | 4x16: 231 and 306 s, 10.04 GiB (`air-lld` 6.52) |
| Test shards 1 to 16 (`test`) | 4x24 | 206 to 480 s on 12 vCPUs with 12 shards (runs 44682883296, 44685968782). About 150 s of each is the download of the test products. | 4x16: 206 to 728 s (shard 5 spent 390 s on downloads). Shard 8: 13.26 GiB used, 6.4 GiB compressed, memory pressure at warning, 0.06 GiB swapped; over the 12.3 GiB that 16 GiB allows. 4x24: 216 to 488 s; shard 8 17.84 GiB of footprints, the others at most 12.01 GiB (shard 7 10.25 GiB, shard 9 7.88 GiB). Shards 7 and 9 ran on 12x32 until issue #469 was fixed: their tests read the admission ledger before the checkpoint store's callback had dropped its last export, whose manifest permit is charged until then. On 4 vCPUs that callback can still be running at the request's final event. |
| Coverage (pushes to `main` only) | 4x16 | 56 to 196 s on 12 vCPUs. | not run on pull requests |
| PR lint | 4x16 | 35 s on 12 vCPUs. | 4x16: 31 s, 6.04 GiB |

Every macOS job runs `.github/scripts/ci-resource-sampler.py`, a copy of the
sampler of Layr-Labs/d-inference, from its checkout on. Its last step,
"Report the resource samples", prints the memory peak of each class, the
three largest processes, the busy vCPUs and the sampled time scaled to 4 and
8 vCPUs. Use these numbers to change a size. The Shard map job runs
`.github/scripts/test-runner-sizes.py`. It fails when a macOS job names no
size, a size that is not signed, or a size other than the one its table
gives, or when its label does not name its run.

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
before you push. Upstream's workflows name actions by tag, as in
`actions/checkout@v7`. This organization starts only workflows whose actions
are pinned to a full commit SHA, so a workflow with a tag fails at startup,
even one whose jobs only run upstream. Pin each such reference to the commit
its tag names; the Shard map job's `.github/scripts/test-action-pins.py`
fails on one.

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
