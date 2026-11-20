#!/usr/bin/env python3
"""Every macOS job that runs on Darkbloom names a signed VM size.

A Darkbloom macOS job without a size gets the size that the repository's
approval pins. Each macOS job here names its own size instead: in its runner
label (`darkbloom-macos-arm64-4x16`), or, for a lane of
`darkbloom-route.yml`, in the lane's size column. This test fails when:

- a job asks for a Darkbloom Mac without a size, or with a size that the
  image pipeline does not sign (4x16, 4x24, 8x16 and 12x32);
- a Linux route lane names a size (the route refuses one);
- a job's size is not the size that SIZES gives it, or SIZES names a job
  that no workflow has (so a size changes only on purpose);
- `.github/actionlint.yaml` does not list a used label, or still lists the
  label without a size.

FORKDIFF.md (Job sizes) gives each size and the data behind it.

Run: python3 .github/scripts/test-runner-sizes.py
"""

import re
import unittest
from pathlib import Path

GITHUB = Path(__file__).resolve().parent.parent
WORKFLOWS = GITHUB / "workflows"
ACTIONLINT = GITHUB / "actionlint.yaml"
SIGNED = {"4x16", "4x24", "8x16", "12x32"}
MACOS = "darkbloom-macos-arm64"
# (workflow, job) for a job, (workflow, "lane <name>") for a route lane: its size.
SIZES = {
    ("fork-tests.yml", "build"): "8x16",
    ("fork-tests.yml", "metallib"): "4x16",
    ("fork-tests.yml", "test"): "4x24",
    ("fork-tests.yml", "coverage"): "4x16",
    ("pull_request.yml", "lint"): "4x16",
}

JOB = re.compile(r"(?m)^  ([A-Za-z0-9_-]+):[ \t]*(?:#.*)?\n")
RUNS_ON = re.compile(r"(?m)^    runs-on:(.*)$")
MACOS_LABELS = re.compile(
    r"\[self-hosted, macOS, ARM64, " + MACOS + r"-([0-9]+x[0-9]+)\]"
)
ROUTE = "    uses: ./.github/workflows/darkbloom-route.yml"
LANES = re.compile(r"(?m)^      lanes: \|\n((?:        .*\n|[ \t]*\n)*)")
EXPRESSION = re.compile(r"\$\{\{.*?\}\}")


def jobs(text):
    """Return {job id: text} of a workflow's `jobs:` mapping (two-space job keys)."""
    text = "\n" + text
    if "\njobs:\n" not in text:
        return {}
    parts = JOB.split(text.split("\njobs:\n", 1)[1])
    return dict(zip(parts[1::2], parts[2::2]))


def lane_lines(job):
    """Return the fields of each line of a route job's `lanes:` block.

    A `${{ }}` expression counts as one field.
    """
    match = LANES.search(job)
    if not match:
        return []
    return [
        EXPRESSION.sub("EXPRESSION", line).split()
        for line in match.group(1).splitlines()
        if line.strip()
    ]


def macos_sizes(workflows):
    """Return ({(workflow, job or "lane <name>"): size}, [errors]).

    `workflows` is {workflow file name: text}.
    """
    sizes, errors = {}, []
    for workflow, text in sorted(workflows.items()):
        for name, job in jobs(text).items():
            where = f"{workflow}: job {name}"
            if ROUTE in job:
                for fields in lane_lines(job):
                    lane, profile = fields[0], fields[1] if len(fields) > 1 else ""
                    size = fields[4] if len(fields) > 4 else None
                    if profile == "macos_arm64":
                        if size not in SIGNED:
                            errors.append(f"{where}: macOS lane {lane}: no signed size")
                        sizes[(workflow, f"lane {lane}")] = size
                    elif size is not None:
                        errors.append(f"{where}: lane {lane} is not macOS: no size")
            runs_on = RUNS_ON.findall(job)
            if not runs_on:
                continue
            value = runs_on[0].split(" #", 1)[0].strip()
            if len(runs_on) > 1 or not value:
                errors.append(f"{where}: write runs-on once, on one line")
                continue
            if MACOS not in value:
                continue
            match = MACOS_LABELS.fullmatch(value)
            if not match:
                errors.append(
                    f"{where}: runs-on {value} names no size; use [self-hosted, "
                    f"macOS, ARM64, {MACOS}-<size>], a size of {sorted(SIGNED)}"
                )
                continue
            if match.group(1) not in SIGNED:
                errors.append(f"{where}: size {match.group(1)} is not signed")
            sizes[(workflow, name)] = match.group(1)
    return sizes, errors


def actionlint_labels():
    text = ACTIONLINT.read_text()
    return set(re.findall(r"(?m)^    - ([A-Za-z0-9._-]+)\s*$", text))


class RunnerSizes(unittest.TestCase):
    def setUp(self):
        self.workflows = {
            path.name: path.read_text() for path in sorted(WORKFLOWS.glob("*.yml"))
        }

    def test_every_macos_job_names_its_signed_size(self):
        sizes, errors = macos_sizes(self.workflows)
        self.assertEqual(errors, [])
        self.assertLessEqual(set(SIZES.values()), SIGNED)
        self.assertEqual(sizes, SIZES)

    def test_actionlint_lists_the_sized_labels_only(self):
        labels = actionlint_labels()
        for size in sorted(set(SIZES.values())):
            self.assertIn(f"{MACOS}-{size}", labels)
        self.assertNotIn(MACOS, labels)

    def test_an_unsized_or_unsigned_job_fails(self):
        # The check must block, not only pass: each of these is refused.
        job = "jobs:\n  a:\n    runs-on: [self-hosted, macOS, ARM64, {}]\n"
        bare = job.format(MACOS)
        unsigned = job.format(f"{MACOS}-6x12")
        block = f"jobs:\n  a:\n    runs-on:\n      - self-hosted\n      - {MACOS}\n"
        lane = (
            "jobs:\n  route:\n"
            "    uses: ./.github/workflows/darkbloom-route.yml\n"
            "    with:\n      lanes: |\n"
            "        mac macos_arm64 xcode-27 ${{ inputs.mac && 1 || 0 }}\n"
            "        lin linux_arm64 ubuntu-24.04 1 4x16\n"
        )
        for text in (bare, unsigned, block):
            with self.subTest(text=text):
                self.assertNotEqual(macos_sizes({"x.yml": text})[1], [])
        self.assertEqual(len(macos_sizes({"x.yml": lane})[1]), 2)
        sized = job.format(f"{MACOS}-4x16")
        self.assertEqual(macos_sizes({"x.yml": sized}), ({("x.yml", "a"): "4x16"}, []))


if __name__ == "__main__":
    unittest.main()
