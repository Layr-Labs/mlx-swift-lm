#!/usr/bin/env python3
"""Every action a workflow or a local action uses is pinned to a commit.

This organization's Actions policy starts only workflows whose actions are
pinned to a full commit SHA, and it checks the actions those actions use
too. A workflow that names an action by a tag or a branch, as in
`actions/checkout@v7`, fails at startup, before any job's `if:` is read:
even a job that runs only in the upstream repository fails here. Upstream's
workflows name actions by tag, so a merge from upstream can bring one in.

This test fails on each `uses:` of another repository's action or reusable
workflow that names no 40-character commit SHA. Local actions and workflows
(`./…`) and Docker images (`docker://…`) are not checked.

Run: python3 .github/scripts/test-action-pins.py
"""

import re
import unittest
from pathlib import Path

GITHUB = Path(__file__).resolve().parent.parent
USES = re.compile(r"""(?m)^[ \t]*(?:-[ \t]+)?uses:[ \t]*["']?([^\s"'#]+)""")
PINNED = re.compile(r"[^@\s]+@[0-9a-f]{40}")
LOCAL = ("./", "docker://")


def unpinned(text):
    """Return the `uses:` references in `text` that name no commit SHA."""
    return [
        ref
        for ref in USES.findall(text)
        if not ref.startswith(LOCAL) and not PINNED.fullmatch(ref)
    ]


def files():
    """The workflows and local actions under `.github`."""
    paths = [*GITHUB.glob("workflows/*.yml"), *GITHUB.glob("workflows/*.yaml")]
    paths += [*GITHUB.glob("actions/**/action.yml"), *GITHUB.glob("actions/**/action.yaml")]
    return sorted(paths)


class ActionPins(unittest.TestCase):
    def test_every_action_is_pinned(self):
        found = {str(p.relative_to(GITHUB)): unpinned(p.read_text()) for p in files()}
        self.assertGreater(len(found), 1)
        self.assertEqual({path: refs for path, refs in found.items() if refs}, {})

    def test_a_tag_a_branch_or_a_short_sha_fails(self):
        # The check must block, not only pass: each of these is refused.
        for ref in (
            "actions/checkout@v7",
            "actions/checkout@main",
            "actions/checkout@3d3c42e",
            "owner/repo/.github/workflows/ci.yml@v1",
        ):
            with self.subTest(ref=ref):
                self.assertEqual(unpinned(f"    steps:\n      - uses: {ref}\n"), [ref])
                self.assertEqual(unpinned(f"    uses: '{ref}'\n"), [ref])
        pinned = (
            "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n"
            "      - uses: ./.github/actions/select-xcode\n"
            "    uses: ./.github/workflows/darkbloom-route.yml\n"
            "      - uses: docker://alpine:3\n"
            "      # - uses: actions/checkout@v4\n"
        )
        self.assertEqual(unpinned(pinned), [])


if __name__ == "__main__":
    unittest.main()
