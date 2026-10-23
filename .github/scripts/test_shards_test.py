#!/usr/bin/env python3
"""Tests for test_shards.py. They need no Xcode and no Swift build."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_shards  # noqa: E402

TESTS = [
    "M.A/testOne",
    "M.A/testTwo",
    "M.B/testOne",
    "M.UnitTests/U/check()",
    "M.KernelTests/K1/run()",
    "M.KernelTests/K2/run(dtype:)",
    "M.free()",
    "M.Paged/exclusive()",
    "M.Paged/other()",
]


def make_map(shards, groups=None):
    return {
        "groups": groups if groups is not None else [
            {"name": "unit", "strict": True, "metallib": False, "filters": [r"\.UnitTests/"]},
            {"name": "paged", "strict": True, "filters": [r"^M\.Paged/"], "skips": ["exclusive"]},
            {"name": "exclusive", "strict": True, "extra": True, "filters": ["exclusive"],
             "env": {"EXCLUSIVE": "1"}},
            {"name": "exclusive-main", "filters": ["exclusive"]},
        ],
        "shards": shards,
    }


GOOD = [
    {"runs": [{"groups": ["unit"]}, {"groups": ["paged"]}, {"units": ["M.A"], "groups": ["exclusive-main"]}]},
    {"runs": [{"units": ["M.KernelTests/K1"]}, {"rest": True}, {"groups": ["exclusive"]}]},
]


def problems_of(data, tests=TESTS):
    runs = test_shards.resolve_runs(data)
    return test_shards.partition(runs, tests)


class PartitionTests(unittest.TestCase):
    def test_good_map_puts_each_test_in_one_main_run(self):
        owners, problems = problems_of(make_map(GOOD))
        self.assertEqual(problems, [])
        self.assertEqual(owners["M.B/testOne"], ["shard2-run2"])
        self.assertEqual(owners["M.free()"], ["shard2-run2"])
        self.assertEqual(owners["M.Paged/exclusive()"], ["shard1-run3"])

    def test_new_test_lands_in_the_rest_run(self):
        owners, problems = problems_of(make_map(GOOD), TESTS + ["M.New/added()"])
        self.assertEqual(problems, [])
        self.assertEqual(owners["M.New/added()"], ["shard2-run2"])

    def test_test_in_two_runs_fails(self):
        shards = [{"runs": [{"units": ["M.A"]}, {"units": ["M.A"]}, {"rest": True}]}]
        _, problems = problems_of(make_map(shards, groups=[]))
        self.assertTrue(any("more than one main run" in p for p in problems))

    def test_skipped_and_never_selected_test_fails(self):
        shards = [{"runs": [{"groups": ["paged"]}, {"rest": True}]}]
        groups = [{"name": "paged", "strict": True, "filters": [r"^M\.Paged/"], "skips": ["exclusive"]}]
        _, problems = problems_of(make_map(shards, groups))
        self.assertIn("M.Paged/exclusive() is in no main run.", problems)

    def test_stale_unit_fails(self):
        shards = [{"runs": [{"units": ["M.Gone"]}, {"rest": True}]}]
        _, problems = problems_of(make_map(shards, groups=[]))
        self.assertTrue(any("M.Gone selects no test" in p for p in problems))

    def test_extra_run_without_tests_fails(self):
        groups = [{"name": "x", "strict": True, "extra": True, "filters": ["nothing"]}]
        shards = [{"runs": [{"groups": ["x"]}, {"rest": True}]}]
        _, problems = problems_of(make_map(shards, groups))
        self.assertTrue(any("selects no test" in p for p in problems))

    def test_mixed_settings_in_one_run_fail(self):
        shards = [{"runs": [{"groups": ["unit", "paged"]}, {"rest": True}]}]
        with self.assertRaises(SystemExit):
            test_shards.resolve_runs(make_map(shards))

    def test_unit_pattern_escapes_the_id(self):
        self.assertEqual(test_shards.unit_pattern("M.KernelTests/K2"), r"^M\.KernelTests/K2/")
        self.assertEqual(test_shards.unit_pattern("M.free()"), r"^M\.free\(\)")


class RunTests(unittest.TestCase):
    def test_strict_run_uses_the_nested_suite_script(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        paged = next(r for r in runs if r["groups"] == ["paged"])
        command = test_shards.run_command(paged, Path("/r"))
        self.assertTrue(command[0].endswith("scripts/run-nested-suite.sh"))
        self.assertEqual(command[1:5], [r"(?:^M\.Paged/)", "--no-parallel", "--skip", "exclusive"])

    def test_rest_run_skips_every_main_filter(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        rest = next(r for r in runs if r["rest"])
        command = test_shards.run_command(rest, Path("/r"))
        self.assertNotIn("--filter", command)
        self.assertIn(r"^M\.A/", command)
        self.assertIn(r"\.UnitTests/", command)
        self.assertNotIn("1", command)

    def test_environment_sets_and_unsets_the_group_variables(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        os.environ["EXCLUSIVE"] = "stale"
        try:
            exclusive = next(r for r in runs if r["groups"] == ["exclusive"])
            paged = next(r for r in runs if r["groups"] == ["paged"])
            self.assertEqual(test_shards.run_environment(exclusive, runs, Path("/p"))["EXCLUSIVE"], "1")
            self.assertNotIn("EXCLUSIVE", test_shards.run_environment(paged, runs, Path("/p")))
        finally:
            del os.environ["EXCLUSIVE"]


class VerifyTests(unittest.TestCase):
    def write_run(self, root, run_id, extra, log, xml=None):
        folder = root / run_id
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"id": run_id, "extra": extra}))
        (folder / "log.txt").write_text(log)
        if xml:
            (folder / "xunit-swift-testing.xml").write_text(xml)

    def verify(self, runs, tests):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in runs:
                self.write_run(root, *run)
            (root / "tests.txt").write_text("\n".join(tests) + "\n")
            return subprocess.run(
                [sys.executable, str(Path(test_shards.__file__)), "verify",
                 "--list", str(root / "tests.txt"), "--results", str(root)],
                capture_output=True, text=True, env={**os.environ, "GITHUB_STEP_SUMMARY": ""})

    XML = ('<testsuites><testsuite><testcase classname="M.KernelTests.K1" name="run()" time="1.5"/>'
           '</testsuite></testsuites>')
    LOG = ("Test Case '-[M.A testOne]' started.\n"
           "Test Case '-[M.A testOne]' passed (0.010 seconds).\n")

    def test_each_test_once_passes(self):
        result = self.verify([("s1-r1", False, self.LOG, self.XML)],
                             ["M.A/testOne", "M.KernelTests/K1/run()"])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_missing_test_fails(self):
        result = self.verify([("s1-r1", False, self.LOG)], ["M.A/testOne", "M.KernelTests/K1/run()"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("M.KernelTests/K1/run() did not execute", result.stdout)

    def test_test_in_two_main_runs_fails(self):
        result = self.verify([("s1-r1", False, self.LOG), ("s2-r1", False, self.LOG)], ["M.A/testOne"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than one run", result.stdout)

    def test_extra_run_does_not_count_as_a_second_execution(self):
        result = self.verify([("s1-r1", False, self.LOG), ("s2-r1", True, self.LOG)], ["M.A/testOne"])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_list_reader_keeps_only_test_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.txt"
            path.write_text("[0/1] Planning build\nM.A/testOne\nM.free()\nBuild complete!\n")
            self.assertEqual(test_shards.read_list(path), ["M.A/testOne", "M.free()"])


if __name__ == "__main__":
    unittest.main()
