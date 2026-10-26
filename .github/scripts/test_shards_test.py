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

    def test_test_that_a_group_skip_leaves_out_lands_in_the_rest_run(self):
        shards = [{"runs": [{"groups": ["paged"]}, {"rest": True}]}]
        groups = [{"name": "paged", "strict": True, "filters": [r"^M\.Paged/"], "skips": ["exclusive"]}]
        owners, problems = problems_of(make_map(shards, groups))
        self.assertEqual(problems, [])
        self.assertEqual(owners["M.Paged/exclusive()"], ["shard1-run2"])
        self.assertEqual(owners["M.Paged/other()"], ["shard1-run1"])

    def test_group_skip_does_not_change_the_other_parts_of_a_run(self):
        shards = [{"runs": [{"groups": ["paged"], "tests": ["M.Paged/exclusive()"]}, {"rest": True}]}]
        groups = [{"name": "paged", "strict": True, "filters": [r"^M\.Paged/"], "skips": ["exclusive"]}]
        owners, problems = problems_of(make_map(shards, groups))
        self.assertEqual(problems, [])
        self.assertEqual(owners["M.Paged/exclusive()"], ["shard1-run1"])

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
    def test_exact_command_selects_each_test_by_its_id(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        paged = next(r for r in runs if r["groups"] == ["paged"])
        command = test_shards.run_command(paged, Path("/r"), 1, ["M.Paged/other()", "M.A/testOne"])
        self.assertEqual(command[:8], ["swift", "test", "--skip-build", "--no-parallel",
                                       "--filter", r"^M\.Paged/other\(\)", "--filter", r"^M\.A/testOne$"])
        self.assertNotIn("--skip", command)
        self.assertIn("/r/events-1.jsonl", command)

    def test_runs_with_skips_or_unanchored_filters_select_by_full_id(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        by_groups = {tuple(r["groups"]): r for r in runs}
        self.assertTrue(test_shards.needs_exact(by_groups[("paged",)]))
        self.assertTrue(test_shards.needs_exact(by_groups[("unit",)]))
        units_only = next(r for r in runs if r["units"] == ["M.KernelTests/K1"])
        self.assertFalse(test_shards.needs_exact(units_only))
        command = test_shards.run_command(units_only, Path("/r"), 1, None)
        self.assertEqual(command[4:6], ["--filter", r"^M\.KernelTests/K1/"])

    def test_strict_group_fails_on_a_skip_or_zero_tests(self):
        groups = [{"name": "paged", "strict": True, "filters": [r"^M\.Paged/"]},
                  {"name": "free", "filters": [r"^M\.free"]}]
        shards = [{"runs": [{"groups": ["free", "paged"], "units": ["M.A"]}, {"rest": True}]}]
        run = test_shards.resolve_runs(make_map(shards, groups))[0]
        chosen = test_shards.selection(run, [run], TESTS)
        passed = ("Test Case '-[M.A testOne]' started.\nTest Case '-[M.A testOne]' passed (0.010 seconds).\n"
                  "Test Case '-[M.A testTwo]' started.\nTest Case '-[M.A testTwo]' skipped (0.001 seconds).\n")
        ran = events(started("M.Paged/other()/P.swift:1:1", 1), ended("M.Paged/other()/P.swift:1:1", 2),
                     started("M.Paged/exclusive()/P.swift:2:1", 2), ended("M.Paged/exclusive()/P.swift:2:1", 3),
                     started("M.free()/F.swift:1:1", 3), ended("M.free()/F.swift:1:1", 4))
        skip = events(started("M.Paged/other()/P.swift:1:1", 1), ended("M.Paged/other()/P.swift:1:1", 2),
                      ("event", {"kind": "testSkipped", "testID": "M.Paged/exclusive()/P.swift:2:1"}),
                      ("event", {"kind": "testSkipped", "testID": "M.free()/F.swift:1:1"}))
        none = events(("event", {"kind": "testSkipped", "testID": "M.free()/F.swift:1:1"}))
        cases = {"ran": (ran, []), "skip": (skip, ["paged skipped 1 tests, for example M.Paged.exclusive()."]),
                 "none": (none, ["paged executed zero tests."])}
        for name, (stream, expected) in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "log-1.txt").write_text(passed)
                (folder / "events-1.jsonl").write_text(stream)
                self.assertEqual(test_shards.run_problems(run, chosen, folder, TESTS), expected, name)

    def test_run_that_executed_nothing_fails(self):
        shards = [{"runs": [{"units": ["M.A"]}, {"rest": True}]}]
        run = test_shards.resolve_runs(make_map(shards, groups=[]))[0]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "log-1.txt").write_text("error: no tests\n")
            self.assertEqual(test_shards.run_problems(run, ["M.A/testOne"], folder, TESTS),
                             ["the run executed zero tests."])

    def test_later_attempt_selects_each_remaining_test_exactly(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        paged = next(r for r in runs if r["groups"] == ["paged"])
        command = test_shards.run_command(paged, Path("/r"), 2, ["M.A/testOne", "M.free()/F.swift:3:4"])
        self.assertEqual(command[4:8], ["--filter", r"^M\.A/testOne$", "--filter", r"^M\.free\(\)"])
        self.assertIn("/r/events-2.jsonl", command)

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


def events(*records):
    return "\n".join(json.dumps({"version": 0, "kind": kind, "payload": payload})
                     for kind, payload in records) + "\n"


def started(test_id, at):
    return ("event", {"kind": "testStarted", "testID": test_id, "instant": {"absolute": at}})


def ended(test_id, at):
    return ("event", {"kind": "testEnded", "testID": test_id, "instant": {"absolute": at}})


class VerifyTests(unittest.TestCase):
    TESTS = ["M.A/testOne", "M.KernelTests/K1/run()", "M.KernelTests/K2/run()"]
    GOOD_MAP = {"groups": [], "shards": [{"runs": [{"rest": True}]}]}
    LOG = ("Test Case '-[M.A testOne]' started.\n"
           "Test Case '-[M.A testOne]' passed (0.010 seconds).\n")
    EVENTS = events(("test", {"kind": "suite", "id": "M.KernelTests"}),
                    ("test", {"kind": "function", "id": "M.KernelTests/K1/run()/K1.swift:3:4"}),
                    started("M.KernelTests/K1/run()/K1.swift:3:4", 1.0),
                    ended("M.KernelTests/K1/run()/K1.swift:3:4", 2.5),
                    started("M.KernelTests/K2/run()/K2.swift:3:4", 3.0))

    def verify(self, runs, tests=None, data=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, extra, files in runs:
                folder = root / run_id
                folder.mkdir(parents=True)
                (folder / "run.json").write_text(json.dumps({"id": run_id, "extra": extra,
                                                             "attempts": len(files) // 2}))
                for name, text in files.items():
                    (folder / name).write_text(text)
            (root / "tests.txt").write_text("\n".join(tests or self.TESTS) + "\n")
            (root / "map.json").write_text(json.dumps(data or self.GOOD_MAP))
            return subprocess.run(
                [sys.executable, str(Path(test_shards.__file__)), "--map", str(root / "map.json"), "verify",
                 "--list", str(root / "tests.txt"), "--results", str(root),
                 "--durations", str(root / "durations.json")],
                capture_output=True, text=True, env={**os.environ, "GITHUB_STEP_SUMMARY": ""})

    def test_each_test_once_passes_and_a_stopped_test_counts(self):
        result = self.verify([("s1-r1", False, {"log-1.txt": self.LOG, "events-1.jsonl": self.EVENTS})])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_missing_test_fails_and_names_its_run(self):
        result = self.verify([("s1-r1", False, {"log-1.txt": self.LOG, "events-1.jsonl": ""})])
        self.assertEqual(result.returncode, 1)
        self.assertIn("M.KernelTests/K1/run() did not execute. The map puts it in shard1-run1.", result.stdout)

    def test_test_in_two_attempts_fails(self):
        files = {"log-1.txt": self.LOG, "events-1.jsonl": self.EVENTS, "log-2.txt": self.LOG}
        result = self.verify([("s1-r1", False, files)])
        self.assertEqual(result.returncode, 1)
        self.assertIn("M.A/testOne executed more than once", result.stdout)

    def test_test_in_two_main_runs_fails(self):
        files = {"log-1.txt": self.LOG, "events-1.jsonl": self.EVENTS}
        result = self.verify([("s1-r1", False, files), ("s2-r1", False, {"log-1.txt": self.LOG})])
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than once", result.stdout)

    def test_extra_run_does_not_count_as_a_second_execution(self):
        files = {"log-1.txt": self.LOG, "events-1.jsonl": self.EVENTS}
        result = self.verify([("s1-r1", False, files), ("s2-r1", True, {"log-1.txt": self.LOG})])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_skipped_suite_counts_its_tests(self):
        skipped = events(("test", {"kind": "suite", "id": "M.KernelTests"}),
                         ("event", {"kind": "testSkipped", "testID": "M.KernelTests"}))
        result = self.verify([("s1-r1", False, {"log-1.txt": self.LOG, "events-1.jsonl": skipped})])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_list_reader_keeps_only_test_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.txt"
            path.write_text("[0/1] Planning build\nM.A/testOne\nM.free()\nBuild complete!\n")
            self.assertEqual(test_shards.read_list(path), ["M.A/testOne", "M.free()"])


if __name__ == "__main__":
    unittest.main()
