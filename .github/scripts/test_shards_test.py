#!/usr/bin/env python3
"""Tests for test_shards.py. They need no Xcode and no Swift build."""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_a_shard_with_two_rest_runs_fails(self):
        shards = [{"runs": [{"units": ["M.A"]}, {"rest": True}, {"rest": True}]}]
        with self.assertRaises(SystemExit):
            test_shards.resolve_runs(make_map(shards, groups=[]))

    def test_a_map_without_a_rest_run_fails(self):
        shards = [{"runs": [{"units": ["M.A"]}]}]
        with self.assertRaises(SystemExit):
            test_shards.resolve_runs(make_map(shards, groups=[]))

    def test_a_run_selects_what_one_of_its_parts_selects(self):
        shards = [{"runs": [{"groups": ["paged"], "units": ["M.A", "M.free()"], "tests": ["M.Paged/exclusive()"]},
                            {"units": ["M.KernelTests/K1"]}, {"rest": True}]}]
        tests = TESTS + ["M.AB/testOne", "M.free()/F.swift:1:1", "M.KernelTests/K10/run()", "N.A/testOne"]
        for run in test_shards.resolve_runs(make_map(shards))[:2]:
            for test in tests:
                self.assertEqual(test_shards.selects(run, test),
                                 any(test_shards.part_selects(part, test) for part in run["parts"]), (run["id"], test))

    def test_unit_pattern_escapes_the_id(self):
        self.assertEqual(test_shards.unit_pattern("M.KernelTests/K2"), r"^M\.KernelTests/K2/")
        self.assertEqual(test_shards.unit_pattern("M.free()"), r"^M\.free\(\)")


class RestSplitTests(unittest.TestCase):
    NEW = ["M.NewBig/a()", "M.NewBig/b()", "M.NewBig/c()", "M.NewMid/a()", "M.NewMid/b()"]
    REST_RUNS = {"shard1-run4", "shard2-run2"}

    def rest_owners(self, owners):
        return {test: owned for test, owned in owners.items() if owned[0] in self.REST_RUNS}

    def data(self, first=5, second=5):
        shards = json.loads(json.dumps(GOOD))
        shards[0]["runs"].append({"rest": True})
        shards[0]["estimated_seconds"], shards[1]["estimated_seconds"] = first, second
        return make_map(shards)

    def test_new_units_split_over_the_rest_runs_by_test_count(self):
        owners, problems = problems_of(self.data(), TESTS + self.NEW)
        self.assertEqual(problems, [])
        self.assertEqual(self.rest_owners(owners), {
            "M.NewBig/a()": ["shard1-run4"], "M.NewBig/b()": ["shard1-run4"], "M.NewBig/c()": ["shard1-run4"],
            "M.KernelTests/K2/run(dtype:)": ["shard1-run4"],
            "M.NewMid/a()": ["shard2-run2"], "M.NewMid/b()": ["shard2-run2"],
            "M.B/testOne": ["shard2-run2"], "M.free()": ["shard2-run2"]})

    def test_the_split_starts_from_the_estimated_seconds(self):
        owners, problems = problems_of(self.data(first=20), TESTS + self.NEW)
        self.assertEqual(problems, [])
        rest = self.rest_owners(owners)
        self.assertEqual(len(rest), 8)
        self.assertEqual({tuple(owned) for owned in rest.values()}, {("shard2-run2",)})

    def test_the_split_does_not_depend_on_the_order_of_the_list(self):
        tests = TESTS + self.NEW
        owners, _ = problems_of(self.data(), tests)
        self.assertEqual(problems_of(self.data(), tests[::-1])[0], owners)

    def test_each_rest_run_selects_its_share_and_together_every_test_once(self):
        tests = TESTS + self.NEW
        runs = test_shards.resolve_runs(self.data())
        owners, _ = test_shards.partition(runs, tests)
        shares = [test_shards.selection(run, runs, tests) for run in runs if run["rest"]]
        self.assertEqual(shares, [["M.KernelTests/K2/run(dtype:)", *self.NEW[:3]],
                                  ["M.B/testOne", "M.free()", *self.NEW[3:]]])
        rest = sorted(test for share in shares for test in share)
        self.assertEqual(rest, sorted(self.rest_owners(owners)))
        self.assertEqual(len(rest), len(set(rest)))

    def test_the_main_run_with_the_default_settings_runs_the_share(self):
        tests = TESTS + self.NEW
        runs = test_shards.resolve_runs(self.data())
        first = test_shards.shard_selections([r for r in runs if r["shard"] == 1], runs, tests)
        second = test_shards.shard_selections([r for r in runs if r["shard"] == 2], runs, tests)
        share1 = ["M.KernelTests/K2/run(dtype:)", *self.NEW[:3]]
        # Shard 1: unit (no Metal library) and paged (default) come before M.A, the last default run.
        self.assertEqual(first["shard1-run3"], (["M.A/testOne", "M.A/testTwo", "M.Paged/exclusive()"], share1, None))
        self.assertEqual(first["shard1-run4"], (share1, [], "shard1-run3"))
        self.assertEqual(first["shard1-run2"][1:], ([], None))
        # Shard 2: the extra exclusive pass sets an environment variable, so K1 runs the share.
        self.assertEqual(second["shard2-run1"][1:], (["M.B/testOne", "M.free()", *self.NEW[3:]], None))
        self.assertEqual(second["shard2-run2"][2], "shard2-run1")
        self.assertEqual(second["shard2-run3"][1:], ([], None))

    def test_a_shard_without_a_default_run_that_selects_tests_keeps_its_rest_run(self):
        groups = [{"name": "unit", "strict": True, "metallib": False, "filters": [r"\.UnitTests/"]},
                  {"name": "gone", "filters": [r"^M\.Gone/"]}]
        shards = [{"runs": [{"groups": ["unit"]}, {"rest": True}]},
                  {"runs": [{"groups": ["gone"]}, {"rest": True}]}]
        runs = test_shards.resolve_runs(make_map(shards, groups))
        for shard in (1, 2):
            selections = test_shards.shard_selections([r for r in runs if r["shard"] == shard], runs, TESTS)
            rest_id = f"shard{shard}-run2"
            self.assertIsNone(selections[rest_id][2])
            self.assertEqual(selections[f"shard{shard}-run1"][1], [])
        owners, _ = test_shards.partition(runs, TESTS)
        shares = [test_shards.shard_selections([r for r in runs if r["shard"] == s], runs, TESTS)[f"shard{s}-run2"][0]
                  for s in (1, 2)]
        self.assertEqual(sorted(shares[0] + shares[1]),
                         sorted(t for t, o in owners.items() if o[0] in ("shard1-run2", "shard2-run2")))

    def test_the_main_run_selects_the_share_by_full_id_after_its_own_filters(self):
        runs = test_shards.resolve_runs(self.data())
        k1 = next(r for r in runs if r["units"] == ["M.KernelTests/K1"])
        command = test_shards.run_command(k1, Path("/r"), 1, None, ["M.B/testOne", "M.NewMid/a()"])
        self.assertEqual(command[6:12], ["--filter", r"^M\.KernelTests/K1/", "--filter", r"^M\.B/testOne$",
                                        "--filter", r"^M\.NewMid/a\(\)"])

    def test_a_main_run_whose_rest_tests_did_not_execute_fails(self):
        runs = test_shards.resolve_runs(self.data())
        k1 = next(r for r in runs if r["units"] == ["M.KernelTests/K1"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "log-1.txt").write_text("")
            (folder / "events-1.jsonl").write_text(events(started("M.KernelTests/K1/run()/K.swift:1:1", 1),
                                                          ended("M.KernelTests/K1/run()/K.swift:1:1", 2)))
            tests = TESTS + self.NEW
            self.assertEqual(test_shards.run_problems(k1, ["M.KernelTests/K1/run()"], folder, tests, ["M.NewMid/a()"]),
                             ["the run executed none of the rest tests."])
            self.assertEqual(test_shards.run_problems(k1, ["M.KernelTests/K1/run()"], folder, tests), [])

    def run_shards(self, root, shards, stop=None, drop=()):
        """Run shards with a stand-in for swift test that executes what its filters select.

        `stop` makes the first attempt stop at that test (started, not ended); `drop` never executes.
        Returns each command as (run, log, executed tests) and each shard's exit code.
        """
        tests = TESTS + self.NEW
        commands = []

        def fake_stream(command, env, log, timeout):
            filters = [command[i + 1] for i, a in enumerate(command) if a == "--filter"]
            ran = [t for t in tests if any(re.search(f, t) for f in filters) and t not in drop]
            stopped = log.name == "log-1.txt" and stop in ran
            if stopped:
                ran = ran[:ran.index(stop) + 1]
            records = [e for i, t in enumerate(ran) for e in (started(t, i), ended(t, i + 0.5))]
            Path(command[command.index("--event-stream-output-path") + 1]).write_text(
                events(*(records[:-1] if stopped else records)))
            log.write_text("")
            commands.append((log.parent.name, log.name, ran))
            return 1 if stopped else 0

        (root / "map.json").write_text(json.dumps(self.data()))
        (root / "tests.txt").write_text("\n".join(tests) + "\n")
        done = subprocess.CompletedProcess([], 0, stdout="/bin\n")
        codes = {}
        with mock.patch.object(test_shards, "stream", fake_stream), \
                mock.patch.object(test_shards.subprocess, "run", return_value=done), \
                mock.patch.object(test_shards, "find_metallibs", return_value=[]), \
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(root), "GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            for shard in shards:
                args = argparse.Namespace(map=root / "map.json", shard=shard, results=root / "results" / str(shard),
                                          metallib=root / "mlx.metallib", list=root / "tests.txt")
                codes[shard] = test_shards.command_run(args)
        return commands, codes

    def verify_results(self, root):
        return subprocess.run([sys.executable, test_shards.__file__, "--map", str(root / "map.json"), "verify",
                               "--list", str(root / "tests.txt"), "--results", str(root / "results")],
                              capture_output=True, text=True, env={**os.environ, "GITHUB_STEP_SUMMARY": ""})

    def test_every_test_runs_once_and_no_rest_run_starts_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands, codes = self.run_shards(root, (1, 2))
            self.assertEqual(codes, {1: 0, 2: 0})
            started_runs = [name for name, _, _ in commands]
            self.assertNotIn("shard1-run4", started_runs)
            self.assertNotIn("shard2-run2", started_runs)
            self.assertIn("shard1-run3 runs the 4 tests", (root / "results/1/shard1-run4/log-1.txt").read_text())
            self.assertIn(("shard2-run1", "log-1.txt", ["M.B/testOne", "M.KernelTests/K1/run()", "M.free()",
                                                        *self.NEW[3:]]), commands)
            verify = self.verify_results(root)
            self.assertEqual(verify.returncode, 0, verify.stdout)
            self.assertIn("Each listed test executed exactly once in the main pass.", verify.stdout)

    def test_a_later_attempt_runs_the_own_and_rest_tests_that_did_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands, codes = self.run_shards(root, (1, 2), stop="M.B/testOne")
            self.assertEqual(codes, {1: 0, 2: 1})
            self.assertIn(("shard2-run1", "log-2.txt", ["M.KernelTests/K1/run()", "M.free()", *self.NEW[3:]]),
                          commands)
            verify = self.verify_results(root)
            self.assertEqual(verify.returncode, 0, verify.stdout)

    def test_a_main_run_fails_when_none_of_its_share_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, codes = self.run_shards(Path(directory), (2,), drop=("M.B/testOne", "M.free()", *self.NEW[3:]))
            self.assertEqual(codes, {2: 1})

    def test_a_run_with_its_own_process_or_a_preparation_does_not_run_the_share(self):
        groups = [{"name": "own", "own_process": True, "filters": [r"^M\.B/"]},
                  {"name": "prep", "before": [["true"]], "filters": [r"^M\.free\(\)"]}]
        shards = [{"runs": [{"units": ["M.A"]}, {"groups": ["own"]}, {"groups": ["prep"]}, {"rest": True}]}]
        runs = test_shards.resolve_runs(make_map(shards, groups))
        self.assertEqual(test_shards.rest_host(runs[-1], runs)["id"], "shard1-run1")

    def test_check_accepts_the_split_and_verify_names_the_rest_run_of_a_missing_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.json").write_text(json.dumps(self.data()))
            (root / "tests.txt").write_text("\n".join(TESTS + self.NEW) + "\n")
            tool = [sys.executable, test_shards.__file__, "--map", str(root / "map.json")]
            check = subprocess.run([*tool, "check", "--list", str(root / "tests.txt")],
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stdout)
            self.assertIn("8 tests are in no shard of the map, so the rest runs of 2 shards", check.stdout)
            verify = subprocess.run([*tool, "verify", "--list", str(root / "tests.txt"), "--results", str(root)],
                                    capture_output=True, text=True, env={**os.environ, "GITHUB_STEP_SUMMARY": ""})
            self.assertEqual(verify.returncode, 1)
            self.assertIn("M.NewMid/a() did not execute. The map puts it in shard2-run2.", verify.stdout)


class RunTests(unittest.TestCase):
    def test_exact_command_selects_each_test_by_its_id(self):
        runs = test_shards.resolve_runs(make_map(GOOD))
        paged = next(r for r in runs if r["groups"] == ["paged"])
        command = test_shards.run_command(paged, Path("/r"), 1, ["M.Paged/other()", "M.A/testOne"])
        self.assertEqual(command[:10], ["swift", "test", "--build-system", "native", "--skip-build", "--no-parallel",
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
        self.assertEqual(command[6:8], ["--filter", r"^M\.KernelTests/K1/"])

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
        self.assertEqual(command[6:10], ["--filter", r"^M\.A/testOne$", "--filter", r"^M\.free\(\)"])
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


class HostedShardTests(unittest.TestCase):
    def data(self):
        data = make_map(json.loads(json.dumps(GOOD)))
        data["hosted_tests"] = ["M.free()"]
        data["shards"].append({"runner": "hosted", "runs": [{"tests": ["M.free()"]}]})
        return data

    def command(self, data, *arguments, tests=TESTS):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.json").write_text(json.dumps(data))
            (root / "tests.txt").write_text("\n".join(tests) + "\n")
            args = [arg.replace("{list}", str(root / "tests.txt")) for arg in arguments]
            if args[0] == "plan":
                args += ["--output", str(root / "map.json")]
            result = subprocess.run(
                [sys.executable, test_shards.__file__, "--map", str(root / "map.json"), *args],
                capture_output=True, text=True)
            return result, json.loads((root / "map.json").read_text())

    def test_matrix_separates_native_and_hosted_shards(self):
        for runner, expected in [("all", [1, 2, 3]), ("native", [1, 2]), ("hosted", [3])]:
            result, _ = self.command(self.data(), "matrix", "--runner", runner)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_plan_preserves_the_exact_hosted_exception(self):
        data = self.data()
        result, planned = self.command(data, "plan", "--list", "{list}", "--shards", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(planned["groups"], data["groups"])
        self.assertEqual(planned["hosted_tests"], ["M.free()"])
        self.assertEqual(planned["shards"][-1]["runs"], [{"tests": ["M.free()"]}])
        self.assertEqual(planned["shards"][-1]["runner"], "hosted")
        self.assertEqual(len(planned["shards"]), 3)
        self.assertEqual([shard["runs"][-1] for shard in planned["shards"][:2]], [{"rest": True}] * 2)
        owners, problems = problems_of(planned)
        self.assertEqual(problems, [])
        self.assertEqual(owners["M.free()"], ["shard3-run1"])

    def test_hosted_runs_must_match_the_configured_identities(self):
        data = self.data()
        data["hosted_tests"] = ["M.A/testOne"]
        result, _ = self.command(data, "matrix")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must match hosted_tests exactly", result.stderr)

    def test_hosted_runs_cannot_silently_expand_a_suite(self):
        data = self.data()
        data["shards"][-1]["runs"] = [{"units": ["M.A"]}]
        result, _ = self.command(data, "matrix")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must name their exact test IDs", result.stderr)

    def test_plan_rejects_a_stale_hosted_identity(self):
        result, _ = self.command(self.data(), "plan", "--list", "{list}", "--shards", "2",
                                 tests=[test for test in TESTS if test != "M.free()"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configured hosted test is absent", result.stderr)


if __name__ == "__main__":
    unittest.main()
