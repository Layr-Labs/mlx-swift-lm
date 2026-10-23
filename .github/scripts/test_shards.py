#!/usr/bin/env python3
"""Split the package tests into shards and check that each test runs once.

The Fork Tests workflow runs the tests in parallel jobs. .github/test-shards.json
assigns the tests to the jobs. Each shard is a list of runs, and each run is
one `swift test --skip-build` process.

A run selects tests with --filter and --skip regular expressions. It gets them
from two sources:

- "groups": named selections in the "groups" list of the map. A group can be
  strict, set environment variables, prepare a fixture, or run before the
  Metal library is staged. A strict group fails when it skips a test or
  executes zero tests. A group with "extra": true is an additional pass with
  special settings. Its tests also run in the main pass.
- "units": top-level suites or test classes, for example
  "MLXLMTests.CBv2EndToEndTests" or "MLXLMTests.KernelTests/GLM5NextMTPTests".
  A unit selects every test whose ID starts with "<unit>/".
- "tests": single test IDs. The plan command puts a test here when a group
  skip leaves it out of its suite, for example the exclusive paged kernel
  test in the main pass.

A run with "rest": true selects every test that no other main run selects.
So a new test always runs, also when nobody adds it to the map. The shard
lists the tests, finds those tests and runs only them, by their full IDs.

Commands:
  check   Read the output of `swift test list` and fail when a test is in no
          main run or in more than one, or when an extra run selects no test.
  run     Run one shard and write the logs and event streams to a folder.
          When a test process stops early, a new attempt runs the tests
          that did not execute. The run stays failed.
  verify  Read the reports of all shards and fail when a listed test did not
          execute exactly once in the main pass. Write the test durations.
  plan    Make a new map from the test list and the durations of a run.
  matrix  Print the shard numbers as a JSON list for the workflow matrix.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / ".github" / "test-shards.json"
SPECIAL = set(".^$*+?()[]{}|\\")


def escape(text: str) -> str:
    """Escape a test ID for NSRegularExpression, Swift Regex and Python re."""
    return "".join("\\" + c if c in SPECIAL else c for c in text)


def unit_pattern(unit: str) -> str:
    """Select the tests of a suite, or one test function outside a suite."""
    return "^" + escape(unit) + ("" if unit.endswith(")") else "/")


# Test IDs


def read_list(path: Path) -> list[str]:
    """Return the test IDs in the output of `swift test list`."""
    tests = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.[^ ]*/[^ ]", line) or re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][^ /]*\(.*\)$", line
        ):
            tests.append(line)
    return tests


def canonical(test_id: str) -> str:
    """Return one spelling for an ID from the list, a log or a report."""
    test_id = re.sub(r"/[^/]+\.swift:\d+:\d+$", "", test_id)
    return test_id.replace("/", ".")


def unit_of(test_id: str, containers: set[str]) -> str:
    parts = re.sub(r"/[^/]+\.swift:\d+:\d+$", "", test_id).split("/")
    if len(parts) == 1:
        return parts[0]
    if parts[0] in containers and len(parts) > 2:
        return parts[0] + "/" + parts[1]
    return parts[0]


def containers_of(tests: list[str]) -> set[str]:
    """Return the top-level suites that only hold other suites."""
    children: dict[str, set[str]] = {}
    leaves: set[str] = set()
    for test in tests:
        parts = re.sub(r"/[^/]+\.swift:\d+:\d+$", "", test).split("/")
        if len(parts) > 2:
            children.setdefault(parts[0], set()).add(parts[1])
        elif len(parts) == 2:
            leaves.add(parts[0])
    return {top for top, names in children.items() if len(names) > 1 and top not in leaves}


# The map


def load_map(path: Path) -> dict:
    data = json.loads(path.read_text())
    names = [group["name"] for group in data["groups"]]
    if len(names) != len(set(names)):
        raise SystemExit("error: two groups in the map have the same name.")
    return data


SETTINGS = (("extra", False), ("metallib", True), ("env", {}), ("before", []))


def resolve_runs(data: dict) -> list[dict]:
    """Return every run of the map with its parts and settings.

    A part is one group, one unit or one test. Each part keeps its own
    filters and skips, so a skip of one group does not change another part.
    """
    groups = {group["name"]: group for group in data["groups"]}
    runs = []
    for shard_index, shard in enumerate(data["shards"], start=1):
        for run_index, run in enumerate(shard["runs"], start=1):
            run_id = f"shard{shard_index}-run{run_index}"
            members = [groups[name] for name in run.get("groups", [])]
            for key, default in SETTINGS:
                values = {json.dumps(member.get(key, default)) for member in members}
                if run.get("units") or run.get("tests"):
                    values.add(json.dumps(default))
                if len(values) > 1:
                    raise SystemExit(f"error: {run_id} mixes groups or units with different {key}.")
            first = members[0] if members else {}
            parts = [{"name": m["name"], "filters": m.get("filters", []), "skips": m.get("skips", []),
                      "strict": bool(m.get("strict"))} for m in members]
            parts += [{"name": unit, "filters": [unit_pattern(unit)], "skips": [], "strict": False}
                      for unit in run.get("units", [])]
            parts += [{"name": test, "filters": [exact_pattern(test)], "skips": [], "strict": False}
                      for test in run.get("tests", [])]
            if not parts and not run.get("rest"):
                raise SystemExit(f"error: {run_id} selects nothing.")
            runs.append({
                "id": run_id,
                "shard": shard_index,
                "groups": run.get("groups", []),
                "units": run.get("units", []),
                "tests": run.get("tests", []),
                "rest": bool(run.get("rest")),
                "parts": parts,
                "strict": [part["name"] for part in parts if part["strict"]],
                "extra": bool(first.get("extra", False)),
                "metallib": bool(first.get("metallib", True)),
                "env": first.get("env", {}),
                "before": first.get("before", []),
                "timeout_minutes": max([m.get("timeout_minutes", 30) for m in members] or [30]),
            })
    if len([run for run in runs if run["rest"]]) != 1:
        raise SystemExit("error: the map must have exactly one rest run.")
    return runs


@functools.lru_cache(maxsize=None)
def compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def part_selects(part: dict, test_id: str) -> bool:
    if not any(compiled(f).search(test_id) for f in part["filters"]):
        return False
    return not any(compiled(s).search(test_id) for s in part["skips"])


def selects(run: dict, test_id: str) -> bool:
    """Return whether a run that is not the rest run selects a test."""
    return any(part_selects(part, test_id) for part in run["parts"])


def partition(runs: list[dict], tests: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Return the main runs that select each test, and the problems found.

    The rest run selects each test that no other main run selects.
    """
    owners: dict[str, list[str]] = {test: [] for test in tests}
    problems = []
    for run in runs:
        if run["rest"]:
            continue
        chosen = [test for test in tests if selects(run, test)]
        run["selected"] = len(chosen)
        for part in run["parts"]:
            if not any(part_selects(part, test) for test in chosen):
                problems.append(f"{run['id']}: {part['name']} selects no test.")
        if run["extra"]:
            continue
        for test in chosen:
            owners[test].append(run["id"])
    for run in runs:
        if run["rest"]:
            rest = [test for test, owned in owners.items() if not owned]
            run["selected"] = len(rest)
            for test in rest:
                owners[test].append(run["id"])
    for test, runs_of_test in owners.items():
        if len(runs_of_test) > 1:
            problems.append(f"{test} is in more than one main run: {', '.join(runs_of_test)}.")
    return owners, problems


def selection(run: dict, runs: list[dict], tests: list[str]) -> list[str]:
    """Return the listed tests that a run selects."""
    if run["rest"]:
        owners, _ = partition([other for other in runs if not other["rest"]], tests)
        return [test for test, owned in owners.items() if not owned]
    return [test for test in tests if selects(run, test)]


# check


def command_check(args: argparse.Namespace) -> int:
    runs = resolve_runs(load_map(args.map))
    tests = read_list(args.list)
    if not tests:
        print("::error::The test list is empty.")
        return 1
    _, problems = partition(runs, tests)
    for run in runs:
        kind = "extra" if run["extra"] else ("rest" if run["rest"] else "main")
        print(f"{run['id']}: {kind}, {run['selected']} tests")
        if run["rest"] and run["selected"]:
            print(f"::warning::{run['selected']} tests are in no shard of the map, so the rest run "
                  "runs them. Run test_shards.py plan to add them to the map.")
    print(f"{len(tests)} tests in the list.")
    for problem in problems[:200]:
        print(f"::error::{problem}")
    if problems:
        print(f"{len(problems)} problems. Run test_shards.py plan to make a new map.")
        return 1
    print("Each listed test is in exactly one main run.")
    return 0


# run

ERROR_LINE = re.compile(
    r"(/[^ ]+:[0-9]+: )?error: .*|Test .+ recorded an issue.*|(Fatal error|Segmentation fault).*"
    r"|(unexpected )?signal (code )?11.*"
)


def annotate_errors(log: Path) -> None:
    seen = []
    for line in log.read_text(errors="replace").splitlines():
        if "CoreData: error:" in line:
            continue
        match = ERROR_LINE.search(line)
        if match and match.group(0) not in seen:
            seen.append(match.group(0))
        if len(seen) >= 50:
            break
    for line in seen:
        print("::error::" + line.replace("%", "%25"))


def expand(value: str) -> str:
    return value.replace("{temp}", os.environ.get("RUNNER_TEMP", "/tmp"))


def run_command(run: dict, folder: Path, attempt: int, exact: list[str] | None) -> list[str]:
    """Return the swift test command of one attempt of a run.

    With `exact`, the command selects these tests by their full IDs. A later
    attempt runs only the tests that did not execute because the test process
    stopped.
    """
    outputs = ["--xunit-output", str(folder / f"xunit-{attempt}.xml"),
               "--event-stream-output-path", str(folder / f"events-{attempt}.jsonl"),
               "--event-stream-version", "0"]
    if exact is not None:
        patterns = [exact_pattern(test) for test in exact]
    else:
        patterns = [f for part in run["parts"] for f in part["filters"]]
    filters = [a for pattern in patterns for a in ("--filter", pattern)]
    return ["swift", "test", "--skip-build", "--no-parallel", *filters, *outputs]


def needs_exact(run: dict) -> bool:
    """Swift Testing matches a filter also against the source file of a test.

    A filter that starts with ^ matches only the start of the ID, so the
    source file cannot match it. The other runs select their tests by full
    ID, so that each run executes exactly the tests that the map gives it.
    """
    return run["rest"] or any(part["skips"] or any(not f.startswith("^") for f in part["filters"])
                              for part in run["parts"])


def exact_pattern(test_id: str) -> str:
    """Select one test. An XCTest ID has no parentheses, so it needs an end anchor."""
    test_id = re.sub(r"/[^/]+\.swift:\d+:\d+$", "", test_id)
    return "^" + escape(test_id) + ("" if test_id.endswith(")") else "$")


def run_environment(run: dict, all_runs: list[dict], profiles: Path) -> dict[str, str]:
    env = dict(os.environ)
    for other in all_runs:
        for key in other["env"]:
            env.pop(key, None)
    for key, value in run["env"].items():
        if value is not None:
            env[key] = expand(value)
    env["LLVM_PROFILE_FILE"] = str(profiles / "%p-%m.profraw")
    return env


def stream(command: list[str], env: dict[str, str], log: Path, timeout: float) -> int:
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace")
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            if time.monotonic() > deadline:
                process.kill()
                print(f"::error::The run took more than {timeout / 60:.0f} minutes.")
                return 124
        return process.wait()


def find_metallibs() -> list[str]:
    build = ROOT / ".build"
    return [str(p) for p in build.rglob("*.metallib")] if build.exists() else []


MAX_ATTEMPTS = 10


def command_run(args: argparse.Namespace) -> int:
    all_runs = resolve_runs(load_map(args.map))
    runs = [run for run in all_runs if run["shard"] == args.shard]
    if not runs:
        print(f"::error::The map has no shard {args.shard}.")
        return 1
    results = args.results
    results.mkdir(parents=True, exist_ok=True)
    listed = read_list(args.list)
    if not listed:
        print(f"::error::{args.list} lists no test.")
        return 1
    profiles = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "profraw"
    profiles.mkdir(parents=True, exist_ok=True)
    failed = []
    summary = []
    staged = False
    ordered = [run for run in runs if not run["metallib"]] + [run for run in runs if run["metallib"]]
    for run in ordered:
        if run["metallib"] and not staged:
            bin_path = subprocess.run(["swift", "build", "--show-bin-path"], cwd=ROOT, check=True,
                                      capture_output=True, text=True).stdout.strip()
            subprocess.run([str(ROOT / "scripts" / "stage-test-metallib.sh"), str(args.metallib), bin_path],
                           cwd=ROOT, check=True)
            staged = True
        if not run["metallib"]:
            libraries = find_metallibs()
            if libraries:
                print("\n".join(libraries))
                print(f"::error::The build folder contains a Metal library before {run['id']}.")
                failed.append(run["id"])
                continue
            subprocess.run([str(ROOT / "scripts" / "check-unit-test-files.sh")], cwd=ROOT, check=True)
        folder = results / run["id"]
        folder.mkdir(parents=True, exist_ok=True)
        meta = {k: run[k] for k in ("id", "shard", "groups", "units", "rest", "strict", "extra", "env")}
        meta["units"] = len(run["units"])
        name = ", ".join(run["groups"]) or ("rest" if run["rest"] else f"{len(run['units'])} units")
        print(f"::group::{run['id']}: {name}", flush=True)
        start = time.monotonic()
        status = 0
        env = run_environment(run, all_runs, profiles)
        for step in run["before"]:
            status = subprocess.run([expand(a) for a in step], cwd=ROOT, env=env).returncode or status
        attempt = 1
        chosen = selection(run, all_runs, listed)
        if run["rest"] and chosen:
            print(f"::warning::{len(chosen)} tests are in no shard of the map. The rest run runs "
                  "them. Run test_shards.py plan to add them to the map.", flush=True)
        if run["rest"] and not chosen:
            (folder / "log-1.txt").write_text("Each listed test is in another run of the map.\n")
            print("Each listed test is in another run of the map.", flush=True)
        elif not chosen:
            (folder / "log-1.txt").write_text("The run selects no listed test.\n")
            print(f"::error::{run['id']} selects no listed test.", flush=True)
            status = 1
        elif status == 0:
            exact = chosen if needs_exact(run) else None
            status = stream(run_command(run, folder, 1, exact), env, folder / "log-1.txt",
                            run["timeout_minutes"] * 60)
            # A test process that stops, for example on a fatal error, does
            # not run the tests after the one that stopped it. The next
            # attempt runs those tests. The run stays failed.
            while status != 0 and status != 124 and attempt < MAX_ATTEMPTS:
                done, _ = executed_tests(folder, listed=listed)
                remaining = [t for t in chosen if canonical(t) not in done]
                if not remaining:
                    break
                attempt += 1
                print(f"::warning::{run['id']}: the test process stopped. Attempt {attempt} runs "
                      f"the {len(remaining)} tests that did not execute.", flush=True)
                stream(run_command(run, folder, attempt, remaining), env, folder / f"log-{attempt}.txt",
                       run["timeout_minutes"] * 60)
            for problem in run_problems(run, chosen, folder, listed):
                print(f"::error::{run['id']}: {problem}", flush=True)
                status = status or 1
        else:
            (folder / "log-1.txt").write_text("The preparation command failed.\n")
        elapsed = time.monotonic() - start
        print("::endgroup::", flush=True)
        meta.update({"status": status, "attempts": attempt, "seconds": round(elapsed, 1)})
        (folder / "run.json").write_text(json.dumps(meta, indent=1))
        if status != 0:
            annotate_errors(folder / "log-1.txt")
            print(f"::error::{run['id']} ({name}) failed with exit code {status}.", flush=True)
            failed.append(run["id"])
        summary.append(f"| {run['id']} | {name} | {'strict' if run['strict'] and not run['units'] else 'main'}"
                       f"{' extra' if run['extra'] else ''} | {elapsed:.0f} s | {attempt} | "
                       f"{'failed' if status else 'passed'} |")
    profile_files = sorted(str(p) for p in profiles.glob("*.profraw"))
    if profile_files:
        subprocess.run(["xcrun", "llvm-profdata", "merge", "-sparse", *profile_files,
                        "-o", str(results / "coverage.profdata")], check=False)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(f"### Shard {args.shard}\n\n| Run | Selection | Mode | Time | Attempts | Result |\n"
                         "|---|---|---|---|---|---|\n" + "\n".join(summary) + "\n\n")
    return 1 if failed else 0


def run_problems(run: dict, chosen: list[str], folder: Path, listed: list[str]) -> list[str]:
    """Return the problems of a finished run, other than test failures.

    A run fails when it executed none of its tests. A strict group fails when
    it executed none of its tests or skipped one. A skipped test in a strict
    group means that a precondition is broken, for example a missing Metal
    library.
    """
    executions, skipped = executed_tests(folder, listed=listed)
    problems = []
    if not any(canonical(test) in executions for test in chosen):
        problems.append("the run executed zero tests.")
    for part in run["parts"]:
        if not part["strict"]:
            continue
        keys = [canonical(test) for test in chosen if part_selects(part, test)]
        if not any(key in executions for key in keys):
            problems.append(f"{part['name']} executed zero tests.")
        skips = [key for key in keys if key in skipped]
        if skips:
            problems.append(f"{part['name']} skipped {len(skips)} tests, for example {skips[0]}.")
    return problems


# verify

XCTEST_CASE = re.compile(
    r"Test Case '-\[([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)"
)
XCTEST_START = re.compile(r"Test Case '-\[([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\]' started")


def attempt_tests(folder: Path, attempt: int) -> tuple[dict[str, float], set[str], set[str]]:
    """Return the tests that one attempt executed with their durations, the skipped suites and tests.

    XCTest tests come from the log. Swift Testing tests come from the event
    stream, which the test process writes line by line, so a process that
    stops keeps the record of the tests before it. A test that started and
    did not end counts as executed: it is the test that stopped the process.
    """
    found: dict[str, float] = {}
    log = folder / f"log-{attempt}.txt"
    text = log.read_text(errors="replace") if log.exists() else ""
    for match in XCTEST_START.finditer(text):
        found.setdefault(canonical(f"{match[1]}.{match[2]}/{match[3]}"), 0.0)
    skipped: set[str] = set()
    for match in XCTEST_CASE.finditer(text):
        found[canonical(f"{match[1]}.{match[2]}/{match[3]}")] = float(match[5])
        if match[4] == "skipped":
            skipped.add(canonical(f"{match[1]}.{match[2]}/{match[3]}"))
    skipped_suites: set[str] = set()
    events = folder / f"events-{attempt}.jsonl"
    started: dict[str, float] = {}
    suites: set[str] = set()
    if events.exists():
        for line in events.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload", {})
            if record.get("kind") == "test":
                if payload.get("kind") == "suite":
                    suites.add(canonical(payload.get("id", "")))
                continue
            test_id = payload.get("testID")
            if not test_id:
                continue
            key = canonical(test_id)
            kind = payload.get("kind")
            instant = payload.get("instant", {}).get("absolute", 0.0)
            if key in suites:
                if kind == "testSkipped":
                    skipped_suites.add(key)
            elif kind == "testStarted":
                started[key] = instant
                found.setdefault(key, 0.0)
            elif kind == "testEnded" and key in started:
                found[key] = max(0.0, instant - started[key])
            elif kind == "testSkipped":
                found.setdefault(key, 0.0)
                skipped.add(key)
    return found, skipped_suites, skipped


def attempts_of(folder: Path) -> list[int]:
    numbers = [int(m[1]) for p in folder.glob("log-*.txt") if (m := re.match(r"log-(\d+)\.txt$", p.name))]
    return sorted(numbers)


def executed_tests(folder: Path, listed: list[str] | None = None) -> tuple[dict[str, list[float]], set[str]]:
    """Return each test that a run executed with one duration for each execution, and the skipped tests.

    A skipped test counts as executed. A test in a skipped suite counts as
    executed and skipped.
    """
    executions: dict[str, list[float]] = {}
    skipped: set[str] = set()
    for attempt in attempts_of(folder):
        found, skipped_suites, skipped_here = attempt_tests(folder, attempt)
        skipped |= skipped_here
        for key, seconds in found.items():
            executions.setdefault(key, []).append(seconds)
        if skipped_suites and listed:
            for test in listed:
                key = canonical(test)
                if key not in found and any(key.startswith(suite + ".") for suite in skipped_suites):
                    executions.setdefault(key, []).append(0.0)
                    skipped.add(key)
    return executions, skipped


def command_verify(args: argparse.Namespace) -> int:
    tests = read_list(args.list)
    listed = {canonical(test): test for test in tests}
    owners, _ = partition(resolve_runs(load_map(args.map)), tests)
    seen: dict[str, list[str]] = {}
    durations: dict[str, float] = {}
    problems = []
    stopped = []
    folders = sorted(p.parent for p in args.results.rglob("run.json"))
    for folder in folders:
        meta = json.loads((folder / "run.json").read_text())
        executed, _ = executed_tests(folder, listed=tests)
        if meta.get("attempts", 1) > 1:
            stopped.append(f"{meta['id']} ({meta['attempts']} attempts)")
        if meta["extra"]:
            if not executed:
                problems.append(f"The extra run {meta['id']} executed no test.")
            continue
        for test, runs in executed.items():
            seen.setdefault(test, []).extend([meta["id"]] * len(runs))
            durations[listed.get(test, test)] = sum(runs)
    missing_by_run: dict[str, int] = {}
    for key, test in sorted(listed.items()):
        runs = seen.get(key, [])
        if not runs:
            owner = ", ".join(owners.get(test, [])) or "no run"
            missing_by_run[owner] = missing_by_run.get(owner, 0) + 1
            problems.append(f"{test} did not execute. The map puts it in {owner}.")
        elif len(runs) > 1:
            problems.append(f"{test} executed more than once: {', '.join(runs)}.")
    unlisted = sorted(set(seen) - set(listed))
    print(f"{len(listed)} listed tests, {len(seen)} executed in {len(folders)} runs.")
    if stopped:
        print(f"::warning::A test process stopped and a new attempt continued in: {', '.join(stopped)}.")
    if unlisted:
        print(f"::warning::{len(unlisted)} executed tests are not in the list, for example {unlisted[:5]}.")
    for run, count in sorted(missing_by_run.items()):
        print(f"::error::{count} tests of {run} did not execute.")
    for problem in problems[:100]:
        print(f"::error::{problem}")
    if args.durations:
        args.durations.write_text(json.dumps(dict(sorted(durations.items())), indent=0) + "\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        once = sum(1 for key in listed if len(seen.get(key, [])) == 1)
        with open(step_summary, "a") as handle:
            handle.write(f"### Each test ran once\n\n{len(listed)} listed tests. {once} executed exactly "
                         f"once in the main pass. {len(problems)} problems.\n\n")
    if problems:
        print(f"{len(problems)} problems.")
        return 1
    print("Each listed test executed exactly once in the main pass.")
    return 0


# plan


def command_plan(args: argparse.Namespace) -> int:
    data = load_map(args.map)
    tests = read_list(args.list)
    durations = json.loads(args.durations.read_text()) if args.durations else {}
    containers = containers_of(tests)
    overhead = data.get("overhead_seconds", 25)
    groups = {group["name"]: group for group in data["groups"]}

    group_runs = []
    claimed: set[str] = set()
    for group in data["groups"]:
        part = {"filters": group.get("filters", []), "skips": group.get("skips", [])}
        chosen = [test for test in tests if part_selects(part, test)]
        if not group.get("extra"):
            claimed.update(chosen)
        group_runs.append((group, sum(durations.get(test, 1.0) for test in chosen)))

    units: dict[str, list[str]] = {}
    for test in tests:
        units.setdefault(unit_of(test, containers), []).append(test)
    free = {unit: members for unit, members in units.items() if not any(test in claimed for test in members)}
    leftover = [test for test in tests if test not in claimed and unit_of(test, containers) not in free]

    # Each shard has one main process. A group with its own settings, for
    # example an environment variable, runs in its own process.
    count = args.shards
    loads = [float(overhead)] * count
    shards: list[dict] = [{"groups": [], "alone": [], "units": [], "tests": []} for _ in range(count)]
    rest_shard = count - 1

    def lightest() -> int:
        return min(range(count), key=lambda index: loads[index])

    def alone(group: dict) -> bool:
        return any(group.get(key, default) != default for key, default in SETTINGS)

    for group, seconds in sorted(group_runs, key=lambda item: -item[1]):
        index = lightest()
        if alone(group):
            shards[index]["alone"].append(group["name"])
            loads[index] += seconds + overhead
        else:
            shards[index]["groups"].append(group["name"])
            loads[index] += seconds

    for unit, members in sorted(free.items(), key=lambda item: -sum(durations.get(t, 1.0) for t in item[1])):
        index = lightest()
        shards[index]["units"].append(unit)
        loads[index] += sum(durations.get(test, 1.0) for test in members)

    for test in leftover:
        index = lightest()
        shards[index]["tests"].append(test)
        loads[index] += durations.get(test, 1.0)

    data["shards"] = []
    for index, shard in enumerate(shards):
        runs = [{"groups": [name]} for name in shard["alone"] if not groups[name].get("metallib", True)]
        if shard["groups"] or shard["units"] or shard["tests"]:
            run = {"groups": sorted(shard["groups"])} if shard["groups"] else {}
            run["units"] = sorted(shard["units"])
            if shard["tests"]:
                run["tests"] = sorted(shard["tests"])
            runs.append(run)
        runs += [{"groups": [name]} for name in shard["alone"] if groups[name].get("metallib", True)]
        if index == rest_shard:
            runs.append({"rest": True})
        data["shards"].append({"estimated_seconds": round(loads[index]), "runs": runs})
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for index, load in enumerate(loads, start=1):
        print(f"shard {index}: about {load:.0f} s")
    return 0


def command_matrix(args: argparse.Namespace) -> int:
    data = load_map(args.map)
    resolve_runs(data)
    print(json.dumps(list(range(1, len(data["shards"]) + 1))))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--list", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--metallib", type=Path, required=True)
    run.add_argument("--list", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--list", type=Path, required=True)
    verify.add_argument("--results", type=Path, required=True)
    verify.add_argument("--durations", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("--list", type=Path, required=True)
    plan.add_argument("--durations", type=Path)
    plan.add_argument("--shards", type=int, required=True)
    plan.add_argument("--output", type=Path, default=DEFAULT_MAP)
    commands.add_parser("matrix")
    args = parser.parse_args()
    return {"check": command_check, "run": command_run, "verify": command_verify,
            "plan": command_plan, "matrix": command_matrix}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
