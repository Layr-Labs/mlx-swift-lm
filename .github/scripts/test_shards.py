#!/usr/bin/env python3
"""Split the package tests into shards and check that each test runs once.

The Fork Tests workflow runs the tests in parallel jobs. .github/test-shards.json
assigns the tests to the jobs. Each shard is a list of runs, and each run is
one `swift test --skip-build` process.

A run selects tests with --filter and --skip regular expressions. It gets them
from two sources:

- "groups": named selections in the "groups" list of the map. A group can be
  strict, set environment variables, prepare a fixture, or run before the
  Metal library is staged. A strict run uses scripts/run-nested-suite.sh, which
  fails when the run skips a test or executes zero tests. A group with
  "extra": true is an additional pass with special settings. Its tests also
  run in the main pass.
- "units": top-level suites or test classes, for example
  "MLXLMTests.CBv2EndToEndTests" or "MLXLMTests.KernelTests/GLM5NextMTPTests".
  A unit selects every test whose ID starts with "<unit>/".

A run with "rest": true selects every test that no other main run selects.
So a new test always runs, also when nobody adds it to the map.

Commands:
  check   Read the output of `swift test list` and fail when a test is in no
          main run or in more than one, or when an extra run selects no test.
  run     Run one shard and write the logs and test reports to a folder.
  verify  Read the reports of all shards and fail when a listed test did not
          execute exactly once in the main pass. Write the test durations.
  plan    Make a new map from the test list and the durations of a run.
  matrix  Print the shard numbers as a JSON list for the workflow matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
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


def resolve_runs(data: dict) -> list[dict]:
    """Return every run of the map with its filters, skips and settings."""
    groups = {group["name"]: group for group in data["groups"]}
    runs = []
    for shard_index, shard in enumerate(data["shards"], start=1):
        for run_index, run in enumerate(shard["runs"], start=1):
            members = [groups[name] for name in run.get("groups", [])]
            settings = {
                key: {json.dumps(member.get(key, default)) for member in members}
                for key, default in (("strict", False), ("extra", False), ("metallib", True),
                                     ("env", {}), ("before", []))
            }
            for key, values in settings.items():
                if len(values) > 1:
                    raise SystemExit(
                        f"error: shard {shard_index} run {run_index} mixes groups with different {key}."
                    )
            first = members[0] if members else {}
            if run.get("units") and first.get("strict"):
                raise SystemExit(f"error: shard {shard_index} run {run_index} mixes units and strict groups.")
            filters = [f for member in members for f in member.get("filters", [])]
            filters += [unit_pattern(unit) for unit in run.get("units", [])]
            runs.append({
                "id": f"shard{shard_index}-run{run_index}",
                "shard": shard_index,
                "groups": run.get("groups", []),
                "units": run.get("units", []),
                "rest": bool(run.get("rest")),
                "filters": filters,
                "skips": [s for member in members for s in member.get("skips", [])],
                "strict": bool(first.get("strict", False)),
                "extra": bool(first.get("extra", False)),
                "metallib": bool(first.get("metallib", True)),
                "env": first.get("env", {}),
                "before": first.get("before", []),
                "timeout_minutes": max([m.get("timeout_minutes", 30) for m in members] or [30]),
            })
    rests = [run for run in runs if run["rest"]]
    if len(rests) != 1:
        raise SystemExit("error: the map must have exactly one rest run.")
    main_filters = [f for run in runs if not run["extra"] and not run["rest"] for f in run["filters"]]
    rests[0]["skips"] = main_filters
    for run in runs:
        if not run["filters"] and not run["rest"]:
            raise SystemExit(f"error: {run['id']} selects nothing.")
    return runs


def selects(run: dict, test_id: str) -> bool:
    if run["filters"] and not any(re.search(f, test_id) for f in run["filters"]):
        return False
    return not any(re.search(s, test_id) for s in run["skips"])


def partition(runs: list[dict], tests: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Return the main runs that select each test, and the problems found."""
    owners: dict[str, list[str]] = {test: [] for test in tests}
    problems = []
    for run in runs:
        chosen = [test for test in tests if selects(run, test)]
        run["selected"] = len(chosen)
        for unit in run["units"]:
            if not any(re.search(unit_pattern(unit), test) for test in chosen):
                problems.append(f"{run['id']}: the unit {unit} selects no test.")
        if run["extra"]:
            if not chosen:
                problems.append(f"{run['id']} ({', '.join(run['groups'])}) selects no test.")
            continue
        for test in chosen:
            owners[test].append(run["id"])
    for test, runs_of_test in owners.items():
        if not runs_of_test:
            problems.append(f"{test} is in no main run.")
        elif len(runs_of_test) > 1:
            problems.append(f"{test} is in more than one main run: {', '.join(runs_of_test)}.")
    return owners, problems


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


def run_command(run: dict, results: Path) -> list[str]:
    folder = results / run["id"]
    xunit = ["--xunit-output", str(folder / "xunit.xml")]
    skips = [a for s in run["skips"] for a in ("--skip", s)]
    if run["strict"]:
        combined = "|".join(f"(?:{f})" for f in run["filters"])
        return [str(ROOT / "scripts" / "run-nested-suite.sh"), combined, "--no-parallel", *skips, *xunit]
    filters = [a for f in run["filters"] for a in ("--filter", f)]
    return ["swift", "test", "--skip-build", "--no-parallel", *filters, *skips, *xunit]


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
            handle.write(line)
            if time.monotonic() > deadline:
                process.kill()
                print(f"::error::The run took more than {timeout / 60:.0f} minutes.")
                return 124
        return process.wait()


def find_metallibs() -> list[str]:
    build = ROOT / ".build"
    return [str(p) for p in build.rglob("*.metallib")] if build.exists() else []


def command_run(args: argparse.Namespace) -> int:
    all_runs = resolve_runs(load_map(args.map))
    runs = [run for run in all_runs if run["shard"] == args.shard]
    if not runs:
        print(f"::error::The map has no shard {args.shard}.")
        return 1
    results = args.results
    results.mkdir(parents=True, exist_ok=True)
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
        (folder / "run.json").write_text(json.dumps({k: run[k] for k in (
            "id", "shard", "groups", "units", "rest", "strict", "extra", "env")}, indent=1))
        name = ", ".join(run["groups"]) or ("rest" if run["rest"] else f"{len(run['units'])} units")
        print(f"::group::{run['id']}: {name}")
        start = time.monotonic()
        status = 0
        env = run_environment(run, all_runs, profiles)
        for step in run["before"]:
            status = subprocess.run([expand(a) for a in step], cwd=ROOT, env=env).returncode or status
        log = folder / "log.txt"
        if status == 0:
            status = stream(run_command(run, results), env, log, run["timeout_minutes"] * 60)
        else:
            log.write_text("The preparation command failed.\n")
        elapsed = time.monotonic() - start
        print("::endgroup::")
        text = log.read_text(errors="replace")
        if status == 0 and not run["strict"] and not re.search(
            r"Executed [1-9][0-9]* tests?|Test run with [1-9][0-9]* tests?", text
        ):
            print(f"::error::{run['id']} executed zero tests.")
            status = 1
        if status != 0:
            annotate_errors(log)
            print(f"::error::{run['id']} ({name}) failed with exit code {status}.")
            failed.append(run["id"])
        summary.append(f"| {run['id']} | {name} | {'strict' if run['strict'] else 'main'}"
                       f"{' extra' if run['extra'] else ''} | {elapsed:.0f} s | "
                       f"{'failed' if status else 'passed'} |")
    profile_files = sorted(str(p) for p in profiles.glob("*.profraw"))
    if profile_files:
        subprocess.run(["xcrun", "llvm-profdata", "merge", "-sparse", *profile_files,
                        "-o", str(results / "coverage.profdata")], check=False)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(f"### Shard {args.shard}\n\n| Run | Selection | Mode | Time | Result |\n"
                         "|---|---|---|---|---|\n" + "\n".join(summary) + "\n\n")
    return 1 if failed else 0


# verify

XCTEST_CASE = re.compile(
    r"Test Case '-\[([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)"
)
XCTEST_START = re.compile(r"Test Case '-\[([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\]' started")


def executed_tests(folder: Path) -> dict[str, float]:
    """Return the canonical ID and duration of each test that a run executed."""
    found: dict[str, float] = {}
    log = folder / "log.txt"
    text = log.read_text(errors="replace") if log.exists() else ""
    for match in XCTEST_START.finditer(text):
        found.setdefault(canonical(f"{match[1]}.{match[2]}/{match[3]}"), 0.0)
    for match in XCTEST_CASE.finditer(text):
        found[canonical(f"{match[1]}.{match[2]}/{match[3]}")] = float(match[5])
    for report in folder.glob("*.xml"):
        try:
            root = ElementTree.parse(report).getroot()
        except ElementTree.ParseError:
            print(f"::warning::{report} is not valid XML.")
            continue
        for case in root.iter("testcase"):
            name = case.get("name", "")
            if not name.endswith(")"):
                continue
            test = canonical(f"{case.get('classname', '')}.{name}")
            found[test] = found.get(test, 0.0) + float(case.get("time", "0") or 0)
    return found


def command_verify(args: argparse.Namespace) -> int:
    tests = read_list(args.list)
    listed = {canonical(test): test for test in tests}
    seen: dict[str, list[str]] = {}
    durations: dict[str, float] = {}
    extras_without_tests = []
    folders = sorted(p.parent for p in args.results.rglob("run.json"))
    for folder in folders:
        meta = json.loads((folder / "run.json").read_text())
        executed = executed_tests(folder)
        if meta["extra"]:
            if not executed:
                extras_without_tests.append(meta["id"])
            continue
        for test, seconds in executed.items():
            seen.setdefault(test, []).append(meta["id"])
            durations[listed.get(test, test)] = seconds
    problems = []
    for key, test in sorted(listed.items()):
        runs = seen.get(key, [])
        if not runs:
            problems.append(f"{test} did not execute.")
        elif len(runs) > 1:
            problems.append(f"{test} executed in more than one run: {', '.join(runs)}.")
    unlisted = sorted(set(seen) - set(listed))
    for run in extras_without_tests:
        problems.append(f"The extra run {run} executed no test.")
    print(f"{len(listed)} listed tests, {len(seen)} executed in {len(folders)} runs.")
    if unlisted:
        print(f"::warning::{len(unlisted)} executed tests are not in the list, for example {unlisted[:5]}.")
    for problem in problems[:200]:
        print(f"::error::{problem}")
    if args.durations:
        args.durations.write_text(json.dumps(dict(sorted(durations.items())), indent=0) + "\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(f"### Test coverage of the shards\n\n{len(listed)} listed tests. "
                         f"{len(listed) - len(problems)} executed exactly once in the main pass. "
                         f"{len(problems)} problems.\n\n")
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
        run = {"filters": group.get("filters", []), "skips": group.get("skips", []), "rest": False}
        chosen = [test for test in tests if selects(run, test)]
        if not group.get("extra"):
            claimed.update(chosen)
        group_runs.append((group, sum(durations.get(test, 1.0) for test in chosen)))

    units: dict[str, list[str]] = {}
    for test in tests:
        units.setdefault(unit_of(test, containers), []).append(test)
    free = {unit: members for unit, members in units.items() if not any(test in claimed for test in members)}
    leftover = sum(durations.get(test, 1.0) for test in tests
                   if test not in claimed and unit_of(test, containers) not in free)

    count = args.shards
    loads = [0.0] * count
    shards: list[dict] = [{"strict": [], "lenient": [], "alone": [], "units": []} for _ in range(count)]
    rest_shard = count - 1
    loads[rest_shard] += leftover + overhead

    def lightest() -> int:
        return min(range(count), key=lambda index: loads[index])

    for group, seconds in sorted(group_runs, key=lambda item: -item[1]):
        index = lightest()
        if (group.get("extra") or group.get("env") or group.get("before") or not group.get("metallib", True)
                or (group.get("skips") and not group.get("strict"))):
            shards[index]["alone"].append(group["name"])
            loads[index] += seconds + overhead
        elif group.get("strict"):
            if not shards[index]["strict"]:
                loads[index] += overhead
            shards[index]["strict"].append(group["name"])
            loads[index] += seconds
        else:
            if not shards[index]["lenient"] and not shards[index]["units"]:
                loads[index] += overhead
            shards[index]["lenient"].append(group["name"])
            loads[index] += seconds

    for unit, members in sorted(free.items(), key=lambda item: -sum(durations.get(t, 1.0) for t in item[1])):
        index = lightest()
        if not shards[index]["lenient"] and not shards[index]["units"]:
            loads[index] += overhead
        shards[index]["units"].append(unit)
        loads[index] += sum(durations.get(test, 1.0) for test in members)

    data["shards"] = []
    for index, shard in enumerate(shards):
        runs = [{"groups": [name]} for name in shard["alone"] if not groups[name].get("metallib", True)]
        if shard["strict"]:
            runs.append({"groups": sorted(shard["strict"])})
        if shard["lenient"] or shard["units"]:
            run = {"groups": sorted(shard["lenient"])} if shard["lenient"] else {}
            run["units"] = sorted(shard["units"])
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
