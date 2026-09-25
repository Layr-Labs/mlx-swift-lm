#!/usr/bin/env bash
# Run one test suite with `swift test --skip-build --filter <filter>`.
#
# The script fails when the run executed zero tests or skipped a test.
# `swift test` exits with 0 when the filter matches no test, so the exit code
# alone does not show that the suite ran. The suites that use this script need
# no model weights. The caller sets any environment variable that a test
# needs: scripts/run-paged-kernel-tests.sh sets
# DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 for one test. A skipped test means
# that a precondition is broken, for example a missing Metal library.
#
# Usage: scripts/run-nested-suite.sh <filter> [more swift test arguments]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <filter> [more swift test arguments]" >&2
    exit 2
fi
filter=$1
shift

log=$(mktemp -t nested-suite)

# tee keeps the test output in the job log. pipefail keeps the exit code of
# swift test.
swift test --skip-build --filter "$filter" "$@" 2>&1 | tee "$log"

# XCTest prints "Executed 3 tests". Swift Testing prints "Test run with 3 tests".
if ! grep -qE 'Test run with [1-9][0-9]* test|Executed [1-9][0-9]* test' "$log"; then
    echo "::error::$filter executed zero tests."
    exit 1
fi

# XCTest prints "with 1 test skipped". Swift Testing prints one line for each
# skipped test, for example "✘ Test name() skipped: reason", and counts the
# skipped test as passed. The symbol at the start of that line is different
# in different Swift versions, so the pattern accepts any symbol there.
if grep -qE '^[^A-Za-z0-9]*(Test|Suite) .+ skipped[.:]|with [1-9][0-9]* tests? skipped|skipped [1-9][0-9]* test' "$log"; then
    echo "::error::$filter skipped one or more tests."
    exit 1
fi
