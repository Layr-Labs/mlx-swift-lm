#!/usr/bin/env bash
# Test scripts/run-nested-suite.sh and scripts/run-paged-kernel-tests.sh with
# a fake swift command.
#
# The fake swift prints a given output and exits with a given code. Each case
# sets them, runs one of the scripts, and checks its exit code. Some cases
# also check the calls that the fake swift got. The test needs no Xcode.
#
# Usage: scripts/test-run-nested-suite.sh
set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cases=$work/cases
calls=$work/calls
mkdir -p "$work/bin" "$cases"
cat > "$work/bin/swift" <<'FAKE'
#!/usr/bin/env bash
# Fake swift. Record the arguments and one environment variable. Then print
# the output and exit with the code that the case gave for the --filter value,
# or for "default".
filter=""
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--filter" ]]; then
        filter=$argument
    fi
    previous=$argument
done
echo "$* | DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=${DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST-unset}" >> "$FAKE_SWIFT_CALLS"
directory=$FAKE_SWIFT_CASES/default
if [[ -n "$filter" && -d "$FAKE_SWIFT_CASES/$filter" ]]; then
    directory=$FAKE_SWIFT_CASES/$filter
fi
cat "$directory/output"
exit "$(cat "$directory/status")"
FAKE
chmod +x "$work/bin/swift"

total=0
failed=0

# give <filter or "default"> <swift exit code> <swift output>
give() {
    mkdir -p "$cases/$1"
    printf '%s\n' "$3" > "$cases/$1/output"
    echo "$2" > "$cases/$1/status"
}

# check <expected exit code> <case name> <command> [arguments]
# The command starts without DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST.
check() {
    local expected=$1 name=$2 status=0
    shift 2
    total=$((total + 1))
    : > "$calls"
    env -u DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST \
        FAKE_SWIFT_CASES="$cases" FAKE_SWIFT_CALLS="$calls" PATH="$work/bin:$PATH" \
        "$@" > "$work/log" 2>&1 || status=$?
    if [[ "$status" -eq "$expected" ]]; then
        echo "ok: $name"
    else
        echo "FAILED: $name: exit code $status, expected $expected"
        sed 's/^/    /' "$work/log"
        failed=$((failed + 1))
    fi
    rm -rf "${cases:?}"/*
}

# expect_calls <case name> <expected calls, one line for each call>
expect_calls() {
    total=$((total + 1))
    if [[ "$(cat "$calls")" == "$2" ]]; then
        echo "ok: $1"
    else
        echo "FAILED: $1: the fake swift got these calls:"
        sed 's/^/    /' "$calls"
        failed=$((failed + 1))
    fi
}

# expect_suite <expected exit code> <case name> <swift exit code> <swift output>
expect_suite() {
    give default "$3" "$4"
    check "$1" "$2" "$script_directory/run-nested-suite.sh" SomeTests --no-parallel
}

xctest_none=$'\t Executed 0 tests, with 0 failures (0 unexpected) in 0.000 (0.000) seconds'
testing_none='✔ Test run with 0 tests in 0 suites passed after 0.001 seconds.'
testing_pass='✔ Test run with 4 tests in 1 suite passed after 0.001 seconds.'

# run-nested-suite.sh

expect_suite 0 "Swift Testing suite passes" 0 \
    "$xctest_none"$'\n'"$testing_pass"
expect_calls "swift gets the filter and the extra arguments" \
    "test --skip-build --filter SomeTests --no-parallel | DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=unset"
expect_suite 0 "XCTest suite passes" 0 \
    $'\t Executed 8 tests, with 0 failures (0 unexpected) in 0.004 (0.005) seconds\n'"$testing_none"
expect_suite 1 "No test executed" 0 \
    "$xctest_none"$'\n'"$testing_none"
expect_suite 1 "No test output" 0 ""
expect_suite 1 "XCTest skips one test" 0 \
    $'\t Executed 3 tests, with 1 test skipped and 0 failures (0 unexpected) in 2.946 (2.947) seconds\n'"$testing_none"
expect_suite 1 "XCTest skips two tests" 0 \
    $'\t Executed 3 tests, with 2 tests skipped and 0 failures (0 unexpected) in 2.946 (2.947) seconds\n'"$testing_none"
expect_suite 1 "Swift Testing skip with the Swift 6.1 symbol" 0 \
    '✘ Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect_suite 1 "Swift Testing skip with the Swift 6.3 symbol" 0 \
    '➜ Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect_suite 1 "Swift Testing skip without a reason" 0 \
    '➜ Test decodeMicrobenchmark() skipped.'$'\n'"$testing_pass"
expect_suite 1 "Swift Testing skip without a symbol" 0 \
    'Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect_suite 1 "Swift Testing suite skip" 0 \
    '➜ Suite "CBv2 paged safety" skipped: "missing fixture"'$'\n'"$testing_pass"
expect_suite 0 "Test output that contains the word skipped" 0 \
    '◇ Test "scaled tensors on explicitly skipped paths are rejected" started.'$'\n''[cbv2-query-block] window=1024 dVsAbsolute=skipped'$'\n'"$testing_pass"
expect_suite 3 "swift test fails" 3 \
    '✘ Test run with 4 tests in 1 suite failed after 0.001 seconds with 1 issue.'

# run-paged-kernel-tests.sh

kernel_pass='✔ Test run with 14 tests in 1 suite passed after 0.001 seconds.'
kernel_fail='✘ Test run with 14 tests in 1 suite failed after 0.001 seconds with 1 issue.'
single_pass='✔ Test run with 1 test in 1 suite passed after 0.001 seconds.'
single_fail='✘ Test run with 1 test in 1 suite failed after 0.001 seconds with 1 issue.'
kernel_calls="test --skip-build --filter CBv2PagedKernelTests --no-parallel --skip decodeBatchCompositionInvariance | DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=unset
test --skip-build --filter decodeBatchCompositionInvariance --no-parallel | DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1"

give CBv2PagedKernelTests 0 "$kernel_pass"
give decodeBatchCompositionInvariance 0 "$single_pass"
check 0 "Paged kernel tests pass" \
    env DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 "$script_directory/run-paged-kernel-tests.sh"
expect_calls "The first process runs without the variable, the second with it" "$kernel_calls"

give CBv2PagedKernelTests 1 "$kernel_fail"
give decodeBatchCompositionInvariance 0 "$single_pass"
check 1 "Paged kernel tests fail when the first process fails" \
    "$script_directory/run-paged-kernel-tests.sh"
expect_calls "The second process runs after the first process fails" "$kernel_calls"

give CBv2PagedKernelTests 0 "$kernel_pass"
give decodeBatchCompositionInvariance 1 "$single_fail"
check 1 "Paged kernel tests fail when the second process fails" \
    "$script_directory/run-paged-kernel-tests.sh"

give CBv2PagedKernelTests 0 "$xctest_none"$'\n'"$testing_none"
give decodeBatchCompositionInvariance 0 "$single_pass"
check 1 "Paged kernel tests fail when the first process executes zero tests" \
    "$script_directory/run-paged-kernel-tests.sh"

echo "$((total - failed)) of $total cases passed."
[[ "$failed" -eq 0 ]]
