#!/usr/bin/env bash
# Test scripts/run-nested-suite.sh with a fake swift command.
#
# Each case gives the fake swift an output and an exit code. The case then
# checks the exit code of run-nested-suite.sh. The test needs no Xcode.
#
# Usage: scripts/test-run-nested-suite.sh
set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir "$work/bin"
cat > "$work/bin/swift" <<'FAKE'
#!/usr/bin/env bash
# Fake swift: record the arguments, print the output, exit with the code.
printf '%s\n' "$*" > "$FAKE_SWIFT_ARGUMENTS"
cat "$FAKE_SWIFT_OUTPUT"
exit "$FAKE_SWIFT_STATUS"
FAKE
chmod +x "$work/bin/swift"

total=0
failed=0

# expect <expected exit code> <case name> <swift exit code> <swift output>
expect() {
    local expected=$1 name=$2 swift_status=$3 output=$4 status=0
    total=$((total + 1))
    printf '%s\n' "$output" > "$work/output"
    FAKE_SWIFT_OUTPUT="$work/output" FAKE_SWIFT_STATUS="$swift_status" \
        FAKE_SWIFT_ARGUMENTS="$work/arguments" PATH="$work/bin:$PATH" \
        "$script_directory/run-nested-suite.sh" SomeTests --no-parallel \
        > "$work/log" 2>&1 || status=$?
    if [[ "$status" -eq "$expected" ]]; then
        echo "ok: $name"
    else
        echo "FAILED: $name: exit code $status, expected $expected"
        sed 's/^/    /' "$work/log"
        failed=$((failed + 1))
    fi
}

xctest_none=$'\t Executed 0 tests, with 0 failures (0 unexpected) in 0.000 (0.000) seconds'
testing_none='✔ Test run with 0 tests in 0 suites passed after 0.001 seconds.'
testing_pass='✔ Test run with 4 tests in 1 suite passed after 0.001 seconds.'

expect 0 "Swift Testing suite passes" 0 \
    "$xctest_none"$'\n'"$testing_pass"
expect 0 "XCTest suite passes" 0 \
    $'\t Executed 8 tests, with 0 failures (0 unexpected) in 0.004 (0.005) seconds\n'"$testing_none"
expect 1 "No test executed" 0 \
    "$xctest_none"$'\n'"$testing_none"
expect 1 "No test output" 0 ""
expect 1 "XCTest skip" 0 \
    $'\t Executed 3 tests, with 1 test skipped and 0 failures (0 unexpected) in 2.946 (2.947) seconds\n'"$testing_none"
expect 1 "Swift Testing skip with the Swift 6.1 symbol" 0 \
    '✘ Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect 1 "Swift Testing skip with the Swift 6.3 symbol" 0 \
    '➜ Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect 1 "Swift Testing skip without a reason" 0 \
    '➜ Test decodeMicrobenchmark() skipped.'$'\n'"$testing_pass"
expect 1 "Swift Testing skip without a symbol" 0 \
    'Test skip() skipped: "missing fixture"'$'\n'"$testing_pass"
expect 1 "Swift Testing suite skip" 0 \
    '➜ Suite "CBv2 paged safety" skipped: "missing fixture"'$'\n'"$testing_pass"
expect 0 "Test output that contains the word skipped" 0 \
    '◇ Test "scaled tensors on explicitly skipped paths are rejected" started.'$'\n''[cbv2-query-block] window=1024 dVsAbsolute=skipped'$'\n'"$testing_pass"
expect 3 "swift test fails" 3 \
    '✘ Test run with 4 tests in 1 suite failed after 0.001 seconds with 1 issue.'

total=$((total + 1))
arguments=$(cat "$work/arguments")
if [[ "$arguments" == "test --skip-build --filter SomeTests --no-parallel" ]]; then
    echo "ok: swift gets the filter and the extra arguments"
else
    echo "FAILED: swift got the arguments: $arguments"
    failed=$((failed + 1))
fi

echo "$((total - failed)) of $total cases passed."
[[ "$failed" -eq 0 ]]
