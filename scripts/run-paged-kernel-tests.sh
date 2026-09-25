#!/usr/bin/env bash
# Run CBv2PagedKernelTests in two test processes.
#
# decodeBatchCompositionInvariance compares GPU results bit for bit. It runs
# only when DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1, and it must run alone in
# its process. The first process runs all other tests in the suite with the
# variable unset. The second process runs that one test with the variable set.
# The second process runs even when the first one fails.
set -uo pipefail
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
status=0

env -u DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST \
    "$script_directory/run-nested-suite.sh" CBv2PagedKernelTests --no-parallel \
    --skip decodeBatchCompositionInvariance || status=$?

env DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 \
    "$script_directory/run-nested-suite.sh" decodeBatchCompositionInvariance \
    --no-parallel || status=$?

exit "$status"
