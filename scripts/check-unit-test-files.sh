#!/usr/bin/env bash
# Check that each Swift file under a Tests/<Target>/Unit/ folder declares its
# code in `extension UnitTests`. The CI step "Unit tests (no GPU)" selects the
# unit tests by the name of the UnitTests suite, so a unit test outside that
# suite would not run in that step.
#
# Usage: scripts/check-unit-test-files.sh [repository root]
set -euo pipefail

root=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
bad=0
while IFS= read -r -d '' file; do
    if ! grep -q '^extension UnitTests ' "$file"; then
        echo "::error file=${file#"$root"/}::This file is under a Unit folder but has no 'extension UnitTests'."
        bad=1
    fi
done < <(find "$root"/Tests/*/Unit -name '*.swift' -print0 2>/dev/null)
exit "$bad"
