#!/usr/bin/env bash
# Check that each Swift file under a Tests/<Target>/Unit/ folder declares its
# code in `extension UnitTests`. The CI step "Unit tests (no GPU)" selects the
# unit tests by the name of the UnitTests suite, so a unit test outside that
# suite would not run in that step.
#
# Also check that each Tests/<Target>/TestTypeTags.swift declares the same tags
# and the same UnitTests suite as Tests/MLXLMTests/TestTypeTags.swift. The
# comments in the files can be different.
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

# declarations <file>: the lines that declare a tag or the UnitTests suite.
declarations() {
    grep -E '@Tag static var|@Suite\(\.tags\(\.unit\)\)|enum UnitTests' "$1" | sed -E 's/^[[:space:]]+//'
}

reference=$root/Tests/MLXLMTests/TestTypeTags.swift
if [[ -f "$reference" ]]; then
    for file in "$root"/Tests/*/TestTypeTags.swift; do
        [[ "$file" == "$reference" ]] && continue
        if ! diff <(declarations "$reference") <(declarations "$file"); then
            echo "::error file=${file#"$root"/}::The tag declarations are not the same as in Tests/MLXLMTests/TestTypeTags.swift."
            bad=1
        fi
    done
fi
exit "$bad"
