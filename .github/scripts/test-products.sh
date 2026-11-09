#!/usr/bin/env bash
# Hand the Fork Tests build from the build job to the test and coverage jobs
# as an artifact, as Layr-Labs/d-inference does (scripts/ci-test-products.py).
# The Actions cache cannot do it: Blacksmith's cache refuses this repository.
#
#   .github/scripts/test-products.sh pack ARCHIVE     (build job, after the build)
#   .github/scripts/test-products.sh unpack ARCHIVE   (test job, after checkout)
#
# The archive holds .build without what no test step reads: SwiftPM's bare
# clones (repositories), each target's <Target>.build directory (object
# files), ModuleCache, index, Modules, the host tools (-tool) and the .dSYM
# bundles. It keeps the test bundle, the build database, the checkouts and
# workspace-state.json. SwiftPM's products hold absolute paths and were built
# with one Xcode, so unpack fails when the workspace path or the developer
# directory is not the build job's.
set -euo pipefail

if [[ $# -ne 2 ]] || [[ $1 != pack && $1 != unpack ]]; then
    echo "usage: $0 pack|unpack ARCHIVE" >&2
    exit 2
fi
archive=$2
workspace=${GITHUB_WORKSPACE:?}
developer_dir=${DEVELOPER_DIR:-$(xcode-select -p)}
cd "$workspace"

if [[ $1 == pack ]]; then
    [[ -d .build ]] || { echo "No .build in $workspace. Run swift build first." >&2; exit 1; }
    printf '%s\n' "$workspace" > .build/fork-tests-workspace.txt
    printf '%s\n' "$developer_dir" > .build/fork-tests-developer-dir.txt
    mkdir -p "$(dirname "$archive")"
    start=$(date +%s)
    find .build \( -path .build/repositories \
        -o -path '.build/*-apple-*/*/ModuleCache' \
        -o -path '.build/*-apple-*/*/index' \
        -o -path '.build/*-apple-*/*/Modules' \
        -o -path '.build/*-apple-*/*/Modules-tool' \
        -o -path '.build/*-apple-*/*/*.build' \
        -o -path '.build/*-apple-*/*/*-tool' \
        -o -path '.build/*-apple-*/*/*-tool.product' \
        -o -name '*.dSYM' \) -prune -o \( -type f -o -type l \) -print0 \
        | tar --null -T - -cf - \
        | zstd -T0 -3 --long=31 -q -f -o "$archive"
    echo "Packed $(du -sh "$archive" | cut -f1) in $(( $(date +%s) - start )) s"
else
    start=$(date +%s)
    zstd -dc --long=31 -q "$archive" | tar -xpf -
    built_in=$(cat .build/fork-tests-workspace.txt)
    if [[ $built_in != "$workspace" ]]; then
        echo "::error::The build ran in $built_in and this job runs in $workspace. SwiftPM's products hold absolute paths."
        exit 1
    fi
    built_with=$(cat .build/fork-tests-developer-dir.txt)
    if [[ $built_with != "$developer_dir" ]]; then
        echo "::error::The build used the Xcode at $built_with and this job uses $developer_dir."
        exit 1
    fi
    echo "Unpacked in $(( $(date +%s) - start )) s"
fi
