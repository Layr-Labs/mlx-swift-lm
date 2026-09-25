#!/usr/bin/env bash
# Copy mlx.metallib to the places where the tests look for it.
#
# MLX first looks for mlx.metallib in the folder of the binary that contains
# MLX. For the tests, that binary is in the .xctest bundle, in Contents/MacOS.
# This script copies the library into every .xctest bundle in the bin folder,
# and into the bin folder for the other binaries.
#
# Usage: scripts/stage-test-metallib.sh <mlx.metallib> <swift build bin folder>
set -euo pipefail

if [[ $# -ne 2 || ! -f "$1" || ! -d "$2" ]]; then
    echo "usage: $0 <mlx.metallib> <swift build bin folder>" >&2
    exit 2
fi
metallib=$1
bin_directory=$2

cp "$metallib" "$bin_directory/mlx.metallib"
echo "Staged $bin_directory/mlx.metallib"

found=0
for bundle in "$bin_directory"/*.xctest; do
    [[ -d "$bundle" ]] || continue
    mkdir -p "$bundle/Contents/MacOS"
    cp "$metallib" "$bundle/Contents/MacOS/mlx.metallib"
    echo "Staged $bundle/Contents/MacOS/mlx.metallib"
    found=1
done
if [[ "$found" -ne 1 ]]; then
    echo "No .xctest bundle in $bin_directory. Run swift build --build-tests first." >&2
    exit 1
fi
