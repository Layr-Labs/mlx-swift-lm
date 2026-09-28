#!/usr/bin/env bash
# Build the full MLX Metal library (mlx.metallib) with cmake.
#
# `swift build` makes no Metal library. The pinned mlx-swift excludes the MLX
# kernel sources and nojit_kernels.cpp from its Cmlx target, so SwiftPM builds
# MLX in JIT mode. Without a staged mlx.metallib, every MLX operation in the
# tests traps with "Failed to load the default metallib". (xcodebuild compiles
# the .metal files in Source/Cmlx/mlx-generated into a partial
# default.metallib of about 459 symbols. This workflow does not use
# xcodebuild.)
#
# This script compiles every kernel ahead of time (-DMLX_METAL_JIT=OFF). It
# uses the MLX source that SwiftPM checked out for the mlx-swift revision in
# Package.swift.
#
# Usage: scripts/build-metallib.sh <output file>
# Run it from the package root after `swift build`.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <output file>" >&2
    exit 2
fi
output=$1

mlx_source=.build/checkouts/mlx-swift/Source/Cmlx/mlx
# MLX compiles the NAX kernels only for a deployment target of 26.2 or later.
deployment_target=26.2
# The full library has about 17,000 symbols. A JIT-mode build
# (-DMLX_METAL_JIT=ON) or an incomplete build has far fewer. The check at the
# end rejects a library with fewer than 10,000 symbols.
minimum_symbols=10000

if [[ ! -f "$mlx_source/mlx/version.h" ]]; then
    echo "No MLX source at $mlx_source. Run swift build first." >&2
    exit 1
fi

build_directory=$(mktemp -d)
trap 'rm -rf "$build_directory"' EXIT

echo "MLX source: $mlx_source at $(git -C "$mlx_source" rev-parse HEAD)"
cmake -S "$mlx_source" -B "$build_directory" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DMLX_METAL_JIT=OFF \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$build_directory" --target mlx-metallib -j "$(sysctl -n hw.ncpu)"

mkdir -p "$(dirname "$output")"
cp "$build_directory/mlx/backend/metal/kernels/mlx.metallib" "$output"

symbols=$(xcrun -sdk macosx metal-nm "$output" | wc -l | tr -d ' ')
bytes=$(stat -f %z "$output")
echo "Wrote $output: $symbols symbols, $bytes bytes"
if (( symbols < minimum_symbols )); then
    echo "The library has fewer than $minimum_symbols symbols. It is a JIT-mode or incomplete build." >&2
    exit 1
fi
