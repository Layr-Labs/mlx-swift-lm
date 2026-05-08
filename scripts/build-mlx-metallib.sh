#!/bin/bash
# Build MLX's Metal shader library when testing the local SwiftPM package.
#
# mlx-swift's Package.swift builds the C++ target, but SwiftPM does not compile
# the Metal shaders. MLX looks for a colocated mlx.metallib at runtime, so this
# script mirrors the needed CMake shader step for local tests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLX_SWIFT_DIR="${MLX_SWIFT_DIR:-"$REPO_ROOT/../mlx-swift"}"
SRC_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx-generated/metal"
OUT_DIR="${1:-"$REPO_ROOT/.build/mlx-metallib"}"
CACHE_DIR="${CLANG_MODULE_CACHE_PATH:-"/tmp/mlx-swift-metal-clang-cache"}"

if [ ! -d "$SRC_DIR" ]; then
    echo "Missing MLX generated Metal source directory: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR"
rm -f "$OUT_DIR"/*.air "$OUT_DIR"/mlx.metallib

while IFS= read -r -d '' FILE; do
    REL="${FILE#"$SRC_DIR"/}"
    STEM="${REL%.metal}"
    STEM="${STEM//\//_}"
    xcrun -sdk macosx metal \
        -x metal \
        -Wall \
        -Wextra \
        -fno-fast-math \
        -Wno-c++17-extensions \
        -Wno-c++20-extensions \
        -fmodules-cache-path="$CACHE_DIR" \
        -c "$FILE" \
        -I"$SRC_DIR" \
        -o "$OUT_DIR/$STEM.air"
done < <(find "$SRC_DIR" -type f -name '*.metal' -print0)

xcrun -sdk macosx metallib "$OUT_DIR"/*.air -o "$OUT_DIR/mlx.metallib"

if [ "${INSTALL_TEST_BUNDLE:-0}" = "1" ]; then
    TEST_EXEC_DIR="$REPO_ROOT/.build/arm64-apple-macosx/debug/mlx-swift-lmPackageTests.xctest/Contents/MacOS"
    if [ ! -d "$TEST_EXEC_DIR" ]; then
        echo "Missing SwiftPM test executable directory: $TEST_EXEC_DIR" >&2
        echo "Run swift build --build-tests first." >&2
        exit 1
    fi
    cp "$OUT_DIR/mlx.metallib" "$TEST_EXEC_DIR/mlx.metallib"
fi

echo "$OUT_DIR/mlx.metallib"
