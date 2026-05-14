#!/usr/bin/env bash
# Cold-cache benchmark for .mlxpack loads. Displaces target file from page
# cache using LRU pressure (reads ~80 GiB of unrelated files), then times
# one cold load + one warm load.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <file.mlxpack> <iterations>" >&2
    exit 64
fi
PACK="$1"
ITERS="$2"

BIN="/Users/gaj/Documents/Builds/MLX-Swift/mlx-swift-lm-opt/.build/arm64-apple-macosx/release/BenchLoad"

ALL_DISPLACE=()
while IFS= read -r f; do ALL_DISPLACE+=("$f"); done < <(
    find /Users/gaj/.cache/huggingface/hub -type f -name "*.safetensors" \
        -size +1000M ! -path "*$PACK*" 2>/dev/null
)
echo "displacement pool: ${#ALL_DISPLACE[@]} files"
echo "target:            $PACK"
echo "iterations:        $ITERS"
echo ""

GROUP_SIZE=16
for i in $(seq 1 "$ITERS"); do
    echo "=== iteration $i ==="
    start=$(( ((i - 1) * GROUP_SIZE) % ${#ALL_DISPLACE[@]} ))
    t0=$(python3 -c "import time; print(time.time())")
    for ((j=0; j<GROUP_SIZE; j++)); do
        idx=$(( (start + j) % ${#ALL_DISPLACE[@]} ))
        dd if="${ALL_DISPLACE[$idx]}" of=/dev/null bs=1m 2>/dev/null
    done
    t1=$(python3 -c "import time; print(time.time())")
    echo "  displaced in $(python3 -c "print(f'{($t1-$t0):.1f}')")s"

    out=$("$BIN" --load-mlxpack "$PACK" 2 0 2>&1)
    cold=$(echo "$out" | grep -E "^  \[run 1\]" | awk '{print $3}')
    warm=$(echo "$out" | grep -E "^  \[run 2\]" | awk '{print $3}')
    echo "  cold: ${cold} ms"
    echo "  warm: ${warm} ms"
done
