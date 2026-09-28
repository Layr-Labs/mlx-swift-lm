# MiMo FP32 short-forward router GEMV

`Libraries/MLXLLM/Models/MiMo/MiMoV26DecodeRouter.swift` adapts `_ROUTER_GEMV_SOURCE`
and its dispatch from [jundot/omlx #3990](https://github.com/jundot/omlx/pull/3990),
exact head [e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb](https://github.com/jundot/omlx/blob/e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb/omlx/patches/mimo_v2/decode_fast.py),
declared Apache-2.0. Preserve [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).

The original algorithm follows MLX's `GemvKernel` in `gemv.h`,
Copyright © 2023-2024 Apple Inc., under MIT. Preserve the Apple copyright and
SDK [LICENSE](../../LICENSE). The local comparison uses core commit
`3fa8f25e6451174d7b06be372c3a24272b77d88e`: `matmul.cpp` selects
BM1/BN8/SM1/SN32/TM4/TN4 when K >= 16N, and `gemv.h` defines the same K stride,
shuffle tree and ordered eight-group fold. Its newer wide GEMV explicitly
excludes FP32 output, so it does not already cover this FP32 router operation.

Swift adaptation Copyright © 2026 Eigen Labs. Changes add explicit B1/1–7-row,
dtype, device and aligned-shape guards; preserve the model's declared operand
cast before FP32 widening; and keep all expert selection and normalization in
the existing router. The helper retains no arrays across calls and defaults
off. One-row and independent per-row equality are intended gates, not observed
results. Batched GEMM comparisons must record maximum ULP and exact expert/
greedy decisions separately. No approximate math or quantization is introduced.
