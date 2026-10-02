# MiMo distinct-expert MXFP4 short-forward kernels

`MiMoV26DecodeExpertMetal.swift` and `MiMoV26DecodeExperts.swift` adapt
[jundot/omlx #3990](https://github.com/jundot/omlx/pull/3990), exact head
[`e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb`](https://github.com/jundot/omlx/blob/e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb/omlx/patches/mimo_v2/moe_decode.py).
The original declares Apache-2.0; retain [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).

The FP4/E8M0 decoding, `qdot<float,16,4>` lane mapping, FP32 reduction and
activation operator arithmetic originate in MLX, Copyright © 2024 Apple Inc.
The pinned comparison is core `3fa8f25e6451174d7b06be372c3a24272b77d88e`, files
`fp4.h`, `fp8.h`, `fp_quantized.h`, `unary_ops.h`, `binary_ops.h` and the typed
primitive emission in `backend/metal/compiled.cpp`. Retain the Apple copyright
and full MIT license in SDK [LICENSE](../../LICENSE).

Swift adaptation Copyright © 2026 Eigen Labs. Changes include a MiMo ownership
and stock-module gate; actual default-stream detection; shape/dtype/packed-layout
checks; and duplicate-slot support. The upstream member list assumes each expert
occurs at most once per row. This version collects one representative per token
row and expert, then writes the identical result to every corresponding routed
slot. Down projection is paired with the helper's gate/up output, so duplicated
slots have identical activations. No caller-supplied distinct activations are
silently combined. Native packed codes and scales are read directly.

SwiGLU retains the activation dtype at sigmoid, gate-times-sigmoid and final
up-product boundaries. The pinned `compiledSiluProduct` composes `MLXNN.silu`
and multiply; `compiledSilu` uses multiply/sigmoid; Metal compiled emission gives
every intermediate its declared dtype. The same stable sigmoid expression is
used here. Prepared exhaustive finite BF16/FP16 gate tests compare the actual
helper function with `compiledSiluProduct`, including varied fractional/negative
up values. Source inspection does not establish JIT or numerical equality.

The kernel defaults on; exact `DARKBLOOM_MIMO_DECODE_EXPERTS=0` / `false` /
`no` / `off` restores SwitchGLU for one process. Unsupported shape, type,
module, activation or CPU stream returns to existing SwitchGLU. The original
router, combine, cache, MTP and target normalization remain unchanged. Speed and
memory claims belong to the native and full-model measurement records.
