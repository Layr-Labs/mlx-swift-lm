# MiMo short-forward residual and norm kernels

`Libraries/MLXLLM/Models/MiMo/MiMoV26DecodeKernels.swift` adapts the add/RMS and
combine/residual/RMS mechanisms from `omlx/patches/mimo_v2/decode_fast.py`,
[jundot/omlx #3990](https://github.com/jundot/omlx/pull/3990), exact head
[`e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb`](https://github.com/jundot/omlx/blob/e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb/omlx/patches/mimo_v2/decode_fast.py).
That source declares `SPDX-License-Identifier: Apache-2.0`. Preserve the full
license in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) when distributing this code.

The RMS reduction was transcribed upstream from MLX's `rms_single_row` in
`mlx/backend/metal/kernels/rms_norm.metal`, Copyright © 2024 Apple Inc.,
distributed under the MIT license. The adaptation was compared with the local
core pinned at `3fa8f25e6451174d7b06be372c3a24272b77d88e`; its single-row reduction
and dispatch geometry are the same. Preserve the Apple copyright and the full
MIT text in the SDK [LICENSE](../../LICENSE).

Swift adaptation Copyright © 2026 Eigen Labs. Changes include narrow GPU/B1/
dtype/shape guards; unchanged Swift router/expert calls; explicit FP32 product
rounding; request-local normalized-state carry through both ordinary forwarding
and the managed CBv2 trunk; source-weight reads on every call; and independent
storage-bit, greedy-token and complete retained-cache tests. No weight
quantization, attention math or target-to-MTP normalization policy was changed.

The kernels default on through `MiMoV26DecodeDefaults`; exact
`DARKBLOOM_MIMO_FUSED_DECODE_NORMS=0` / `false` / `no` / `off` restores the
stock residual and norm operations for one process. While the norm switch is
unset, the opt-in scalar-dense rectangular verifier
(`DARKBLOOM_MIMO_RECTANGULAR_SCALAR_DENSE=1`) keeps unfused norms, because its
eligibility requires them. Source review and upstream performance claims do not
establish Swift compilation, kernel dispatch, exactness, MTP qualification or a
speedup. Those require separately recorded native test and benchmark receipts
at the integrated source and binary identities.
