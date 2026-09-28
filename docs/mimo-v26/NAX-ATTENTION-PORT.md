# MiMo NAX attention: native-rounding candidate

Default off, selected by `DARKBLOOM_MIMO_V26_NAX_ATTENTION` only after actual
MiMo/device/stream/dtype/shape gates. Implementation:
`Libraries/MLXLMCommon/MiMoV26NAXAttention.swift` and
`MiMoV26NAXAttentionMetalSources.swift`. See [qualification](qualification.md);
this page records arithmetic and attribution, not a runtime pass.

The literal oMLX3994 one-pass kernel is not the selected native baseline.
The Swift adaptation keeps native Q scaling, native-rounded QK scores,
FP32 max/exp-sum with the pinned `fast::exp`, normalized probabilities rounded
to their native dtype, then FP32 P×V accumulation and native output rounding.
It uses three score passes. Finite-min boolean-mask behavior is preserved for
real keys; physical padding does not become an attended key. There is no
`exp2`/online-rescaling substitution or tolerance-only losslessness claim.

The ordinary attention call retains its original unblocked contract. Managed
`CBv2AttentionV1` preserves the EXISTING q128 decomposition and per-block
K-prefix/window slices: no full-attention shortcut runs before those bounds.
One real cache update precedes attention; spans, softcap and serialized MTP
keep their established paths. Runtime batch/head/token/inner strides cover
lazy noncontiguous views without host evaluation. Unsupported cases fall back.

`MiMoV26BlockBatchAttention` is a separately gated, admitted grouping of those
same exact blocks, not a chunk/default or visibility change.
`MiMoV26NAXAttentionKeyRanges` is a distinct source helper preserving the order
of each of the three passes; its retained states need a real admitted native
owner before serving integration. Split-key matrix scoring is separate again.

Attribution: adapted from
[jundot/omlx3994](https://github.com/jundot/omlx/pull/3994), head
`1c487861d1c1d4c82a2b2920dd69c064eec85e18`, Apache-2.0,
Copyright 2025 oMLX contributors. Apple MLX NAX primitives are MIT-licensed.
Retain [LICENSE-OMLX](LICENSE-OMLX), [Apache-2.0](LICENSE-APACHE-2.0), and the
Apple/MIT text embedded in `MiMoV26NAXMetalSources.swift`.
Changes to three-pass arithmetic, stride handling and visibility are intentional
adaptations; upstream timing is not a result for this variant.
