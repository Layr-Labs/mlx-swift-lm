# MiMo NAX attention: native-rounding experiment

This candidate is OFF by default and is not numerically or operationally
qualified. It requires the separately frozen NAX gather packet's embedded
Apple/MIT primitive headers and capability/stream helpers, but the gather flag
need not be enabled. Set DARKBLOOM_MIMO_V26_NAX_ATTENTION=1 only in an owned,
qualification process before first use. Removing the flag restores the stock
attention path. No chunk size, cache layout, cache update or memory limit changes.

The literal #3994 kernel is rejected for this task's summation-order-only
contract. The pinned MLX fallback multiplies Q by a native-dtype scale, rounds
QK scores to that dtype, computes precise softmax with native-dtype output, and
multiplies those normalized native-dtype probabilities by V. Its Metal softmax
uses fast::exp. The upstream fused kernel scales FP32 QK after the dot product,
uses exp2 and online rescaling, multiplies unnormalized FP32 P by V, and
normalizes the accumulated output. Those are extra rounding/operation changes.

This source adaptation preserves the selected baseline's rounding boundaries:
native scaled Q; native rounded scores; FP32 softmax max/sum using the same
fast::exp; native rounded normalized P; FP32 P@V accumulation and one output
cast. Avoiding a score tensor while retaining the global-max exponent arguments
requires three QK passes: max, exp-sum, then normalized P@V. Native false-mask
entries use the same dtype finite minimum; physical K padding is excluded.
There is no exp2 or online exponential rescaling in the candidate. The extra
QK passes and native scaled-Q temporary have unknown performance and memory
effects. This is not the upstream one-pass benchmark implementation.

Eligibility requires native 192/128 heads, Q64/KV4 or KV8, matching BF16/FP16
inputs, qL>8, valid causal or boolean mask, compatible sink dtype, and a GPU
stream/build/device for which the pinned core reports NAX support. Small,
unsupported or non-NAX cases return to the original function. No host read of
lazy-array strides is used: the kernel consumes runtime batch/head/sequence/
inner-dimension strides. Unit inner strides retain the native tile loaders;
non-unit inner strides use the same fragment loads with an explicit column
stride. No Q/K/V contiguity copy is introduced for layout purposes; the
baseline-prescribed Q scaling itself remains an eager graph operation.

The ordinary MiMo attention call and the actual managed contiguous cache both
have scoped integration. A default-false immutable cache flag is set only by
MiMoV26ContiguousLayerCache's constructor hunk. Common full attention tries
the fused function after exactly one existing row update, before the existing
q128 loop. Sliding-window attention retains its existing q128 bounds, slices,
causal/window masks and ordering, replacing only eligible SDPA calls. Span
overlays, softcap, attention diagnostic packets and serialized MTP stay on
their existing routes. <=8-row calls decline. Other model/cache constructors
retain false. No prefix, paged, or block-batched SWA support is claimed.

Five unrun XCTest methods include eight exact uniform/strided comparisons,
20 baseline maxULP diagnostics, shape/dtype/mask gating, and two actual common
cache dispatch/state cases with a serial-query exclusion check. Hardware cases
require DARKBLOOM_TEST_MIMO_NAX_ATTENTION=1. The managed test additionally needs
the route flag at process start. Encoding counts are not GPU completion proof.
MaxULP logging plus shape/finiteness checks are diagnostic only; no allclose
threshold is used as a lossless acceptance test.

Before acceptance, complete compilation/JIT, numerical contract attribution,
production heads and lengths, exact greedy tokens, complete target/MTP/cache
state, attention ownership, sink/window/span/softcap/serial regressions, memory,
lifecycle and matched server A/B. Preserve failures and skips. This port
supplies source and tests, not evidence that any of those gates passed.

Attribution: adapted from jundot/omlx #3994 at
1c487861d1c1d4c82a2b2920dd69c064eec85e18 (Apache-2.0, 2025 oMLX contributors);
NAX/attention primitives derive from Apple MLX (MIT).
Full notices are in the companion gather packet's LICENSE-OMLX and embedded
MiMoV26NAXMetalSources.swift. The changed source files prominently identify
the three-pass and stride adaptations. Upstream exact source is retained in
the private evidence packet, without implying that its benchmark result applies.
