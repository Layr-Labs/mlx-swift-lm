# MiMo native MXFP4 NAX gather candidate

This is an experimental source port of oMLX PR #3995, head
47876fbc310fbb311cd382c4eba8572ec4368308, against MLX-Swift-LM
6f3d171fb7270ba18fb2432ab4f4aab5ed4b6114 plus the current native MiMo work.
It is OFF by default. No compilation, GPU evaluation, numerical test or
performance measurement has been performed for this source packet.

Set DARKBLOOM_MIMO_V26_NAX_GATHER=1 before first use only in an owned,
qualification run. The actual core NAX gate must also be true; hardware names
alone do not override unavailable OS/build support. CPU streams, unsupported
dtypes/shapes/quantization, unaligned or unsorted inputs and ordinary model
owners keep the existing projection. The opt-in is latched for the process.
Restart with the flag absent or 0 to select the baseline.

Production eligibility is a MiMo-owned SwitchGLU, exact QuantizedSwitchLinear
class without a linear bias, sorted uint32 indices, one already gathered row
per index, 256 experts, at least 1024 assignments, uint32 packed 4-bit weights
and uint8 group-32 MXFP4 scales without affine biases. Activations and outputs
remain BF16 or FP16 in their original dtype. Supported K/N are 4096/2048,
4096/4096 and 2048/4096. This patch leaves gate/up split; 4096/4096 is ready for
a separately qualified fused load. No weights or precision are converted.

The one-group scan emits single-expert tiles. The matmul uses the original
16x32x16 NAX operations and FP32 accumulators. BM128/db applies at 65–128
assignments per expert, BM64/db below that, and BM64/seg above 128. Both
projection dimensions are aligned; the upstream ragged K/N and affine routes
are deliberately outside this MiMo packet. Row offsets are Int32, avoiding
the upstream Int16 overflow above 32768 assignments.

The embedded Metal text is adapted from oMLX's Apache-2.0 implementation.
The primitive headers are mechanically flattened from local MLX
3fa8f25e6451174d7b06be372c3a24272b77d88e. The unused affine quantizer and unused
fp8_e4m3/scale-round-up helpers are omitted. Apple copyright and the full MIT
license are retained in MiMoV26NAXMetalSources.swift. The complete oMLX license,
including Copyright 2025 oMLX contributors, is in LICENSE-OMLX. No root NOTICE
exists in the upstream recursive tree at the pinned head.

The host preserves the current default stream and lazy graph. It adds no eval,
synchronize, canary, catch-all completion fallback, array cache or weight
resident copy. Kernel creation/evaluation faults must flow to the provider's
existing required fault/retirement handling. Offline qualification replaces
the upstream wrapper's first-forward canary; an enabled run is not qualified
merely because the Metal source was encoded. Diagnostics count encodings only.

Six XCTest methods are supplied, all UNRUN. Three hardware methods cover
16 bit-exact pairings: BF16/FP16, BM64/BM128, seg/db, sparse and empty experts,
partial expert runs, strided input and more than 32768 rows. They print native
output maxULP and require zero plus raw byte equality against the unchanged
stock kernel (safe slices for oversized rows). Three other methods cover
production plan/selection, owner propagation and CPU stream rejection.
DARKBLOOM_TEST_MIMO_NAX_GATHER=1 explicitly enables the hardware methods; absent
capability yields a recorded XCTest skip, never a numerical pass.

Before activating for service, the integrator must run those tests and real
production-shaped gate/up/down comparisons, verify actual encoded and completed
routes, compare full-model greedy tokens and full target/MTP/cache state with
baseline, and report maxULP. Any accepted nonzero differences must be attributable
solely to allowed floating summation order; no tolerance-only claim suffices.
Retain exact original artifact hashes, natural EOS, counterbalanced server A/B,
warmup, effective chunk sizes, timing spans, residency, native ownership and
cancel/unload/reload checks. No chunk size, cache policy, memory cap or safety
reserve changes are part of this candidate.
