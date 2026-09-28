# MiMo MXFP4 joint gate/up candidate

This is a task-specific joint-gather variant of #3993's projection intent.
It is not a literal port of its post-load weight concatenation. It depends on
the frozen #3995 packet's MiMo owner flag, plans, stream/capability helpers,
pinned Metal header and licenses. Existing gate/up modules and checkpoint
tensors stay split and unmodified. The candidate is OFF by default:
DARKBLOOM_MIMO_V26_NAX_GATE_UP=1 opts in before first use. Its flag is independent
of DARKBLOOM_MIMO_V26_NAX_GATHER; failed eligibility returns to the same
projectExpert calls, including that independently selected single-gather route.

The current SwitchGLU already calls gatherSort once for >=64 assignments.
gatherSort flattens indices, builds order/inverse once, and replicates token
rows through order.floorDivide(topK). Both up and gate consume the same
already-gathered x and sorted index vector. Down consumes activated rows under
that same index vector; the inverse permutation is used afterward. This patch
does not duplicate or remove those operations.

The only integration hunk is inside the existing split gate/up branch.
MiMo-owned eligible native MXFP4 projections can produce two arrays jointly;
the existing compiledSiluProduct, down projection and inverse mapping follow
unchanged. Non-MiMo owners, fused gate_up_proj topology, custom subclasses,
affine or mismatched quantization, linear biases, unsupported dtypes/geometry,
CPU streams, non-NAX builds/devices and small/unsorted calls keep their prior
path. Production shape is E256, K4096, N2048, uint32 codes, uint8 group32 scales,
native BF16/FP16 activations and >=1024 sorted assignments.

The joint kernel shares one tile scan and loads each A fragment once. It reads
gate and up weights/scales separately, dequantizes each through the unchanged
native MXFP4 decoder, and applies each A fragment to two independent FP32
accumulators. Each output has the same K order and one final native dtype cast.
There is no cross-projection reduction, bias, activation or output-dtype change.
Packed weights are never concatenated, rewritten, requantized or persistently
copied, and no module tree or native loader topology changes.

This variant uses the segmented schedule and BM64/BM128. Two single-buffer
BF16 weight tiles occupy 2*64*(64+8)*2 = 18432 threadgroup bytes. Dual double
buffering would occupy 36864 bytes, so it is deliberately not provided.
Maintaining two FP32 accumulators and two weight fragments adds register
pressure and may lower occupancy. The joint path removes one scan and one
matmul launch versus two #3995 calls and halves their duplicated A-fragment
loads for matched output columns. It does not reduce weight bytes or FLOPs.
These are source properties, not measured speedups; it may lose to the
single-projection db schedule at some route distributions.

Four unrun XCTest methods cover 54 exact output comparisons: two dtypes,
two tile heights, partial/empty/sparse experts, independent gate/up seeds,
stock and separate-NAX oracles, >32768 rows against safe stock slices, and
composition through one real gatherSort, compiledSiluProduct, down and inverse.
Comparisons print maxULP and require zero plus raw byte equality. Hardware
tests require DARKBLOOM_TEST_MIMO_NAX_GATE_UP=1 and actual NAX support.
No GPU work, compilation or test execution was performed in this packet.

Before acceptance, complete source/JIT compilation, those exact oracles,
native production shapes/strides, effective joint dispatch (not just flag),
full-model greedy and full state agreement, lifetime and memory checks,
and same-build counterbalanced server A/B against the separately optimized
gather path as well as the baseline. Preserve decode/other-family fallback,
natural EOS, failures and skips. No chunk, scheduler, cache, reserve or default
activation changes belong to this candidate.

Sources: jundot/omlx #3993 head a68cab68ae40eeab7062ba7c1eff4ddffa06e373
(joint projection motivation) and #3995 head
47876fbc310fbb311cd382c4eba8572ec4368308 (MXFP4 loader/tile arithmetic).
The former's actual diff/tests are retained in the parent packet. Apple/MIT
primitive code and full license are embedded in MiMoV26NAXMetalSources.swift;
the oMLX Apache-2.0 text and copyright are in docs/mimo-v26/LICENSE-OMLX.
Modified kernel files identify this new two-input/two-accumulator adaptation.
