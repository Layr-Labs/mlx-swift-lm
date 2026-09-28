# MiMo native MXFP4 NAX gather candidate

Default off. `MiMoV26NAXGatherQMM` adapts
[jundot/omlx3995](https://github.com/jundot/omlx/pull/3995), exact head
`47876fbc310fbb311cd382c4eba8572ec4368308`. This is a native MXFP4 path, not an
affine-only port. See [qualification](qualification.md) for independent gates.

Eligibility is MiMo-owned `SwitchGLU`, exact bias-free `QuantizedSwitchLinear`,
sorted uint32 indices, one gathered row per index, 256 experts, at least 1024
assignments, uint32 packed 4-bit weights, uint8 group 32 scales and native
BF16/FP16 activations. Supported K/N shapes are 4096/2048, 4096/4096 and 2048/4096.
CPU/non-NAX/unsupported cases retain the existing projection. The process-latched
`DARKBLOOM_MIMO_V26_NAX_GATHER` flag does not override any eligibility check.

The single-expert tile scan uses Int32 row offsets, including above 32768 rows.
Native fragment operations and FP32 accumulators are retained; the schedule
selects its bounded segmented/double-buffered tiles from route occupancy.
General affine and unaligned/ragged projection dimensions are outside this
adaptation. No checkpoint conversion, loader-topology change, resident weight
copy, implicit eval/synchronize or first-forward canary is added.

Sources are `Libraries/MLXLMCommon/MiMoV26NAXGatherQMM.swift`,
`MiMoV26NAXMetalSources.swift` and the MiMo-scoped seam in `SwitchLayers.swift`.
Tests include actual stock comparisons, route eligibility, strides, sparse
experts and large row offsets; encoding counts alone are not completion proof.

oMLX code is Apache-2.0, Copyright 2025 oMLX contributors; retain
[LICENSE-OMLX](LICENSE-OMLX). Primitive headers derive from Apple MLX
`3fa8f25e6451174d7b06be372c3a24272b77d88e`; Apple copyright and the complete MIT
text remain in the embedded Metal source. Upstream speed measurements do not
qualify the Swift wrapper, full model, state lifetime or serving defaults.
