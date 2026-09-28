# MiMo MXFP4 joint gate/up and row-map candidates

These default-off adaptations preserve split checkpoint weights and the native
loader topology. `MiMoV26NAXGateUp` reads gate/up weights and scales separately,
reuses gathered activations and keeps independent FP32 accumulators with native
output rounding. It does not concatenate all resident weights. The ordinary
`SwitchGLU` projection/activation/down/unsort path remains the fallback.

`DARKBLOOM_MIMO_V26_NAX_GATE_UP` selects joint projection.
`DARKBLOOM_MIMO_V26_NAX_SWIGLU` separately requests the epilogue, which preserves
rounded gate/up values and the pinned sigmoid/multiply stages.
`DARKBLOOM_MIMO_V26_NAX_ROW_MAP` additionally allows sorted activation loads
through the existing route order rather than materializing repeated input rows.
The row-map route retains the same inverse mapping for down/weighted output.
These are distinct opt-ins, not a post-load weight-fusion flag.

Eligibility remains narrow: actual MiMo owner/default SiLU, split exact
quantized modules, native MXFP4 group 32 / 4-bit codes, compatible BF16/FP16 shapes,
sorted routes and actual NAX GPU support. Fused stored topology, custom
activations, unsupported streams and other models retain their previous paths.
No weights, numeric baseline, chunk policy, reserve or defaults change.

Implementation: `Libraries/MLXLMCommon/MiMoV26NAXGateUp.swift`,
`MiMoV26NAXGateUpMetalSources.swift` and `SwitchLayers.swift`.
Joint projection shares activation loads but still reads both weight matrices
and maintains two accumulators; source reasoning is not a measured speed gain.
See [qualification](qualification.md) for native bits, complete state and
full-model greedy/serving gates.

Reference heads:

- [oMLX3993](https://github.com/jundot/omlx/pull/3993):
  `a68cab68ae40eeab7062ba7c1eff4ddffa06e373` (joint-projection motivation).
- [oMLX3995](https://github.com/jundot/omlx/pull/3995):
  `47876fbc310fbb311cd382c4eba8572ec4368308` (MXFP4 fragment/tile operations).
- [oMLX4022](https://github.com/jundot/omlx/pull/4022):
  `d6dbc9970c6df99b6a8589746e6ff8ddc38c33ff` (SwiGLU epilogue).
- [oMLX4029](https://github.com/jundot/omlx/pull/4029):
  `498685bae7b536cbd0d7d5e145b0327544e2fb06` (sorted row-map reuse).

Retain the Apache-2.0/oMLX notices in [LICENSE-OMLX](LICENSE-OMLX) and Apple/MIT
primitive text in `MiMoV26NAXMetalSources.swift`. The Swift modifications keep
the selected native rounding boundaries; reference tests/timings are not a
qualification of this composition.
