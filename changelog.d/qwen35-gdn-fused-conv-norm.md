### Qwen 3.5: two more Gated DeltaNet launches folded, bit for bit

- `GatedDeltaConvolutionQKNorm` runs the decode step's convolution, SiLU and
  q/k RMS norms in one kernel (`GatedDeltaConvolution` and `GatedDeltaQKNorm`'s
  expressions), writing v directly; the convolution output's q and k channels
  are no longer written and read back.
- `GatedDeltaOutputNorm` runs the gated output norm (`MLXFast.rmsNorm` and the
  SiLU-gated float32 tail) in one kernel.
- Each is the same values as the kernels and ops it replaces, on the GPU
  families listed in `ResidualRMSNorm.verifiedFamilies`. Off with
  `DARKBLOOM_GDN_FUSED_CONV_QK_NORM=0` and `DARKBLOOM_GDN_FUSED_OUTPUT_NORM=0`
  (diagnostic).
