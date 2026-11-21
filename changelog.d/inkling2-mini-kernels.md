### Inkling 2 Mini: row-exact decode kernels

- Decode steps and verify windows of Inkling 2 Mini read each projection
  weight once for up to 64 rows (FP32 accumulation; a row's result depends on
  the row only, not on how many rows share the call), read each selected
  expert once per step through a gate-up → softcapped SwiGLU → down → combine
  sequence, and compute the router's logits and top-8 selection in one launch.
  Convolution states are staged in packed form, which removes the per-token
  copies; loop messages run in one launch per pass, segmented products and
  experts split K eight ways, and one-row output-projection and gate-tail
  inputs are staged once per threadgroup. Attention preparation gives each
  relative-state head its own unit, small one-row products run in 32-thread
  threadgroups, the one-row mix and message-gate down projections share one
  launch, one-row experts use a dedicated GEMV, and the router runs as one
  launch with per-call arrival counters. Every launch key carries all of its
  kernel's template values, so cached launches are shared safely by several
  engines on one model. Decode bits differ once from the
  unfused path, the same way for every row and verify position. A row alone equals the same row among any company, and a verify
  position equals the decode step that computes the same token.
- The kernels are reached through early returns in the decoder layer, the
  sparse MoE and the row-exact matmul; prefill is unchanged.
  `DARKBLOOM_INKLING2_KERNELS=0` keeps the plain MLX decode path and
  `DARKBLOOM_INKLING2_FUSED_LAYER=0` the unfused layer application. Tests:
  `Inkling2KernelsTests`, `Inkling2ExpertKernelTests`, `Inkling2RowExactnessProbeTests`.
