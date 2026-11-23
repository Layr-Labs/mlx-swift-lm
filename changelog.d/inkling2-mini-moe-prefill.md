### Inkling 2 Mini: prefill GEMMs on neural accelerators

- On GPUs with neural accelerators, Inkling 2 Mini's prompt chunks compute the
  experts as grouped GEMMs (expert plans grouped without a sort, exact NVFP4
  and MXFP8 conversion to BF16 tiles, SwiGLU in the gate-up epilogue, FP32
  combine in ascending expert order, empty experts skipped) and the dense
  projections with a fixed-tile GEMM whose K order is fixed, so a row's result
  does not depend on the chunk length. Full expert tiles skip the per-fragment
  bounds checks. Short chunks under chunk-invariant prefill take the expert
  kernel without padding.
- `DARKBLOOM_INKLING2_MOE_PREFILL=0` keeps the sorted `gather_qmm` route and
  `DARKBLOOM_INKLING2_PREFILL_GEMM=0` the padded MLX calls. GPUs without neural
  accelerators keep the existing path. Tests: `Inkling2MoEPrefillTests`,
  `Inkling2PrefillGEMMTests`.
