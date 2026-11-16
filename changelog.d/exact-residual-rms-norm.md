### Match MLX's prebuilt norm in the fused residual norm

- `ResidualRMSNorm` (the fused residual add and RMSNorm of the Qwen 3.5 decode
  path) now rounds the norm's mean and orders its weight product the way MLX's
  prebuilt norm does in this process, as `RMSNormMeanForm` measures it. Before,
  it wrote `acc / D + eps` and `w * (x * inv)`; with MLX built by Xcode for this
  fork the prebuilt norm uses `fma(acc, 1 / D, eps)`, and its float32 kernel for
  rows wider than 4,096 computes `(w * inv) * x`, so the fused norm differed
  from the unfused add and norm on some rows. Where a form is not measured, the
  unfused ops run.
