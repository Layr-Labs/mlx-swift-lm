### Fuse Gemma 4's norm chains around the feed-forward

- Gemma 4 decode steps and verify windows run each layer's norm chains around the
  feed-forward as two kernels instead of thirteen launches, including the next
  layer's input norm. Outputs keep their bits. One-stream decode of Gemma 4
  26B-A4B is 7.5-11 % faster on an M5 Ultra; 4 to 32 streams gain 1-6 %.
- `RMSNormMeanForm` measures, once per row width, how MLX's prebuilt norm rounds
  its mean (the compiler that built MLX's metallib decides), so a kernel copy
  writes the same bits. The new kernels run only where that form is measured,
  on M5 GPUs (family 17), and after they write the unfused ops' bytes on the
  rows that separate the two forms; a Metal error or a difference keeps the
  unfused ops and logs once. `DARKBLOOM_GEMMA4_FUSED_NORMS=0` keeps the unfused
  ops (the rollback).
