### Fuse Gemma 4's attention prologue, routing, GeGLU and expert sum

- Gemma 4 decode steps and verify windows run each paged attention layer's q/k/v
  norms and RoPE as one kernel (five launches today, plus one value norm per
  request at 2-8 streams), the expert routing after the router's projection as
  one kernel per row (five launches today), the approximate-GELU products of the
  dense MLP and the experts as one kernel each (two today: MLX's compile splits
  them), and the experts' weighted sum inside the feed-forward tail kernel (two
  launches today). Outputs keep their bits. A one-stream decode step of Gemma 4
  26B-A4B runs 701 kernels instead of 1,061, and its longest dependent chain is
  371 instead of 491.
- `RoPEForm` and `SoftmaxForm` measure how MLX's prebuilt rope and softmax round
  in this process, and `ExactGeGLU` checks every bf16/f16 gate value against
  `compiledGeGLU`, so the kernel copies write the same bits; each runs only where
  its check passed.
