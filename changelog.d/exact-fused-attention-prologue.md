### Fuse Gemma 4's attention prologue and expert routing

- Gemma 4 decode steps and verify windows run each paged attention layer's q/k/v
  norms and RoPE as one kernel (five launches today, plus one value norm per
  request at 2-8 streams), and the expert routing after the router's projection
  as one kernel per row (five launches today). Outputs keep their bits.
- `RoPEForm` and `SoftmaxForm` measure how MLX's prebuilt rope and softmax round
  in this process (which product the compiler fused into a multiply-add, and
  precise or fast reciprocals), so the kernel copies write the same bits. The
  new kernels run only where exactly one form matches.
