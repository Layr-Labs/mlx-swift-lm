### GPT-OSS: decode steps with fewer launches, bit for bit

- CBv2 decode steps of GPT-OSS (one token per row, up to 15 rows) run
  `GPTOSSFusedNorms`: the attention prologue (the q/k/v biases, YaRN's scale
  and RoPE in the measured `RoPEForm`), the router (its bias, `argPartition`'s
  stable top-k and `softmax(precise: true)` in the measured `SoftmaxForm`),
  the gate/up bias inside the compiled SwiGLU, and two residual kernels: the
  output projection's bias, the residual add and the post-attention norm; and
  the experts' biases, routing weights and sum (as MLX's small column
  reduction adds them), the residual add and the next norm (the measured
  `RMSNormMeanForm`).
- Each writes the values of the ops it replaces. Off with
  `DARKBLOOM_GPTOSS_FUSED_NORMS=0`, or per piece with
  `DARKBLOOM_GPTOSS_FUSED_PROLOGUE=0`, `DARKBLOOM_GPTOSS_FUSED_ROUTING=0` and
  `DARKBLOOM_GPTOSS_FUSED_EXPERT_BIAS=0` (diagnostic).
