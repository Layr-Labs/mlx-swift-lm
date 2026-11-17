### Qwen 3.5 A3B: the sparse block's end and the next residual norm in one kernel

- On one-stream decode, the qwen3_5 sparse block hands the next residual
  norm its parts before their sum (`Qwen35DecoderLayer.Pending.experts`), and
  `Qwen35ExpertTail` writes the routed experts' `weightedExpertSum` (as MLX's
  small column reduction adds the rows, for at most 8 experts), the shared
  expert's `sigmoid(gate) * shared`, their sum, the residual add and the
  RMSNorm in one kernel, with the same values. Off with
  `DARKBLOOM_QWEN35_FUSED_EXPERT_TAIL=0` (diagnostic).
