### Qwen 3.5: the existing one-kernel SwiGLU and shared-expert combine at one stream

- `Qwen3NextMLP.callAsFunction` (one-stream decode, prompt chunks, the A3B
  shared expert) now takes `Qwen35DecodeFusions.siluMultiply` when the qwen3_5
  decode fusions are on, as batched decode rows already do: one compiled
  kernel instead of `silu` and a multiply.
- The qwen3_5 sparse block's shared-expert combine
  `combined + sigmoid(gate) * shared` takes the compiled kernel the qwen4 path
  uses (`Qwen4ExpFusions.sharedExpertCombine`) instead of a sigmoid, a
  multiply and an add.
- Both trace the same ops at the same dtypes, so the values are the same.
  `DARKBLOOM_QWEN35_DECODE_FUSIONS_V2=0` keeps the separate kernels
  (diagnostic).
