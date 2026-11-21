### Prompt chunks that round like the reference rows (opt-in, dense Qwen 3.5)

- With `DARKBLOOM_PREFILL_ROW_INVARIANCE=1`, the engine builds each prompt chunk
  of a dense Qwen 3.5 model (Qwen 3.5 9B, Qwen 3.8 27B) in MLX's prefill-row
  scope (`GPU.withPrefillRowInvariance`). Quantized products of a chunk then
  split K as the engine's longest chunk does (at least 4,096 rows), head-dim
  256 attention on NAX GPUs fuses from 16 queries. A prompt then has the same
  bits however it is cut on multiples of 128, so a prefix restored at such a
  position continues a cold run bit for bit.
- Off by default. Long chunks keep today's bits; prompts or prompt tails of
  256 rows or fewer (M3) or under 1,024 (M5) move to the long-chunk rounding.
  `CBv2PrefillRowInvariance.defaultOn` turns a qualified family on by default.
  Decode rows, MTP verification and drafts are unchanged. Gemma 4 is not
  qualified.
