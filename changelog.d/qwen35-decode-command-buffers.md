### Qwen 3.5 family: plain decode steps commit less often

- Plain decode steps of `Qwen35Model` and its MoE subclass (Qwen 3.5 9B and
  35B-A3B, Qwen 3.6 35B-A3B, Qwen 3.8 27B) commit a Metal command buffer every
  100 operations, as GPT-OSS, Nemotron, Gemma 4 and GLM decode steps already
  do. Before, every routed expert bank exceeded MLX's 50 MB input bound, so
  decode committed after almost every MoE operation and waited on MLX's ten
  buffers in flight. Outputs keep their bits (`Qwen35DecodeCommandBufferTests`).
  Prefill chunks and MTP rounds keep MLX's defaults.
  `DARKBLOOM_QWEN35_DECODE_COMMAND_BUFFERS=0` (or `off`, `false`, `no`)
  restores MLX's defaults for those steps.
