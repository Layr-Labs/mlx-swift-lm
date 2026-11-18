### Qwen 3.5 family: plain decode steps commit fewer command buffers

- Plain (non-speculative) decode steps of every model the CBv2 engine serves as
  `Qwen35Model` or its MoE subclass (Qwen 3.5 9B and 35B-A3B, Qwen 3.6 35B-A3B,
  Qwen 3.8 27B) commit a Metal command buffer about every 100 kernel dispatches,
  as GPT-OSS, Nemotron and Gemma 4 decode steps already do. Before, every routed
  expert bank exceeded MLX's 50 MB input bound, so decode committed after almost
  every MoE operation and waited on MLX's ten buffers in flight. Outputs keep
  their bits (`Qwen35DecodeCommandBufferTests`). Prefill chunks, MTP rounds and
  the Prism Hadamard text pack keep MLX's defaults; cluster Qwen sessions set
  their own, identical limits. Measured on Qwen 3.6 35B-A3B (M5 Ultra, rows
  that don't draft): one stream 133.7 → 159.5 tok/s, four 96.4 → 105.1, eight
  64.1 → 69.3. `DARKBLOOM_QWEN35_DECODE_COMMAND_BUFFERS=0` (or `off`, `false`,
  `no`) restores MLX's defaults for those steps.
