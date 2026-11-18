### Qwen 3.5 family: MTP rounds commit fewer command buffers

- MTP rounds of the Qwen 3.5 family whose rows each carry at most 8 tokens (the
  assistant's draft steps and the target's verification windows, or a prompt
  chunk that short) commit a Metal command buffer about every 100 kernel
  dispatches, as plain decode steps now do and as Gemma 4's and GLM's rounds
  already do. Outputs keep their bits by construction (only the partitioning
  changes); `Qwen35DecodeCommandBufferTests.verificationRowLimitsKeepExactBytes`
  checks verify-shaped rows through the target.
  `DARKBLOOM_QWEN35_MTP_COMMAND_BUFFERS=0` (or `off`, `false`, `no`) restores
  MLX's defaults for those rounds.
