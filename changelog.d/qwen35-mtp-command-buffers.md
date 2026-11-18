### Qwen 3.5 family: MTP rounds commit fewer command buffers

- MTP rounds of the Qwen 3.5 family whose rows each carry at most 8 tokens (the
  assistant's draft steps and the target's verification windows) commit a Metal
  command buffer about every 100 kernel dispatches, as plain decode steps now
  do and as Gemma 4's and GLM's rounds already do. Outputs keep their bits
  (`Qwen35DecodeCommandBufferTests.verificationRowLimitsKeepExactBytes`).
  `DARKBLOOM_QWEN35_MTP_COMMAND_BUFFERS=0` (or `off`, `false`, `no`) restores
  MLX's defaults for those rounds.
