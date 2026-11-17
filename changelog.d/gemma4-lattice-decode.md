### Gemma 4 decode: QAT-lattice experts compile, MTP rounds commit less often

- The compiled feed-forward now takes expert banks of 4 bits in groups of 32
  (Google's QAT lattice, as in the 26B-A4B "C" build). Those banks kept the
  eager graph before. Outputs keep their bits (`Gemma4CompiledFeedforwardTests`).
- MTP rounds whose rows each carry at most 8 tokens (decode and verification
  rows) now commit a Metal command buffer every 100 operations, as plain decode
  steps already do. Outputs keep their bits. `DARKBLOOM_GEMMA4_MTP_COMMAND_BUFFERS=0`
  (or `off`, `false`, `no`) restores MLX's defaults for those rounds.
