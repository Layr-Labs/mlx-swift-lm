### Gemma 4 decode: the compiled feed-forward covers 8 to 16 rows

- Decode steps of 8 to 16 rows, and MTP verification of 4 to 8 rows of two
  tokens, now run the compiled feed-forward. At top-8 those rows sort their
  expert assignments; the trace sorts them exactly as the eager graph does.
  Outputs keep their bits (`Gemma4CompiledFeedforwardTests`). Inside a
  `Gemma4BatchDecodeScope` frame those sizes stay eager.
  `DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD_SORTED=0` (or `off`, `false`, `no`)
  keeps those rows on the eager graph.
