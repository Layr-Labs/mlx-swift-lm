### Gemma 4 decode: the compiled feed-forward covers two-draft verification

- MTP verification rows of three tokens (`[1, 3]` and `[2, 3]`, below 64
  expert assignments) now run the compiled feed-forward, as one- and
  two-token rows already do. Outputs keep their bits
  (`Gemma4CompiledFeedforwardTests`). Scoped prompt or batch-decode projections
  keep those rows eager. `DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD=0` turns the
  whole trace off.
