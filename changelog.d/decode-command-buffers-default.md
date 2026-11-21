### Larger decode command buffers for every model

- Plain decode steps (every row one token) of a model that declares no
  command-buffer limits of its own now commit every 100 operations, the value
  the Gemma 4, GPT-OSS, Nemotron, GLM and Flint families already use, so the
  CPU encodes further ahead of the GPU. The partitioning never changes a
  result. Models that declare limits keep them; prefill chunks and
  speculative rounds keep MLX's defaults. A process that sets
  `MLX_MAX_OPS_PER_BUFFER` or `MLX_MAX_MB_PER_BUFFER` (the cluster ranks pin
  16 operations) keeps its own limits, and
  `DARKBLOOM_CBV2_DECODE_COMMAND_BUFFERS=0` turns the default off.
