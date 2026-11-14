### Opt-in: read a long context once per kv head in paged decode

- With large head dims, a kv head's query heads split over several pass-A
  threadgroups (the Qwen 3.5 27B's full attention, Gemma 4's global layers),
  and each threadgroup reads the partition's keys and values. With
  `DARKBLOOM_CBV2_PAGED_SIDE_BY_SIDE_TOKENS=<n>`, decode from n attended
  tokens runs those head groups side by side in one threadgroup and reads them
  once. The partitions and each head's order of operations are unchanged, so
  the output has the same bits (checked on GPUs in tests and in the engine at
  65,536 and 131,072 tokens). Off by default: per attention layer it was
  faster at long context on an M5 Ultra (1.08x for the 27B shape and 1.22x for
  Gemma 4 global layers at 65,536 tokens), but whole decode steps on an M5 Max
  were not faster.
