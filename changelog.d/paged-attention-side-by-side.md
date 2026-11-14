### Read a long context once per kv head in paged decode

- With large head dims, a kv head's query heads split over several pass-A
  threadgroups (the Qwen 3.5 27B's full attention, Gemma 4's global layers),
  and each threadgroup read the partition's keys and values. From 49,152
  attended tokens the head groups now run side by side in one threadgroup
  and read them once. The partitions and each head's order of operations
  are unchanged, so the output has the same bits. Measured per attention
  layer on an M5 Ultra (the plain kernel's text replayed): 1.08x for the 27B
  shape and 1.22x for Gemma 4 global layers at 65,536 tokens, 1.20x and
  1.79x at 131,072. `DARKBLOOM_CBV2_PAGED_SIDE_BY_SIDE_TOKENS` sets the
  length; 0 turns it off.
