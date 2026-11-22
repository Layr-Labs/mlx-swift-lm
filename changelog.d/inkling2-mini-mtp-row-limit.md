### Inkling 2 Mini: draft on at most 4 decode rows by default

- The DSpark drafter's shipped configuration speculates on at most 4 decode
  rows (`Inkling2DSparkDrafter.defaultSpeculativeRows`, was 8). A step with
  more decode rows runs every row as plain decode, which gives the same
  tokens. On an M5 Ultra, all-rows-decoding throughput with the drafter on
  rises from 237 to 257 tok/s at 5 rows, 263 to 289 at 6 and 302 to 334 at 8,
  level with the drafter off; 4 rows keep the drafter's gain (250.6 vs 222.9
  off). `DARKBLOOM_INKLING2_MTP_MAX_ROWS` sets another limit.
