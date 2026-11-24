### Inkling 2 Mini's DSpark drafter holds requests past 41K tokens

- The DSpark drafter sizes a private paged pool for each request's whole
  bound (prompt plus output plus its block). It granted that pool its pages
  plus a fixed 4 MiB, but the segmented pool charges whole segments: one
  poison page per 32 usable pages, power-of-two segment sizes, a separate
  group for the sliding layers and each segment's allocation footprint. Past
  41,432 tokens of prompt plus output the charge passed the grant, and the
  request's first step failed with `capacityExhausted` (for example needed
  210,730,909 bytes, available 209,780,736 at a 46,966-token prompt).
- The drafter now grants at least the pool's own charge for its row,
  computed by the pool (`PagedKVPool.segmentedRowReservationBytes`). Requests
  that fit before are unchanged; their grant does not shrink.
