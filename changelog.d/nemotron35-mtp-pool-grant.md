### Nemotron 3.5 MTP holds requests past 165K tokens

- The Nemotron 3.5 MTP assistant sizes a private paged pool for each
  request's whole bound (prompt plus output plus its draft padding). It
  granted that pool its pages plus a fixed 4 MiB, but the segmented pool
  charges whole segments: one poison page per segment, power-of-two segment
  sizes and each segment's allocation footprint. With 2 BF16 KV heads of 128
  (Nemotron 3.5 Lightning) the charge passed the grant at 165,849 tokens of
  prompt plus output, in bands above that and at every length from 176,121
  tokens. The request's first step then failed with `capacityExhausted`
  (2,113,407 bytes short at a 262,144-token request).
- The assistant now grants at least the pool's own charge for its row,
  computed by the pool (`PagedKVPool.segmentedRowReservationBytes`). Requests
  that fit before are unchanged; their grant does not shrink. The charge stays
  near 1,049 bytes per token, inside the assistant's admission charge of at
  least 20 KiB per token. Tests: `NemotronH35MTPPoolSizingTests`.
