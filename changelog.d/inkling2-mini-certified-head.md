### Exact certified sampling head (Inkling 2 Mini on by default)

- A model can pick a greedy or temperature-only token from a cheap proxy of
  its unembedding plus exact logits for the candidate rows, and fall back to
  the full product whenever the proxy cannot certify the result. The token is
  always the one the full product gives: greedy compares exact logits, and
  temperature sampling uses the sampler's own noise without advancing it.
  Rows with any other logits transform, logprobs, a token constraint or a
  reasoning budget never take it, and a row with non-finite values fails
  closed to the full product.
- Engine pieces: `CBv2CertifiedTokens` (the request and model seam),
  `DefaultSamplerV2` (one `masksLogits` predicate decides both the hard mask
  and whether a row may be offered), `LogitsPipelineV2.appliesOnlyTemperature`,
  `SamplerV2.pendingNoise`, and the engine loop and adapter that offer it. A
  model opts in; models that do not are unchanged. `DARKBLOOM_CERTIFIED_TOKENS=0`
  disables the seam.
- Inkling 2 Mini opts in for up to 8 decoding rows. The candidate-row count
  is a runtime value in every kernel and the load warm-up covers 1–64 rows, so
  no kernel variant is compiled mid-request; `DARKBLOOM_INKLING2_CERTIFIED=0`
  keeps every step on the full product. Its 4-bit coded projections for one to
  four rows are present but off (`DARKBLOOM_INKLING2_CODED_PROJECTIONS=1`).
  Its kernel-variant census (used by tests) is switched and read under one lock.
  Tests: `Inkling2CertifiedHeadTests`, `CBv2CertifiedTokenTests`.
