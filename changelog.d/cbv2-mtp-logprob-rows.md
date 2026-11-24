### Rows that request logprobs can draft on exactly verified targets (opt-in)

- `CBv2MTPConfig.logprobSpeculation` (off by default) lets a request with
  `topLogprobs > 0` take part in MTP rounds when the target verifies with
  `.rectangularExact` and acceptance is `.exact` (Inkling 2's DSpark drafter
  verifies that way). Each committed token's logprob and top logprobs are
  gathered from the verify window's raw logits in the same way ordinary decode
  gathers them, so the token, its logprob and the top list (members, order
  and ties) equal the drafter-off engine's bit for bit.
- On any other verification mode or acceptance rule the opt-in has no effect,
  and such rows stay target-only as before. Rows with a token constraint stay
  target-only. `DARKBLOOM_MTP_LOGPROB_ROWS=0` turns the opt-in off for the
  process.
