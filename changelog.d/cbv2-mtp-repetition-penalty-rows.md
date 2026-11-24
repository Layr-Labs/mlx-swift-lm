### CBv2 MTP: rows with a repetition penalty can draft (opt-in)

- A row with `repetition_penalty != 1` was always target-only under MTP. With
  the new per-engine opt-in `CBv2MTPConfig.repetitionPenaltySpeculation`
  (default off), such a row drafts when the drafter uses target-prefix
  acceptance and the sampler re-derives the penalty at every verify position
  (`CBv2StepSampler.supportsMTPRepetitionPenaltyVerification`; the default
  sampler does). Each position is penalized as ordinary decode would after
  sampling the drafts before it: every token among the last
  `repetitionContextSize` tokens of prompt, confirmed output and those drafts
  has a positive logit divided by the penalty and a negative one multiplied,
  before presence/frequency penalties, with the same float32 ops. These rows
  always accept exactly, so the committed tokens are the MTP-off engine's at
  T 0 and at seeded T > 0; a rejected draft never reaches a later window.
  Constrained rows keep their exclusion. Tests:
  `CBv2MTPRepetitionPenaltyVerifyTests`, `CBv2MTPRepetitionPenaltyEngineTests`.
- `DARKBLOOM_MTP_REPETITION_PENALTY_ROWS=0` (or `false`, `no`, `off`) keeps
  these rows target-only even where an engine opted in. Without an opt-in
  nothing changes.
