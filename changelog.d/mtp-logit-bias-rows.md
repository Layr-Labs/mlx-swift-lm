### MTP: rows with a `logit_bias` can draft, with exact verification (opt-in)

- A request with a `logit_bias` used to stay target-only for its whole life:
  no drafts, so it decoded at plain-step speed. With
  `CBv2MTPConfig.logitBiasSpeculation` on, such a row drafts like any other.
  Every verify position adds the row's bias before its other transforms
  (`CBv2StepSampler.supportsMTPLogitBiasVerification`), exactly as ordinary
  decode's first stage does. The committed tokens are therefore the ones the
  drafter-off engine commits, greedy and sampled alike, and bias rows always
  accept exactly.
- Off by default. `DARKBLOOM_MTP_LOGIT_BIAS_ROWS=0` is the process kill switch
  over an opt-in. Rows with a token constraint keep the historical exclusion.
