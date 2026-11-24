### CBv2 MTP: drafter opt-ins for depth-zero chaining, the assistant-copy failure rule and the cost curve

- Depth-zero plans run as chained ordinary decode only for a drafter that opts
  in (`CBv2MTPRequestStatefulDrafter.supportsDepthZeroChaining`, default
  false). The Qwen 3.5 inline assistant and Inkling 2's DSpark drafter opt in,
  so both behave as before. `DARKBLOOM_CBV2_MTP_DEPTH0_CHAINED=0` still turns
  it off for every drafter. Tests: `Qwen35MTPOrdinaryHistoryTests`.
- When the settled assistant copy of a prefix checkpoint fails, only an
  assistant that opts in
  (`CBv2HistoricalMTPPrefixCheckpointCoding.refusedAssistantCopyKeepsDonorStep`)
  keeps the split rule: a device error fails the cohort, and any other error
  drops only the checkpoints. Inkling 2's DSpark drafter opts in. MiMo V2.6's
  assistant gets the earlier rule back: any error discards the steps of the
  rows that had checkpoints. Tests: `CBv2SettledAssistantCopyFailureTests`.
- `DARKBLOOM_CBV2_MTP_COST_CURVE` can only turn the cost-curve depth choice off
  (`=0`). `=1` no longer turns it on for a family whose configuration does not
  opt in (`CBv2MTPConfig.costCurveDepthSelection`; only Inkling 2 does).
  Tests: `CBv2MTPCostCurveSwitchTests`.
