### Inkling 2 Mini: MTP picks the draft depth from the measured per-depth costs

- The marginal depth policy can pick each row's depth as the argmax of
  expected committed tokens over the measured cost of that window depth
  (`CBv2MTPRawCostEstimator.costRatios(upTo:)`,
  `CBv2MTPMarginalDepthPolicy.selectDepthByCost`), instead of comparing
  drafts against one linear head-step cost ratio. Round costs flatten with
  depth, and the linear ratio, weighted by the depths already chosen, kept
  several-row steps at depth 1. `CBv2MTPConfig.costCurveDepthSelection`
  selects it; Inkling 2's DSpark configuration sets it; other families are
  unchanged. `DARKBLOOM_CBV2_MTP_COST_CURVE=0` or `=1` overrides.
- Served on an M5 Ultra: 4 streams code 63 tok/s per stream (was 50), prose
  86-88 (was 70-73); 6 streams code 50 (was 44-45), prose 62-63 (was 59); one
  stream unchanged. Tokens are unaffected.
