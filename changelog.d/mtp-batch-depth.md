### MTP: one weak row no longer stops a whole batch from drafting

- For drafters whose rows commit their own accepted prefix (Inkling 2's
  DSpark), a plan of several drafting rows took the MINIMUM of the rows' own
  depth choices. One row with poor acceptance, or a low-confidence previous
  token, then held every row at plain decode, although each row still commits
  its target token at any depth. The plan now takes the depth with the most
  committed tokens per round cost over the whole rectangle
  (`CBv2MTPMarginalDepthPolicy.selectBatchDepthByCost`). The plan's rows that
  cannot draft count one token each at every depth.
- A row whose depth policy chose no draft also never drafted again, since
  only rounds updated its acceptance estimate. Each target-only decode step
  now moves the estimate 1/64 of the way back to the prior
  (`CBv2MTPRequestAcceptanceState.relaxTowardPrior`), so the row is offered
  drafts again after a few dozen steps. This applies to the same drafters
  only; drafters that commit a common width keep the fixed estimate.
- Under exact acceptance (the default), tokens are unchanged: every depth
  verifies exactly. Under an operator-set typical acceptance, a different
  depth changes which drafts are kept, within typical acceptance's contract.
  `DARKBLOOM_CBV2_MTP_BATCH_DEPTH=0` restores the minimum and the fixed
  estimate.
