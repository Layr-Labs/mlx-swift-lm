### CBv2 MTP: coupled drafting for Inkling 2's DSpark drafter (default off)

- `CBv2MTPConfig.coupledProposals` (default false) lets a block drafter that
  adopts `CBv2MTPCoupledBlockDrafter` propose, for each sampled row under exact
  target-prefix acceptance, the keyed race its verify draw runs at that output
  index (`CBv2StepSampler.mtpCoupledDraftProposal`; the default sampler
  forwards to `mtpCoupledProposal`) instead of the draft's argmax. Draft index
  j races with the noise of output index `generatedTokenCount + j`.
- Output is unchanged: exact acceptance commits only the target's own draws,
  so only acceptance can move. Greedy rows, rows in a grammar, typical
  acceptance and drafters that do not adopt the protocol keep their own
  rule, so with the setting off (the default) nothing changes.
- Inkling 2's DSpark drafter adopts it
  (`Inkling2DSparkDrafter.defaultMTPConfig(coupledProposals:)`): a coupled
  round forms each step's full draft logits (one full base product per round
  when the certified head is on) and feeds the selected token to the next
  Markov step. Its context cache never holds drafts, so a rejected proposal
  leaves nothing to roll back.
- Tests: `Inkling2DSparkCoupledDraftingTests` (seeded sampled output with
  coupling on equals coupling off and the drafter-off engine; greedy rows and
  typical acceptance never couple; a selection drives the Markov chain; a
  rejected or discarded coupled round leaves the drafter state a greedy round
  leaves), `CBv2MTPCoupledProposalsConfigTests`.
