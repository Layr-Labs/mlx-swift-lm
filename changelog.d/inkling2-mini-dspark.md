### Inkling 2 Mini: DSpark drafter and drafter prefix checkpoints

- Inkling 2 Mini drafts with the block drafter its checkpoint ships in
  `mtp.safetensors` (declared by `config.json` `mtp_config`; three layers,
  blocks of 8, reading the target's layer applications 5, 23 and 41 and its
  final norm). The drafter loads only for its exact target, the verify window
  adapts to acceptance, and greedy outputs are the same with the drafter on
  and off. Its prompt chunks run the same prefill path as the plain model.
- The engine gains a batched-draft seam for drafters that keep per-request
  state, a configurable limit on how many rows speculate (default 8), and
  durable prefix checkpoints that carry the drafter's state next to the
  target's (a separate checkpoint layout; the drafter copy is reserved through
  admission before it exists; media is refused; identity and salt unchanged).
- Checkpoint capture: a refused drafter copy now drops only that checkpoint
  instead of the live step; a restored request installs the drafter's capture
  context after checking the restored tokens equal the prompt prefix, so later
  turns can publish deeper checkpoints; capture outcomes are counted
  (`CBv2CheckpointCaptureEvents`, positions and sizes only). Tests:
  `Inkling2DSparkTests`, `Inkling2DSparkEngineTests`, `Inkling2DSparkPrefixEngineTests`.
- Depth-zero plans of drafters that keep ordinary history run as chained
  decode steps instead of separate seed steps (default on;
  `DARKBLOOM_CBV2_MTP_DEPTH0_CHAINED=0` restores the earlier steps). Those
  history steps read only hidden rows, so they take the model's certified
  tokens like a drafter-off step (default on;
  `DARKBLOOM_CBV2_MTP_HISTORY_CERTIFIED=0` keeps them on the full product).
  Tests: `Inkling2DSparkDepthZeroChainedTests`.
- The drafter keeps one reach pipeline per process and bounds the prompt
  evidence it stores. Certified draft selection over the unembedding
  (`DARKBLOOM_INKLING2_DSPARK_CERTIFIED`), early submission
  (`DARKBLOOM_INKLING2_DSPARK_PIPELINED`) and the compiled Markov chain
  (`DARKBLOOM_INKLING2_DSPARK_COMPILED_CHAIN`) are opt-in; default drafting is
  unchanged. The compiled chain is built when the drafter is created, so one
  drafter can serve several engines. Tests: `Inkling2DSparkCertifiedTests`.
