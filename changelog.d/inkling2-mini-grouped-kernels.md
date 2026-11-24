### Inkling 2 Mini: grouped expert kernels, one-launch row states, shared-page verify attention

- Decode steps of 2 to 8 rows compute the routed experts with grouped GEMVs:
  each selected expert is read once per step and serves every (row, slot) pair
  that picked it, with each pair's sums in the one-row order, so a row's result
  is the same alone and in a batch. The fused router issues its weight loads up
  front and selects by rank over the threadgroup.
  `DARKBLOOM_INKLING2_EXPERT_GROUP=0` keeps the per-row tiles, and
  `DARKBLOOM_INKLING2_ROUTER_RANK=0` the earlier select rounds.
- Engine steps pack their rows' convolution and state-space states in one
  launch instead of one concatenation per layer
  (`DARKBLOOM_INKLING2_STATE_GATHER=0` keeps the concatenation).
- Paged attention over a verification window stages each key/value page once
  per threadgroup for all of the window's queries, in the same page order, so
  an 8-token window reads the history once instead of four times.
- Outputs are unchanged: greedy and teacher-forced results equal the previous
  kernels on real weights, alone and batched, with the drafter on and off.
  Tests: `Inkling2ExpertKernelTests`, `Inkling2KernelsTests`,
  `Inkling2PagedAttentionTests`.
