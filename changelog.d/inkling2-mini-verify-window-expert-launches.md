### Inkling 2 Mini: mixed expert launches for verify windows, split verify attention past 7 partitions

- Steps of 9 to 32 rows (drafter verify windows of several streams, whose
  consecutive tokens share experts) run the expert GEMVs as planned grouped
  launches that take each expert's pairs in blocks of up to four, NVFP4
  weights decoded once per block, with work items of four or more pairs run
  as MMA tiles in the same launch. Smaller steps keep the grouped GEMVs and
  larger ones the tiles. Every form runs the same chains per output.
  `DARKBLOOM_INKLING2_EXPERT_GROUP_BLOCK=0` keeps the earlier launches;
  `DARKBLOOM_INKLING2_EXPERT_GROUP_FORM`, `_TILE_AT`, `_FROM` and `_ROWS`
  choose the form.
- Verification windows with two or more query blocks take the split attention
  kernel past 7 partitions (was 9). `DARKBLOOM_INKLING2_ATTN_FUSED_CAP=0` turns
  the cap off; another number sets it.
- Outputs are unchanged: the new launches give the tiles' bits and each row's
  bits alone for NVFP4, MXFP8 and dense experts at 2 to 33 rows
  (`Inkling2ExpertKernelTests`); greedy output, batched-equals-alone and
  drafter on/off equal the previous snapshot on real weights.
