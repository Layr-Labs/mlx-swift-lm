### Inkling 2 Mini: exact prefill stage kernels, in-order NVFP4 expert GEMVs, split verify attention past 9 partitions

- Prompt chunks run the attention input projections as one fused stage, the
  post-attention and post-MLP convolutions with their residual adds as single
  kernels, and the message gate's FP32 products with their tail as one kernel.
  The prefill router picks its experts with the decode select kernel. Each
  kernel computes the same values in the same order as the chain it replaces.
  `DARKBLOOM_INKLING2_PROJECTION_FUSION=0`, `DARKBLOOM_INKLING2_STAGE_FUSION=0`,
  `DARKBLOOM_INKLING2_PREFILL_ROUTER_SELECT=0` and
  `DARKBLOOM_INKLING2_PREFILL_FP32_TILES=0` keep the earlier chains.
- NVFP4 expert GEMVs decode their weights in k-slot order straight from the
  stored bytes, and grouped GEMVs for 2 to 8 rows take their work items from
  the existing plan launch (`DARKBLOOM_INKLING2_EXPERT_INORDER=0`,
  `DARKBLOOM_INKLING2_EXPERT_GROUP_PLAN=0` keep the earlier forms).
- Verification windows with two or more query blocks take the split attention
  kernel past 9 partitions (`DARKBLOOM_INKLING2_ATTN_FUSED_CAP=0` keeps the
  fused kernel).
- Outputs are unchanged: prefill digests, teacher-forced logits and greedy
  output equal the previous kernels on real weights, alone and batched, with
  the drafter on and off. Tests: `Inkling2ProjectionFusionTests`,
  `Inkling2StageFusionTests`, `Inkling2ExactRouterTests`,
  `Inkling2ExpertKernelTests`.
