# Qwen 3.8 27B DFlash2 runner

This standalone executable is pinned to the target and DFlash2 artifacts used
by `youssofal/MTPLX#335`. It validates every required config, tokenizer, index,
and weight digest at construction, loads the BF16 DFlash2 checkpoint, installs
its measured affine W4/group-64 runtime, and runs either target-only
autoregressive decoding or the DFlash2 speculative lane.

Both prompt paths stream fixed 2,048-token chunks and project only the final
target vocabulary row per chunk. DFlash2 additionally projects the five fixed
post-block target captures directly into its typed sink-plus-window caches.
Before timing, the runner executes an input-independent 2,048-token warm and
every decode width from 1 through 8 on throwaway state. It then releases the
warm allocator cache and derives row 50's resident-weight budget from the live
post-warm footprint.

The optimized target route includes the retained row-21 Q/K fusion,
row-24/26 evaluation ladders, row-48 residual boundary, construction-routed
affine-Q4 M4--M8 projections, and DFlash innovation-tape rollback. At cached
lengths 16,384 through 32,767 only, verify widths 6--8 use a grouped-GQA
dispatch with the source's six-queries-per-KV-head layout. This is the
Swift adaptation of the source per-head route and stays within its measured
two-BF16-ULP bound on the real 16K geometry under MLX 0.32.2.
The wide recurrent path consumes the fixed 10,240-wide conv output directly,
fusing Q/K normalization, recurrence, and fp32 innovation-tape construction.
All model, dtype, packing, and geometry checks happen before warmup; an enabled
decode route dispatches directly without a custom-to-stock failure fallback.

Draft projections remain on stock `quantized_matmul`, matching the promoted
source receipt where `custom_draft_qmv.active_modules == 0`. Reusing the target
verification kernels in the greedy draft changes near-tied proposals and is
therefore not installed.

For short contexts, logical width 7 routes directly to physical width 8. Both
widths occupy the same padded M8 target-projection cost class on this Swift
backend, while width 8 can commit one more draft token. The policy decision is
made once per speculative cycle; installed projection routes perform no
eligibility checks or fallback accounting in their hot path.

Build the executable and its source-matched MLX Metal library:

```sh
scripts/build-qwen38-dflash2-runner.sh
```

Run DFlash2:

```sh
.build/release/qwen38-dflash2-runner \
  --target /path/to/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --draft /path/to/z-lab--Qwen3.8-27B-DFlash2 \
  --mode dflash2 \
  --prompt 'Explain why speculative decoding preserves the target distribution.' \
  --max-tokens 1024 \
  --receipt /tmp/qwen38-dflash2.json
```

For the PR #335 performance workload, pass the exact 1,024-token programming
prompt as token IDs and request the source-matched same-prompt conditioner:

```sh
.build/release/qwen38-dflash2-runner \
  --target /path/to/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --draft /path/to/z-lab--Qwen3.8-27B-DFlash2 \
  --mode dflash2 \
  --tokens-file /path/to/python-programming-1024.tokens.json \
  --conditioner-tokens 1024 \
  --max-tokens 1024 \
  --dflash-width 8 \
  --receipt /tmp/qwen38-dflash2-conditioned-1k.json
```

The conditioner generates 1,024 output tokens outside the measured interval.
The runner then clears the allocator cache and peak-memory counter before the
fresh measured pass: 1,024 input tokens of prefill plus 1,024 generated output
tokens. The fixed width is a construction-time benchmark control selected by
the matched MLX 0.32.2 gate; the adaptive short-context route remains the
default for general use. The receipt records both conditioner counts
explicitly.

The unchanged target-only control does not construct or load the draft:

```sh
.build/release/qwen38-dflash2-runner \
  --target /path/to/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --mode ar \
  --prompt 'Explain why speculative decoding preserves the target distribution.' \
  --max-tokens 1024 \
  --receipt /tmp/qwen38-ar.json
```

Both paths use target temperature `1.0`, top-p `0.95`, top-k `20`, and seed
`42`. Draft proposals are greedy, matching the pinned DFlash2 benchmark. The
JSON receipt stamps the exact Swift base, `mlx-swift`, MLX, MTPLX PR head,
Yukon source, DFlash2 source, target artifact, and draft artifact revisions
alongside the applied runtime settings and token digest. The runner serializes
GPU ownership through `/tmp/mtplx-gpu-exclusive.lock` and
requires the measured 512 MiB / 50-operation MLX command-buffer contract.
On the measured 96-GiB-and-larger machine class it requests a wired limit of
active bytes plus 64 MiB, capped 256 MiB below MLX's recommended working set;
smaller machines remain on the stock unwired route.

The startup acknowledgement is required by the source MTPLX distribution's
Apache-2.0 `NOTICE` terms and is written to standard error, outside all timed
prefill and decode intervals.
