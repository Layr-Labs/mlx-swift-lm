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
