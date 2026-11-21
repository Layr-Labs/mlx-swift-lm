### Inkling 2 Mini: native looped model on paged KV

- Thinking Machines' Inkling 2 Mini (`inkling2_loop`) runs natively from its
  own checkpoint: 24 stored decoder layers, layers 6–17 run three passes (48
  layer applications per token), 128 experts top-8 with sigmoid routing and
  softcapped SwiGLU, learned relative attention bias and per-query sinks (no
  RoPE), short causal convolutions, block attention residuals and loop
  messages. Experts load in their native formats (NVFP4 in layers 0–19 with
  per-expert F32 scales, MXFP8 in layers 20–23); everything else stays BF16.
  Nothing is converted or requantized. FP32 router and gate products run as
  row-wise FP32, never TF32.
- The model has no contiguous cache: it declares `requiresNativePagedKV` and
  keeps 48 cache slots per sequence (8 full-attention, 40 sliding-window 1,024)
  plus each slot's convolution state. Its paged attention kernel computes the
  relative bias and sinks in-kernel; decode uses two half-partitions merged in
  a fixed order; prompt chunks use the GPU's neural accelerators exactly where
  MLX's own NAX check allows (elsewhere they take the decode arithmetic), and
  keys below a prefill block's window read as zero. The
  pool's segment count is bound at run time (two binding counts per pass), so a
  new pool size compiles no new kernel in the middle of a request. One
  attention core serves every engine on a loaded model: its step plans are kept
  per engine (keyed by the engine's page pool) and its mutable state is behind a
  lock.
- `PagedLayerCache` gains a package hook that gives a model kernel the
  calling rows' own page tables and reserved write slots; page ids are
  bounds-checked, a missing segment falls back to the gathered copy, and the
  existing write validation is unchanged; `poolIdentity` names the page pool a
  cache belongs to. Complete prefix checkpoints extend
  to models with sliding-window layers and recurrent state; their identity,
  salt and durability gates are unchanged.
- A model capability `requiresFixedPrefillChunkGrid` makes the scheduler end
  prompt chunks only at multiples of the chunk size or the prompt end (a chunk
  that does not fit waits; it is never shortened), with prefix checkpoints on
  the same grid. Inkling 2 Mini opts in, so a row's tokens do not depend on
  what else shares the step. Other models do not set it and are unchanged.
- `Inkling2Model.invariantPrefillBlockRows` (default nil) instead makes prompt
  chunks bit-identical at any boundary with fixed-row dense calls;
  `DARKBLOOM_INKLING2_INVARIANT_PREFILL=1` sets it at load. Concatenated
  attention projections follow `DARKBLOOM_INKLING2_CONCAT_PROJECTIONS`
  (default on). Tests: `Tests/MLXLMTests/Inkling2*`,
  `Tests/MLXLMTests/Integration/Inkling2/`.
