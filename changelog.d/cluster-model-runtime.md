### Preserve cluster histories, numerical paths and media ownership

- Keep per-request Qwen, Gemma, Nemotron and GPT-OSS batch histories independent,
  preserve singleton projection/expert arithmetic, and retain complete MiMo
  paged histories through rollback and supported transfer seams.
- Preserve FP32 Qwen expert partials across tensor collectives and expose
  independent tensor batch positions with explicit per-row arithmetic ownership.
- Prepare Qwen image/video inputs through the actual processor and expose
  request-owned vision/MRoPE interfaces. The optional encoded cluster carrier
  fails closed on ordinary local engines; native readiness remains authoritative.
- Bind captured logprob metadata to ordered generation events so producer-ahead
  execution cannot drain future metadata into an earlier stream frame.
- Keep registered geometry and admission regressions meaningful on CI without
  requiring full model-sized allocations, and pin the matching native Swift
  arithmetic changes for standalone SDK builds.
- Tiny fixtures and CPU/module checks do not qualify a complete model, RDMA,
  automatic topology, every supported modality or production throughput. Native
  capture support, tool/JSON constraints, MTP owner integration and full-artifact
  tensor qualification remain separate gates.
- Preserve request-owned Qwen tensor MTP assistant histories in charged paged
  storage, with bounded capture, restore, transactional rollback and retirement.
- Bound the inline Qwen assistant's trusted-history flushes to evaluated blocks
  (`historyFlushRows`), so a first draft never builds a whole-history rectangle
  and an ordinary decode stretch cannot keep one root per committed token.
- Under `Qwen35BatchDecodeScope.verifyRows`, a batched target-verify rectangle
  projects every row as its own single-request window: one `[B·S, H]` quantized
  product changes kernel and reduction order once B·S reaches the device's vector
  limit, which let concurrent speculative requests emit different near-tie tokens.
- Let those verify rows share a quantized call while their combined vector count
  stays below MLX's vector-kernel limit for the device (`Qwen35QuantizedVectorLimit`,
  which mirrors `get_qmv_batch_limit`): below it each vector's arithmetic does not
  depend on how many vectors share the call, so the rows keep their single-request
  results with fewer launches (qmv_wide reads the weights once per tile of up to five
  vectors). CPU work, other quantizers, overridden forwards and the 64/128-wide quad
  kernel keep one call per row.
- Where it measured faster and byte-equal, project a cohort's verify rows with one
  four-bit kernel (`Qwen35WideQuantizedKernel`, a port of `qmv_wide_impl` that keeps
  each vector's lanes, sub-chunk sums and shuffle ladder) instead of MLX's calls: up
  to eight vectors share a weight read instead of five. The layout follows the GPU
  family (`Qwen35WideQuantizedKernel.layout`): one output row per thread on Apple
  family 17 (M5 Max); two per thread for float32 inputs on family 15 (M3 Ultra), whose
  bfloat16 projections keep MLX's calls; other families keep MLX's calls.
- Pack a Qwen 3.5 sparse block's shared-expert float32 partial behind its expert
  rows in one tensor collective (`Qwen35TensorParallelSwitchLinear.carrying`) when
  the shared expert's down projection is a `Qwen35TensorParallelPackedPartial`;
  every value is unchanged and other models keep their path.
- Under a batched decode or verify scope, run each row's router, shared expert and
  selected experts as that row's own call computes them and sum every row's packed
  float32 partials in one collective (`withoutCollective`, `sum`) instead of one
  per row. Each element is still summed on its own, so every rounded value equals
  the row's own call.
- Let a CBv2 forward submit its decoder layers in strides
  (`Qwen35LayerSubmission.stride`, task-local, 0 by default): `asyncEval` on the
  hidden rows lets the GPU and a tensor rank's collectives run the layers already
  built while the host builds the rest. Evaluation order only.
- Remove the Qwen tensor shards' own attention page pool (`PagedAttentionKVPool`,
  `PagedAttentionKVCache`), its in-place decode kernel (`PagedAttentionDecode`) and
  the `KVCacheStepAttending` seam of `attentionWithCacheUpdate`: the cluster's
  tensor shards keep keys and values in the continuous-batching paged host.
- Share the live/checkpoint KV precision selection (`PagedKVQuantizationSelection`,
  `PagedKVQuantizationPolicy`), the segmented serving pool builder
  (`PagedKVBackend.segmented`) and the kernel preflight core (`PagedKernelSmokePlan`)
  in MLXLMCommon, so every owner builds a paged pool one way; owners that drive
  their own steps check refused writes, finish and discard each step through
  `PagedKVBackend` as `EngineV2` does. Engine behavior and pool configurations are
  unchanged.
- Multiply any number of vectors by an affine 2/4/8-bit weight with one fixed
  arithmetic (`AffineQuantizedVectors`): every vector gets the bits it gets alone,
  whatever the call's vector count, position or kernel layout, and the arithmetic
  is a function of the GPU family and weight packing only. A scoped Qwen 3.5
  decode step or verify window (`Qwen35BatchDecodeScope.exactVectors`) routes its
  affine projections through it, so a verify window's position equals the plain
  step that computes the same token. Decode-shaped forwards also fuse launches
  without changing any value (`ResidualRMSNorm`, the Gated DeltaNet convolution
  step and q/k normalization, SwiGLU and the attention output gate), each enabled
  only on GPU families measured byte-equal to the unfused operations.
