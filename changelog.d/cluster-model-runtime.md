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
