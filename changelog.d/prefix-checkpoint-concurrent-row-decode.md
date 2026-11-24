### Restore a lossy prefix checkpoint's coded rows on several cores, into its pages

- A complete prefix checkpoint stored with the balanced (lossy) profile is
  restored by decoding every coded row back to the native dtype. The restore
  decoded each row on the staging thread, copied it 16 KiB at a time into the
  staged pages, and created those pages as copied host zeros. Models that
  store their sliding-window layers coded (Gemma 4) paid this on every cache
  hit, also for a one-block hit: on an M5 Ultra a 1,024-token Gemma 4 26B hit
  staged in about 141 ms, against about 30 ms for Qwen 3.6, whose checkpoints
  are native.
- The streaming row decoder now decodes the whole rows of a transfer call
  concurrently (at most eight threads, at most half the active cores), 1,024
  rows at a time. Rows of a paged native layer go straight to their page
  addresses (`CBv2PagedCheckpointStorage.rowTargets`, the addresses `append`
  writes); other destinations keep the ordered, batched sink calls. A refused
  row fails the transfer with the same error as before.
- Provider-staged checkpoint segments wrap zero-fill-on-demand pages when the
  live pool does (`PagedKVPoolConfig.hostZeroedSegments`), as admission growth
  already does, instead of copying host zeros into new buffers.
- Restored bytes are unchanged. The concurrent decode's scratch is charged
  with the import's decoder scratch.
  `DARKBLOOM_PREFIX_CACHE_PARALLEL_STAGE=0` keeps every row on the calling
  thread and the original sink path. Tests: `CBv2ParallelRowDecodeTests`,
  `NativeQuantizedPagedCheckpointTests.concurrentRestoreIsBitIdentical`.
