# CBv2 paged-attention decode microbenchmark (WS-C)

Decode-attention step time: the CBv2 paged backend (two-pass flash-decoding
Metal kernel over paged slabs) vs a v1-style per-row SDPA dispatch vs the
legacy dense rectangular batch (uniform lengths — the dense engine's best
case, unreachable under real mixed-length traffic).

- Shape: GPT-OSS-20B attention — 64 query heads / 8 KV heads, head dim 64,
  fp16 KV, page size 16, partition 256 tokens.
- Each step: append 1 token per row (slab slice-update writes) + attention
  dispatch + blocking `eval` (so numbers include the per-step host/submit
  overhead a single-layer step pays; a real model amortizes the submit over
  all layers in one graph).
- Machine: Apple M4 Max 128 GB, macOS 26.5, Xcode 26.4. NOTE: run recorded
  while sibling engine-v2 worktrees were building on the same machine —
  treat small (<10%) deltas as noise.
- Reproduce: `DARKBLOOM_CBV2_PAGED_BENCH=1 swift test --filter
  CBv2PagedBenchmarkTests` (10 warmup / 40 timed steps per scenario).

## Results (2026-07-01, commit at time of writing)

| B | context | engine | ms/step | KV GB/s | tokens/s |
|---|---------|--------|---------|---------|----------|
| 1 | 512 | paged-kernel | 1.075 | 1.0 | 930 |
| 1 | 512 | v1-per-row-sdpa | 0.790 | 1.4 | 1266 |
| 1 | 512 | legacy-dense-batch | 0.800 | 1.4 | 1250 |
| 2 | 512 | paged-kernel | 1.122 | 2.0 | 1782 |
| 2 | 512 | v1-per-row-sdpa | 1.301 | 1.7 | 1537 |
| 2 | 512 | legacy-dense-batch | 0.817 | 2.7 | 2449 |
| 4 | 512 | paged-kernel | 1.371 | 3.2 | 2917 |
| 4 | 512 | v1-per-row-sdpa | 2.287 | 1.9 | 1749 |
| 4 | 512 | legacy-dense-batch | 0.826 | 5.4 | 4845 |
| 8 | 512 | paged-kernel | 1.680 | 5.3 | 4762 |
| 8 | 512 | v1-per-row-sdpa | 4.461 | 2.0 | 1793 |
| 8 | 512 | legacy-dense-batch | 0.968 | 9.2 | 8268 |
| 1 | 4096 | paged-kernel | 1.135 | 7.4 | 881 |
| 1 | 4096 | v1-per-row-sdpa | 0.971 | 8.7 | 1030 |
| 1 | 4096 | legacy-dense-batch | 0.967 | 8.7 | 1034 |
| 2 | 4096 | paged-kernel | 1.470 | 11.5 | 1361 |
| 2 | 4096 | v1-per-row-sdpa | 1.648 | 10.3 | 1214 |
| 2 | 4096 | legacy-dense-batch | 1.104 | 15.3 | 1812 |
| 4 | 4096 | paged-kernel | 1.872 | 18.1 | 2136 |
| 4 | 4096 | v1-per-row-sdpa | 3.036 | 11.1 | 1318 |
| 4 | 4096 | legacy-dense-batch | 1.471 | 23.0 | 2719 |
| 8 | 4096 | paged-kernel | 2.639 | 25.6 | 3031 |
| 8 | 4096 | v1-per-row-sdpa | 5.631 | 12.0 | 1421 |
| 8 | 4096 | legacy-dense-batch | 2.359 | 28.7 | 3391 |
| 1 | 16384 | paged-kernel | 1.831 | 18.4 | 546 |
| 1 | 16384 | v1-per-row-sdpa | 1.612 | 20.8 | 620 |
| 1 | 16384 | legacy-dense-batch | 1.512 | 22.2 | 661 |
| 2 | 16384 | paged-kernel | 2.416 | 27.8 | 828 |
| 2 | 16384 | v1-per-row-sdpa | 2.690 | 25.0 | 744 |
| 2 | 16384 | legacy-dense-batch | 2.129 | 31.6 | 939 |
| 4 | 16384 | paged-kernel | 3.366 | 40.0 | 1188 |
| 4 | 16384 | v1-per-row-sdpa | 3.734 | 36.0 | 1071 |
| 4 | 16384 | legacy-dense-batch | 1.629 | 82.5 | 2455 |
| 8 | 16384 | paged-kernel | 4.306 | 62.5 | 1858 |
| 8 | 16384 | v1-per-row-sdpa | 5.930 | 45.4 | 1349 |
| 8 | 16384 | legacy-dense-batch | 2.342 | 114.9 | 3417 |

## Gate status (spec C-paged-backend §5)

| Requirement | Status |
|---|---|
| Beat v1 per-row SDPA at B ≥ 2 | **PASS** — all 9 (B, context) cells, up to 2.7x at B=8 |
| Within 15% of single-request SDPA at B = 1 | **PARTIAL** — 16k ctx passes; 4k misses by ~2%; 512 misses by ~36% (≈0.28 ms absolute) |

The B=1 gap is a constant per-step overhead, not a bandwidth gap: the paged
path issues two custom-kernel dispatches (partition pass + merge pass) plus
a per-step `seqinfo` host→device upload, while the baseline is MLX's single
fused `scaledDotProductAttention` primitive. The gap shrinks as context
grows (16 k passes — the kernel reaches higher GB/s than SDPA at B=8) and
vanishes into the batch win at B ≥ 2.

Per the spec, the backend is ship-optional: v2 can ship on the v1
(contiguous per-row SDPA) backend where single-request latency dominates,
with the paged backend enabled for batched service (the product target is
B=3–4). Follow-up to close B=1 short-context: a fused single-dispatch
variant for rows with one partition, constructed to be bitwise-identical to
partition-pass + trivial merge (w = exp2(0) = 1 exactly) so batch-composition
invariance is preserved, plus device-side seqinfo maintenance to drop the
per-step upload.

## History

- 2026-07-01 v1 (single-pass kernel, one threadgroup per (row, kv head)):
  failed the gate broadly — only B*KVH threadgroups meant the GPU idled at
  low batch (B=1 16k: 3.52 ms vs 0.90 ms SDPA, 9.5 GB/s).
- 2026-07-01 v2 (two-pass flash-decoding, 256-token partitions, ≤8
  simdgroups/threadgroup): all B ≥ 2 cells pass; B=1 within 15% at 16k,
  near-miss at 4k, miss at 512 (dispatch overhead-bound).
