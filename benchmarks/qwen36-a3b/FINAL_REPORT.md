# EigenLabs Qwen3.6 35B-A3B Swift Optimization Report

## Outcome

The retained construction profile combines an 8192-token solo prefill stripe, fixed-K2 inline MTP, an exact E256/top8 BF16 row-owned router finalizer, and an exact BF16 output-owned expert reduction. It generates exactly 100 greedy tokens on the frozen 129-token Python prompt.

On the Apple M5 Max benchmark host, the three independent invariant-clean full-profile runs reached 234.6, 232.8, and 232.1 steady-state decode tok/s. The median is **232.8 tok/s**, 1.663x the unchanged 140.0 tok/s control and 16.4% above the 200 tok/s requirement.

The construction-time `--output-parity byte-exact` route reached 109.9, 103.9, and 107.2 tok/s, for a **107.2 tok/s median**. All three 100-token arrays are byte-for-byte identical to the stock serial control. Byte-exact is therefore an explicit correctness datapoint, not the throughput default; `--output-parity fast` remains the default.

The 8192-token prefill stripe also fixes the reported long-context prefill bottleneck. In alternating matched three-run measurements, it reduced median TTFT by 42.6% at 8K and 34.2% at 32K.

## Exact pins and artifacts

| item | value |
|---|---|
| model | `EigenLabs/Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8` |
| model revision | `73a03825c2226177f3e679210965dba3508cdee8` |
| mlx-swift-lm base | `ab73a827c9dde6f8802507003aa0be71605aab8e` |
| mlx-swift dependency | `606d28cfa8c1d66b2975d3162a4aac9756835c5f` |
| frozen prompt SHA-256 | `ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de` |
| release benchmark SHA-256 | `c27f0793bf48a41395065e81cb9268a3d1cee32b67c3b6d8b3bee18dbec69690` |
| release metallib SHA-256 | `e3733805ff8549df57aebaf83efe7f491cfeadb851a1b8c8e2d3f476a5c1e0ea` |

Construction inspection fixes the target contract at H2048, E256/top8, I512, 40 layers, affine W4/g64 target experts, W8/g64 target routers/shared gates, and MXFP8/g32 inline-MTP matrices. The target affine kernel was not transplanted into the differently packed assistant.

## Decode comparison

| mode | output contract | repetitions | median steady decode | versus stock |
|---|---|---:|---:|---:|
| stock autoregressive control | byte-exact | 1 | 140.0 tok/s | 1.000x |
| optimized K2, `--output-parity byte-exact` | byte-exact | 3 | **107.2 tok/s** | 0.766x |
| optimized K2, `--output-parity fast` (default) | target-authoritative | 3 | **232.8 tok/s** | **1.663x** |

The byte-exact K2 route is 23.4% slower than stock and the fast route is 2.172x its throughput. This is why output arithmetic is a named construction choice rather than an unstated qualification on one headline number.

### Fast route receipts

| run | prefill | steady decode | generated | accepted |
|---|---:|---:|---:|---:|
| rep 1 | 102.8 ms | 234.6 tok/s | 100 | 64/68 |
| rep 2 | 103.3 ms | 232.8 tok/s | 100 | 64/68 |
| rep 3 | 103.4 ms | 232.1 tok/s | 100 | 64/68 |
| median | 103.3 ms | **232.8 tok/s** | 100 | 64/68 |

The exact invocation is recorded in every Markdown receipt. The historical fast repetitions omitted the new flag, which is equivalent to its `fast` default. A current explicit-flag smoke reached 233.5 tok/s and records `rectangular-target-authoritative` in both JSON and Markdown:

```sh
.build/arm64-apple-macosx/release/BenchCBv2 --model /Users/davidtai/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 100 --profile full --prefill-chunk 8192 --solo-stripe 8192 --mtp-depth 2 --output-parity fast --prompt-file benchmarks/qwen36-a3b/python-prompt.txt --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-mlx-swift-lm-eigenlabs-qwen36-a3b-fast-smoke --receipt benchmarks/qwen36-a3b/full-k2-c8192-fast-flag-smoke-20260825.json --label qwen36-a3b-full-k2-c8192-fast-flag-smoke --out benchmarks/qwen36-a3b/full-k2-c8192-fast-flag-smoke-20260825.md
```

The raw JSON receipts contain all 129 prompt token IDs, all 100 generated token IDs, phase timestamps, peak memory, artifact contract, construction routes, exact revisions, and GPU lock owner:

- `full-k2-c8192-hotpath-clean-rep{1,2,3}-20260824.json`
- `full-k2-c8192-hotpath-clean-rep{1,2,3}-20260824.md`
- `full-k2-c8192-fast-flag-smoke-20260825.{json,md}`

### Byte-exact route receipts

| run | prefill | steady decode | generated | accepted |
|---|---:|---:|---:|---:|
| rep 1 | 103.5 ms | 109.9 tok/s | 100 | 63/70 |
| rep 2 | 103.3 ms | 103.9 tok/s | 100 | 63/70 |
| rep 3 | 103.8 ms | 107.2 tok/s | 100 | 63/70 |
| median | 103.5 ms | **107.2 tok/s** | 100 | 63/70 |

The matched command differs from the explicit fast smoke only at `--output-parity byte-exact` and the output filenames. Every receipt records schema 2, `outputParity=byte-exact`, `verificationRoute=serial-byte-exact`, and `verify=serial_target`. Raw evidence is `full-k2-c8192-byte-exact-rep{1,2,3}-20260825.{json,md}`.

## Prefill receipt

| prompt | stock median TTFT | retained median TTFT | TTFT reduction | stock throughput | retained throughput | uplift |
|---|---:|---:|---:|---:|---:|---:|
| 8192 | 2986.7 ms | 1714.8 ms | 42.6% | 2742.8 tok/s | 4777.3 tok/s | 1.742x |
| 32768 | 14647.6 ms | 9640.7 ms | 34.2% | 2237.1 tok/s | 3398.9 tok/s | 1.519x |

Raw matched reports are `prefill-{stock,c8192}-L{8192,32768}-rep{1,2,3}-20260824.md`. The retained peak memory was 23.86 GiB at 8K and 29.09 GiB at 32K. A 16K stripe was rejected because it was 4.6% slower at 32K and used 26.4% more peak memory than the 8K stripe.

The cause of slow stock prefill is construction geometry: the default 512-token solo stripe fragments long prompts into too many target dispatches. The retained 8192-token stripe is installed once at engine construction. The optimized decode closures route only on logical M; artifact metadata and dtype are not revalidated, and no environment read, engagement counter, or invariant-based stock fallback exists in the measured hot path.

## Arithmetic boundary

The row-owned router and expert-reduction kernels have exact unit parity and leave the K2 generated stream unchanged relative to the same K2 route with stock target kernels.

The fast K2 route uses rectangular target verification (logical target M3). It is greedy and the target model remains authoritative, but matrix geometry changes one near-tie relative to serial one-row autoregressive evaluation: output token 96 is 17 rather than stock token 18. The explicit serial target-verification route reproduces all stock tokens exactly, but its new three-run K2 median is 107.2 tok/s, so it cannot satisfy the speed requirement. The fast receipt must therefore not be described as byte-identical to serial stock decoding.

## Candidate disposition

The retained winner stack is:

- 8192-token construction-time prefill chunk and solo stripe.
- Fixed-K2 inline assistant route, bound at construction.
- Row-owned E256/top8 BF16 target router finalizer for runtime logical M1...M16.
- Output-owned BF16 target expert reduction for logical M1/M2.

Not throughput winners or not installed:

- 16K prefill stripe: 32K latency and memory regression.
- Fixed K1/K3/K4: slower than K2.
- Byte-exact serial MTP verification: retained as an opt-in correctness mode and measured datapoint, but not the throughput winner because it is slower than stock.
- Target affine expert kernels in the MXFP8 assistant: incompatible physical packing.
- Unmeasured GDN and whole-MoE hypotheses: not performance claims and not part of the retained route.

See `OPTIMIZATION_LEDGER.md` for the complete candidate ledger and individual receipts.
