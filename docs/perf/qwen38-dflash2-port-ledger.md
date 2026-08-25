# Qwen 3.8 DFlash2 Swift Port Ledger

## Pinned inputs

| Component | Revision | Swift disposition |
|---|---|---|
| `Layr-Labs/mlx-swift-lm` | `ab73a827c9dde6f8802507003aa0be71605aab8e` | Requested target anchor |
| `Layr-Labs/mlx-swift-lm` | `6036b2bff2d832d96f8136ecfb5d09978e1edc43` | Current-main PR base; includes the requested anchor and Qwen fused-input follow-up |
| `Layr-Labs/qwen-3.8-mtp-challenge` | `eb5eadc7a165047d4321ce883b9ff30894d8bd19` | Arithmetic and runner provenance |
| `davidtai/dflash-mlx` | `c5b76ddb62bdefb6eeef1282641842edcf23a1b8` | DFlash2 algorithm authority |
| `youssofal/MTPLX` | `9a6f48e69f9c8c6932d0f005c364844b2bf33e9c` | PR #335 head; final policy, route, and receipt authority |
| Qwen 3.8 target | `123db8bcc7101455b00d9aad36c0e760c6e7de02` | Exact target artifact |
| DFlash2 draft | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Exact BF16 checkpoint; recursively installed as affine W4/G64 at construction |

## Target substrate decisions

| Candidate | Status | Evidence and action |
|---|---|---|
| Copy Yukon `Qwen35FastEngine` | Rejected before runtime | Pinned readiness disables the custom engine, and its uniform Q4/G64 loader cannot represent the target's Q4/G32 base plus 74 Q8/G64 overrides. The provisional copy was removed. |
| Existing `MLXLLM.Qwen35TextModel` | Retained | Already loads heterogeneous per-path quantization and owns Qwen GDN speculative replay. Added only a fixed-segment DFlash2 captured forward. |
| Dynamic capture hook/set lookup | Rejected by design | The measured target forward executes six fixed layer segments and captures the five post-block outputs directly. |
| Native Qwen MTP | Excluded | DFlash2 runner construction never enables or loads the native MTP head. |
| Input-independent shape warm | Retained | A throwaway 2,048-token session executes target/draft widths 1 through 8 before any timing; allocator scratch is cleared before residency sizing. |
| Row-50 wired residency | Retained | On hosts with at least 96 GiB, the post-warm live footprint plus 64 MiB is capped 256 MiB below MLX's recommended working set. |
| Row-53 command-buffer profile | Retained | Process settings are installed before MLX construction at 512 MiB and 50 operations. |
| Rows 21/24/26/48 | Retained | Fixed Q/K RMSNorm+partial-RoPE, phase-specific evaluation ladders, and fused residual/RMSNorm boundaries use the exact target dimensions and execution phases. |
| GQA widths 6--8 | Retained with Swift-specific execution layout | Only cached lengths 16,384--32,767 route. The pinned MLX revision returns NaNs for the Python source's one-query-head/16K-KV SDPA allocation; packing all four KV heads into one grouped-GQA dispatch is bit-exact on the production geometry (`max_abs=0`). |
| DFlash innovation tape | Retained | Wide verification fuses Q/K normalization and recurrence from the 10,240-wide conv output while writing fp32 innovations. Rejected-prefix replay recomputes only normalized K from that output and combines it with innovation, G, and the pre-state. |
| Direct GDN projection replacement | Rejected and removed | Replacing the source-shaped GDN projection views expanded installation from 232 to 424 target modules and regressed the forced-M8 1K run to 66.456 tok/s. The retained fused one-allocation GDN input path remains intact. |

The captured feature contract is the concatenation of unnormalized post-block
outputs after layers `[5, 19, 33, 47, 61]`, width `5 * 5120 = 25600`.

The prompt path runs in fixed 2,048-token chunks, projects each capture chunk,
and seeds the five typed draft caches immediately. Each draft cache preserves
the pinned 64-token sink, 2,048-token sliding tail, and absolute RoPE positions
while advancing a separate logical offset. The first proposal therefore never
projects or retains a full 16K prompt.

The target-only control uses the same fixed chunk cadence and final-row-only
vocabulary projection without constructing the draft or capture projection.

## Artifact discrepancy

PR #335 records target config SHA-256
`533e833dedb9e7b6a8ee22ab4f2fc034bcf6ded9d8693e5ebcc9d5f159b62a3b`.
The bytes served by its pinned target revision hash to
`913b57d11eb131d9e1bb7316ea8729b6bcd110b59f8a08840a83ba2e524f370d`.
The two files differ only by the remote revision's added
`mtplx_mtp_quantization` metadata; this runner disables native MTP. The Swift
manifest pins both complete byte digests explicitly and records the digest
actually used in every receipt. The pinned draft config digest matches PR #335:
`873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.

## Final policy disposition

Retained from PR #335:

- DFlash2 physical block 8 and layers `[5, 19, 33, 47, 61]`.
- Row-11 plus row-15 position-EMA width selection below 16,384 prompt tokens.
- Swift cost alignment from logical width 7 to physical width 8. The target's
  width-7 affine-Q4 route already pads to M8, so the extra draft position uses
  the same target-projection cost class and can commit one additional token.
- Direct fixed-M8 route at 16,384 prompt tokens and above.
- Target sampler temperature 1.0, top-p 0.95, top-k 20, seed 42.
- Target-authoritative longest matching draft prefix.

Explicitly absent: later adaptive rows, CopySpec, DDTree, sparse prefill,
batched serving, HTTP scheduling, and concurrency claims.

## Swift projection routes

The projection installer first validates affine Q4 packing, group size, module
ownership, and every live `(K,N)` shape, then replaces eligible owners once.
The installed class uses a construction-time route table; the enabled hot path
does not recheck model metadata and never catches a custom-kernel failure to
fall back silently.

The target verification stack and the draft do not share a route table. The
target uses the retained PR #335 width islands below. The promoted source
receipts report `custom_draft_qmv.active_modules == 0`, so production draft
widths use stock `quantized_matmul`. Installing target verification arithmetic
in the draft changes selector numerics and is not part of the retained stack.

| Logical rows | Retained target Swift route |
|---|---|
| 4 | Exact split-K direct-nibble projection; two K partitions for N >= 4096 and four otherwise. |
| 5 | Exact M5 split-K projection. |
| 6 | Barrier-free K1 for `(5120,10240)` and `(5120,17408)`; exact K2 elsewhere. |
| 7 | Exact Metal 4 BM8 islands for output `(6144,5120)` and linear-Z `(5120,6144)`; remaining eligible shapes pad once to BM16. |
| 8 | Exact Metal 4 BM8 islands for attention K/V `(5120,1024)`, linear QKV `(5120,10240)`, and MLP `(5120,17408)`; exact-M8 output remains excluded; remaining eligible shapes pad once to BM16. |

The target installs 232 optimized projection modules while preserving 192
named GDN views that already share the fused physical input allocation. The
Q4/G64 draft remains on the source receipt's stock projection implementation.

The source turbo-Q8 projection candidate was implemented and exact-parity
tested against the real 73 Q8/G64 target owners, then rejected at its first
matched full-workload gate. It installed 305 total target projections but ran
at 68.005 tok/s, below both the retained route and the 68.351 tok/s unchanged
width-policy control. No Q8 custom route remains installed.

## Verification ledger

| Gate | Status | Receipt |
|---|---|---|
| Package/core revision manifest | Pass | `ArtifactManifestTests` |
| Real target/draft geometry | Pass | `ConfigurationTests`, `Qwen38DFlash2ConfigurationTests` |
| Fixed capture plan and direct surface | Pass | `Qwen38DFlash2TargetCapturePlanTests` |
| Acceptance prefix | Pass | `DFlash2AcceptanceTests` |
| Context split | Pass | `DFlash2ContextPolicyTests` |
| Source-matched metallib | Pass | `scripts/build-qwen38-dflash2-runner.sh` builds and stages the metallib beside the release executable. |
| Dynamic convolution and noncausal mask | Pass | Metallib-backed numeric fixtures cover both operations. |
| Typed context-cache/window plan | Pass | `DFlash2ContextCachePlanTests` covers long-prompt and incremental spans. |
| Width-2 and wide target cache commit | Pass | Direct snapshot, innovation-tape, and replay routes pass exact fixed-shape parity. |
| M4--M8 and padded-M16 affine Q4 kernels | Pass | `QuantizedProjectionKernelTests` on the real K/N/group geometries. |
| Rows 21 and 48 fused kernels | Pass | `Qwen38DFlashQKKernelTests` and `Qwen38DFlashBoundaryKernelTests`. |
| DFlash innovation and replay | Pass | `Qwen38DFlashTapeKernelTests` compares the production 10,240-wide conv-output route with stock output, final state, and every strict accepted prefix. |
| Long-context GQA widths 6--8 | Pass | `Qwen38DFlashGQATests` uses the retained 16,384-token KV geometry and matches native GQA bit-exactly. The incompatible per-head allocation is covered by the causal investigation above, not installed. |
| Real-artifact construction | Pass | Pinned local target and draft construct, warm widths 1--8, and install the projection counts above. |
| Exact conditioned 1K throughput | Pre-MLX-0.32.2 baseline pass | Exact 1,024-token programming input, untimed same-prompt 1,024-output conditioner, then a fresh measured 1,024-token output budget under the exclusive GPU lock. Retained runs: 73.216, 67.394, 73.514, 73.077, and post-route-cleanup 73.431 tok/s; mean 72.126, median 73.216, identical token SHA-256 `614179983a59e4bf674a92a752ce11459b8976280c1afa10de4d5ae000f62eba`. Path-sanitized raw metric receipts: [`run 1`](receipts/qwen38-dflash2-swift/cost-aligned-widths-conditioned-run-1.json), [`run 2`](receipts/qwen38-dflash2-swift/cost-aligned-widths-conditioned-run-2.json), [`run 3`](receipts/qwen38-dflash2-swift/cost-aligned-widths-conditioned-run-3.json), [`post-cleanup run`](receipts/qwen38-dflash2-swift/final-conditioned-run.json), and [`post-route-cleanup run`](receipts/qwen38-dflash2-swift/final-conditioned-run-2.json). Operator-home prefixes are rendered as `~`; metric fields are unchanged. |
| Width-policy behavior | Pass | The production-scheduling diagnostic completed in 204 cycles with 826 accepted draft tokens. Its width histogram is M3: 3, M4: 18, M5: 29, M6: 39, M8: 115, with no M7 dispatch. The unchanged policy needed 222 cycles and accepted 804 draft tokens. Raw [`retained diagnostic`](receipts/qwen38-dflash2-swift/cost-aligned-widths-diagnostic.json) and [`unchanged-policy diagnostic`](receipts/qwen38-dflash2-swift/stock-width-policy-diagnostic.json). |
| Turbo-Q8 candidate | Rejected | 68.005 tok/s on the exact conditioned workload; raw [`receipt`](receipts/qwen38-dflash2-swift/rejected-q8-conditioned-run.json). The custom route was removed. |
| Matched 16K route | Pending | Run only after the short-context acceptance gate. |

No per-token, per-layer, or per-cycle engagement counters are present in the
timed lane. Python row names remain provenance; only Swift parity tests and
matched real-artifact measurements promote a Swift route.
