# Qwen 3.8 DFlash2 Swift Port Ledger

## Pinned inputs

| Component | Revision | Swift disposition |
|---|---|---|
| `Layr-Labs/mlx-swift-lm` | `ab73a827c9dde6f8802507003aa0be71605aab8e` | PR base |
| `Layr-Labs/qwen-3.8-mtp-challenge` | `eb5eadc7a165047d4321ce883b9ff30894d8bd19` | Arithmetic and runner provenance |
| `davidtai/dflash-mlx` | `c5b76ddb62bdefb6eeef1282641842edcf23a1b8` | DFlash2 algorithm authority |
| `youssofal/MTPLX` | `26e27b78d5299dcafb319844283ac50a137bfee5` | Final policy, route, and receipt authority |
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
- Direct fixed-M8 route at 16,384 prompt tokens and above.
- Target sampler temperature 1.0, top-p 0.95, top-k 20, seed 42.
- Target-authoritative longest matching draft prefix.

Explicitly absent: cost-aligned widths, later adaptive rows, CopySpec, DDTree,
sparse prefill, batched serving, HTTP scheduling, and concurrency claims.

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
| Width-2 and wide target cache commit | Implemented; exclusive-GPU test pending | Direct snapshot/tape routes are covered by `CBv2MTPCaptureVerifyTests`. |
| Real-artifact construction/parity | Pending | Requires local pinned snapshots. |
| Matched 1K/16K ABBA | Pending | Requires verified runner and exclusive GPU lock. |

No per-token, per-layer, or per-cycle engagement counters are present in the
timed lane. Swift-specific kernel claims remain pending until matched real-shape
profiling; Python row names are provenance, not evidence that a Swift dispatch
is faster.
