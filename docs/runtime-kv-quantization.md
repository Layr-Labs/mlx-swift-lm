# Runtime KV quantization

Last updated: 2026-10-09.

`PagedKVPool.Config.quantization` selects optional packed attention storage.
The SDK default remains native. Serving policy selects one profile before
backend construction; changing weight precision does not select KV precision.
`PagedKVQuantization` describes K4/V4, K8/V4 or K8/V8 with affine groups of 64,
FP32 scale/offset metadata, deterministic normalized signed Hadamard keys and
an original native recent band of 128 confirmed tokens. Its stable fingerprint
also records native-only layer owners. Current mixed readers require equal K/V
head widths. Unsupported geometry fails before writes.

## Live storage and attention

Packed pages contain codes and metadata for **every** stored row. Separately
owned original FP16, BF16 or FP32 arrays retain recent128, all pending prefill
tokens and speculative tokens. The packed mirror of the recent band is already
present when those tokens age out. Windows at most128 and configured shared
assistant target owners stay native. Recurrent and auxiliary state stays native.

`PagedQuantizedLayerOperation` captures the pre-write view, appends owned rows
and calls direct mixed attention. Keys are rotated after projection/RoPE;
queries use that basis only for packed historical rows. Values retain their
original basis. `pagedattention.metal` combines original and packed rows using
bounded online-softmax tiles, including GQA, sinks, softcap and absolute masks.
It does not allocate a decoded full-history cache or query-by-history scores.
Oversized window prefill owns a bounded copy before ring writes overwrite
history. Shared borrowers use their owner's immutable view.

Qwen4 compact QSA reconstructs only its selected rows with
`PagedQuantizedSelectedGather`, including inverse key rotation and an exact
original recent/pending overlay. Selection order, duplicates and invalid-index
padding retain the native reader's semantics. Its output and per-segment
metadata stay charged through completion; the native selected reader refuses
packed storage. The sparse algorithm is preserved beyond its budget crossover.

`attendQuantizedReadOnly` reads committed history plus an original native
ephemeral canvas without advancing offsets or reserving future pages. Native
block diffusion uses this path during denoising and appends only encoder commits.

## Completion, accounting and failure

Page codes, scale/offset bytes, poison/padding, native generations and bounded
workspace reserve capacity before allocation. `bytesInUse` includes native
bands; `pagedStorage.livePageBytes` describes pages alone. Neither is a peak
device-memory measurement.

Native generations survive speculative commit/rollback and actual GPU
completion. Ordinary quantized decode disables successor chaining during
retirement. Adaptive target-prefix MTP therefore compares with measured
isolated ordinary decode; native backends retain the committed chained
baseline. `PagedKVBackend.supportsOrdinaryDecodeChaining` supplies the actual
availability to `CBv2MTPRoundDriver.build`, without relabeling isolated samples
as chained. `finishQuantizedStorageStep` evaluates completion roots and drains
the issuing stream before compacting original state to recent128. Copy
destinations remain attached to their charged owners even on partial failure.
Failure-only cleanup drains submitted work and retires rows; it never evaluates
a failed graph to invent completion. Teacher-forced forwards use the same
completion lifecycle.

Diffusion media must preserve each accepted whole visual block.
`DiffusionGemmaPrefillGeometry.maximumVisualBlockTokens` defines its closed
bound, including coalesced spans. Surrounding markers remain ordinary chunks;
paged configuration must quote that bound alongside text and canvas lengths.

`PagedQuantizedKernelSmoke.smokeShapes` selects actual packed geometries,
including shared-query widths and native-owner exceptions, with supported
query/native dtype pairs. `runtimeSmoke` evaluates write, native and raw-code
gather, selected gather and mixed attention variants using bounded fixtures.
Provider construction can run these in a child process to catch fatal compiler
failures, then repeat in its own process before publishing a pool. Per-segment
value offsets and selected-row counts are runtime inputs, so buffer growth and
selection length do not create new transfer/selected shader specializations.

## Complete checkpoints

The distinct `affine-paged-full-recurrent-v1` and
`affine-paged-historical-attention-v1` formats represent packed K/V as uint8
streams shaped `[1, H, bytesPerHead]`. Each head stores all coded rows followed
by its exact original native recent band. Original dtype, bit widths, codec,
rotation, band length and native-only owners belong in numerical identity.
Native groups retain their original typed tensor layout.

At a captured frontier, pages remain pinned and window/native copies retain
their source owners until evaluation and synchronization. Fragmented import
copies the original band privately before publishing a row. Its input DTOs stay
charged as scratch until the active row's separately charged copy completes;
only physical page destinations transfer into the pool floor. Import and later
aging preserve the original coded mirror, with no second quantizer.

Legacy untyped native tensor snapshots lack this band contract and packed pools
refuse them before allocation or mutation. Provider consumers must bind packed
complete formats explicitly and bypass legacy resident indexes until those
indexes carry equivalent ownership and numerical identities. This SDK change
does not extend native-block diffusion's separate prefix snapshot codec.

## Native execution with compressed checkpoints

`EngineV2` separately accepts `checkpointQuantization: PagedKVQuantizationConfig?`.
Its default is `nil`: it does not change native checkpoints or explicit live
packed inference. To compress durable checkpoints without executing packed
attention, leave `PagedKVPoolConfig.quantization` nil and supply the checkpoint
profile. Resolve the effective setting before constructing the store identity:

```swift
let checkpointProfile = CBv2CompleteCheckpointStorageQuantization.resolve(
    requestedProfile, layerKinds: layerKinds, layerDTypes: observedDTypes,
    pagedConfig: pagedConfig, hasAssistantState: hasPersistentAssistant)
// Include checkpointProfile?.identity, native dtypes and geometry in the
// store's numericsFingerprint before passing that store to EngineV2.
let engine = EngineV2(
    model: model, layerKinds: layerKinds, backend: backend, cacheProvider: caches,
    completePrefixCache: store, checkpointQuantization: checkpointProfile,
    completeCheckpointKVDTypes: observedDTypes)
```

The observed dtype argument enables generic contiguous historical targets that
have no model-provided dtype contract. It must come from the loaded model's
bounded native probe, not a requested weight precision. A supplied table cannot
override a conflicting model declaration or a package-issued native binding.
Export validates the actual native arrays against that table. With both new
arguments omitted, previously ineligible symmetric contiguous targets remain
ineligible. Explicit observed types also enable their uncompressed native
`native-contiguous-historical-attention-v1` format.

The resolver retains the existing codec when there is no eligible owner, an
unsupported dtype/geometry/profile, a live-packed pool, persistent assistant
state, or Qwen4 selected-index state. Configured native owners and windows no
larger than the exact recent band remain native. Family policy must keep MiMo
and native-block diffusion on their existing native paths. These are cache
encoding decisions, not reasons to change or disable native inference.

Four distinct formats identify lossy storage restored to native execution:

- `native-paged-affine-checkpoint-full-recurrent-v1`
- `native-paged-affine-checkpoint-historical-attention-v1`
- `native-contiguous-affine-checkpoint-full-recurrent-v1`
- `native-contiguous-affine-checkpoint-historical-attention-v1`

Their manifests include `checkpointQuantization` and `checkpointNativeDTypes`.
Every encoded K/V head contains only its older coded rows, followed by the exact
last `min(recentTokenCount, retainedTokens)` native rows. Unlike the live-packed
format, there is no packed mirror of that recent band. Convolution/recurrent
state and other permitted auxiliary tensors retain their native bytes. Model,
build, numerics, profile, dtype, owner/window geometry, tenant, prefix and tensor
descriptors are checked before creating an import stage. Native/existing packed
formats refuse these new metadata fields rather than reinterpret them.

Export reads immutable native page maps, captured window copies or contiguous
views one row at a time on the checkpoint queue. Scalar affine encoding and
Hadamard rotation allocate no MLX conversion graph or full-history tensor.
Each read reserves 64 KiB of bounded CPU conversion workspace; the returned
bounded Data remains the store's IO allocation. Import admits native page or
contiguous-capacity destinations, native auxiliary state and 64 KiB of conversion
scratch before allocation. It buffers at most one fragmented row, validates
coefficients, inverse-rotates K, and writes the native destination directly.
Window modulo placement, shared ownership and the full native serving promise
are unchanged. Cancellation retains native destinations and charges until the
required completion boundary, not merely until a public transfer handle closes.

Older reconstructed rows are approximate. Re-exporting native reconstructed
rows runs the encoder again; native dtype rounding and rotation can accumulate
error. There is no original-byte equality or no-drift guarantee for this mode.
Exact recent rows are copied without numeric conversion. The existing live
packed format's exact-code re-export and aging guarantees are unchanged.

## Focused validation

Build first, then stage the source-matched core metallib as described by the
existing test workflow. Run:

```sh
scripts/run-nested-suite.sh \
  'CBv2RuntimeKVStorageTests|CBv2MixedQuantizedAttentionTests|QuantizedCompleteCheckpointTests|Qwen3VLPagedQuantizedTests|PagedQuantizedKernelSmokeTests|Qwen4QuantizedSelectedPageCopiesTests' \
  --no-parallel
scripts/run-nested-suite.sh DiffusionGemmaPagedStateTests --no-parallel
scripts/run-nested-suite.sh \
  'NativeCheckpointRowCodecTests|NativeQuantizedPagedCheckpointTests|NativeQuantizedContiguousCheckpointTests|Qwen35NativeQuantizedCheckpointTests|HistoricalWindowCheckpointEngineTests.testContiguous' \
  --no-parallel
DARKBLOOM_TEST_NATIVE_COMPLETE_PREFIX=1 scripts/run-nested-suite.sh \
  CBv2NativeCompleteCheckpointTransferTests.testLossyNativePartialImportKeepsChargeUntilRealCompletion \
  --no-parallel
```

The suites cover actual f16/bf16/f32 storage and fragmented checkpoint import,
aging across the native band, D64/128/256/512 mixed attention, sinks/masks/GQA,
window wrap, source-copy failure ownership, read-only diffusion, Qwen3-VL
wrappers and compact Qwen4 QSA at 16,389 tokens. These are storage/operator contracts; whole-model quality and
performance require paired artifact-specific serving runs.
