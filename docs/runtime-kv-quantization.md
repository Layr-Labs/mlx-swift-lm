# Runtime KV quantization

> Last updated: 2026-10-09

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

Optional authenticated [token-byte topology](checkpoint-byte-topology.md)
describes these original role streams, including the coded mirror overlapping
the native band. Loaded-codec equality precedes import allocation; absent
legacy records preserve canonical serialization and opaque streams are not
assigned an inferred profile.

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

## Focused validation

Build first, then stage the source-matched core metallib as described by the
existing test workflow. Run:

```sh
scripts/run-nested-suite.sh \
  'CBv2RuntimeKVStorageTests|CBv2MixedQuantizedAttentionTests|QuantizedCompleteCheckpointTests|Qwen3VLPagedQuantizedTests|PagedQuantizedKernelSmokeTests|Qwen4QuantizedSelectedPageCopiesTests' \
  --no-parallel
scripts/run-nested-suite.sh DiffusionGemmaPagedStateTests --no-parallel
```

The suites cover actual f16/bf16/f32 storage and fragmented checkpoint import,
aging across the native band, D64/128/256/512 mixed attention, sinks/masks/GQA,
window wrap, source-copy failure ownership, read-only diffusion, Qwen3-VL
wrappers and compact Qwen4 QSA at 16,389 tokens. These are storage/operator contracts; whole-model quality and
performance require paired artifact-specific serving runs.
