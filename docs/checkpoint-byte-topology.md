# Authenticated checkpoint token and byte topology

> Last updated: 2026-10-09

`CBv2CompleteCheckpointManifest.tokenByteTopologies` describes byte ranges in
the existing target K/V descriptor streams. It carries geometry authenticated
with the manifest; it does not encode tensors again, grant a cache hit, or
replace complete-checkpoint endpoint state.

## Canonical streams

Every topology identifies a target tensor index, loaded model layer, K or V
role, native dtype, role width, head count, absolute token interval and optional
semantic attention window. K and V have independent widths and byte strides.
Native dtype belongs to each role; FP32 scale/offset metadata remains FP32.

| Component | Original bytes and order within each head |
|---|---|
| `native` | All retained native rows in the original FP16, BF16 or FP32 dtype. |
| `affineMirror` | Every coded row, including rows overlapping the original native recent band; each row contains codes, FP32 scales and FP32 offsets. |
| `nativeRecent` | The separately stored original recent band following the complete affine mirror. |

`headStrideBytes` includes every component. `components` returns at most two
values in physical stream order, never a list of pages. `affineRoleBytes`
describes the code region, scale/offset locations, metadata width and total
affine row stride. The role's versioned `encodingIdentity` binds profile,
rotation, width and native dtype. Native-exempt target owners remain explicit;
shared borrowers have no separate target tensor or topology.

The recent-band overlap is required. As those tokens age, the already stored
code bytes become authoritative without a second quantizer. Omitting their
mirror would change restored history. Sliding rows retain their true absolute
start, including wrapped windows; `isFullAttentionHistory` distinguishes them
from full history. A byte range is always a view into the original descriptor
stream, not a reconstructed native tensor.

## Consumer API and validation

| API | Contract |
|---|---|
| `validatedTokenByteTopologies()` | Validates the complete manifest once and returns present records or bounded derivable legacy native records. |
| `validatedTokenByteTopology(tensorIndex:)` | Returns one validated target role, or `nil` when that tensor stays endpoint-owned. |
| `byteSpan(head:component:absoluteTokenStart:tokenCount:)` | Checks the head, component and token interval, then returns checked original byte offset/count and element width. |

Present records must cover every target K/V descriptor exactly once in sorted
tensor order. Structural validation checks geometry, head extents, native-band
placement and the complete attention owner map before deriving offsets. Owners
have unique model-layer identities; a borrower names an earlier true owner with
matching geometry and native dtype. Span multiplication and addition reject
overflow and negative operands.

`CBv2CompleteCheckpointCodec.plan` independently derives the expected topology
from the loaded codec, then requires exact equality before allocating an import
plan or mutating destination state. Changing a rotation/profile without changing
the descriptor size still fails this check. Manifest self-consistency alone is
not loaded-model authorization.

## Legacy and endpoint-owned state

Absent or `nil` topology preserves legacy canonical serialization: the field is
omitted, and decode/re-encode does not invent records. Native full/historical
descriptors with closed owner geometry can yield a token-byte view. Opaque
legacy affine `uint8` streams cannot yield an inferred profile or native-band
meaning and remain endpoint-owned. Complete auxiliary-only checkpoints also
omit the field because they contain no owning target K/V roles.

Recurrent, convolution, index and all MTP/assistant tensors remain complete
endpoint state. A target K/V view never authorizes truncating or substituting
those tensors. Package-issued native asymmetric and MiMo producers continue
emitting `nil`; their original serialization, permits and loaded-owner checks
remain unchanged. Native-block diffusion's distinct codec is unchanged.

## Host ownership and positive allowances

The decoder bounds the new record array to 4,096 entries before member decoding
or capacity reservation. Unknown-length containers stop before a 4,097th entry.
The existing provider I/O reservation must already cover parsing when decoding
begins; the later SDK manifest permit does not retroactively pay for parsing.

`maximumTokenByteTopologyHostBytes` quotes a positive envelope for two topology
arrays at twice the maximum record capacity, a transient integer-to-topology
index with allocation overhead at that capacity, and 64 KiB for headers and one
at-most-two-component iteration. Providers add this allowance to their existing
I/O scratch instead of treating unused capacity as credit. Additional retained
component collections and page descriptors require separate charges.

Constructed or adopted manifests with present records add the allowance to
their host metadata permit. Value copies share that owner and its reservation;
closing an export does not release metadata still held by a manifest alias.
Legacy `nil` manifests retain the original metadata permit. No GPU, token,
admission or activation reserve is lowered by these byte views.

## Validation and source map

After the full SDK test build and source-matched Metal staging, run:

```sh
scripts/run-nested-suite.sh \
  'CheckpointByteTopologyTests|QuantizedCompleteCheckpointTests|QuantizedHistoricalCheckpointBoundaryTests|PagedCompleteCheckpointCodecTests|Qwen35PagedCompleteCheckpointTests|CBv2CompleteCheckpointTests|CompleteCheckpointMetadataOwnershipTests|CBv2ElasticWindow(Storage|ModelParity|Checkpoint)Tests|CBv2WindowLifetimeBackingTests' \
  --no-parallel
```

`CheckpointByteTopologyTests` is a pure Swift unit suite under `Unit/Cache`;
the fork CI unit selector is `\.UnitTests/`. The quantized checkpoint suites
also run in the packed-storage CI selector and the whole-package pass.
Fixtures exercise all K4/V4, K8/V4 and K8/V8 profiles, FP16/BF16/FP32 roles,
window starts, complete coded/native overlap, native-exempt shared owners,
auxiliary state, loaded-codec mismatch, invalid owner maps, bounded decode and
metadata alias lifetimes, including auxiliary-only legacy export descriptors
and planning. They do not load network model weights or establish
whole-model quality/performance.

| Concern | Source under `Libraries/MLXLMCommon/ContinuousBatchingV2/Prefix/` |
|---|---|
| Typed components and checked byte spans | `CheckpointTokenByteTopology.swift` (`CBv2CheckpointTokenByteTopology`) |
| Whole-manifest and legacy geometry validation | `CheckpointManifestByteTopology.swift` (`validatedTokenByteTopologies`) |
| Expected loaded-codec records | `CheckpointCodecByteTopology.swift` (`checkpointTokenByteTopologies`), `CompleteCheckpointCodec.swift` (`plan`) |
| Optional bounded wire records | `CompleteCheckpointManifestCoding.swift` (`decodeTokenByteTopologies`) |
| Shared host ownership and permits | `CompleteCheckpointManifestMemory.swift` (`CBv2CheckpointManifestMemory`) |

See [runtime KV quantization](runtime-kv-quantization.md) for packed storage and
completion, and [complete checkpoint retention](prefix-checkpoint-retention.md)
for capture, publication and retirement.
