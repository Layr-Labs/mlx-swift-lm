# Native MiMo V2.6

This directory documents the dedicated `mimo_v2` implementation, its explicit
serving boundaries and third-party adaptations. It is not a model catalog entry
or a release qualification. Weights are not included in this repository.

- [Composed component execution record](qualified-composition-20260928.md): current114 scope, committed test inputs and remaining gates.
- [Earlier component execution record](qualified-cohort.md): historical89 scope and its corrected test inputs.
- [OpenRouter media working-set check](media-working-set-20260930.md): exact encoded fixtures, bounded vision execution and measured scope.
- [Qualification contract](qualification.md): entry points, numerical/state,
  native ownership and endpoint gates.
- [Implementation provenance](provenance.md): architecture, processors and
  selected artifact versus source-code identity.
- [Implementation references](implementation-references.md): exact oMLX/MLX
  reference pins, adapted scope, exclusions and license requirements.
- [Native-rounding NAX attention](NAX-ATTENTION-PORT.md): three-pass arithmetic
  and exact query visibility.
- [Sorted MXFP4 gather](NAX-GATHER-PORT.md) and
  [joint gate/up](NAX-GATE-UP-PORT.md): native packed input and guarded dispatch.
- [Decode residual/norm](DECODE-KERNEL-ATTRIBUTION.md),
  [router](DECODE-ROUTER-ATTRIBUTION.md) and
  [distinct experts](DECODE-EXPERT-ATTRIBUTION.md): narrow short-forward variants.
- [Apache-2.0](LICENSE-APACHE-2.0), [oMLX notices](LICENSE-OMLX), and the existing
  [SDK MIT license](../../LICENSE): redistribution obligations.

## Code map

Model-specific source lives in each owning module's `Models/MiMo/` folder;
the tool parser stays under `MLXLMCommon/Tool/Parsers/MiMo/`. MiMo tests live
under their existing test target's `MiMo/` folder. These are source folders,
not new Swift modules. Shared continuous-batching, cache and budget code stays
in its existing common infrastructure folders.

| Concern | Entry point |
|---|---|
| Configuration / target | `Libraries/MLXLLM/Models/MiMo/MiMoV26Configuration.swift`, `MiMoV26Text.swift`, `MiMoV26Attention.swift`, `MiMoV26MoE.swift` |
| Strict native load and lifetime | `Libraries/MLXVLM/Models/MiMo/MiMoV26ModelFactory.swift`; `Libraries/MLXVLM/Models/MiMo/MiMoV26SerialLoad.swift`, `MiMoV26LoadedModel.swift` |
| Contiguous target rows | `Libraries/MLXLLM/Models/MiMo/MiMoV26CBv2.swift` (`MiMoV26CBv2Adapter`, `MiMoV26CBv2Backend`) |
| Trained heads | `Libraries/MLXLLM/Models/MiMo/MiMoV26MTPAssistant.swift`, `MiMoV26MTPState.swift`, `MiMoV26MTPPrefixCheckpoint.swift` |
| Media / encoded inputs | `Libraries/MLXVLM/Models/MiMo/MiMoV26MultimodalProcessor.swift`; `Libraries/MLXVLM/Models/MiMo/MiMoV26EncodedVisualDecoder.swift`, `MiMoV26EncodedAudioDecoder.swift` |
| Native complete-prefix contract | `Libraries/MLXVLM/Models/MiMo/MiMoV26NativePrefixProducer.swift`; `Libraries/MLXLMCommon/ContinuousBatchingV2/CBv2NativeCompletePrefixWork.swift` |
| Native target-only paging | `Libraries/MLXVLM/Models/MiMo/MiMoV26NativePagedProducer.swift`; `Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/NativePagedModelBinding.swift` |

The companion provider owns advertisement, artifact admission, process budgets,
HTTP/coordinator routing and deployment. A successful SDK call does not enable
those surfaces. Generic TokenIterator generation is not this native entry point.

## Managed vision working memory

`MiMoV26VisionTower.forwardBounded` processes one temporal grid at a time and
synchronously evaluates the projection, each transformer block, and the merger.
Its mandatory checkpoint runs through the existing `MiMoV26FailedMediaWork`
owner before evaluation. Completed frame features remain retained; earlier
layer graphs do not remain queued across the whole image or video.
Successful checkpoints retire completed scratch registrations from the native
owner registry while preserving the preparation owner, reservation and loan.
Failure never performs that retirement. This is necessary for actual managed
memory use to follow the same bound as direct component execution.

`MiMoV26VisionWorkingSet.frameBytes` therefore charges the largest frame/block
working set. The published 64-wide heads on the default Metal stream use a
fused-kernel buffer quote; CPU/custom streams and other geometry retain the
full-attention score allowance. FP32 intermediates and allocator slack remain.
`MiMoV26ManagedVisualCommitment` adds
all decoded inputs, pixel preparation, patch backing and output features;
target KV, codec weights and the host's OS/activation reserves remain separate.
This is a source-derived reservation, not a measured resident-memory claim.

Owned PCM processing similarly checkpoints every encoder block and RVQ
codebook step. `MiMoV26AudioWorkingSet` charges retained mels/features and the
largest frontend, encoder or actual quantizer tile. Original encoder grouping,
padding, masks and numerical operations remain unchanged. The separately
authenticated codec weights and final audio-patch graph are still charged.
Required checkpoint failure retains the original native owners and reservation;
it is never converted to a healthy media-capacity refusal.

The OpenRouter fixtures in `Tests/MLXLMTests/Resources/MiMoOpenRouter` cover
the two JPEG sizes, silent QuickTime video and AAC-bearing MP4 seen in the
2026-09-30 incident. `MiMoV26OpenRouterMediaTests` decodes those exact bytes,
checks their geometry and reservation bounds, and compares bounded vision
execution with the original lazy implementation on a tiny native tower.
`MiMoV26NativeMediaDeadlineTests.testMediaReservationRefusalLeavesTextEngineUsable`
checks media refusal followed by real text generation on the same tiny engine.
Full-size signed-provider qualification remains a separate release gate.
