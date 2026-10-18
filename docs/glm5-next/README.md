# GLM5 Next component provenance and notices

> Last verified: 2026-10-16

This page records sources and notices for the native GLM5 Next (GLM-5.3-Flash)
implementation the provider serves. It covers:

- Target assembly with the `GLM5NextExecutionPolicy.serving` preset.
- Bounded loading of the NVFP4 hybrid artifact (`GLM5NextWeightInventory.Storage.nvfp4Hybrid`):
  NVIDIA NVFP4 routed experts and dense MLPs, affine 4-bit non-expert matrices, and the BF16
  vision tower kept as stored.
- Native paged CBv2 execution with 2048-token steps, bounded paged history reads and a split
  step price (`GLM5NextPagedModelBinding.validateStepEnvelope`).
- Rectangular trained-MTP verification with the complete-prefix cache and per-request MTP memory
  (`GLM5NextMTPExecution.memoryPlan`).
- Image rows: prepared causal media embeddings on the protected engine, with fused vision
  attention on the default GPU stream.

Model quality and performance evidence are separate records. Synthetic
comparisons here establish equation and kernel parity, not full-model quality.

The existing [SDK MIT license](../../LICENSE) is unchanged. The identified
Transformers adaptations retain Apache-2.0 licensing and attribution; the full
license is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). This notice does not
relicense unrelated SDK code or model artifacts.

## Adaptation scope

| Native file | Reference scope and local adaptation |
|---|---|
| [GLM5NextConfiguration.swift](../../Libraries/MLXLLM/Models/GLM5NextConfiguration.swift) | Official checkpoint schema, Transformers configuration and SGLang geometry/NextN references; explicit Swift decoding and validation, not model registration. |
| [GLM5NextHyperConnection.swift](../../Libraries/MLXLLM/Models/GLM5NextHyperConnection.swift) | Transformers `Glm5NextTextHyperConnection.forward` and decoder residual mixing; Swift/MLX collapse and expansion with explicit dtype and epsilon boundaries. |
| [GLM5NextLinearAttention.swift](../../Libraries/MLXLLM/Models/GLM5NextLinearAttention.swift) | Transformers causal convolution, forget gate, recurrent Kimi delta attention and gated RMSNorm; projected inputs and caller-owned state, sequential reference recurrence. |
| [GLM5NextLinearKernel.swift](../../Libraries/MLXLLM/Models/GLM5NextLinearKernel.swift), [GLM5NextLinearChunkedRecurrence.swift](../../Libraries/MLXLLM/Models/GLM5NextLinearChunkedRecurrence.swift) | Transformers recurrent KDA, with MLX-VLM's vector kernel and flash-linear-attention's chunk form as execution references; whole-step FP32 recurrence and chunk-parallel prefill (`.serving` uses the chunked form). |
| [GLM5NextMoE.swift](../../Libraries/MLXLLM/Models/GLM5NextMoE.swift) | Transformers clipped MLP, experts, sigmoid router and shared-expert composition; native split projection banks and validation. |
| [GLM5NextSparseAttention.swift](../../Libraries/MLXLLM/Models/GLM5NextSparseAttention.swift) | Transformers learned pool selection and attention, with MLX-VLM as the absorbed-MLA reference; stateless explicit-history selection and selected-latent execution, not an owned serving cache. |
| [GLM5NextTextModel.swift](../../Libraries/MLXLLM/Models/GLM5NextTextModel.swift), [GLM5NextAttentionLayers.swift](../../Libraries/MLXLLM/Models/GLM5NextAttentionLayers.swift) | Transformers layer/backbone composition and final hidden boundary, using the separately tested native math. |
| [GLM5NextAbsorbedAttention.swift](../../Libraries/MLXLLM/Models/GLM5NextAbsorbedAttention.swift) | Transformers DSA attention equations, with MLX-VLM as the absorbed-MLA reference; the serving form (`.absorbedDenseHistory` in `.serving`) on views of the packed kv_b; the reference graph remains the member default. |
| [GLM5NextExecutionPolicy.swift](../../Libraries/MLXLLM/Models/GLM5NextExecutionPolicy.swift), [GLM5NextHyperConnectionKernel.swift](../../Libraries/MLXLLM/Models/GLM5NextHyperConnectionKernel.swift), [GLM5NextHyperConnectionFusion.swift](../../Libraries/MLXLLM/Models/GLM5NextHyperConnectionFusion.swift), [GLM5NextMoEKernel.swift](../../Libraries/MLXLLM/Models/GLM5NextMoEKernel.swift), [GLM5NextSparseSelection.swift](../../Libraries/MLXLLM/Models/GLM5NextSparseSelection.swift) | The same Transformers mHC, expert-combine and pool-selection equations; one load-time execution policy whose `.serving` preset is the provider configuration (fused Metal mHC collapse/expand, fused expert combine, parallel threshold pool selection). |
| [GLM5NextMoEPrefillKernel.swift](../../Libraries/MLXLLM/Models/GLM5NextMoEPrefillKernel.swift) | Adapted from oMLX (Apache-2.0, Copyright oMLX contributors; [notice](../mimo-v26/LICENSE-OMLX)) PRs #3995, #3993, #4022 and #4029 through the pinned MiMo port in `MiMoV26NAXMetalSources` (Apple MLX MIT primitives, notice retained there): segmented single-expert NAX tiles, joint gate/up, SwiGLU epilogue and row-map reads. Modified for affine 4-bit group dequantization with activation-dtype scales and offsets, the clamped-GLU epilogue, token rows read through the sort order and inverse by scatter. |
| [GLM5NextMTP.swift](../../Libraries/MLXLLM/Models/GLM5NextMTP.swift) | SGLang and vLLM trained NextN normalization, shifted input, residual DSA/MoE and shared head; no random assistant fallback. |
| [GLM5NextVision.swift](../../Libraries/MLXVLM/Models/GLM5NextVision.swift), [GLM5NextProcessor.swift](../../Libraries/MLXVLM/Models/GLM5NextProcessor.swift) | Transformers vision equations and image/video geometry, one frame at a time with checked evaluation and an explicit sampling policy. `GLM5NextVisionAttentionKernel.automatic` uses MLX fused attention only on the default GPU stream for head dimensions 64/80/128 and the query-tiled reference graph elsewhere; explicit `.fused` refuses off that stream. |
| [GLM5NextArtifactLoader.swift](../../Libraries/MLXLLM/Models/GLM5NextArtifactLoader.swift), [GLM5NextWeightInventory.swift](../../Libraries/MLXLLM/Models/GLM5NextWeightInventory.swift) | Local closed-inventory validation and bounded source-range loading; no download or publication client. |
| [GLM5NextCBv2.swift](../../Libraries/MLXLLM/Models/GLM5NextCBv2.swift), [GLM5Next state owners](../../Libraries/MLXLMCommon/ContinuousBatchingV2/GLM5Next/) | Native request ownership, transfer accounting, paged history reads, step pricing, complete-prefix checkpoints and embedding-spliced media prefill. |

The NextN adaptations retain the trained equations and checkpoint mapping, not
the upstream schedulers, CUDA/Triton kernels or dependencies. The implementation does
not claim cross-backend bit identity; source snapshots and numerical tests are
separate evidence.

### Pipeline stage boundary

`GLM5NextTextStagePlan` selects a nonempty contiguous range of **global** text
layer indices from the validated full inventory. `loadTextStage` retains the
artifact reader's complete hash/header checks but reads only the selected
payloads. It calls the supplied admission hook before each tensor allocation;
packed expert banks retain the ordinary bounded materialization and source
retirement contract. No full-model construction precedes stage selection.

With `GLM5NextArtifactLoadRoots`, fused input projections also retain their
concatenated buffers and member views before the admission callback and native
evaluation. This applies to dense/shared MLP, DSA, KDA and NextN construction.
A refusal or a throw after evaluation leaves those arrays with the construction
owner until explicit fenced cleanup; successful model adoption retires the
temporary aliases. Direct constructors keep checked local evaluation when no
materializer is supplied.

The first stage embeds tokens (or accepts prepared embeddings). Every
intermediate stage emits the complete `[B,T,hc_mult,hidden_size]` residual
streams. For the published four-stream, 4096-wide BF16 configuration, this is
32 KiB per token. Only the final stage takes the stream mean, applies the
trained final norm and reads logits. Global indices preserve the original
KDA/DSA mapping and sparse-attention layer identity.

Stage caches belong to their stage instance. A prepared candidate builds on a
copy, and publishes only after the caller's evaluation/join/fault-check and
downstream acknowledgement fence succeeds. Discard, a throwing fence and a
changed destination cannot publish that candidate. Copies retain recurrent
checkpoints; unsupported KDA truncation requires saved-boundary replay.

The final stage can explicitly load the trained embedding for NextN. The plan
charges that duplicate embedding in addition to its own layers and readout;
MTP tensors remain a separate partition. Stage ownership does not itself
provide a distributed scheduler, RDMA transport, paged cache integration,
model admission or a cluster-serving qualification claim. Those consumer
boundaries require their own tests and full-artifact runs.

### Execution policy

`GLM5NextExecutionPolicy` is chosen once per target, through
`GLM5NextArtifactLoader(executionPolicy:)` or
`GLM5NextTextModel(executionPolicy:)`. Every layer keeps it, the MTP layer
must match it, and paged backends built for the target take its metadata form.
No forward entry point accepts a per-call override. Every member defaults to
the unchanged reference graph; `GLM5NextExecutionPolicy.serving` is the preset
the provider loads. No member adds resident bytes, LOAD bytes or per-token
decode weight reads.

The clamped GLU reference uses separate sigmoid and multiply operations on
the caller's current device, retaining the activation dtype at each boundary.
It does not reuse MLXNN's global compiled SiLU trace, which can retain a CPU
device from an earlier model and change later GPU rounding. The decode tests
prime CPU BF16 traces before comparing the actual GPU MoE paths; packed-weight
tests retain exact equality against explicit activation operations.

| Member | `.serving` | Other choices and exactness |
|---|---|---|
| `sparseAttention` | `.absorbedDenseHistory` | `.absorbed`; mathematically equivalent to the reference, not bit-identical (below). |
| `sparseSelection` | `.parallelThreshold`, `deviceRanges`, `queryBlocks` | Exact: the same selected tokens. |
| `pagedMetadataReads` | `.deviceTable` | Exact: the same copied bytes. |
| `hyperConnection` | `.nativeNormFused` | `.nativeRequired`; reference rounding boundaries with FP32 sums in a fixed tree, not bitwise. |
| `routedExperts` | `.fusedDecodeNAXPrefill` | `.nativeCombine` (bitwise), `.nativeCombineSortedPrefill`, `.fusedDecode`; sorted/NAX prefill changes accumulation order. |
| `linearAttention` | `.nativeRecurrenceChunkedPrefill` | `.nativeRecurrence`; reference equations and FP32 state, reassociated sums. Chunked FP32 GEMMs may run at TF32 on Neural-Accelerator GPUs. |
| `normalization` | `.fusedKernel` | MLX fused RMSNorm and the index LayerNorm kernel. |
| `linearDecodeStep` | `.fusedEveryStep` | `.composed`, `.fusedKernel`. |
| `routing` | `.nativeKernel` | `.reference`. |
| `projections` | `.fusedInputs` | `.separate`; fused input projections are not bitwise. |
| `sparseDecode` | `.nativeKernels` | `.reference`. |
| `sparsePrefill` | `.nativeKernel` | `.reference`. |
| `prefillProjections` | `.mlx` (default-off: `.nativeGEMM`) | Opt-in native prefill GEMM. |
| `stepEvaluation` | `.deferred` (default-off: `.earlyAsync`) | Opt-in early asynchronous evaluation. |

MLX compiles one Metal library per distinct custom-kernel template tuple, so
the decode kernels take history-dependent extents at run time: the reference
selector reads rows and pools from its score shape and is specialized only on
the token budget, the selected-latent read takes the history length from the
last entry of the row's page table (both metadata forms), and asymmetric
segment reads take their token count from the output shape. Segment page
bounds and value offsets remain template values, so a new pool segment
compiles once; a growing history does not.

Native mHC, routed-expert and linear-attention choices run only on the default
GPU stream and throw elsewhere; there is no fallback. The parallel selector and the native
mHC and combine kernels use 256-thread groups; MLX refuses such a dispatch with
an error, never a fallback, if a pipeline admits fewer threads.

#### Whole-step linear attention

The reference preprocesses and recurs one token at a time. Both
`linearAttention` members instead convolve, normalize and gate the whole step
once per KDA layer, for every batch row, through the CBv2 contiguous and paged
targets and MTP verification alike, which share the target's layers:

- **`.nativeRecurrence`** runs `GLM5NextLinearKernel` once over all T tokens:
  one launch replaces the per-token recurrence for decode and prefill alike.
- **`.nativeRecurrenceChunkedPrefill`** keeps that kernel for steps shorter
  than `linearAttentionChunkSize` (32) and runs longer prefill steps in the
  WY/UT chunk form of `GLM5NextLinearChunkedRecurrence`: batched matrix
  products within each 32-token chunk and one sequential step per chunk.
  In-chunk decay ratios are factored through a mid-chunk pivot, which bounds
  each factor against overflow at the configured gate lower bound (-5);
  construction refuses a bound that does not admit the chunk size. Decayed
  terms may underflow to subnormals or zero, as in the recurrence. The
  in-chunk triangular system is solved by forward substitution with
  elementwise FP32 products, as in flash-linear-attention's
  `naive_chunk_kda`; a truncated Neumann product is unstable for nearly
  parallel keys with beta near one. The other in-chunk and per-chunk products
  are GEMMs, which MLX runs at TF32 precision on Neural-Accelerator GPUs by
  default.

A padded token must not advance any state. The whole step convolves each row's
valid tokens in order, after the previous valid history, and gives padded
tokens zero log decay and beta, which makes their recurrence step the exact
identity. Beta keeps the reference's FP32 sigmoid, projection-dtype rounding and
FP32 recurrence. Projection weights are read once per step, as before.

Neither member adds resident or LOAD bytes, but the chunk form adds
unmeasured activation transients per KDA layer step. At a 2,048-token stripe
(H = 64, K = V = 128) each FP32 `[T, H, 128]` intermediate is 64 MiB and each
`[H, N, C, C]` matrix 16 MiB. The chunk form materializes about 18 of the
former and a few of the latter, versus the kernel's 64 MiB output and 4 MiB
state. Its static upper bound is about 1.2 GiB per layer step if nothing were
released early; after MLX releases consumed buffers, roughly 0.3 to 0.5 GiB
stays live during the chunk scan. It scales linearly with the stripe. This
must be measured and covered by the activation allowance before the member is
used at that stripe size.

The official oracle's text cases have 17 and 19 tokens. They exercise the
kernel branch of both members; the chunk form is checked against the same
oracle at chunk sizes 4, 8 and 16 through the reference policy's internal
whole-step qualification seam.

#### Absorbed sparse attention

`.absorbed` computes `(q W_k)·c` and `W_v Σ p c` over only the non-padding
selection columns, directly on the packed `kv_b`:

- **Weights.** Each head's `[Dq, C]` key rows use MLX
  `quantized_matmul(transpose: false)`, which contracts the rows and expands the
  packed, group-quantized C axis. The `[Dv, C]` value rows use
  `transpose: true`. Both are batched over heads and queries, through reshape and
  slice views whose inner matrices are row-contiguous. Nothing is dequantized,
  copied or kept resident, so per-token weight reads stay at `kv_b`'s bytes and
  LOAD accounting is unchanged.
- **Numerics.** Q4 math runs in the scale dtype, as the reference `kv_b` call
  does, because FP32 inputs would stream per-call FP32 scale copies. Metal
  accumulates in FP32; with several queries its tiled kernels also round
  dequantized weights to BF16, as the reference `kv_b` call does. The
  reference rounds expanded K/V to BF16; this path rounds `q W_k` and `Σ p c`
  instead. The result is mathematically equivalent, not bit-identical. MLX's
  CPU quantized kernels round every BF16 operation in both paths.
- **Batching.** Contiguous sources batch scores and softmax over query blocks
  of at most 64 MiB of FP32 latents. Paged sources keep one priced read per
  query and use each read in place as its own block, so no unpriced stacked
  copy exists. Paged reads omit the selection's static padding columns through
  an explicit read contract; the full-width reference contract is unchanged.
- **Qualification.** No real-checkpoint qualification of this path exists.

On the default CPU stream, vision patch projection uses an explicit 64-lane
FP32 fused multiply-add reduction, folds to four lanes, and sums those lanes
left to right before adding bias. This preserves the canonical CPU projection
across Apple CPU generations. Biased FP32 vision linears pad their output width
to a multiple of 32 before slicing, so Accelerate's partial output block does
not change the bias epilogue. Parameter names, shapes and stored dtypes remain
unchanged. CPU patch reads pass the owner checkpoint, evaluation and fault checks before host access, and
cancellation is checked between patch rows. The Metal projection path and
fused vision attention policy are unchanged. The frozen image/video oracle
and end-to-end tolerances remain unchanged; these component checks do not
qualify full-model distributed image/video serving.

## Immutable source identities

Hashes below are SHA-256 of complete source-file bytes, not Git blob IDs or
model-weight hashes.

### Transformers

Canonical revision: `f0705f991a71f05dad9569336b181119f5546f93`.

| Source | Bytes | SHA-256 |
|---|---:|---|
| [configuration_glm5_next.py](https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/configuration_glm5_next.py) | 13,944 | `0a4df8f0238fd6e903fc94e7d89c0feee0bbea39bd8c92ccb42506e1b135d644` |
| [modeling_glm5_next.py](https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py) | 103,774 | `78e23c624010751f1578254145f313db2f3266579a0f9a396a3ff6cb5779f43d` |

The generated source files identify `modular_glm5_next.py` as their upstream
origin. The table pins the generated files actually used as references.

Earlier component citations have these verified relationships:

- mHC reference [modeling_glm5_next.py at
  `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`](https://github.com/huggingface/transformers/blob/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/glm5_next/modeling_glm5_next.py)
  is byte-identical to the canonical file, including the SHA-256 above.
- Recurrence reference [modeling_glm5_next.py at
  `4cc2aa84301c9aa210b5513dab9fefae03981f6e`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/glm5_next/modeling_glm5_next.py)
  is 103,722 bytes, SHA-256
  `caa38805e6bb7f8772b7016f98897ca0408090fd00c314ea6c8e63c543b586cf`.
  The complete-file diff is the absence of
  `_tp_plan = {"lm_head": "colwise_gather_output"}` in that revision; the
  recurrence equations are unchanged.
- Configuration reference [configuration_glm5_next.py at
  `d4f69a869d66eeba7af0cc19a585552b14ddd6f2`](https://github.com/huggingface/transformers/blob/d4f69a869d66eeba7af0cc19a585552b14ddd6f2/src/transformers/models/glm5_next/configuration_glm5_next.py)
  is byte-identical to the canonical configuration file.

### SGLang

Reference revision: `a6a8c56015b8dbca4147634fa9ef22c7c836de05`.

| Source | Bytes | SHA-256 |
|---|---:|---|
| [configs/glm5_next.py](https://github.com/sgl-project/sglang/blob/a6a8c56015b8dbca4147634fa9ef22c7c836de05/python/sglang/srt/configs/glm5_next.py) | 11,851 | `f4d0a006216ccb9117f3f84de76d71e9cf115b255538ee36d2da2a255cd88b83` |
| [models/glm5_next.py](https://github.com/sgl-project/sglang/blob/a6a8c56015b8dbca4147634fa9ef22c7c836de05/python/sglang/srt/models/glm5_next.py) | 75,925 | `438d03db9457282185f9ce9ee77da36d37582cea1412e4524f9bbc4e50fd15a3` |
| [models/glm5_next_nextn.py](https://github.com/sgl-project/sglang/blob/a6a8c56015b8dbca4147634fa9ef22c7c836de05/python/sglang/srt/models/glm5_next_nextn.py) | 4,307 | `3fad675ff95cbb7bbe5ff6e4bcf549e18a430e1ce7d0a06da003e0c1fac587c2` |

### vLLM

The secondary trained-MTP reference is vLLM at
[`273a04964ed9d1e389c4a7cb9d660e8294782f8d`](https://github.com/vllm-project/vllm/tree/273a04964ed9d1e389c4a7cb9d660e8294782f8d):
`vllm/models/glm5next/common/mtp.py`, `common/model.py`,
`vllm/models/glm5next/nvidia/ops/fused_eh_norm.py` and
`vllm/v1/spec_decode/llm_base_proposer.py`. Its
[Apache-2.0 license](https://github.com/vllm-project/vllm/blob/273a04964ed9d1e389c4a7cb9d660e8294782f8d/LICENSE)
applies to those adaptations; the pinned root `NOTICE` path is absent. No vLLM
runtime or bundled vendor kernel is included.

### MLX-VLM

The secondary absorbed-MLA reference is
[mlx_vlm/models/glm5_next/language.py](https://github.com/Blaizzy/mlx-vlm/blob/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/glm5_next/language.py)
at `4623cf9d2633c5ee2b1ac125eeae51423447c41f`: 43,767 bytes, SHA-256
`478b285c862e09a6c2c9289af702e5644f50445a2aea3a1c5de086455f1c0d9c`.
It is MIT-licensed; its notice is retained below separately from the Apache
adaptations.

## Copyright, license and NOTICE check

Transformers source notice:

> Copyright 2026 the HuggingFace Team. All rights reserved.

Its pinned repository license also retains:

> Copyright 2018- The Hugging Face team. All rights reserved.

SGLang NextN source notice:

> Copyright 2023-2024 SGLang Team

Both identified upstream projects use Apache License, Version 2.0. The included
license text is an exact copy of the pinned [Transformers
LICENSE](https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/LICENSE),
SHA-256 `77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049`.
The pinned [SGLang LICENSE](https://github.com/sgl-project/sglang/blob/a6a8c56015b8dbca4147634fa9ef22c7c836de05/LICENSE)
has SHA-256 `1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747`.

The complete pinned Git trees were checked for NOTICE files. Transformers has
none. SGLang has no root or model/configuration NOTICE; it does include
[python/sglang/kernels/aot/THIRDPARTYNOTICES.txt](https://github.com/sgl-project/sglang/blob/a6a8c56015b8dbca4147634fa9ef22c7c836de05/python/sglang/kernels/aot/THIRDPARTYNOTICES.txt),
covering FlashInfer, TensorRT-LLM, DeepGEMM and flash-attention material. Those
AOT kernels and bundled components are not part of the adaptations listed
above. The pinned MLX-VLM tree has no NOTICE file. This records the inspected
source scope, not a blanket clearance for additional upstream material.

Apache-2.0 section 4 requires the license copy, retention of applicable upstream
notices, prominent change notices in modified files, and applicable NOTICE
attributions where supplied. This page does not replace per-file change notices
or final review of the distributed source and binary contents.

### MLX-VLM MIT notice

The following preserves the pinned [MLX-VLM
LICENSE](https://github.com/Blaizzy/mlx-vlm/blob/4623cf9d2633c5ee2b1ac125eeae51423447c41f/LICENSE),
SHA-256 `f74c448746be3376e27dee43e2be9ec9d55f4820cd7466c603a6c7d1fd2d25b8`,
for the identified secondary reference; it does not replace the SDK's root
license or the Apache notices above.

```text
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model artifacts are a separate identity

The copied official configuration fixture retains the model's
[MIT notice](LICENSE-MODEL). It does not relicense the native source ports.

The official configuration reference is
[`zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda).
Its `config.json` SHA-256 is
`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`.

The served NVFP4 hybrid artifact is converted from
[`nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348):
its NVFP4 experts and dense MLPs are repacked losslessly, BF16 non-expert
matrices are quantized to affine 4-bit and the vision tower is kept as stored.

The earlier raw Q4 candidate is
[`Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP/tree/76add2a341a1cd90ad0e86bb69839ea9c35827c6),
not a shipped SDK model. Its raw assets have these known discrepancies:

- Its `chat_template.jinja` SHA-256 is
  `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`,
  unlike official
  `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`.
  They differ in null-content handling, concatenation and loop behavior.
- Its embedding and output-head headers contain Q4 U32 weights
  `[154880,512]` with BF16 scales and offsets `[154880,64]`, despite its card's
  BF16 embedding/output claim. Tensor headers take precedence for load geometry;
  they do not prove numerical provenance or quality.

These observations neither repair nor normalize the source template.

The separately identified `EigenLabs/GLM-5.3-Flash-MLX-4bit-mtp` artifact ships
template SHA-256
`46b2e2137e2759355b3e0e3e90cf4bd172b568437a75588af5c457cc8c095b2e`. It is the
official template above with its four numeric attribute accesses (`.0.`)
written as `[0].` subscripts, which the pinned Swift-Jinja 2.3.6 can parse.
The official template remains the rendering reference. The `glm5` tool-call
frames it emits are unchanged, so the parser fixtures still follow the official
source. No model-weight redistribution permission, publication, deployment
approval or qualification is implied by this provenance record.
Source-code licensing and the artifact's own license, bytes and prompt contract
must remain separately identified.
