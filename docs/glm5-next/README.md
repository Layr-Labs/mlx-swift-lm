# GLM5 Next component provenance and notices

> Last verified: 2026-10-14

This page records sources and notices for the unregistered native GLM5 Next
candidate. Target assembly, trained MTP math, bounded artifact loading, media
processing and CBv2 state ownership are implemented source, **not a qualified
provider integration**. Full-checkpoint loading, complete prefix reuse, native
MTP execution, media serving, default-path performance and model quality remain
separate acceptance gates. Synthetic comparisons do not establish those gates.

The existing [SDK MIT license](../../LICENSE) is unchanged. The identified
Transformers adaptations retain Apache-2.0 licensing and attribution; the full
license is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). This notice does not
relicense unrelated SDK code or model artifacts.

## Adaptation scope

| Native file | Reference scope and local adaptation |
|---|---|
| [GLM5NextConfiguration.swift](../../Libraries/MLXLLM/Models/GLM5NextConfiguration.swift) | Official checkpoint schema, Transformers configuration and SGLang geometry/NextN references; explicit Swift decoding and validation, not model registration. |
| [GLM5NextHyperConnection.swift](../../Libraries/MLXLLM/Models/GLM5NextHyperConnection.swift) | Transformers `Glm5NextTextHyperConnection.forward` and decoder residual mixing; Swift/MLX collapse and expansion with explicit dtype and epsilon boundaries. |
| [GLM5NextLinearAttention.swift](../../Libraries/MLXLLM/Models/GLM5NextLinearAttention.swift) | Transformers causal convolution, forget gate, recurrent Kimi delta attention and gated RMSNorm; projected inputs and caller-owned state, sequential recurrence rather than the chunk-matrix optimization. |
| [GLM5NextMoE.swift](../../Libraries/MLXLLM/Models/GLM5NextMoE.swift) | Transformers clipped MLP, experts, sigmoid router and shared-expert composition; native split projection banks and validation. |
| [GLM5NextSparseAttention.swift](../../Libraries/MLXLLM/Models/GLM5NextSparseAttention.swift) | Transformers learned pool selection and attention, with MLX-VLM as the absorbed-MLA reference; stateless explicit-history selection and selected-latent execution, not an owned serving cache. |
| [GLM5NextTextModel.swift](../../Libraries/MLXLLM/Models/GLM5NextTextModel.swift), [GLM5NextAttentionLayers.swift](../../Libraries/MLXLLM/Models/GLM5NextAttentionLayers.swift) | Transformers layer/backbone composition and final hidden boundary, using the separately tested native math. |
| [GLM5NextMTP.swift](../../Libraries/MLXLLM/Models/GLM5NextMTP.swift) | SGLang and vLLM trained NextN normalization, shifted input, residual DSA/MoE and shared head; no random assistant fallback. |
| [GLM5NextVision.swift](../../Libraries/MLXVLM/Models/GLM5NextVision.swift), [GLM5NextProcessor.swift](../../Libraries/MLXVLM/Models/GLM5NextProcessor.swift) | Transformers vision equations and image/video geometry, with bounded frame/query work and explicit sampling policy. |
| [GLM5NextArtifactLoader.swift](../../Libraries/MLXLLM/Models/GLM5NextArtifactLoader.swift), [GLM5NextWeightInventory.swift](../../Libraries/MLXLLM/Models/GLM5NextWeightInventory.swift) | Local closed-inventory validation and bounded source-range loading; no download or publication client. |
| [GLM5NextCBv2.swift](../../Libraries/MLXLLM/Models/GLM5NextCBv2.swift), [GLM5Next state owners](../../Libraries/MLXLMCommon/ContinuousBatchingV2/GLM5Next/) | Native request ownership, transfer accounting and engine seams; no complete-cache or provider registration claim. |

The NextN adaptations retain the trained equations and checkpoint mapping, not
the upstream schedulers, CUDA/Triton kernels or dependencies. The candidate does
not claim cross-backend bit identity; source snapshots and numerical tests are
separate evidence.

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

The selected raw Q4 candidate under investigation is
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

These observations neither repair nor normalize the source template. No
template replacement, model-weight redistribution permission, publication,
deployment approval or qualification is implied by this provenance record.
Source-code licensing and the artifact's own license, bytes and prompt contract
must remain separately identified.
