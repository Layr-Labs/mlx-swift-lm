# Ternary Bonsai 2 native serving

The `prism_hadamard_qwen35` factories load the published
[Prism 2-bit MLX artifact](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/tree/3f926b415992eaa2ae9dd7b573706494d6bbf787)
using the existing dense Qwen3.8-27B / `qwen3_5` backbone. This is not the native
Qwen4 Flash-Next model.

The 27B contract below remains separate from the private 9B preparation in
the qualification section. Private local execution of the inspected 9B
artifact does not establish a public release, production eligibility or
transfer of the 27B qualification.

## Artifact contract

- Schema 2, affine 2-bit/group 128 with FP16 packed scales. Published language
  normalizers remain FP32; native normalization promotes downstream activations
  and KV/recurrent convolution rows to FP32. Do not infer state precision from
  embedding or packed scale storage alone.
- Packed recurrent prefill materializes the small convolution carry before
  retaining it. A slice otherwise owns the entire prefill-chunk allocation;
  this is a bit-preserving storage change, not a precision or recurrence change.
  Other model families and single-token decode keep their existing path.
- Signed block-Hadamard metadata in `hadamard.json`; FP32 transform arithmetic
  followed by restoration of the incoming dtype. Generic BF16 weight conversion
  is not applied to this artifact.
- Explicit grouped GDN layout, without a second output-head permutation.
- The selected weights contain 2,390 tensors: 402 sign vectors, 333 vision
  tensors and no MTP/draft tensors. The config declares vision support and
  `mtp_num_hidden_layers: 0`. A text-generation pipeline tag is not evidence
  that the vision tower is absent.
- No conversion or requantization occurs. The loader validates the transform
  manifest, duplicated sign vectors, packed dtypes/shapes and module mapping
  before returning the model. Unsupported packing contracts fail recoverably.

## Serving

`PrismHadamardQwen35TextModel` supports the text factory without loading vision
weights. `PrismHadamardQwen35` retains the vision tower and delegates native
CBv2 state to the existing Qwen text model; image/video preprocessing uses the
Qwen3-VL processor and native causal vision prefill. Generic VLM media generation
is rejected explicitly; use the native provider route for media.

The 27B artifact has no MTP assistant. Do not attach a similarly named checkpoint's
head or advertise MTP. Native maximum context is 262144; physical memory and
request admission remain separate from that architectural limit.

## Qualification

### Private 9B image processor preparation

The inspected schema-2 dense 9B profile uses 32 layers, hidden width 4096,
vocabulary 248320 and block-4096 packed modules. `PrismHadamardImageProfile`
also checks attention, recurrent and vision geometry before selecting
`PrismHadamardImageProcessor`. That route reads the checkpoint's image
`size.shortest_edge` and `size.longest_edge` pixel bounds through the existing
Qwen image preprocessing path. It preserves the default 1,280-token serving
budget, capped further by the checkpoint maximum, and rejects a checkpoint
minimum that exceeds this budget. For the inspected 65536-pixel minimum, a
64×64 image becomes 256×256 before patchification.

Ordinary Qwen processors, the block-1024 27B profile and other unadmitted Prism
geometries retain their existing image policy. Explicit processing bounds still
apply and must form a valid minimum/maximum pair: a maximum below the checkpoint
minimum requires an explicit compatible minimum as well. This preserves the
shared checkpoint path's rejection of inconsistent bounds. This is checkpoint
geometry combined with serving policy, not uncapped
reference-processor equivalence. The shared checkpoint-bound decoder preserves
Qwen4's existing bound precedence, defaults and validation. The provider's one-image-at-a-time vision
tower, device-buffer admission and MLX error handling remain required; decoded
image bounds are not a memory-admission grant. Video policy is unchanged.

Source tests and bounded private model comparisons do not establish full-model
vision or serving qualification. Broader tensor numerics, tokenizer equivalence,
single/multiple-image semantics, memory and API gates remain separate. No
tokenizer, chat-template or model artifact bytes are rewritten, and this profile
does not enable MTP or transfer the 27B memory/cache/performance qualification.

### Native Qwen3.5 vision activation

`Qwen3VLConfiguration.VisionConfiguration.visionMLPApproximationForConstruction`
in `Libraries/MLXVLM/Models/Qwen3VL.swift` selects the vision-block MLP
activation from the explicit checkpoint declaration. This mapping applies to
nested vision `model_type` values `qwen3_5`, `qwen3_5_vision` and
`qwen3_5_moe_vision`:

| Explicit `hidden_act` | Native `GELU.Approximation` |
| --- | --- |
| `gelu_pytorch_tanh` | `.tanh` |
| `gelu` | `.none` (exact GELU) |
| Missing or unrecognized | `.fast` (existing compatibility behavior) |

The mapping reads the optional declaration, so an absent field is not
mistaken for an explicitly requested exact GELU. Existing Qwen3-VL/Qwen4
policies and the patch merger's exact GELU remain unchanged. This affects
native construction without rewriting checkpoint configuration or weights.

The inspected private 9B artifact declares `gelu_pytorch_tanh`. Correct
activation construction does not establish full-model numerical equivalence
or answer quality.

### Explicit 9B MTP sidecar

The private 9B MLX revision `b94a52f3ce9047919dce00392a0fc6b8a7968abe`
publishes a separate `mtp/model.safetensors`: 486,582,808 bytes, SHA-256
`5b338202dce501905ad2a6ac44f07a471a9b59143a98327b87a0a884eeea7f4d`.
Its 15 F16 tensors describe one full-attention head. The accompanying config
incorrectly retains `text_config.mtp_num_hidden_layers: 0`; the strict loader
rejects it. A separately recorded local derivation changes only that count to
one, retains `block_size: 2`, and leaves every weight and base-config byte
unchanged. Keep this directory outside the base checkpoint's recursive weight
search. This is an explicit local candidate, not automatic release discovery.

`PrismHadamardQwen35.loadValidated9BMTPAssistant` validates the dense 9B
geometry, complete target/sidecar text configuration, exact tensor names,
shapes, declared packing and `format=mlx`. It binds the existing target's transformed
embedding and output projection. The base F32 norms stay unchanged; converted
MLX norm weights receive no additional offset. Capability remains off until
binding succeeds. Other Prism sizes are rejected by this binding.

Revision `f4e28a59531731c41b6dfac88cf271bd58f0a287` replaces only the
optional head with affine 4-bit/group-64 matrices: 136,884,309 bytes, SHA-256
`8968d94855ba38809a4ff5b6e20f4856757b64fdbbbd84511e4bea371764faba`.
The strict binder accepts its exact 31-tensor inventory: eight U32 packed
matrices, their F16 scales/biases, and seven F16 norms. Both quantization
declarations must agree; mixed layouts, different group sizes/bit widths,
missing or additional tensors and incompatible dtypes/shapes fail closed.
The original F16 path remains supported for rollback. The new publisher config
still needs the separately recorded zero-to-one layer-count correction. Base
weights, tokenizer, template and processor bytes are unchanged. Native MLX
quantized modules consume the supplied head directly; no weight conversion,
base precision change or GGUF/DFlash binding is introduced.

Draft depth is capped at two. The Hadamard output head cannot use the ordinary
quantized shortlist optimization, so this binding disables it. Request-state
admission reserves at least F32 head KV and hidden history, with the existing
allocation rounding and rollback ownership. Provider qualification uses exact
acceptance; serial and rectangular target verification remain separately
measured modes. Native sidecar loading and numerical head fixtures do not
establish full generation, cache, cancellation, sampling or performance
qualification. The provider retains an explicit local-path opt-in and a plain
decoding baseline; no MTP speedup or production eligibility is implied.

### Default performance profile

This performance follow-up leaves the published weights unchanged. Both eligible
paths are **on by default** when their overrides are absent. Exact `1` also
enables them; explicit `0` disables them. Other explicit spellings stay disabled.
Overrides latch per process and must be set before starting the provider.

To restore the previous schedule and repeated constant conversion:

```sh
DARKBLOOM_BONSAI_PREFILL_CARRY_ASYNC=0
DARKBLOOM_BONSAI_F16_CONSTANT_CACHE=0
```

The first submits already-compacted recurrent carry arrays earlier, only during
eligible packed text prefill. Native arithmetic, deferred-host-input handling,
paged write validation, final evaluation and request retirement stay intact.
Short/decode, media-position and captured-verification paths retain the previous
scheduling. The second reuses the exact FP16-to-FP32 quantization-constant
conversion the native packed operator already performs. It does not change
weights, quantization, activation/KV precision, gates or model architecture.
Descriptor and execution-stream changes invalidate that bounded reuse; tracing
falls back to the native operation. The generic `MLX_QUANTIZED_CONSTANT_CACHE=0`
rollback remains effective.

Constant reuse retains approximately 1.60 GB (decimal) of additional converted
constants for this artifact. This is a real residency cost, not a smaller model
or a free cache. Earlier carry evaluation reduces temporary prefill peaks; report
MLX active/cache and whole-process footprint separately. Neither mechanism is
MTP: the artifact has no assistant heads.

Independent frozen raw-logit, all-layer recurrent-state and paged-KV comparisons
cover both M3 Ultra and M5 Max at batch sizes 1 and 2. Native long-context,
media/cohort, cancellation/readmission, reload and encrypted-fixture checks pass
within the dependent provider's recorded scope. Ordinary API serving, lifecycle
and late-error framing pass; the original strict tool-success matrix remains
24/28, with four baseline-reproduced copy-value failures. Signed persistence and
hosted qualification remain separate open gates, not production sign-off.
Rejected or low-impact gather/sign-fusion prototypes are
not part of this candidate. No new numerical baseline or tolerance is accepted.

Focused configuration/shape, transform and HTTP regressions are included. The
dependent provider PR records full-model/API/image execution, 1K/10K/20K/50K
benchmarks, memory tradeoffs and open gates. Large decode improvements do not
imply a large or universal prefill improvement. Build success alone is not a
production, full-context, hosted OpenRouter or model-quality pass.

The wire parser uses Qwen structured tool frames. Tokenizer/template bytes and
the published quantization stay unchanged. The pack runtime's older text-only
note and stale README entry in its file manifest are superseded for identity by
the pinned config, actual tensor inventory and Hugging Face Git/LFS hashes.

## Composition and attribution

The public baseline uses the runtime sources approved in Layr-Labs/mlx-swift
PR #27 (`97b7f13830baa0aa761c979bcb4e94fc86b20cb6`, merged as
`35e55a6b9f53db22a01fa0b6fb38afef8dd3100d`). This follow-up pins the public
`Layr-Labs/mlx-swift` performance commit
`d7c1d3dcd114aa0e96ac6030da633ec7d4bd8254` for default-on exact constant reuse. Merge the
dependency first, then repin to its observed merged commit and recheck the
composed build. No floating branch or inaccessible private dependency is used.
The required Hadamard/2-bit primitives
already exist in the pinned MLX core; no whole-fork substitution is required.
The signed-transform layers are adapted from Prism's MIT-licensed Swift work
at `6d3a84de28225d1f5bc0a56f5c781596997242f9`; pack semantics are checked against
the runtime distributed with the immutable model revision above.

### Public 9B artifact status

The 9B checkpoint is publicly readable at
[revision f4e28a59531731c41b6dfac88cf271bd58f0a287](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/tree/f4e28a59531731c41b6dfac88cf271bd58f0a287),
verified on 2026-10-15. The earlier private inspection and F16 rollback pins
above remain historical evidence. Public availability does not change the
strict sidecar layer-count check, establish a final distribution namespace,
or qualify the full model. The Prism drafter requires rectangular verification;
the provider preserves that mode with exact acceptance and depth two. Its
generic automatic work-cap selector is not the Prism verification path.
