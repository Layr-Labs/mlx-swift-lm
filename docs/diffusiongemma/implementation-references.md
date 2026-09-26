# DiffusionGemma implementation references

The native block-diffusion work is under implementation. Registration, full-model
execution, API/cache/concurrency/multimodal qualification and performance acceptance
are not implied by configuration or sampler component tests.

Architecture and generation reference: Google/Hugging Face Transformers,
commit `c587bc884db2c2e31fc2b8102314656b17aa07b1`,
[`models/diffusion_gemma`](https://github.com/huggingface/transformers/tree/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/diffusion_gemma).
Copyright 2026 the HuggingFace Team. Apache License 2.0;
the license text is retained at [LICENSE-APACHE-2.0](../qwen4/LICENSE-APACHE-2.0).

Independent Apple-silicon implementation reference: MLX-VLM,
commit `e79b0e041677ec4ca5333ba750376bb4e8c434cb`,
[`models/diffusion_gemma`](https://github.com/Blaizzy/mlx-vlm/tree/e79b0e041677ec4ca5333ba750376bb4e8c434cb/mlx_vlm/models/diffusion_gemma)
and [`generate/diffusion.py`](https://github.com/Blaizzy/mlx-vlm/blob/e79b0e041677ec4ca5333ba750376bb4e8c434cb/mlx_vlm/generate/diffusion.py).
Reference-specific defaults are not automatically the released model's contract.
The diffusion-specific vision path follows the same revision's
[`gemma4/vision.py`](https://github.com/Blaizzy/mlx-vlm/blob/e79b0e041677ec4ca5333ba750376bb4e8c434cb/mlx_vlm/models/gemma4/vision.py).
MLX-VLM attribution and MIT text are retained in [LICENSE-MLX-VLM](LICENSE-MLX-VLM).

The configuration and sampler implement the released entropy-bound countdown,
fresh renoising and stable/confident block-finalization semantics. They are not
ordinary autoregressive generation or MTP. Model weights and serving credentials
are not included. Configuration tests cover nondefault semantic round trips and
recoverable rejection of unimplemented controls; they do not qualify the engine.

The selected released architecture has one tied output projection and shared
encoder/decoder text weights plus trained self-conditioning and vision components,
not an auxiliary autoregressive MTP head. A complete artifact must retain those
components. Native canvas generation is not labelled embedded MTP.
The [native sampling contract](native-sampling.md) describes the request-owned
entry points, experimental compilation control and numerical qualification.

Native engine lifecycle uses the common monotonic lease state with confirmed
native-work watermarks. Refined canvases are progress, not public output tokens.
The independent quantum watchdog publishes typed partial usage without touching
device state. `submitWithRetirement` returns an acknowledgment that remains
pending after an early watchdog terminal until actual queue-side cleanup.
Consumers retaining external reservations must honor that acknowledgment.

The common Responses wire decoder maps `input_image`/string `image_url` to the
existing canonical Chat image part without losing media bytes or ordering.
Uploaded-file references are rejected because no file resolver is present.
This also preserves the built-in text-only engine's media refusal; an unsupported
model must not appear to answer an image request by silently dropping its image.

## Native SDK surfaces

`DiffusionGemmaModelFactory` loads a separate `DiffusionGemmaContext` and
`DiffusionGemmaContainer`; it is not registered as an autoregressive
`LanguageModel`. Ordinary AR loading rejects the model rather than driving
`TokenIterator` against a diffusion canvas. The context retains the generation
recipe and normalized template syntax; no checkpoint files are rewritten.

The Hugging Face tokenizer-loader macro uses
`DiffusionGemmaTokenizerConfiguration` for the declared DiffusionGemma processor:
an absent `clean_up_tokenization_spaces` means false, matching its reference
tokenizer. Explicit Boolean values remain authoritative. This preserves literal
punctuation and quoted argument bytes after decoding, without changing token IDs,
the sampler or artifact files. Other processors retain their existing loader
behavior. Custom tokenizer loaders must preserve the same native decode contract.
The synthetic default tests and opt-in released-tokenizer oracle cover both
special-token modes; generation quality is a separate gate.

`generateNative` owns a single request's cache, noise and self-conditioning,
and emits only finalized blocks. Its counters distinguish committed output,
internal canvas work, first committed output and total request time.
`DiffusionGemmaPrefixCheckpoint` provides complete in-process committed encoder
state with exact token and trusted identity checks. A hit restores independent
full/windowed storage at the original positions without model replay. It is
not an encrypted persistent format or evidence of a paged attention dispatch.
Prepared media reuse requires a trusted identity for evaluated features, ordered
spans and attention/position policy. Unbound media remains cold.

An optional `DiffusionGemmaResidentPrefixConfiguration` connects these complete
snapshots to `makeNativeEngine`. Its partition stays inside the slot grant,
and the request estimate charges restore/capture overlap. Compact selection
copies preserve tensor bits while avoiding retention of larger donor buffers.
Staged snapshots are not lookup-visible and failed/canceled native work cannot
publish them. Engine shutdown clears the cache without retaining model weights.
Resident media uses `DiffusionGemmaPrefillGeometry` to preserve whole visual-block
cold boundaries. Exact restore may use a full prompt; appended requests must
restore on their own cold schedule. These primitives are not the full
persistent/paged release gate.

`DiffusionGemmaPersistentPrefixCodec` adds an explicit native portable layout:
committed full/windowed K/V, exact token IDs and virtual ring ordering. Its
identity binds verified artifact, template, build and numerics; per-load codec
tickets prevent adoption by a different owner. Integer bit packing keeps the
full native token list within the existing encrypted-manifest bound without
changing tokens, tensors, context capacity or legacy checkpoint encodings.
Imports allocate only bounded, validated destinations, and keep capacity leases
through partial transfer, adoption and request retirement. The common native
transfer primitive does not authenticate a file by itself: the provider must
authenticate every segment before finishing and retain its process I/O budget.
`CBv2NativeBlockPrefixCache` connects provider-side staging to native request
adoption. `DiffusionGemmaPrefixPersistence` keeps exported donors charged through
asynchronous completion and uses the existing provider-encrypted transport.
The configured cold chunk size binds both resident and portable identity.
With memory retention disabled, snapshots exist only for active capture/import/
donation ownership, not as a hidden persistent RAM tier. Successful disk adoption
reports the snapshot tier. Media manifests bind the typed media digest and the
same media-bound tenant scope as resident reuse. Import planning validates cold
geometry before allocating destinations. The native checkpoint layout is version
2: bound media can address exact stable cold boundaries off the nominal token
grid; text-only and autoregressive alignment rules are unchanged. Version1 native
checkpoints are incompatible and must fall back cold, not be reinterpreted.
A coincidentally aligned truncated endpoint does not replace a stable branch
point. The provider must implement matching native-boundary keys; the SDK never
changes chunking to manufacture reuse. Native media
does not use AR assistant or target-only checkpoint state. Real encrypted media
transport, paged dispatch and signed-process restart remain separate gates.

The current shared vision components select an explicit diffusion contract;
existing Gemma4 defaults remain unchanged. Pinned generated-weight fixtures
separately check vision features, projection, text-cache state and denoising
logits, plus the pre-change Gemma4 vision control. Syntax similarity between
floating-point operations is not numerical equivalence: explicit FP32 power
semantics are necessary where the reference and build-library defaults differ.

Provider/HTTP routing, native cohort scheduling, persistent/paged caches and
full checkpoint media/tool qualification are separate integration requirements;
the native loader and component fixtures do not establish those capabilities.

## Page-backed native arithmetic

`DiffusionGemmaRequestCache` and `DiffusionGemmaGenerationSession` accept an
explicit `PagedKVBackend` for SDK compatibility tests. Committed encoder state
uses real segmented page/ring storage, bounded writes and generation-checked
release. Full/windowed gathers feed the unchanged native SDPA graph; temporary
canvas K/V never enters committed pages. Portable snapshots preserve virtual
ring ordering and can restore into fresh pages without a model forward.
Native steps synchronize their consumers before another request can recycle
pages; failed steps restore the fence boundary and retire only their own rows.

This is not a direct paged-attention kernel or a proven speedup. The existing
scalar paged decode reduction fails raw-bit equality against native diffusion
canvas SDPA for the tested256/512 head dimensions. Its older relative-error
acceptance cannot be substituted for the native lossless gate. In the pinned
MLX revision,256-wide multi-query no-mask attention defaults to the composed
path and512-wide attention has no fused equivalent; different intermediate
rounding is observable. An opt-in `DiffusionGemmaPagedAttentionProbeTests`
records this failed candidate rather than weakening the reference.

`makeNativeEngine` can select an explicit segmented `pagedConfiguration` and
`CBv2ProcessMemoryOwner`. `CBv2NativeBlockPagedMemory` reuses AdmissionV2's
physical floor: nominal request pages overlap actual pool backing; canvas,
gathered views, prefix snapshots and transfer buffers stay additive. Actual
evaluated native imports receive materialization coverage and keep their leases
through final borrower retirement. Host encrypted I/O remains provider-owned,
not a second SDK charge for those same buffers. The pool can bind only one engine.

The provider's explicit page-backed selection is an integration surface, not
proof of the complete paging release matrix. Its default selection remains
contiguous. Concurrency, media, cache/lifecycle, long-context memory/performance
and real-transport qualification remain separate gates; these APIs do not
justify advertising fused paged dispatch or the target throughput.

## Native media preparation

`DiffusionGemmaProcessor` uses the released image geometry and actual pooled-grid
lengths, preserving typed message/tool history and interleaved media ordering.
`DiffusionGemmaVisualEmbeddings` binds evaluated native tower/projector output
to validated prompt spans. Native sessions can chunk media only at complete
visual-block boundaries; they never encode half an image bidirectional block.

Video support must distinguish a frame sequence from a dedicated video input
tensor. The pinned Transformers encoder/config omits native video features,
while [Google's model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)
documents video as image frames. `DiffusionGemmaVideoFrames` implements that
image route at one frame/s, up to60s, retaining the artifact's image token budget
because it declares no separate video-frame budget.
It does not copy the MLX-VLM processor's Gemma4 video patch payload or claim an
audio tower. Real clip/timestamp/ordering tests remain distinct from component
visual-mask fixtures and from image-only numerical tests.

The processor converts decoded input to bounded sRGB8 without dithering, then
uses `DiffusionGemmaBicubicRGB` for the reference's separable integer-coefficient
resize, per-axis clipping/rounding and FP32 rescaling. CoreImage's floating
linear-light bicubic filter is not numerically equivalent. CoreImage render tasks
are awaited while their buffers remain owned, and render failures reject rather
than silently supplying a zero-initialized image. Other model processors are
unchanged. Color-profile/alpha/codec behavior, actual GPU pixel checks and model
semantic/quality qualification remain separate evidence.

The RGB8 resampling contract is adapted from [Pillow 12.1.0 `Resample.c`](https://github.com/python-pillow/Pillow/blob/46f45f674d47b5d8bc54230dda8fe9e214598b87/src/libImaging/Resample.c).
Copyright Secret Labs AB, Fredrik Lundh and contributors, Jeffrey A. Clark and
contributors; the permission and disclaimer are retained in [LICENSE-PILLOW](LICENSE-PILLOW).
`DiffusionGemmaProcessorTests.imagePixelsMatchIndependentReference` is an opt-in
actual-processor gate: bind `DARKBLOOM_DIFFUSION_PIXEL_REFERENCE_LIVE=1` and a
synthetic oracle directory using `DARKBLOOM_DIFFUSION_PIXEL_REFERENCE_DIR`.
It compares original FP32 pixels and preserves separate, non-overwritten outputs;
successful geometry or a correct color-name answer does not substitute for it.

## Reproducing full-artifact tests

`Tests/MLXLMTests/DiffusionGemmaArtifactFixture.swift` pins every file, size and
SHA-256 checksum of `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision
`a7a81407613811e8ba63af92ac0d852b809e191f`. Download that revision with `hf download`
and set `DARKBLOOM_DIFFUSION_MODEL_DIR` to its snapshot directory. The portable
state test verifies the real bytes rather than depending on a private locally
produced manifest. The provider encrypted-handler test imports the same fixture.

The portable-state test also requires the source-matched `mlx.metallib` beside
its executable and the identical bytes in the nested `mlx-swift_Cmlx.bundle`
resource path. In the provider checkout, `scripts/stage-test-metallib.sh` stages
both after the SDK tests are built against the local pinned MLX Swift sources.
The checkpoint identity records the current executable and library hashes;
an old compiler-specific Metal digest is not portable across builds.
