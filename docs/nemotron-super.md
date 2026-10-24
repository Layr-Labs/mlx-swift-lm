# Nemotron Super VL native components

This is incomplete native support for NVIDIA's original mixed NVFP4/FP8/BF16
Nemotron 3.5 Super VL checkpoint. It is not a registered, qualified serving
model. No model weights are included in this repository.

## Checkpoint identity

Source: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, revision
`7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c`. The complete checkpoint has 88 shards
totalling 75,010,188,240 bytes (74,986,817,096 tensor bytes and 23,371,144
bytes of safetensors headers) and 169,505 indexed tensors. Its 4,120 MTP
tensors are already in shard 88; downloading the complete snapshot includes MTP.
The weights use OpenMDW 1.1; preserve the upstream license and attribution.

The checkpoint index SHA256 is
`c368acdbcfe882236af1ec79df4a1fb6127e6ed9f742d59c01c21fe96ddc2741`.
`NemotronSuperCheckpointPin` and the inventory loader reject incompatible
layouts. The pin is public so a host opens every shard against the same SHA256
values that `NemotronSuperInstallation.stage` requires; it grants no serving or
discovery capability. ModelOpt producer scales are part of the computation: generic affine
quantization is not an interchangeable representation.

## Component boundaries

- `MLXLMCommon/Quantization` owns calibrated ModelOpt projections, packed
  NVFP4 expert banks, E4M3 codec and bounded unit-scale FP8 KV storage.
- `MLXLLM/Models/NemotronHModel+ModelOpt.swift` installs calibrated language
  weights after inventory preflight. `NemotronSuperMTPModule` loads the embedded
  head; its eager `forward` is the reference and `graphForward` the draft path.
- `NemotronSuperMTPAssistant` is the opt-in speculative assistant described in
  "Text speculation" below. `NemotronSuperFP8KVHistory` is its rewindable
  unit-scale E4M3 history; `NemotronSuperMTPServing` loads the head and binds
  the assistant to a `NemotronSuperMTPTextTarget`.
- `MLXVLM/Models/NemotronSuperInstallation.swift` owns sequential native loading
  epochs, completion fences and retained resources. The header-derived
  79,972,843,016-byte quote excludes backend allowance and is not measured peak
  memory; its resident payload is 73,217,162,760 bytes, including the
  3,932,160-byte BF16 video embedder. Failed completion retains ownership
  rather than refunding early.
- `NemotronRadio`, `NemotronSuperVisionProjector`, image composition and
  `NemotronSuperVisionPreparation` provide the causal image path.
  `NemotronSuperImageComposition.videoFeatures` runs frame pairs (tubelets)
  through the separate video patch embedder, then the image path's encoder and
  projector.
- `NemotronSuperImagePreprocessing` applies NVIDIA's grid rule and reproduces
  CPU PyTorch 2.12 float32 antialiased-bicubic rounding; its unit tests compare
  outputs bitwise. File decoding belongs to the caller.
- `NemotronSuperVideoPreprocessing` samples at most 256 frames at 2 frames per
  second, labels tubelets with upstream's millisecond-truncated timestamps and
  chooses the 1024-patch frame grid. `NemotronSuperEVS` selects the projected
  video tokens that EVS keeps (vLLM `evs.py`). See "Video input".
- `NemotronSuperImageComposition.activationBytes` and `pinnedActivationBytes`
  bound the MLX working memory of one image's tower and projector evaluation
  from shapes, including MLX's unfused attention fallback for unsupported head
  dimensions. `tubeletActivationBytes` and `pinnedTubeletActivationBytes` bound
  one video tubelet. Integration tests compare them with measured MLX peaks.
- `NemotronSuperRecurrentAdapter` is a causal adapter: prompts and media are
  serial; text decode may batch independent requests (see "Batched text
  decode" below).
  `NemotronSuperMTPTextTarget` is a separate opt-in target-hidden wrapper
  that speculates for text only and serves images target-only (see "Images
  on the speculating owner"); its subclass `NemotronSuperMTPCaptureTarget`
  adds the exact captured verify window.

Paths above are relative to `Libraries/`. Tests under
`Tests/MLXLMTests/Integration/Nemotron`, `Kernel/Quantization`, `Kernel/Vision`
(the ModelOpt NVFP4 and FP8 primitives), `Unit/Models` and `Unit/Vision` (image
and video preprocessing, EVS selection and vision preparation) use tiny
synthetic fixtures, not the full checkpoint. The exception is the gated video
golden described in "Video input".

## Video input

`NemotronSuperVideoPreprocessing` follows NVIDIA's processor and vLLM's
uniform loader: `min(256, frames, floor(seconds * 2))` frames, at least one,
spread with NumPy's truncating `linspace`; one label per two-frame tubelet (an
odd last frame alone) whose timestamps truncate the frame duration to whole
milliseconds; and an even 1024-patch frame grid (1280 x 720 gives 42 x 24
patches, 252 tokens per tubelet). Frames are normalized like images
(`NemotronSuperImagePreprocessing.normalizeRGB`).
`NemotronSuperImageComposition.videoFeatures` pairs frames 2t and 2t + 1 into
tubelet t (the caller duplicates an odd last frame), runs the BF16 1280 x 1536
video patch embedder, then the image path's encoder and projector, and returns
`[T, N, 4096]`. It only builds the graph; the caller bounds the tubelets per
call. `tubeletActivationBytes` bounds one tubelet's working memory and k
tubelets by k times that; measured peaks of one and four synthetic tubelets at
the largest grid stay within it.

EVS prunes the projected tokens without weights: each token scores `1 - cos`
against the same position of the previous tubelet
(`NemotronSuperEVS.dissimilarity`, a lazy pairwise graph that a host may
evaluate in overlapping windows), the first tubelet is always kept, and
`keptPositions(dissimilarity:tubelets:tokens:pruningRate:)` keeps the highest
remaining scores up to `retainedCount`, `max(N, int(T * N * (1 - q)))`.
`NemotronSuperVideoPreprocessing.defaultPruningRate` (q = 0.5) is the model
README's vLLM serving recipe (`--video-pruning-rate 0.5`). The pinned
config.json's `video_pruning_rate` (0.7) is deliberately not used: it drives
HF's in-model EVS, which also assumes square frame grids.

The installation loads the video embedder like every other tower weight:
3,932,160 bytes of resident payload and, through the image mapping's fourfold
staging allowance, 15,728,640 staging bytes, so the LOAD quote grows by
19,660,800 bytes.

`NemotronSuperVideoPreprocessingTests` (`Unit/Vision`) compares sampling,
timestamps, labels, frame grids and EVS selection with upstream Python.
`NemotronSuperEVSTests`, `NemotronRadioTests`,
`NemotronSuperImageCompositionTests` and `NemotronSuperImageActivationTests`
cover scoring, the tubelet patch order, frame packing and the measured peaks.
`NemotronSuperVideoReferenceTests` runs the pinned checkpoint's tower, video
embedder and projector on two tubelets against NVIDIA's modeling code in
float32 (per-token cosine above 0.98, mean above 0.999); it runs only when
`NEMOTRON_SUPER_REFERENCE_DIR` and `NEMOTRON_SUPER_VIDEO_GOLDEN_DIR` are set.

## Activation overflow policy

ModelOpt's PyTorch eager cast has no saturation: an FP8 static per-tensor
activation with |x * multiplier| > 464, or an NVFP4 row whose
amax / (6 * input scale) > 464, becomes NaN. The input scales were calibrated
on text. Image embeddings can exceed them; under that rule an image request
gets NaN logits, and greedy decoding then emits token 0.

Deployment engines (TensorRT-LLM, vLLM) convert these activations with
SATFINITE instead: FP8 activations clamp to ±448, NVFP4 block scales saturate
at 448 and E2M1 values at 6. They never turn an overflow into NaN.

Super follows the deployment semantics. This is a reversible numerics
decision. Each ModelOpt projection and expert bank carries a
`ModelOptActivationOverflow` fixed at load:

- `.eagerNaN` is the default for every constructor and loader, and every
  other user keeps it. Its results are unchanged, bit for bit.
- `.saturating` removes only the overflow mask. NaN inputs stay NaN, and
  infinities saturate like any other overflow. Finite values round through
  the same E4M3/E2M1 encoders, which already saturate, as do MLX's native
  `to_fp8` and its NVFP4 product.

`NemotronSuperInstallation.stage` loads every calibrated language projection
and expert bank with `ModelOptActivationOverflow.nemotronSuper`
(`.saturating`). The MTP head loader does the same. Every Super path quantizes
under its layer's policy, so the plain and speculating owners stay exact
against each other. This covers ordinary and tiled prefill, row-exact batched
decode, captured verify windows, the head and the assistant's output
projection. The two policies give different outputs only on activations that
the eager rule turns into NaN. To revert, select `.eagerNaN` at load.

`ModelOptActivationOverflowTests` checks every quantizer on synthetic values:
the FP8 and NVFP4 (dense and routed) activation kernels, the lazy FP8 path, the
eager FP8 and NVFP4 references, the row-exact path and the NVFP4 expert bank.
Overflows are NaN under `.eagerNaN`. Under `.saturating` they equal a clamped
reference. In-range activations are bit-identical under both policies.
`NemotronSuperActivationOverflowTests` gives a fixture image embedding beyond
one layer's calibration:

- The eager rule produces NaN logits, and both owners fail the request closed
  (below) where greedy decoding would emit token 0.
- The Super policy keeps the logits finite, and the plain and speculating
  owners emit the same tokens.
- Text is bit-identical under both policies.

The full-checkpoint evidence is in the provider pull request.

### Non-finite logits fail closed

EngineV2 never emits a token decided by logits that hold NaN or infinity, for
any model and any request, text or media (`CBv2NonFiniteLogits`). Each step
builds one lazy flag per sampled row in the same graph as the sampled token:
`x - x` is NaN exactly for NaN and ±infinity, so one elementwise pass and one
row reduction decide it. The flags are concatenated after the sampled tokens
and read back with them at the step's existing host sync; the tokens array
that feeds a chained successor is unchanged. An MTP verify window computes the
same flag per position and writes `-1` in place of a flagged target token in
the existing acceptance packet, so no draft matches it.

At finalize a flagged sampled row (plain, chained, MTP seed or decode, sampled
prefill) emits nothing and finishes with
`.error("non-finite logits: the model produced NaN or infinity")`, which the
provider classifies as a generation failure. A verify row fails the same way
when any position it would commit is flagged; positions past its accepted
prefix are discarded anyway and do not fail it. The failing row takes the
existing finalize-time failure path: a discarded chained successor rolls back
its token, a verify row rolls back its recurrent window and releases its
assistant state, reservations follow the ordinary finish, and error finishes
never donate or publish prefix state. Other rows of the batch are unaffected.
`CBv2NonFiniteLogitsTests` covers both backends and the chained path;
`NemotronSuperNonFiniteLogitsTests` (one token embedding spiked beyond the
calibration under the eager rule) covers row-exact batched decode, the
speculating owner's seed decode and its verify window, each beside a healthy
request that keeps its solo tokens.

Only sampled positions carry logits. A non-sampling prompt chunk whose state
already turned non-finite fails at the request's first sampled position, after
any resident prefix blocks of earlier chunks were published.

## Graph execution and measured performance

`ModelOptCalibratedLinear` and `ModelOptNVFP4SwitchLinear` run a device-resident
graph path by default (`execution: .graph`); `checkedForward` stays the eager
reference. No call reads device values: calibration scalars are validated on the
host at init and after every parameter update, routing stays on the device, and
invalid expert IDs give NaN rows plus a validity flag for the owner's fence
(`ModelOptGraphValidity`). Metal kernels reproduce the eager NVFP4 and FP8
activation quantization bit for bit; NVFP4 products are MLX's gathered
`.nvfp4` kernels on the producer bytes; FP8 weights are decoded inside a copy
of MLX's gemv (one row) or per eager tile (several rows). Decode, row-exact
batched decode (each group of up to 16 rows) and MTP verify windows are
bit-identical to the eager reference. Default prefill is not: it runs
`.tiled`, MLX matrix kernels from 32 dense rows and 4 routed rows per expert,
with routed activations and sums in the activation dtype. Producer
bits and scales are unchanged; the summation order and one rounding of the
unscaled sums differ (float32 sums alone: largest difference 5.7e-7 of the row
maximum; with bfloat16 sums each output stays within 2^-7 of its row maximum).
`.rowwise` keeps prefill bit-identical. `calibratedPrefillTokenLimit` caps
graph chunks at 2,048 tokens and checks the larger of the per-layer ModelOpt
workspace and the whole-chunk transient bound against the 24 GiB workspace
ceiling. The whole-chunk bound is 1 GiB + 3.75 MiB per token: 8.5 GiB at
2,048 tokens (`prefillActivationBoundBytes(tokens:)`). The provider's prefill
lane reserves the larger of that bound and the per-layer workspace; the
24 GiB ceiling is not the lane's actual reservation. The larger chunks in the
historical measurements below are not the current managed default.

Full checkpoint (coordinator runs, M3 Ultra 60-core GPU, release build, greedy,
bf16 contiguous KV): eager decode 8,862 ms/token and prefill 0.99 tok/s; graph
path at SDK f495d987: decode 31.9 ms/token with greedy tokens identical to the
eager run (60/60); a 1,826-token prompt prefills at 227 tok/s (`.rowwise`) or
441 tok/s (`.tiled`); MLX peak 81 GB during that prefill.

Prefill per layer on the tiled default (synthetic, release, matrix-product
TFLOPS): MoE 16.8 ms per 512 tokens (10.7 TFLOPS), 104.5 ms per 4096 (13.8),
204.9 ms per 8192 (14.0); Mamba with the fused SSD windows 9.5 / 69.2 /
140.6 ms (11.8 / 13.0 / 12.8); attention 3.4 / 26.5 / 66.6 ms. Projected over
the 40 MoE, 40 Mamba and 8 attention layers: about 572 tok/s at 4096-token
chunks (full checkpoint before the fused SSD: 542 tok/s at one 8192 chunk).
Routed experts run at MLX's NVFP4 matrix rate for expert shapes. The fused SSD
(Super graph prefill only, `setModelOptFusedSSD`) builds each 256-token window's
masked decay matrix in one kernel that reproduces `ssmAttn`'s segment sums
(MLX's strided cumulative sum: the same simd prefix sums per 32-row block),
precise `exp`, C.B product and triangular mask, so outputs and states equal
`ssmAttn` bit for bit. An earlier `cs[i] - cs[j]` form differed in float32
rounding; after the bf16 output rounding that reached 1.5% of a row maximum
on a CI runner GPU (0.5% on M3 Ultra).

Routed fc1 experts in tiled prefill use a grouped NVFP4 GEMM kernel
(`ModelOptNVFP4GroupedGEMM`): per-expert row tiles from the sorted rows,
in-kernel E2M1/E4M3 dequantization, float32 simdgroup products and a fused
scale and relu2 epilogue. At 2048-token chunks it takes 17.9 ms against
20.2 ms for MLX's sorted gather product plus the separate epilogue; at fc2's
K = 2688 MLX stays faster and is kept. Both run near the ~14 TFLOPS that even
dense bfloat16 GEMM reaches at these expert shapes on this GPU, so the MoE
layer (54 ms per 2048 tokens) is within about 4% of that rate.

On NAX GPUs (M5 and later) the grouped GEMM is off by default and fc1 also
uses MLX's sorted NAX gather product: Super prefill on an M5 Ultra runs
2,100 instead of 1,556 tokens/s. The summation order differs, so outputs are
not bit-identical, but teacher-forced accuracy against the true next token
is unchanged over 46,070 positions (top-1 78.16% vs 78.12%, perplexity
2.7238 vs 2.7294). Any summation-order change, TF32 on or off included,
agrees with the other path at about 96% of positions on this model; where the
two expert paths disagree, the NAX path is right 388 times and the grouped
GEMM 371.

On NAX GPUs both routed expert products then run on a dedicated NVFP4 NAX
kernel (`ModelOptNVFP4NAXExperts`, adapted from the GLM prefill kernel):
weights decoded eight codes per word into double-buffered threadgroup tiles,
one tile-scan launch instead of an expert scan in every threadgroup, empty
16-row fragments skipped, and the expert scale (and relu2 for fc1) fused into
the epilogue. It issues the same tensor operations in the same order as MLX's
sorted NAX gather, so its outputs are bit-identical (max error 0 on random
NVFP4 experts at Super's shapes; the full-model teacher-forced score matches
at all 46,070 positions). At 45,056 routed rows over 512 experts on an M5
Ultra, fc1 takes 2.98 ms instead of 4.7 and fc2 2.96 ms instead of 3.75, and
prefill of a 12K-token prompt runs at 2,255 instead of 2,094 tokens/s. The
fastest plans per input width are pinned in `planOverrides`.

These kernels, the dense projections and the chunked SSD below compile at
first use with the OS's Metal compiler. They are validated on macOS 27
(`ModelOptNVFP4NAXExperts.runtimeCompilerSupported`); macOS 26's compiler
rejects some tensor-op kernels ("unsupported deferred-static-alloca-size"),
which MLX reports as a fatal error, so earlier systems keep MLX's prebuilt NAX
kernels (the 2,094 tokens/s path) and all decode changes.

The MoE layer's expert block then runs as one fused route
(`ModelOptNVFP4NAXExperts.fusedRoute`, on with the NAX kernels): one route
launch replaces the argsorts, gathers and scans, fc1 reads the latent rows
through the sort order instead of a sorted copy, fc1's epilogue does fc2's
NVFP4 activation rounding, and the combine reads through the inverse order
instead of an unsort copy, about 850 MB less traffic per MoE layer. The block
takes 6.47 instead of 8.85 ms, bit for bit. The dense FP8 and NVFP4 prefill
projections (`ModelOptNAXDenseLinear`) decode their weights inside the NAX
GEMM instead of writing a BF16 copy per call (FP8) or float32 sums and a
separate scale pass (NVFP4), again bit-identical to MLX's products. Together
they take 12K-token prefill to 2,471 tokens/s on the M5 Ultra, with greedy
outputs and the teacher-forced score unchanged.

The default chunk cap is 2048 tokens. `prefillActivationBoundBytes(tokens:)`,
1 GiB + 3.75 MiB per token (8.5 GiB at 2048), bounds the whole chunk's transient
memory beyond weights and caches, and the owner must admit the bound for the
request that is actually prefilling. The full checkpoint measured an MLX peak
7.50-7.67 GB above the held state at every 2048-token chunk, independent of the
context (to 260K tokens) and of contiguous or paged KV. Each stage evaluated
alone peaks at most 1.0 GB (the Mamba SSD) and a chunk evaluated stage by stage
at 1.9 GB: the rest is MLX encoding ahead of the GPU, which keeps every queued
operation's inputs and outputs until its command buffer completes; evaluating
the residual stream every one or two layers leaves it unchanged. With profiling
(below) `NemotronSuperProfileRecorder(recordsMemory: true)` records each stage's
transient and retained bytes.

Short op chains of the graph path run as fused kernels, each bit-identical to
the chain it replaces (`setSuperFusedKernels(false)` restores the chains for
measurement; the eager reference never uses them):
- `NemotronSuperMoEKernels.select`: the router's sigmoid, bias, stable top-k
  (MLX's merge-sort order, NaN last, ties by index), gathered scores, sum,
  normalization and scaling in one kernel per row; `combine`: the weighted
  expert sum in MLX's small strided-reduction order.
- `NemotronSuperMambaStep`: the one-token conv step (state concat, depthwise
  conv, bias, `silu`, retained state), `computeDt` inside the SSM step kernel,
  and the gated RMS norm.
- `NemotronSuperResidualNorm`: each block's residual add (for MoE layers,
  routed plus shared output) together with the next block's RMS norm.
- `ModelOptRowExact` applies the FP8 activation cast (with the projection's
  overflow policy) inside the exact one-row products.
- Router-produced expert IDs skip the device range check (a top-k permutation
  is always in range) and, below the tiled threshold, the row sort.

The row-exact trunk (MTP verify windows and `.rowExact` batched decode) and the
MTP head's draft step use the same kernels wherever one-token decode does: the
residual-norm block boundaries, the gated norm over every window row, the
expert combine, squared ReLU fused into the shared-expert (and the head's fc1)
products, the one-token SSM step per batched-decode row, and in the head the
fused selection with router-produced IDs. `NemotronSuperMTPFusion.enabled`
(default true; `DARKBLOOM_NEMOTRON_SUPER_MTP_FUSION=0` at launch) restores the
previous op chains for measurement.
The trunk is also submitted every `NemotronSuperMTPWindowSubmission.everyLayers`
blocks (default 12; `DARKBLOOM_NEMOTRON_SUPER_MTP_WINDOW_SUBMIT=<n>` at launch,
0 for a single submission) with `asyncEval`, so the GPU starts while later
layers are encoded; evaluation order only. Measured on the full checkpoint
(M5 Ultra): verify rounds 8% faster, identical tokens. The two switches are
independent; the previous behavior is both set to 0.

Plain decode steps (every row one token, no speculative round) run with larger
Metal command buffers (`NemotronSuperDecodeCommandBuffers`, through
`CBv2DecodeCommandBufferLimitsProviding` and MLX's thread-scoped
`GPU.withCommandBufferLimits`): MLX's defaults commit every 50 operations or
50 MB of distinct inputs, which the expert banks exceed on almost every MoE
operation, so the CPU could not encode ahead of the GPU. With 100 operations
per buffer and no input-size limit, single-request plain decode went from 34.2
to 43.0 tok/s on the full checkpoint (eight rows: 81.9 to 91.4), with the same
tokens and the same memory peaks; prefill chunks and speculative rounds keep
MLX's defaults.

Synthetic weights at the real dimensions (`ModelOptGraphBenchmarkTests`, release
harness, per layer, decode chained 8 layers per graph, median of 25):

| Layer | Decode, graph | Decode, eager | Prefill 512 graph | Prefill 512 eager |
|---|---|---|---|---|
| MoE | 0.43 ms | 5.7 ms | 44.4 ms rowwise, 18.4 ms tiled | 1,436 ms |
| Mamba | 0.24 ms (cached step) | 24.9 ms | 10.4 ms | 555 ms |
| Attention | 0.19 ms | 8.5 ms | 3.9 ms | 167 ms |
| FP8 output head | 1.0 ms | 124 ms | one row | |

Eager prefill runs its admitted 32-token chunks. Exact rowwise prefill is
ALU-bound: MLX's per-row NVFP4 kernel runs near the GPU's scalar peak, and two
exact multi-row kernels (weights decoded once per row tile, rows staged in
threadgroup memory) were bit-identical but not faster. Prefill at about
23 GFLOP per token cannot exceed roughly 900 tok/s on this GPU. Decode remains
about 2-3x above the weight-bandwidth bound; per MoE layer the routed experts
take 0.16 ms, the weighted sum and output latent projection 0.09 ms, the shared
expert 0.09 ms and the router 0.06 ms.

`NEMOTRON_SUPER_PREFILL_PROFILE=1` turns on a per-stage wall-clock profile of
Super layers (MoE router, latent projections, sort, expert fc1/fc2, unsort,
combine and shared expert; Mamba in_proj, conv, SSD, norm/gate and out_proj;
attention q/k/v, attend and o_proj), plus rows-per-expert routing statistics per
MoE layer. Call `NemotronSuperPrefillProfile.reset()` before a forward and print
`NemotronSuperPrefillProfile.report()` after evaluating it. Each stage evaluates
its inputs and outputs, so the profiled forward is slower than an unprofiled
one; use the shares. Results are unchanged. At 2048 tokens with synthetic
weights and near-uniform routing (88 rows per expert median) the routed experts
take 36% of a MoE+Mamba+attention layer triple, Mamba in_proj 18%, the shared
expert 12%, the SSD scan 8%.

`NemotronSuperMambaPrefill.enabled` (on by default on NAX GPUs; elsewhere only
with `allowsGenericLayout`) routes graph prefill chunks of 32 or more tokens
through a chunked path that is not bit-identical to the windows above:
- One kernel reads the in_proj output in place for the convolution (state
  concatenation, depthwise conv, bias, `silu`, retained state), with the
  one-token step kernel's expressions, so its output and state equal the op
  chain's.
- `NemotronSuperSSDChunked` runs the SSD on the convolution output and the
  projected `dt` in place (`computeDt` inside): one threadgroup per head walks
  the prompt in 32-token chunks (`chunkSize`, or 64) with the head's float32
  state as the register accumulators of its products, as the GLM 5.3 prefill
  recurrence keeps each core's state resident. Inside a chunk C.Bᵀ with the
  decay mask, the intra-chunk output, the carried-state output and the state
  update are 16x32x16 NAX tensor ops (`mpp::tensor_ops::matmul2d`, float32
  accumulation, float32 operands by default: `operandType`) on fragments in
  MLX's `BaseNAXFrag` layout. Decay factors and masks are float32 differences
  of the in-chunk cumulative sums.
- The gated norm runs as `NemotronSuperMambaStep.gatedNorm` over every row.
The fragment copies assume the NAX register layout, as MLX's NAX kernels do;
the `.generic` layout routes every operand through threadgroup memory by the
tensors' own coordinates, which is correct on any GPU with Metal Performance
Primitives (macOS 26) and is how `NemotronSuperMambaPrefillTests` checks the
kernel on an M3 Ultra. There, against `ssmAttn` the outputs stay within
0.73% of the largest output (one bf16 step) and the states within 1.1e-6 of
the largest state element; against a float64 sequential scan the chunked
output is closer than `ssmAttn`'s. At Super's shapes (one layer, 2048 tokens,
random weights) the SSD output differs from the current path by at most one
bf16 step (0.33% of its maximum), the SSM state by 1.7e-6 relative, the conv
state not at all; the layer output differs by 2.6% of its maximum (0.6% of
the mean magnitude on average) because out_proj quantizes its input to E4M3
and one-step input changes move whole FP8 steps. On the M3 Ultra (generic
layout, so slower than NAX) the SSD takes 3.5 instead of 10.6 ms and the
layer 30 instead of 38 ms. `compareWithReference(tokens:iterations:seed:layout:operandType:chunkSize:)`
returns these errors and median times for the current GPU.

On the M5 Ultra (native layout) one layer at 2,048 tokens takes 8.0 instead
of 11.3 ms (SSD 1.9 instead of 6.1 ms), and 12K-token prefill runs at 2,881
instead of 2,471 tokens/s. Teacher-forced over 46,070 positions it scores
top-1 78.09% and perplexity 2.7275, against 78.12% / 2.7294 for the grouped
GEMM build and 78.16% / 2.7238 for the bit-identical NAX expert build; the
perplexity difference is the same (+0.1-0.2%) at every position of a 4K
window, so it does not grow with context. A prefix checkpoint should be
restored by a build with the same setting that made it.

## Text speculation

`NemotronSuperMTPAssistant` is a `CBv2MTPRequestStatefulDrafter` bound to the
exact loaded language model. It drafts greedily from each request's own state;
`Settings.maximumSpeculativeRows` (up to eight, rectangular verification only)
lets one round verify that many requests, and because every Super verify product
is row-exact it opts into `commitsAcceptedPrefixPerRow`: each request commits
its own accepted prefix instead of the round's common minimum.
Depth one is the baseline; depths two and three need `allowsDeepDrafts`. It
only reaches `NemotronSuperMTPTextTarget`; image requests never draft (see
"Images on the speculating owner"). `Settings.verification` selects:

- `serial` (`serialTarget`): every column runs the ordinary `[1, 1]` target
  forward; the engine rolls back rejected Mamba transactions and attention KV.
- `rectangularExact`: one target forward over `[last accepted token, k drafts]`
  of each of up to eight requests through `NemotronSuperMTPCaptureTarget`.
  Every position's logits and hidden equal that request's one-token decode bit
  for bit: FP8 projections, the FP8 output head and the BF16 router and latent
  projections use row-exact products (`ModelOptCalibratedLinear.rowExactForward`,
  `nemotronRowExactLinear`) that read each weight once per dispatch of up to 8
  rows, with up to 16 rows per call; NVFP4 shared and routed experts keep one
  single-vector product per row (below MLX's sorted matrix route); convolution,
  SSM and norms keep their per-row arithmetic; and the bank serializes each
  request's attention one query at a time.
  Each Mamba layer stages its per-position conv/SSM states; the engine commits
  the state after the last accepted token and trims attention KV to the
  accepted length. Per-position capture was chosen over recomputing the
  accepted prefix: recomputation reads every target weight a second time,
  while capture writes one conv/SSM state per window position and Mamba
  layer, which the engine charges as recurrent generations. Any other projection, or a
  calibration that fails validation, runs one row at a time.

- Draft graphs use the target's device-resident ModelOpt wrappers:
  `ModelOptNVFP4SwitchLinear.graphForward` for routed experts and
  `ModelOptCalibratedLinear.graphForward` for shared experts and for the
  target's own FP8 output head (`lmHead`), which the assistant shares. No
  routing readback, `.item()` or tile `eval` occurs. A projection whose
  registered calibration is invalid refuses graph construction before any
  history write; that refusal, or invalid device routing, ends the request in
  `requestStateDidFinishEvaluation` before the accept walk.
- History pairs each committed token with the previous trusted target hidden.
  Prompt history is written in canonical `historyChunkRows` (512) chunks as
  soon as a whole chunk has been observed, so only a partial chunk is ever
  queued (in the head dtype); the first draft step writes that partial chunk.
  After it, accepted rows are written one at a time at the next observation or
  draft step. Chained draft rows are always rewound; accepted inputs replay
  with the verifier's hidden before the next draft.
- `NemotronSuperFP8KVHistory` allocates its whole capacity, the request's own
  prompt plus `max_tokens` plus `k + 1` rows, once at its first write and
  writes every append in place; it never grows by concatenation.
- The engine admits the assistant in three parts
  (`CBv2MTPSequenceScaledAllocationProviding`). Per request and fixed: the
  frontier carries and the queued partial chunk, twice. Per reserved token,
  priced for each request's own length through EngineV2's auxiliary ledger:
  the FP8 codes and decoded mirror of K and V, 1,536 bytes per token for the
  pinned configuration. A draft step's one-query attention runs MLX's fused
  vector SDPA (head dim 128, GQA factor 16), which never materializes scores:
  its two-pass FP32 partials are at most 32 heads x 1,024 blocks x 130, about
  17 MB per step, charged with the step; only a head that would fall back to
  unfused attention is charged scores per token. Once per owner
  (`ownerWorking`, reported as `ownerBytes` for the host to reserve), whatever
  the number of requests: every draft step the depth policy allows in one
  round including the FP8 output head, one history chunk, the explicit backend
  scratch allowance, for rectangular verification the window at the round's
  largest position count (the heaviest layer's ModelOpt workspace, the output
  head over every row and FP32 logit rows), and the whole-history copies MLX
  may hold at once. A history write copies its whole buffer when an earlier
  step's attention (held until its command buffer completes) or the engine's
  retained roots of a pipelined step still hold it: measured on a 65,536-row
  history at the pinned head geometry, a depth-3 round evaluated once held two
  extra mirror copies, a depth-1 round or one evaluated per step none.
  `CBv2MTPAllocationLimits.historyCopies` bounds them over every decode width
  N: each of the N speculating rows once per draft step plus once, every other
  request once, each at the configured sequence bound; it follows the depth
  table it is given. Captured verify windows keep each request's ordinary
  three recurrent generations; the window generations beyond them are sized
  once from the depth policy's largest round. None of these are measured
  peaks.
- The head's LOAD construction allowance is four copies of its source
  payload. Dense constructor and cast placeholders are lazy and every one is
  replaced by a loaded array before the loader's only evaluation, so they are
  not charged; `NemotronSuperMTPLoadBoundTests` checks that none is ever
  evaluated. For the pinned 1,769,654,296-byte head payload this is
  7,078,617,184 bytes (6.59 GiB), down from 28.51 GiB.

Synthetic fixtures check bitwise graph/eager head parity, history rewind, forced
rejection at every position, commit of every accepted prefix of a captured
window, cancellation and greedy token equality with the ordinary adapter in
both verification modes. `NemotronSuperMTPRowExactnessTests` checks the
batched products at every Super FP8/NVFP4 shape. Serial verification spends
one target forward per column, so it cannot beat plain decode. With synthetic
weights at the pinned dimensions (`NemotronSuperMTPWindowTimingTests`, a
two-Mamba, two-attention, two-MoE stack plus the 131,072-row head), a captured
window of 2, 3 and 4 positions took 1.14, 1.23 and 1.31 times one decode step,
against 2.0, 3.0 and 4.0 times serially, and matched the serial steps bit for
bit. These are not full-checkpoint numbers; useful speedup still needs
measured full-model acceptance.

## Images on the speculating owner

The MTP targets adopt `CBv2RecurrentMTPRequestScopedMediaFallback`, so the
speculating engine also serves causal image requests. An image request is
target-only for its whole lifetime: it never drafts or verifies and never
holds a carry or assistant state. Its embedding prefill, text chunks and
decode rows run `base`, the ordinary `NemotronSuperRecurrentAdapter`, exactly
as on the plain owner. In a decode step with both kinds of rows, the text rows
run the hidden-returning forward and the image rows the ordinary forward, and
the logits return to their sampler order. Both groups are row-exact, so no
row depends on the other group. Bidirectional spans stay refused at submit.
The MTP step checks an image request (`requiresTargetOnlyMedia`) before its
hidden-returning branches. Every prefill chunk and decode row of the request
therefore takes the ordinary forwards, and the hidden-returning MTP seams
never receive embeddings. Every request is still charged the assistant's
per-request allocation.

`NemotronSuperMTPMediaFallbackTests` checks with synthetic weights that an
image request gives the same greedy tokens as the plain row-exact owner. It
also checks that text requests give their solo tokens with and without an
image request beside them while they speculate, that the image request never
holds MTP state, and that cancelling the image request mid-decode leaves the
text output unchanged and refunds every byte. Both verification modes are
tested.

## Batched text decode

`NemotronSuperRecurrentAdapter(_:maximumDecodeRows:decodeProducts:)` accepts
one `[B, 1]` decode step of B independent requests, the rectangle the CBv2
engine already builds for decode rows. Each row owns its recurrent state and
its attention KV row. Prompt chunks, media prefill and explicit positions stay
single-request; the engine runs them as separate forwards in the same step.
The default initializer keeps the serial owner.

- `.rowExact` (default) runs `superRowExactBatchedDecode`: every row is
  bit-identical to that request's one-token decode. FP8 projections, the BF16
  router and latent projections and the FP8 output head read each weight once
  per dispatch of up to 8 rows (the same row-exact products as the verify
  window, in groups of up to 16 rows); NVFP4 shared and routed experts keep one
  single-vector product per row. Each row's conv/SSM state is its own buffer.
- `.shared` runs `NemotronHModel.cbv2Forward` over the B rows: one M = B
  product per projection. MLX then picks wider kernels whose float32
  summation order differs from M = 1, and committed recurrent rows are views
  of one batch buffer that stays resident until every row has moved on.

`NemotronSuperDecodeCharge` states the admitted bytes per request: every live
recurrent generation (3 x 162.3 MiB for the pinned configuration), attention
KV per token (8 KiB in BF16) and one decode row of workspace, plus for
`.shared` an alias reserve of `(rows - 1)` departed rows per generation. These
are arithmetic bounds, not measured peaks; they add to the activation floor.

Synthetic tests check bitwise equality of logits, recurrent state and KV with
joins, an early finish and a cancelled in-flight row, and EngineV2 concurrent
greedy serving against solo serving. `NemotronSuperBatchedDecodeBenchmark`
times release-build decode steps of synthetic stacks at the pinned dimensions
and projects an 88-layer step from the marginal layer costs. On an M3 Ultra
the row-exact projection was 32.4, 38.1, 52.4, 80.6 and 140.5 ms per step at
B = 1, 2, 4, 8 and 16 (30.8 to 113.9 tokens/s aggregate), bitwise equal to
per-request decode at B = 8. These are synthetic projections with uniform
random routing, not full-checkpoint measurements.

## Serving prompts beside decode rows

Tiled prefill arithmetic depends on the chunk geometry, so a request's prompt
must run in the same chunks whether or not it is served alone. Two opt-in
`CBv2SchedulerConfig` limits make co-scheduling keep that geometry:
`maxPrefillRowsPerStep` (one: a step runs at most one prompt chunk, so one
prefill activation allowance covers every step at any concurrency) and
`decodeStepsBetweenPrefillChunks` (decode-only steps between two chunks while
requests decode). The serving owner also sizes the step budget to hold one
whole chunk plus every decode or verify row, so no chunk is ever shortened.
`NemotronSuperCoScheduledServingTests` checks a burst of requests, plain
batched and speculative, against solo serving token for token.

## Prefix reuse

`NemotronSuperRecurrentAdapter(_:maximumDecodeRows:decodeProducts:exactPrefixCheckpoints:)`
opts the plain text owner into EngineV2's resident recurrent checkpoint bank
(`CBv2HybridPrefixCache`) on the contiguous backend, or on paged storage when
the host opts in (see "Prefix bank on paged KV"): a checkpoint is every
Mamba layer's conv/SSM state plus the attention KV prefix, and a request whose
prompt extends a published prompt restores the deepest one and prefills only
the rest. `supportsPrefixReuse` stays false, because attention-only reuse would
drop the Mamba state.

Tiled prefill output depends on the chunk partition, so the adapter also sets
`alignsRecurrentCheckpointsToPrefillChunks`: the engine captures only at
multiples of its fixed prefill chunk C reached by whole chunks
(`CBv2RecurrentCheckpointGeometry`), never at a 256-token block that ends a
shorter range. A cold prompt runs `[0, C), [C, 2C), ...` and a request restored
at `kC` runs `[kC, (k + 1)C), ...`, the same partition, so it computes exactly
what the cold request does. A shortened, longer or off-grid range disarms
capture for the rest of that prompt, and the engine builds no bank for such a
model under a solo prefill stripe, which would change the cold partition.
Reuse granularity is therefore C (2048 tokens by default): prompts shorter
than C restore nothing.

For the pinned configuration one checkpoint is 170,229,760 bytes (162.3 MiB)
of recurrent state plus 8 KiB of KV per token. The bank keeps at most two
checkpoints per request (the first and the newest) inside its byte cap and
evicts the least recently used unpinned entry.

An image is a run of causal embeddings at scalar positions, the adapter's
only out-of-band input, so the adapter also sets
`supportsCausalMediaCheckpointReuse`: an image request bound by the host to a
`CBv2HybridPrefixIdentity` (a digest of its evaluated embeddings and spans)
checkpoints like text, in a cache scope derived from that identity. A later
turn with the same images at the same positions restores and continues the
cold prefill bit for bit; another identity, an unbound image request and text
never see those checkpoints, and an unbound image request never restores or
publishes. On the speculating owner image requests stay target-only and their
checkpoints hold no assistant history.

The speculating owner opts in the same way
(`NemotronSuperMTPCaptureTarget(capturing:maximumRequests:exactPrefixCheckpoints:)`),
and `NemotronSuperMTPAssistant` checkpoints its prompt history beside the
target state (`CBv2MTPPrefixCheckpointDrafter`). Until its first draft step a
request's assistant state is the FP8 draft history of every whole canonical
`historyChunkRows` chunk observed so far plus the queued partial chunk: the
shifted (next token, previous target hidden) pairs after the last whole chunk
and the frontier hidden. A checkpoint copies the history's FP8 codes and that
queue at the boundary. Restore writes the same codes, and the mirror decoded
elementwise from them, into the request's own history when its bound is
configured, and the first draft step writes the partial chunk in the same
chunk as a cold request, so a restored request builds the same draft history
as a cold one whatever its prefill chunks were. A checkpoint never holds a
speculative round; restore makes a fresh request state. Each checkpoint adds
512 bytes of FP8 codes per token to the bank's charge for the pinned
configuration (two KV heads of 128, K and V), plus fewer than 512 queued rows
of 8 KiB target hidden (BF16, 4,096 wide).

Synthetic fixtures (`NemotronSuperPrefixReuseTests`,
`NemotronSuperPrefixBankLifecycleTests`, tiled BF16 products, BF16 KV, 128-token
chunks) check that restored requests give the same tokens and full-vocabulary
log-softmax as cold requests bit for bit, that the restored conv/SSM state and
KV equal the cold state at the same position, partial-prefix and repeat hits,
eviction under the byte cap, no publication and a full refund on cancellation,
and that unbound image requests never reuse. `NemotronSuperMTPPrefixReuseTests`
(depth-3 exact rectangular speculation, 8-token chunks) checks that a restored
speculative request emits the cold run's tokens with the same rounds, drafted
and accepted tokens, that its restored recurrent state, KV and assistant
queue equal the cold request's own, concurrent restored and cold speculation,
a full refund when a restored request is cancelled, and a target-only image
turn that restores exactly. `NemotronSuperMediaPrefixReuseTests` checks that a
bound image turn restores the cold tokens and full-vocabulary log-softmax bit
for bit, and that another identity, an unbound image and text restore nothing.

### Complete checkpoints

The same chunk-aligned checkpoints can be complete checkpoints
(`CBv2CompleteCheckpointCodec`) that a host stores durably, for example
encrypted on SSD, instead of keeping them in the resident bank. The engine
builds the codec for a recurrent paged model only when the model declares its
attention KV types (`CBv2CompleteCheckpointKVTypeProviding`) and those types
equal the paged pool's observed layer types, and when a persistent assistant
can encode its history (`CBv2MTPPrefixCheckpointCoding`). The adapter and the
MTP target declare `NemotronHModel.observedAttentionKVDTypes()`, resolved once
like the recurrent spec. The host chooses complete checkpoints or the bank:
with a complete store the engine keeps the store's checkpoints, not the bank's.

`NemotronSuperMTPAssistant` encodes a checkpoint at `n` target inputs as its K
and V history codes for the `(n - 1) / historyChunkRows` whole chunks, the
queued pairs and tokens of the partial chunk, and the frontier hidden. The
split depends only on `n`, so the layout does too. The generic recurrent
manifest admits floating assistant tensors only, so the codes are stored
widened to float16 integers in 0...255. The widening is exact and costs a
second byte per code: 1,024 bytes per token for the pinned configuration
(two KV heads of 128, K and V), beside 8 KiB of KV. Decode reads the imported
host buffers without GPU work. It refuses a value that is not an integer
byte, a queued token that is not the prompt's own and a layout of another
`n`, then narrows the codes back. Restore therefore writes the same codes, and
the mirror decoded from them, as the request the checkpoint was taken from.
`NemotronSuperMTPCompleteCheckpointTests` carries encoded checkpoints through
host bytes and checks that a restored request drafts exactly as the cold one.
The assistant declares `CBv2MTPExactHistoryPrefixCheckpointCoding`, so under a
checkpoint profile the codec writes
`native-paged-affine-checkpoint-full-recurrent-mtp-v1`: older attention rows
are coded, while this history, the recurrent state and the recent band stay exact
([runtime KV quantization](runtime-kv-quantization.md#native-execution-with-compressed-checkpoints),
`NemotronSuperLossyCompleteCheckpointTests`).
Image checkpoints have no durable layout and stay resident-only.

## Paged attention KV

The adapter declares `supportsPagedKV` and `requiresNativePagedKV`, as the
Lightning target does: the eight attention layers' BF16 KV may live in the
segmented native paged backend with the observed layer dtypes. The host picks
the backend; contiguous stays the default. Paged and contiguous attention
reduce differently (the paged decode kernel against SDPA), so across backends
results agree within a stated tolerance, not bit for bit. Within paged
storage the batched, speculative and rollback contracts stay exact.

Synthetic evidence (`NemotronSuperPagedAttentionTests`,
`NemotronSuperPagedParityTests`, `NemotronSuperPagedEngineTests`): one attention
layer at the pinned geometry (32 query heads, 2 KV heads of 128) to 16,464
tokens keeps paged outputs within 1/64 of the largest output (prefill chunks
bitwise, decode 2.6e-3 measured) with equal stored KV; the trunk stays within
0.1 of the largest logit with equal greedy choices (float32 3.7e-2 measured,
where a last-bit difference can flip a downstream NVFP4 activation code;
bfloat16 bitwise on this fixture). Paged runs repeat bit for bit, row-exact
batches with joins, an early finish, a cancelled row and sixteen rows equal
each request alone, a rolled-back speculative write restores the exact bytes,
EngineV2 speculation (serial and captured verification) streams the plain
paged greedy tokens, a captured `[1, W]` verification window over paged rows
equals W one-token paged decode steps bit for bit, and concurrent serving
equals solo serving.

Full checkpoint (provider `NemotronSuperPagedLiveTests`, one staging, a
16,000-token prompt in 2,048-token chunks, then 64 decode steps fed the
contiguous choices): paged prefill logits equal contiguous bit for bit at every
chunk end. Paged decode logits deviate by at most 0.30 of the largest logit,
with 2 of 64 greedy choices different, both at a top-two margin of 0.25 (one
BF16 step at that magnitude). That is the same order as the trunk's own
sensitivity to a valid change in attention reduction order: the same contiguous
storage prefilled in 1,024-token chunks deviates by 0.20 at chunk ends and 0.34
in decode, with 4 different choices (margins 0, 0, 0.25 and 0.25). The full
checkpoint was not instrumented per layer to attribute the amplification.
Served through the provider runner, greedy text from paged storage (plain, and
speculating at depth 3) equals contiguous text for counting and whole-context
recall prompts of 131 to 15,855 tokens.

### Prefix bank on paged KV

`CBv2HybridPrefixCacheConfig.permitsPagedStorage` (off by default) lets the
resident checkpoint bank of "Prefix reuse" serve paged storage that
`EngineV2.hostsPagedHybridBank` accepts: a segmented pool of unquantized
(`.pagedFP16`) pages with no step-owned attention, native binding or resident
page-sharing index. A donor's pages return to the pool after publication, so
the bank keeps its own copy of the donor's prompt KV, gathered with
`PagedSequenceKV.gatherRange` before the pages are released and charged to the
bank's byte cap. The bank decides first, from the row geometry
(`PagedSequenceKV.gatheredByteCount`): a donor whose checkpoints only refresh an
existing entry, or that the bank refuses, is never gathered. The gather is built
and scheduled inside an MLX error scope; a failed or mis-sized gather refuses
the publication and drops its write-fence edges. A restore is a `.direct`
adoption that writes the prefix into fresh pages beside the restored conv/SSM
state. `NemotronSuperPagedPrefixReuseTests` checks with synthetic weights that a
restored request equals a cold paged run bit for bit, also after an unrelated
request has reused the donor's pages, that paged storage keeps no bank without
the opt-in, that only a kept publication gathers, and that a failed gather
refuses the publication while serving continues.

## Long context

The language model's attention has no positional encoding
(`max_position_embeddings` 1,048,576) and keeps 8 KiB of BF16 KV per token in
its eight attention layers; the Mamba state, the decode-row workspace and the
graph prefill chunk do not depend on the context. Full checkpoint (M3 Ultra,
256 GB, provider release runner, greedy, thinking off): natural text from the
provider's documentation with three notes at 10, 50 and 90 % depth, each giving
a room colour and a code.

| Prompt tokens | Speculating owner | Codes, in order | TTFT | Decode |
|---:|---|---|---:|---:|
| 31,935 | depth 3, 8 requests | 3 of 3 | 63.4 s | 37.8 tok/s |
| 64,734 | depth 3, 8 requests | 3 of 3 | 137.0 s | 37.2 tok/s |
| 130,058 | depth 3, 2 requests | 3 of 3 | 325.7 s | 25.0 tok/s |
| 261,160 | depth 3, 1 request | 3 of 3 | 851.3 s | 16.5 tok/s |

Rerun on the production head with the decode fusions and the default 1 GiB
prefix bank (eight requests at the served 65,536-token context): 3 of 3 codes
at 31,924 and 64,704 tokens, TTFT 62.5 s and 130.7 s, decode 37.6 and
32.4 tok/s.

Asked as well for the first and last three document titles and a two-sentence
summary of the last document, it named the first three correctly at every
length and summarized the last document correctly at about 32K tokens in both
runs and at 64K in one of two; at 130,058 and 261,160 it summarized an earlier
document as the last and listed wrong or repeated titles. Retrieval holds to
261,160 tokens; whole-document recall degrades beyond 32K. No answer contained
`<unk>`.

In an earlier run under the saturating overflow policy, the provider's gated
`NemotronSuperLongContextLiveTests` prefilled 131,056 tokens in 2,048-token
chunks: all 64 chunk-end logits and 16 decode steps were finite (largest
magnitude 70.5), and each chunk's MLX transient stayed at 7.7 to 8.7 GB,
inside that run's 12 GiB prefill lane, whatever the context. This is historical
evidence, not the current lane reservation described above.

These runs predate the current assistant charge. They charged every request
30,216 bytes per token of the configured window, so an eight-request owner's
LOAD claim limited the served context to 65,536 tokens on 256 GB. The charge
now (`NemotronSuperMTPAssistant+Admission.swift`, detailed in "Text
speculation") has three parts:

- Fixed per request: the frontier carries and the queued partial history
  chunk (fewer than 512 rows of BF16 hidden with their tokens), held twice;
  about 8.5 MB for the pinned configuration.
- Per reserved token of each request's own length, through EngineV2's
  auxiliary ledger: the FP8 codes and BF16 mirror of K and V, 1,536 bytes.
  Per-token FP32 draft scores are added only when the draft attention would
  run unfused SDPA; the pinned head (dim 128, GQA factor 16) runs fused.
- Once per owner, whatever the number of requests: the round's work,
  reported as `CBv2ResolvedMTPAdmission.ownerBytes`, which the host must
  reserve before the first request and keep until retirement.

In those runs MLX active memory peaked at 83.3 GB at 31,935 tokens and
87.1 GB at 261,160, about 75 GB of it weights, while the process footprint
grew with MLX's buffer cache: 97.4, 108.1, 117.9 and 185.2 GB at the four
lengths.

## Qualification limits

The sections above record full-checkpoint evidence on an M3 Ultra: loading,
graph-path decode and prefill, paged attention, retrieval to 261,160 tokens and
concurrent speculative serving through the provider. This repository does not
establish tokenizer or chat-template parity (the host renders prompts),
production discovery, image-format or video-container coverage (the host
decodes media) or full-model video quality; the video golden checks the vision
features of one clip. Whole-document recall degrades beyond 32K tokens. Prefix
reuse on the full checkpoint is measured by the host (the provider's managed
slot), not by this repository. Codec agreement is not fused CUDA attention
parity. The components stay unregistered and opt-in; serving qualification
belongs to the host.
