# Nemotron Super VL native components

This is incomplete native support for NVIDIA's original mixed NVFP4/FP8/BF16
Nemotron 3.5 Super VL checkpoint. It is not a registered, qualified serving
model. No model weights are included in this repository.

## Checkpoint identity

Source: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, revision
`7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c`. The complete checkpoint has 88 shards,
75,010,188,240 weight bytes and 169,505 indexed tensors. Its 4,120 MTP tensors
are already in shard 88; downloading the complete snapshot includes MTP.
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
  79,953,182,216-byte quote excludes backend allowance and is not measured peak
  memory. Failed completion retains ownership rather than refunding early.
- `NemotronRadio`, `NemotronSuperVisionProjector`, image composition and
  `NemotronSuperVisionPreparation` provide the causal image path.
- `NemotronSuperImagePreprocessing` applies NVIDIA's grid rule and reproduces
  CPU PyTorch 2.12 float32 antialiased-bicubic rounding; its unit tests compare
  outputs bitwise. File decoding belongs to the caller.
- `NemotronSuperImageComposition.activationBytes` and `pinnedActivationBytes`
  bound the MLX working memory of one image's tower and projector evaluation
  from shapes, including MLX's unfused attention fallback for unsupported head
  dimensions. Integration tests compare them with measured MLX peaks.
- `NemotronSuperRecurrentAdapter` is a causal adapter: prompts and media are
  serial; text decode may batch independent requests (see "Batched text
  decode" below).
  `NemotronSuperMTPTextTarget` is a separate opt-in target-hidden wrapper
  that speculates for text only and serves images target-only (see "Images
  on the speculating owner"); its subclass `NemotronSuperMTPCaptureTarget`
  adds the exact captured verify window.

Paths above are relative to `Libraries/`. Tests under
`Tests/MLXLMTests/Integration/Nemotron`, `Kernel/Quantization`, `Kernel/Vision`
and `Unit/Models` use tiny synthetic fixtures, not the full checkpoint.

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

- The eager rule produces NaN logits and token 0.
- The Super policy keeps the logits finite, and the plain and speculating
  owners emit the same tokens.
- Text is bit-identical under both policies.

The full-checkpoint evidence is in the provider pull request.

## Graph execution and measured performance

`ModelOptCalibratedLinear` and `ModelOptNVFP4SwitchLinear` run a device-resident
graph path by default (`execution: .graph`); `checkedForward` stays the eager
reference. No call reads device values: calibration scalars are validated on the
host at init and after every parameter update, routing stays on the device, and
invalid expert IDs give NaN rows plus a validity flag for the owner's fence
(`ModelOptGraphValidity`). Metal kernels reproduce the eager NVFP4 and FP8
activation quantization bit for bit; NVFP4 products are MLX's gathered
`.nvfp4` kernels on the producer bytes; FP8 weights are decoded inside a copy
of MLX's gemv (one row) or per eager tile (several rows). Decode, small
batches and verify windows are bit-identical to the eager reference. Prefill
defaults to `.tiled`: MLX matrix kernels from 32 dense rows and 4 routed rows
per expert, with routed activations and sums in the activation dtype. Producer
bits and scales are unchanged; the summation order and one rounding of the
unscaled sums differ (float32 sums alone: largest difference 5.7e-7 of the row
maximum; with bfloat16 sums each output stays within 2^-7 of its row maximum).
`.rowwise` keeps prefill bit-identical. `calibratedPrefillTokenLimit` admits
chunks up to 8192 tokens inside an explicit 12 GiB ModelOpt workspace
(`modelOptGraphPrefillWorkspaceBytes(tokens:)`).

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

The default chunk cap is 2048 tokens. `prefillActivationBoundBytes(tokens:)`,
4 GiB + 4 MiB per token, bounds the whole chunk's transient memory beyond
weights and caches; the full checkpoint measured about 3.7 MB per token, and
the owner must admit the bound for the request that is actually prefilling.

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
  for bit: projections, the router and latent projections use only row-exact
  products that read each weight once per group of up to 16 rows
  (`ModelOptCalibratedLinear.rowExactForward`, `nemotronRowExactLinear`),
  routed experts keep one single-vector product per row (below MLX's sorted
  matrix route), convolution, SSM and norms keep their per-row arithmetic, and
  the bank serializes each request's attention one query at a time.
  Each Mamba layer stages its per-position conv/SSM states; the engine commits
  the state after the last accepted token and trims attention KV to the
  accepted length. Per-position capture was chosen over recomputing the
  accepted prefix: recomputation reads every target weight a second time,
  while capture writes one conv/SSM state per window position and Mamba
  layer, which the engine charges as recurrent generations. Any other projection, or a
  calibration that fails validation, runs one row at a time.

- Draft graphs use the target's device-resident ModelOpt wrappers:
  `ModelOptNVFP4SwitchLinear.graphForward` for routed experts and
  `ModelOptCalibratedLinear.graphForward` for shared experts and the shared FP8
  output head, with the target's `fp8Product`. No routing readback, `.item()`
  or tile `eval` occurs. A projection whose registered calibration is invalid
  refuses graph construction before any history write; that refusal, or
  invalid device routing, ends the request in `requestStateDidFinishEvaluation`
  before the accept walk.
- History pairs each committed token with the previous trusted target hidden.
  Prompt history is primed in fixed chunks; accepted rows replay one at a time.
  Chained draft rows are always rewound; accepted inputs replay with the
  verifier's hidden before the next draft.
- `boundedRequestAllocation` declares the per-request charge the engine admits
  before allocation: the FP8 history and backlog at the sequence bound, every
  chained draft step including the FP8 output head, one history chunk and an
  explicit backend scratch allowance. For the pinned configuration with a
  16,384-token bound, the arithmetic charge is about 0.9 GiB at depth one
  before that scratch allowance; it is not a measured peak. Rectangular
  verification adds the window: the heaviest layer's ModelOpt workspace for
  `k + 1` rows, the output head over every row and FP32 logit rows.
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
  bit-identical to that request's one-token decode. Projections, the router
  and the output head read their weights once per group of up to 16 rows
  (the same row-exact products as the verify window); routed experts keep one
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

## Text prefix reuse

`NemotronSuperRecurrentAdapter(_:maximumDecodeRows:decodeProducts:exactPrefixCheckpoints:)`
opts the plain text owner into EngineV2's resident recurrent checkpoint bank
(`CBv2HybridPrefixCache`) on the contiguous backend: a checkpoint is every
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
model under a solo prefill stripe, which would change the cold partition. Reuse granularity is therefore C (2048 tokens by
default): prompts shorter than C restore nothing.

For the pinned configuration one checkpoint is 170,229,760 bytes (162.3 MiB)
of recurrent state plus 8 KiB of KV per token. The bank keeps at most two
checkpoints per request (the first and the newest) inside its byte cap and
evicts the least recently used unpinned entry.

Media requests carry no checkpoint identity: they never restore or publish,
including later turns that repeat an image. The MTP text target keeps the
adapter's default (no checkpoints): its assistant has no prompt-history
checkpoint.

Synthetic fixtures (`NemotronSuperPrefixReuseTests`,
`NemotronSuperPrefixBankLifecycleTests`, tiled BF16 products, BF16 KV, 128-token
chunks) check that restored requests give the same tokens and full-vocabulary
log-softmax as cold requests bit for bit, that the restored conv/SSM state and
KV equal the cold state at the same position, partial-prefix and repeat hits,
eviction under the byte cap, no publication and a full refund on cancellation,
and that image requests never reuse.

## Qualification limits

No full-checkpoint load, real tokenizer/template parity, long-context quality,
production discovery, paged KV or image prefix reuse, prefix reuse with MTP,
concurrent serving or full-image-format qualification is established here.
Prefix reuse speedup on the full checkpoint is unmeasured. Codec agreement is
not fused CUDA attention parity. Keep these components opt-in until actual
local-provider endpoint tests and exact-head cold/warm text/image benchmarks
pass.
