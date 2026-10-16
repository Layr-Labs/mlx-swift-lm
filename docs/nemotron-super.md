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
layouts. ModelOpt producer scales are part of the computation: generic affine
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
- `NemotronSuperRecurrentAdapter` is a serial causal adapter.
  `NemotronSuperMTPTextTarget` is a separate opt-in text-only target-hidden
  wrapper that refuses media; its subclass `NemotronSuperMTPCaptureTarget`
  adds the exact captured verify window.

Paths above are relative to `Libraries/`. Tests under
`Tests/MLXLMTests/Integration/Nemotron`, `Kernel/Quantization`, `Kernel/Vision`
and `Unit/Models` use tiny synthetic fixtures, not the full checkpoint.

## Performance work still required

High useful throughput and low TTFT are acceptance goals. The approximately
12B active parameters per token, packed weights, hybrid recurrent architecture
and bundled MTP offer potential savings; they do not prove Metal performance.
There is no full-model throughput measurement in this change.

## Text speculation

`NemotronSuperMTPAssistant` is a `CBv2MTPRequestStatefulDrafter` bound to the
exact loaded language model. It drafts greedily for one request at a time.
Depth one is the baseline; depths two and three need `allowsDeepDrafts`. It
only reaches `NemotronSuperMTPTextTarget`, which refuses embeddings, so the
engine rejects media requests at submit. `Settings.verification` selects:

- `serial` (`serialTarget`): every column runs the ordinary `[1, 1]` target
  forward; the engine rolls back rejected Mamba transactions and attention KV.
- `rectangularExact`: one target forward over `[last accepted token, k drafts]`
  through `NemotronSuperMTPCaptureTarget`. Every position's logits and hidden
  equal the one-token decode bit for bit: projections use only row-exact
  batched products (fastpath `.rowwise` NVFP4, gathered BF16 dense, and
  `superRowExactFP8`, which shares the exact FP8 vector product's weight reads
  across up to eight rows), the router, convolution, SSM and norms keep their
  per-row arithmetic, and the bank serializes attention one query at a time.
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

## Qualification limits

No full-checkpoint load, real tokenizer/template parity, long-context quality,
production discovery, paged/prefix reuse, concurrent serving or full-image-format
qualification is established here. Codec agreement is not fused CUDA attention
parity. Keep these components opt-in until actual local-provider endpoint tests
and exact-head cold/warm text/image benchmarks pass.
