# Native Muse Glimmer text decoder

`MuseGlimmerTextModel` (MLXLLM) is a fork-only text tower for Muse Glimmer 30B. It serves
NVIDIA's ModelOpt checkpoint `nvidia/Muse-Glimmer-30B-NVFP4` through ContinuousBatchingV2 on
paged KV. Upstream's `MuseGlimmer` in MLXVLM is unchanged and still loads the MLX 4-bit
checkpoint through the generic factory. The native tower is not registered in a factory: a host
stages it with `MuseGlimmerInstallation`.

Status: text only. Vision, the block drafter and row-exact batched decode are not part of it yet.

## Architecture

52 layers, hidden 6,656, 32 query heads over 2 KV heads of 128, SwiGLU 19,968, vocabulary
202,048. Layers repeat three sliding-window layers (2,048 tokens, RoPE theta 500,000) and one
full-attention layer without position encoding.

The arithmetic is upstream's, operation for operation, so BF16 weights give the same bits in
both towers:

- block norms scale by `1 + weight` in FP32; the norms before a sublayer use `rms_norm_eps`,
  the ones after it `post_norm_eps`;
- `embed_norm` and `qk_norm` have no parameters;
- queries are multiplied by `qk_scale_factor` in FP32 before RoPE, in addition to the usual
  `1 / sqrt(head_dim)` softmax scale;
- the attention output is multiplied by `sigmoid(gate_proj(x))`, where `x` is the normed layer
  input, before `o_proj`;
- the final norm is a plain RMS norm; logits are `lm_head`, times `output_multiplier`, then
  `20 * tanh(x / 20)`.

`MuseGlimmerTextModelTests` holds the two towers to equal bytes on a tiny random model.

## Checkpoint

`MuseGlimmerCheckpointPin` names the revision and the SHA-256 of the index and of each shard.
`MuseGlimmerInstallation.stage(directory:)` hashes the files, then reads each projection's format
from the verified shard headers:

| `weight` dtype | Sidecars | Layer | Count |
|---|---|---|---|
| BF16 | none | `Linear` | 24 |
| F8_E4M3 | `weight_scale`, `input_scale` | `ModelOptCalibratedLinear` (FP8 weights, FP8 input activations) | 180 |
| U8 | `weight_scale`, `weight_scale_2` | `ModelOptWeightOnlyNVFP4Linear` (NVFP4 weights, activations untouched) | 213 |

`hf_quant_config.json` is compared with that reading and never drives it. A missing, extra or
misshapen tensor throws a `MuseGlimmerInstallation.Failure` that names the key. The 809 vision
tensors are verified and listed in the receipt but not loaded; `Installed.visionSources` hands
them to a later vision loader.

Numerics decisions that enter a checkpoint's numerics fingerprint:

- FP8 layers quantize activations with saturating overflow
  (`ModelOptActivationOverflow.museGlimmer`), as deployment runtimes do. An activation beyond the
  calibrated range clamps instead of becoming NaN.
- Weight-only NVFP4 products run MLX's `nvfp4` quantized matmul on the producer's bytes; the
  per-tensor `weight_scale_2` multiplies the output in FP32. With BF16 activations the product
  is rounded to BF16 once by the kernel and once after the scale.

The installer reserves no memory. A host holds its own load admission around `stage`.

## ContinuousBatchingV2

- Layer kinds: `.slidingWindow(2048)` or `.full` per layer, no KV sharing, no sinks.
- Capabilities: `supportsPagedKV`, `requiresNativePagedKV` and `requiresPagedKV`. The last one
  is new and off for every other model; with it `EngineV2` refuses a backend that is not
  `PagedKVBackend`. Build the pool with `segmentSizeBytes` and
  `layerDTypes: model.kvStorageDTypes`.
- Prompt chunks run the whole trunk and write every layer's K/V; only the vocabulary projection
  of unread rows is skipped. Packed prefill is off.
- Complete prefix checkpoints: the model declares historical attention checkpoints and its K/V
  dtypes. Restore replays no tokens.
- Plain decode steps use `MuseGlimmerDecodeCommandBuffers.limits`.

`CBv2SteppableLanguageModelAdapter` drives plain serving. `MuseGlimmerCBv2Adapter` is for a
request-stateful drafter: the round driver refuses such a drafter on the generic adapter, which
is always a recurrent MTP model.

## Target hooks for a block drafter

`mtpHiddenLayerIDs` (default 1, 13, 25, 37, 49) names the layers whose output residual the
drafter conditions on. `forwardWithMTPHidden` and the adapter's `forwardWithHidden` return the
plain logits and those residuals concatenated on the last axis; the prefill variants keep the
hidden of every prompt row and project logits for the last row only. `rawTokenEmbeddings` skips
`embed_norm`; `headProduct` is `lm_head` alone.

## Tests

Tiny synthetic models, no weights:

```bash
swift test --build-system native --skip-build \
  --filter 'MuseGlimmer|ModelOptWeightOnlyNVFP4'
```

Real checkpoint and oracle (skipped unless the variables are set):

```bash
MUSE_GLIMMER_NVFP4_DIR=<snapshot directory> \
MUSE_GLIMMER_ORACLE_DIR=<directory of oracle .npz fixtures> \
swift test --build-system native --skip-build --filter MuseGlimmerRealArtifactTests
```

The oracle is an independent float32 implementation. The FP8 layers round activations to an
8-bit grid, so a last-bit difference near a rounding boundary becomes a visible step: the
oracle's own single-pass and chunked runs differ by 2 to 11% in the hidden states of layers 13
to 49. The tests bound hidden states and logits at about twice the measured distance and
require the oracle's greedy tokens wherever its top-1 margin is not a near-tie.

`MUSE_GLIMMER_BENCH=1` adds a one-row decode and prefill timing on the paged backend.

## Not done

- Vision: upstream's vision classes are private, so loading into them needs a small addition to
  the upstream file.
- The drafter, and complete checkpoints while a persistent-history drafter is active.
- Row-exact batched decode for more than one row, and rectangular verify windows.
- A measured prefill activation bound, load residency accounting and factory registration.
