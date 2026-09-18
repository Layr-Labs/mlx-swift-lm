# Ternary Bonsai 2 27B

The `prism_hadamard_qwen35` factories load the published
[Prism 2-bit MLX artifact](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/tree/3f926b415992eaa2ae9dd7b573706494d6bbf787)
using the existing dense Qwen3.8-27B / `qwen3_5` backbone. This is not the native
Qwen4 Flash-Next model.

## Artifact contract

- Schema 1, affine 2-bit/group 128, FP16 packed scales and activations.
- Signed block-Hadamard metadata in `hadamard.json`; FP32 transform arithmetic
  followed by restoration of the input dtype. Generic BF16 weight conversion
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

The artifact has no MTP assistant. Do not attach a similarly named checkpoint's
head or advertise MTP. Native maximum context is 262144; physical memory and
request admission remain separate from that architectural limit.

## Qualification

Draft implementation: focused configuration/shape and transform regressions are
included. Full-model/API/image execution and performance evidence are being
qualified in the dependent provider PR. Build success alone is not a production,
full-context, concurrency, cache-lifecycle, hosted OpenRouter or model-quality pass.

The wire parser uses Qwen structured tool frames. Tokenizer/template bytes and
the published quantization stay unchanged. The pack runtime's older text-only
note and stale README entry in its file manifest are superseded for identity by
the pinned config, actual tensor inventory and Hugging Face Git/LFS hashes.

## Composition and attribution

Requires Layr-Labs/mlx-swift PR #27, pinned at
`c2de5d17e72ca2d389c2b6c12e310f6f8f5d9af7`. The required Hadamard/2-bit primitives
already exist in the pinned MLX core; no whole-fork substitution is required.
The signed-transform layers are adapted from Prism's MIT-licensed Swift work
at `6d3a84de28225d1f5bc0a56f5c781596997242f9`; pack semantics are checked against
the runtime distributed with the immutable model revision above.
