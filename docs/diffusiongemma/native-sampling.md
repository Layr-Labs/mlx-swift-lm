# Native DiffusionGemma sampling

`DiffusionGemmaDenoisingState` owns one request's canvas, conditioning, completion
flags and history. `DiffusionGemmaGenerationSession` publishes only finalized
blocks. This is diffusion sampling, not autoregressive MTP.

## Entry points and controls

`step(rawLogits:sample:noise:)` retains the arbitrary-callback contract.
`stepNative(rawLogits:samplingTemperature:nextKey:)` selects the known native
sampling operations without tracing caller callbacks. Invalid distributions
are rejected before key advancement. Greedy sampling consumes no categorical
key; native renoising still consumes its original key. State changes only after
validation succeeds.

`DARKBLOOM_DIFFUSION_COMPILED_SAMPLER=1` enables an experimental compiled path.
Unset or any other value retains the original path. Current eligibility is a
single 256-position Int32 canvas, a 262144-token vocabulary, FP32 logits/conditioning,
stability 1, entropy bound 0.1 and confidence threshold 0.005 on the ordinary GPU
stream. Other geometry, storage, controls and CPU/custom streams use the original
sampler. Set the variable before process startup.

The compiled path uses native integer random bits and the same uniform/Gumbel
construction. Exact RNG/clamp constants are explicit inputs: the pinned core's
short decimal literal representation is not guaranteed to preserve every Float.
Request keys/tensors remain inputs, not shared captured state. Sixteen immutable
compiled variants bound graph ownership; MLX retains compilation locking.

## Qualification boundaries

`DiffusionGemmaNativeSamplerTests` checks key order, fallback policies and invalid
distribution rejection. Its opt-in numerical edge test additionally requires
`DARKBLOOM_DIFFUSION_SAMPLER_NUMERICAL_EDGE_LIVE=1`; it uses synthetic logits and
no model checkpoint. Equal keys or common final samples alone are insufficient:
compare uniform/Gumbel values, native state and complete committed output.

Benchmark instrumentation may explicitly arm
`DiffusionGemmaCompiledSamplerDiagnostics`; ordinary serving leaves it disarmed.
First-use compilation, warmed timing, process memory and owner retirement are
separate measurements. A native component or full-generation pilot does not
qualify provider APIs, multimodal requests, caches, concurrency or deployment.

Sources: `Libraries/MLXLMCommon/DiffusionGemmaSampler.swift`,
`Libraries/MLXLMCommon/DiffusionGemmaCompiledSampler.swift`, and
`Libraries/MLXVLM/DiffusionGemmaGenerationSession.swift`.
See [implementation references](implementation-references.md) for architecture,
artifact components, native cache boundaries and licensing.
