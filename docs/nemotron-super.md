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
  weights after inventory preflight. `NemotronSuperMTPModule` loads and executes
  the embedded head eagerly; it is not a speculative assistant.
- `MLXVLM/Models/NemotronSuperInstallation.swift` owns sequential native loading
  epochs, completion fences and retained resources. The header-derived
  79,953,182,216-byte quote excludes backend allowance and is not measured peak
  memory. Failed completion retains ownership rather than refunding early.
- `NemotronRadio`, `NemotronSuperVisionProjector`, image composition and
  `NemotronSuperVisionPreparation` provide the causal image path.
- `NemotronSuperRecurrentAdapter` is a serial causal adapter.
  `NemotronSuperMTPTextTarget` is a separate opt-in text-only target-hidden
  wrapper. It deliberately refuses media and captured-window speculation.

Paths above are relative to `Libraries/`. Tests under
`Tests/MLXLMTests/Integration/Nemotron`, `Kernel/Quantization`, `Kernel/Vision`
and `Unit/Models` use tiny synthetic fixtures, not the full checkpoint.

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
masked decay matrix in one kernel; outputs stay within 2^-7 of each row's
maximum of `ssmAttn` (measured 0.5% at Super shapes, states 2.6e-5).

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

A speculative assistant must bind the exact target, consume trusted shifted
hidden/token history, verify serial depth one, rewind/replay its KV state and
rollback target Mamba/KV state correctly. Image MTP remains unsupported. Enable
speculation only after correctness and measured useful-token gains against the
same plain target; record acceptance, draft/verify/rollback cost and memory.

## Qualification limits

No full-checkpoint load, real tokenizer/template parity, long-context quality,
production discovery, paged/prefix reuse, concurrent serving or full-image-format
qualification is established here. Codec agreement is not fused CUDA attention
parity. Keep these components opt-in until actual local-provider endpoint tests
and exact-head cold/warm text/image benchmarks pass.
