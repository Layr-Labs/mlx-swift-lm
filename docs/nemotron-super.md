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

## Performance work still required

High useful throughput and low TTFT are acceptance goals. The approximately
12B active parameters per token, packed weights, hybrid recurrent architecture
and bundled MTP offer potential savings; they do not prove Metal performance.
There is no full-model throughput measurement in this change.

The eager head still synchronizes routing and calibrated projections with the
host. Opt-in `lazyDecode` and `lazyFP8Forward` helpers avoid those synchronization
points but are not integrated into a drafter. Their callers must admit concurrent
array/graph allowances plus backend scratch and inspect calibration validity at
the owned completion fence. Shared NVFP4 projection work remains.

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
