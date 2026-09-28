# MiMo V2.6 implementation provenance

The native target is `mimo_v2`, distinct from the older MiMo models and from
the separately packaged DFlash assistant. A shared parser or library primitive
does not select the target architecture.

Reference pins:

- Official config and artifact: XiaomiMiMo/MiMo-V2.6-Flash-RL,
  revision `5711b268169967567844e1e560e8a3966da959b1`.
- SGLang implementation and processors:
  `67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b`.
- Consumed Qwen2 audio-patch block behavior: Transformers 5.12.1,
  `ddb849abe009d1089e6c691bfc897f27211c663c`.

The Swift MiMo files contain new implementations and adaptations of native
model, next-N, vision, audio-code and decoded-pixel operations from those
references. SGLang and the relevant Transformers reference code are licensed
under Apache License 2.0; retain the accompanying
[license text](LICENSE-APACHE-2.0) and their attribution when redistributing
adapted portions. The checkpoint's declared weight license does not replace
the code licenses. Other SDK code retains its existing notices.

Adaptation boundaries:

- Target: hybrid full/sliding attention with unequal Q/K and value widths,
  partial RoPE, trained attention sinks, native value scaling and sigmoid MoE
  routing. No substitution with a Qwen target or the older MiMo Flash model.
- Next-N: trained three-head assistant borrowing the target's embeddings and
  readout. The selected driver reuses target post-final-norm features at every
  depth; predictor hidden states are not chained. DFlash is a distinct model.
- Vision: native patch/merge ordering, learned sinks and merger mathematics.
  The selected SGLang path uses global per-frame attention when sinks are absent.
- Audio-code patch encoder: ordered speech tables, per-item repeat-last padding,
  independent four-frame full-attention blocks and trained projections. This
  component alone is not the waveform/RVQ tokenizer.
- Pixels: decoded planar RGB in0–255 units, native bilinear resize without
  antialiasing, ImageNet normalization and spatiotemporal patch order. Image
  codecs, alpha/orientation policy and video sampling are separate boundaries.

Reference implementation discrepancies must remain explicit. CPU scalar or
component comparisons are not whole-checkpoint, serving, losslessness,
performance or production qualification. Capability registration and release
evidence belong to the accepted integration, not this provenance note.
