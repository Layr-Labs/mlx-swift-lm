### Inkling 2 Mini: image input

- `Libraries/MLXVLM/Models/Inkling2/` adds the checkpoint's vision tower, its
  image processor (pixels bit-identical to the transformers processor) and a
  VLM wrapper served through the paged engine. Each image's tower rows replace
  its run of image tokens as causal rows at scalar positions. The tower's
  LayerNorm computes in float32 and rounds once, as the reference does, in one
  fused kernel. Video is not an input of this model and is refused; images
  stay out of durable prefix checkpoints. Tests compare each tower stage with
  the reference's own realizations (`Inkling2CheckpointTests`).
- The wrapper forwards the drafter's target paths, so text requests draft
  through it, and it opts into request-scoped media fallback: in a batch with
  an image row, the image row runs target-only while text rows keep drafting,
  and every row equals its drafter-off result. Tests: `Inkling2VisionTests`,
  `Inkling2VLMEngineTests`, `Inkling2MediaFallbackEngineTests`.
