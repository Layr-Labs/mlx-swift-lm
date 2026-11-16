### Match MLX's sigmoid in the fused Gated DeltaNet convolution

- `GatedDeltaConvolution` (Qwen 3.5's fused convolution step and SiLU) copied
  MLX's `Sigmoid` functor with `metal::exp`; MLX's functor uses
  `metal::precise::exp`. The fused step therefore differed from the unfused
  `conv1d` and `silu` on some values (in every dtype), and Qwen 3.8 27B's
  outputs with the fusion on differed from the unfused ops. The copy now uses
  `metal::precise::exp`, and a test checks the fused step against the unfused
  ops byte for byte.
