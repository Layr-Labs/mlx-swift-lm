### Add a collective-aware Qwen expert down projection

The affine Qwen expert projection can retain float32 channel-shard partials
through an injected collective before rounding and router weighting. Installation
retains packed weights and rejects unsupported quantization or replicated biases.
