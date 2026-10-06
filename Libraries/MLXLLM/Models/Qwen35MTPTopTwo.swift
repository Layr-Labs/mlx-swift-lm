// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Qwen policy entry point preserves its existing shape and lazy reduction.
func qwen35MTPTopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    cbv2TopTwoRows(logits)
}
