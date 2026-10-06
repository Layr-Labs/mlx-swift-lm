// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion keeps Flash-Next residuals in bfloat16 (QSA KV, GDN InT / TB=32,
// affine/gather qmm). Swift `QuantizedLinear` / `gatherQuantizedMM` often
// emit float32, which widens the whole residual after the first HC inject
// miss (N=4) and forces GDN blocked-seq onto TB=16. Pin Qwen4 activations
// back to bf16. Kill: DARKBLOOM_QWEN4_BF16_HIDDEN=0. 27B never calls this.

import Foundation
import MLX
import MLXNN

public enum Qwen4ExpActivation: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_BF16_HIDDEN"
    public static let dtype: DType = .bfloat16

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    /// No-op unless the value is float32 and the pin is on.
    public static func keep(
        _ x: MLXArray,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray {
        guard isEnabled(environment: environment), x.dtype == .float32 else { return x }
        return x.asType(dtype)
    }

    /// Native qmm output is `compute` (bf16). Default ON keeps it there so
    /// the residual does not round-trip back to float32. Kill restores the
    /// previous `asType(x.dtype)` contract.
    public static func nativeOutput(
        _ y: MLXArray, matching x: MLXArray, compute: DType,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray {
        if isEnabled(environment: environment) {
            return y.dtype == compute ? y : y.asType(compute)
        }
        return x.dtype == y.dtype ? y : y.asType(x.dtype)
    }

    /// Conv/PLE recurrent rows follow the embedding compute dtype, not the
    /// packed uint32 quantized weight.
    public static func hiddenDType(from embed: Embedding) -> DType {
        if let quantized = embed as? QuantizedEmbedding {
            switch quantized.scales.dtype {
            case .float16: return .float16
            case .bfloat16: return .bfloat16
            default: break
            }
        }
        switch embed.weight.dtype {
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        default: return dtype
        }
    }
}
