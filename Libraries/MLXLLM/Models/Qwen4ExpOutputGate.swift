// Copyright © 2026 Eigen Labs Inc.

import Foundation

extension Qwen4ExpTextConfiguration {
    /// Validate checkpoint semantics before loading weights, and use the same
    /// mapping when constructing a recurrent layer from a programmatic config.
    func recurrentOutputGate(codingPath: [any CodingKey] = []) throws -> QwenGatedNormActivation {
        switch outputGateType {
        case "sigmoid": return .sigmoid
        case "silu": return .silu
        default:
            throw DecodingError.dataCorrupted(.init(
                codingPath: codingPath + [CodingKeys.outputGateType],
                debugDescription:
                    "Unsupported Qwen4 output_gate_type '\(outputGateType)'; expected sigmoid or silu."))
        }
    }
}
