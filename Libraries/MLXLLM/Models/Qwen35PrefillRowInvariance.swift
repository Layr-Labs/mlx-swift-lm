// Copyright © 2026 Eigen Labs.

import MLXLMCommon

// Dense Qwen 3.5 text models (Qwen 3.5 9B, Qwen 3.8 27B) passed the chunk census and the
// reuse proof on the M3 Ultra and the M5 Max. MoE models and the Prism Hadamard pack
// were not measured, so they keep today's routes.

extension Qwen35Model: CBv2PrefillRowInvarianceProviding {
    public var prefillRowInvarianceFamily: CBv2PrefillRowInvariance.Family? {
        guard !(self is Qwen35MoEModel), !(self is PrismHadamardQwen35TextModel) else {
            return nil
        }
        return languageModel.prefillRowInvarianceFamily
    }
}

extension Qwen35TextModel: CBv2PrefillRowInvarianceProviding {
    public var prefillRowInvarianceFamily: CBv2PrefillRowInvariance.Family? {
        configuration.numExperts == 0 ? .qwen35Dense : nil
    }
}
