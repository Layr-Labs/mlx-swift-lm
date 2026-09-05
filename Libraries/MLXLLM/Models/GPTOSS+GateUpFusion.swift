import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension GPTOSSModel: QuantizationPathAliasing, QuantizationPolicyReceiving {
    public func quantizationPathAliases(for path: String) -> [String] {
        let suffix = ".experts.gate_up_proj"
        guard path.hasSuffix(suffix) else { return [] }
        let base = String(path.dropLast("gate_up_proj".count))
        return [base + "gate_proj", base + "up_proj"]
    }

    /// Row concatenation preserves packed MXFP4/affine bytes. It is legal
    /// only for identical per-half tensor sets and quantization policies;
    /// the shared helper leaves incompatible pairs split and untouched.
    func fuseGateUpWeights(_ weights: [String: MLXArray], enabled: Bool) -> [String: MLXArray] {
        let weights = splitSavedGateUpWeights(weights)
        guard enabled else { return weights }
        return fuseSwitchGLUGateUpWeights(
            weights: weights,
            perLayerQuantization: checkpointPerLayerQuantization,
            moduleName: "experts",
            setFused: { path, fused in
                self.setExpertGateUpLayout(at: path, fused: fused)
            })
    }
}
