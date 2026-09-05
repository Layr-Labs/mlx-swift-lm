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
        guard enabled else { return weights }
        return fuseSwitchGLUGateUpWeights(
            weights: weights,
            perLayerQuantization: checkpointPerLayerQuantization,
            moduleName: "experts",
            setFused: { path, fused in
                let parentPath = String(path.dropLast(".experts".count))
                guard let owner = self.namedModules().first(where: { $0.0 == parentPath })?.1
                    as? MLPBlock else { return }
                guard owner.experts.hasFusedGateUp != fused else { return }
                // Update the direct @ModuleInfo owner. Reconstructing a
                // one-element root path through layers[N] creates a sparse
                // module array that Module.update cannot structurally apply.
                owner.update(modules: ModuleChildren.unflattened([
                    ("experts", owner.experts.withGateUpLayout(fused: fused))
                ]))
            })
    }
}
