// Copyright © 2026 Eigen Labs.
// Native MiMo V2.6 sigmoid routing; no Qwen family substitution.

import MLX
import MLXLMCommon
import MLXNN

final class MiMoV26DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MiMoV26Router: Module {
    let config: MiMoV26Configuration
    let operandDType: DType
    var useFusedDecodeRouter = MiMoV26DecodeRouter.enabledByEnvironment
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ config: MiMoV26Configuration) {
        self.config = config
        operandDType = config.moeRouterDType == "bfloat16" ? .bfloat16 : .float32
        _weight.wrappedValue = MLXArray.zeros([config.routedExpertCount, config.hiddenSize],
                                             dtype: operandDType)
        _correctionBias.wrappedValue = MLXArray.zeros([config.routedExpertCount], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray, rowLocal: Bool = false)
        -> (indices: MLXArray, weights: MLXArray) {
        if rowLocal && MiMoV26RectangularDense.supports(x) {
            // One shared declared-precision conversion per layer/forward.
            // Every dot product AND score-selection/normalization stays [1,1,*].
            // Never rely on common-subexpression elimination of repeated casts.
            let matrix = weight.asType(operandDType).asType(.float32).T
            let routed = (0..<x.dim(1)).map { position in
                let row = x[0..., position..<(position + 1), 0...].contiguous()
                let logits = MiMoV26DecodeRouter.logits(
                    row, weight: weight, operandDType: operandDType, enabled: useFusedDecodeRouter)
                    ?? matmul(row.asType(operandDType).asType(.float32), matrix)
                return select(logits)
            }
            return (concatenated(routed.map(\.indices), axis: 1),
                    concatenated(routed.map(\.weights), axis: 1))
        }
        // Quantize operands to the declared router precision, but do not round
        // the dot-product result to BF16 before sigmoid. The reference CUDA
        // router requests FP32 output for BF16 inputs. This explicit FP32 path
        // remains subject to independent numerical qualification on Metal.
        let logits = MiMoV26DecodeRouter.logits(
            x, weight: weight, operandDType: operandDType, enabled: useFusedDecodeRouter)
            ?? matmul(x.asType(operandDType).asType(.float32),
                      weight.asType(operandDType).asType(.float32).T)
        return select(logits)
    }

    private func select(_ logits: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let originalScores = sigmoid(logits)
        var selection = originalScores + correctionBias.asType(.float32)
        if config.topKGroups < config.expertGroupCount {
            var grouped = unflatten(selection, axis: -1,
                                    shape: [config.expertGroupCount, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let excluded = config.expertGroupCount - config.topKGroups
            let rankedGroups = argPartition(groupScores, kth: excluded - 1, axis: -2)
            let indices = rankedGroups[.ellipsis, ..<excluded, 0...]
            grouped = putAlong(grouped, stopGradient(indices),
                               values: MLXArray(-Float.infinity), axis: -2)
            selection = flattened(grouped, start: -2, end: -1)
        }
        let rankedExperts = argPartition(-selection, kth: config.expertsPerToken - 1, axis: -1)
        let indices = rankedExperts[.ellipsis, ..<config.expertsPerToken]
        var weights = takeAlong(originalScores, indices, axis: -1)
        if config.expertsPerToken > 1 && config.normalizeTopKProbability {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * Float(config.routedScalingFactor ?? 1)
        return (indices, weights)
    }
}

final class MiMoV26MoE: Module, UnaryLayer {
    let gate: MiMoV26Router
    private let useFP32WeightedReduction: Bool
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: MiMoV26Configuration,
         fp32WeightedReduction: Bool = MiMoV26FP32WeightedReduction.isEnabled()) {
        gate = MiMoV26Router(config)
        useFP32WeightedReduction = fp32WeightedReduction
        _switchMLP.wrappedValue = SwitchGLU(inputDims: config.hiddenSize,
                                          hiddenDims: config.moeIntermediateSize,
                                          numExperts: config.routedExpertCount,
                                          weightedReductionProfile: .mimoV26FP32,
                                          mimoV26NAXGather: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        forwardWithWeightedReductionRoute(x).output
    }

    /// Same real router/projections used by callAsFunction, with scalar route
    /// evidence for qualification. No scripted proposal or alternate weights.
    func forwardWithWeightedReductionRoute(_ x: MLXArray, rowLocalRouter: Bool = false)
        -> MiMoV26FP32WeightedReductionResult {
        let routed = gate(x, rowLocal: rowLocalRouter)
        if useFP32WeightedReduction {
            let reduced = switchMLP.callAndMiMoFP32WeightedReduce(
                x, routed.indices, weights: routed.weights, enabled: true)
            return .init(output: reduced.output.asType(x.dtype), route: reduced.route)
        }
        // Keep the default/OFF graph identical to the original MiMo expression.
        let expertOutput = switchMLP(x, routed.indices)
        return .init(output: (expertOutput * routed.weights[.ellipsis, .newAxis])
            .sum(axis: -2).asType(x.dtype), route: .disabled)
    }
}
