// Copyright 2026 the HuggingFace Team. All rights reserved.
// Modified by Eigen Labs in 2026: native Swift/MLX mixed-precision expert modules.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MLX
import MLXLMCommon
import MLXNN

// Math: Glm5NextTextMLP, Glm5NextTextExperts, Glm5NextTextTopkRouter and Glm5NextTextMoE.
// https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py
// Checkpoint: zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda.
// These reusable modules do not register or qualify a complete model.

private func glm5NextClampedGLU(_ gate: MLXArray, _ up: MLXArray, limit: Float) -> MLXArray {
    // Gate has NO lower clamp. Up is clamped independently, before multiplication.
    silu(clip(gate, max: limit)) * clip(up, min: -limit, max: limit)
}

enum GLM5NextComponentLoadingError: Error, Equatable {
    case invalidConfiguration
    case invalidProjection(String)
    case invalidRouterWeights
}

/// The same MLP serves a dense layer and the wider, always-active shared experts.
final class GLM5NextMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int
    let swigluLimit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float = 10) {
        precondition(hiddenSize > 0 && intermediateSize > 0)
        precondition(swigluLimit.isFinite && swigluLimit > 0)
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.swigluLimit = swigluLimit
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    /// Uses loaded projections directly, including QuantizedLinear subclasses.
    init(
        hiddenSize: Int, intermediateSize: Int,
        gateProj: Linear, upProj: Linear, downProj: Linear, swigluLimit: Float = 10
    ) throws {
        guard hiddenSize > 0, intermediateSize > 0, swigluLimit.isFinite, swigluLimit > 0 else {
            throw GLM5NextComponentLoadingError.invalidConfiguration
        }
        for (name, projection, input, output) in [
            ("gate_proj", gateProj, hiddenSize, intermediateSize),
            ("up_proj", upProj, hiddenSize, intermediateSize),
            ("down_proj", downProj, intermediateSize, hiddenSize),
        ] {
            guard projection.weight.ndim == 2, projection.bias == nil,
                (projection as? Quantized).map({ $0.bits > 0 }) ?? true,
                projection.shape == (output, input)
            else {
                throw GLM5NextComponentLoadingError.invalidProjection(name)
            }
        }
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.swigluLimit = swigluLimit
        _gateProj.wrappedValue = gateProj
        _upProj.wrappedValue = upProj
        _downProj.wrappedValue = downProj
        super.init()
    }

    convenience init(_ config: GLM5NextTextConfiguration, intermediateSize: Int? = nil) {
        self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: intermediateSize ?? config.intermediateSize,
            swigluLimit: Float(config.swigluLimit))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(glm5NextClampedGLU(gateProj(x), upProj(x), limit: swigluLimit))
    }
}

/// Three independently quantizable expert banks: `[E, I, D]`, `[E, I, D]`, `[E, D, I]`.
/// Split projections intentionally avoid SwitchGLU's unary-activation classification.
final class GLM5NextExperts: Module {
    let hiddenSize: Int
    let intermediateSize: Int
    let numExperts: Int
    let swigluLimit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(hiddenSize: Int, intermediateSize: Int, numExperts: Int, swigluLimit: Float = 10) {
        precondition(hiddenSize > 0 && intermediateSize > 0 && numExperts > 0)
        precondition(swigluLimit.isFinite && swigluLimit > 0)
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numExperts = numExperts
        self.swigluLimit = swigluLimit
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: hiddenSize, outputDims: intermediateSize, numExperts: numExperts, bias: false
        )
        _upProj.wrappedValue = SwitchLinear(
            inputDims: hiddenSize, outputDims: intermediateSize, numExperts: numExperts, bias: false
        )
        _downProj.wrappedValue = SwitchLinear(
            inputDims: intermediateSize, outputDims: hiddenSize, numExperts: numExperts, bias: false
        )
        super.init()
    }

    /// Uses loaded expert banks directly; no placeholder bank is constructed.
    init(
        hiddenSize: Int, intermediateSize: Int, numExperts: Int,
        gateProj: SwitchLinear, upProj: SwitchLinear, downProj: SwitchLinear,
        swigluLimit: Float = 10
    ) throws {
        guard hiddenSize > 0, intermediateSize > 0, numExperts > 0,
            swigluLimit.isFinite, swigluLimit > 0
        else {
            throw GLM5NextComponentLoadingError.invalidConfiguration
        }
        for (name, projection, input, output) in [
            ("gate_proj", gateProj, hiddenSize, intermediateSize),
            ("up_proj", upProj, hiddenSize, intermediateSize),
            ("down_proj", downProj, intermediateSize, hiddenSize),
        ] {
            guard projection.shape == (numExperts, output, input),
                !projection.parameters().flattened().contains(where: { $0.0 == "bias" })
            else {
                throw GLM5NextComponentLoadingError.invalidProjection(name)
            }
        }
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numExperts = numExperts
        self.swigluLimit = swigluLimit
        _gateProj.wrappedValue = gateProj
        _upProj.wrappedValue = upProj
        _downProj.wrappedValue = downProj
        super.init()
    }

    convenience init(_ config: GLM5NextTextConfiguration) {
        self.init(
            hiddenSize: config.hiddenSize, intermediateSize: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, swigluLimit: Float(config.swigluLimit))
    }

    /// `x` is `[tokens, D]`; indices and FP32 router weights are `[tokens, K]`.
    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        precondition(x.ndim == 2 && x.dim(1) == hiddenSize && x.dtype.isFloatingPoint)
        precondition(indices.ndim == 2 && indices.dim(0) == x.dim(0))
        precondition(indices.dim(1) > 0 && indices.dim(1) <= numExperts)
        precondition(indices.shape == weights.shape && weights.dtype == .float32)

        // The reference visits experts in ascending ID order. Sort the selected
        // slots, not all experts, retaining that per-token addition order.
        let order = argSort(indices, axis: -1)
        let indices = takeAlong(indices, order, axis: -1)
        let weights = takeAlong(weights, order, axis: -1)
        let input = x.expandedDimensions(axes: [-2, -3])
        // No sortedIndices hint: input broadcasts over selected experts rather
        // than carrying one physical row per assignment.
        let gate = gateProj(input, indices)
        let up = upProj(input, indices)
        let projected = downProj(glm5NextClampedGLU(gate, up, limit: swigluLimit), indices)
            .squeezed(axis: -2)

        // PyTorch multiplies by FP32 routing weights, then casts EACH contribution
        // before index_add_ into the input-dtype output. Do not cast only after sum.
        let contributions = (projected.asType(.float32) * weights.expandedDimensions(axis: -1))
            .asType(x.dtype)
        var output = MLXArray.zeros(x.shape, dtype: x.dtype)
        for slot in 0 ..< indices.dim(1) {
            output = output + contributions[0..., slot, 0...]
        }
        return output
    }
}

/// FP32 sigmoid/noaux_tc routing. Correction bias changes selection, never mixture weights.
final class GLM5NextRouter: Module {
    let hiddenSize: Int
    let numExperts: Int
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(
        hiddenSize: Int, numExperts: Int, topK: Int, nGroup: Int = 1, topkGroup: Int = 1,
        normTopkProb: Bool = true, routedScalingFactor: Float = 2.5
    ) {
        precondition(hiddenSize > 0 && nGroup > 0 && numExperts > 0)
        precondition(numExperts.isMultiple(of: nGroup) && numExperts / nGroup >= 2)
        precondition(topkGroup > 0 && topkGroup <= nGroup)
        precondition(topK > 0 && topK <= topkGroup * (numExperts / nGroup))
        precondition(routedScalingFactor.isFinite && routedScalingFactor > 0)
        self.hiddenSize = hiddenSize
        self.numExperts = numExperts
        self.topK = topK
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        _weight.wrappedValue = MLXArray.zeros([numExperts, hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([numExperts])
        super.init()
        freeze(recursive: false, keys: ["e_score_correction_bias"])
    }

    /// Retains loaded router tensors, preserving the FP32 routing computation.
    init(
        hiddenSize: Int, numExperts: Int, topK: Int,
        weight: MLXArray, correctionBias: MLXArray,
        nGroup: Int = 1, topkGroup: Int = 1,
        normTopkProb: Bool = true, routedScalingFactor: Float = 2.5
    ) throws {
        guard hiddenSize > 0, numExperts > 0, nGroup > 0,
            numExperts.isMultiple(of: nGroup), numExperts / nGroup >= 2,
            topkGroup > 0, topkGroup <= nGroup,
            topK > 0, topK <= topkGroup * (numExperts / nGroup),
            routedScalingFactor.isFinite, routedScalingFactor > 0
        else {
            throw GLM5NextComponentLoadingError.invalidConfiguration
        }
        let types: [DType] = [.float16, .bfloat16, .float32]
        guard weight.shape == [numExperts, hiddenSize], types.contains(weight.dtype),
            correctionBias.shape == [numExperts], types.contains(correctionBias.dtype)
        else {
            throw GLM5NextComponentLoadingError.invalidRouterWeights
        }
        self.hiddenSize = hiddenSize
        self.numExperts = numExperts
        self.topK = topK
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        _weight.wrappedValue = weight
        _correctionBias.wrappedValue = correctionBias
        super.init()
        freeze(recursive: false, keys: ["e_score_correction_bias"])
    }

    convenience init(_ config: GLM5NextTextConfiguration) {
        self.init(
            hiddenSize: config.hiddenSize, numExperts: config.nRoutedExperts,
            topK: config.numExpertsPerTok, nGroup: config.nGroup, topkGroup: config.topkGroup,
            normTopkProb: config.normTopkProb,
            routedScalingFactor: Float(config.routedScalingFactor))
    }

    func callAsFunction(_ x: MLXArray)
        -> (logits: MLXArray, weights: MLXArray, indices: MLXArray)
    {
        precondition(x.ndim >= 2 && x.dim(-1) == hiddenSize && x.dtype.isFloatingPoint)
        let flattened = x.reshaped([-1, hiddenSize]).asType(.float32)
        let logits = matmul(flattened, weight.asType(.float32).T)
        let scores = sigmoid(logits)
        var selectionScores = scores + correctionBias.asType(.float32)
        if topkGroup < nGroup {
            let grouped = selectionScores.reshaped([-1, nGroup, numExperts / nGroup])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
            let groupIndices = argPartition(-groupScores, kth: topkGroup - 1, axis: -1)[
                .ellipsis, ..<topkGroup]
            let groupMask = putAlong(
                MLXArray.zeros(groupScores.shape), groupIndices, values: MLXArray(Float(1)),
                axis: -1)
            let expertMask = broadcast(
                groupMask.expandedDimensions(axis: -1),
                to: [flattened.dim(0), nGroup, numExperts / nGroup]
            )
            .reshaped([-1, numExperts])
            selectionScores = which(expertMask .!= 0, selectionScores, -Float.infinity)
        }
        // Like torch.topk(sorted=False), selected slot order and tied-score order
        // are not a stable contract. Experts restore ascending-ID accumulation.
        let indices = argPartition(-selectionScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (logits, weights * routedScalingFactor, indices)
    }
}

final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLM5NextRouter
    @ModuleInfo(key: "experts") var experts: GLM5NextExperts
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP

    init(gate: GLM5NextRouter, experts: GLM5NextExperts, sharedExperts: GLM5NextMLP) {
        precondition(gate.hiddenSize == experts.hiddenSize && gate.numExperts == experts.numExperts)
        precondition(sharedExperts.hiddenSize == experts.hiddenSize)
        _gate.wrappedValue = gate
        _experts.wrappedValue = experts
        _sharedExperts.wrappedValue = sharedExperts
        super.init()
    }

    convenience init(_ config: GLM5NextTextConfiguration) {
        self.init(
            gate: GLM5NextRouter(config), experts: GLM5NextExperts(config),
            sharedExperts: GLM5NextMLP(
                config, intermediateSize: config.moeIntermediateSize * config.nSharedExperts))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routing = gate(x)
        let routed = experts(
            x.reshaped([-1, experts.hiddenSize]), indices: routing.indices, weights: routing.weights
        )
        return routed.reshaped(x.shape) + sharedExperts(x)
    }
}
