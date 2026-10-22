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

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Math: Glm5NextTextMLP, Glm5NextTextExperts, Glm5NextTextTopkRouter and Glm5NextTextMoE.
// https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py
// Checkpoint: zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda.
// These reusable modules do not register or qualify a complete model.

/// Gate has NO lower clamp. Up is clamped independently, before multiplication.
func glm5NextClampedGLUOperations(_ gate: MLXArray, _ up: MLXArray, limit: Float) -> MLXArray {
    let g = clip(gate, max: limit)
    // MLXNN.silu caches its first traced device by C++ default stream, which
    // does not follow Swift's task-local device. A previous CPU model must
    // not move this GPU activation to CPU or change its BF16 rounding.
    // Keep the sigmoid and multiplies on the caller's current stream with
    // the same dtype boundaries as the fused MoE kernels.
    return (g * sigmoid(g)) * clip(up, min: -limit, max: limit)
}

// MLX's compiled elementwise kernel keeps intermediates in FP32 on the GPU,
// so it is not bit-identical to these separate operations; keep them.
private func glm5NextClampedGLU(_ gate: MLXArray, _ up: MLXArray, limit: Float) -> MLXArray {
    glm5NextClampedGLUOperations(gate, up, limit: limit)
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
    /// Set at load for `Projections.fusedInputs`: gate/up are its row views.
    var fusedGateUp: GLM5NextFusedProjection?
    /// `RoutedExperts.fusedDecode` plans of a dense layer; not parameters.
    let decodePlans = GLM5NextMoEDecodeKernel.DensePlanCache()

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

    /// True when MLX runs each of this MLP's quantized products over `rows`
    /// rows as its plain `qmm`, whose K order the routed prefill launches
    /// reproduce bit for bit. Otherwise MLX runs `qmv` (under its batch limit,
    /// at most 33 rows) or `qmm_splitk`, which splits K while
    /// `ceil(N / 32) * ceil(rows / 32)` is under 257 threadgroups and rounds
    /// differently (`mlx/backend/metal/quantized.cpp`).
    func runsPlainQuantizedMatmul(rows: Int) -> Bool {
        let gateUp = fusedGateUp == nil ? intermediateSize : 2 * intermediateSize
        return [gateUp, hiddenSize].allSatisfy {
            Self.mlxRunsPlainQuantizedMatmul(rows: rows, outputs: $0)
        }
    }

    /// MLX's choice for a transposed, non-batched quantized product.
    static func mlxRunsPlainQuantizedMatmul(rows: Int, outputs: Int) -> Bool {
        rows > 33 && ((outputs + 31) / 32) * ((rows + 31) / 32) >= 257
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fusedGateUp {
            let parts = fusedGateUp(x)
            return downProj(glm5NextClampedGLU(parts[0], parts[1], limit: swigluLimit))
        }
        return downProj(glm5NextClampedGLU(gateProj(x), upProj(x), limit: swigluLimit))
    }

    /// A dense layer's feed-forward under its owner's load-time routed-expert
    /// choice. The `fusedDecode` members run one-token steps in two launches
    /// (`GLM5NextMoEDecodeKernel.denseDecode`, bitwise to the composed module:
    /// MLX's `qmv` lane map) when the kernel reads its projections, and refuse
    /// CPU and foreign streams as the MoE does; every other member, longer step
    /// (MLX's `qmv_wide` sums several rows in another lane order) and
    /// unreadable layout is `callAsFunction(_:)`.
    func callAsFunction(
        _ x: MLXArray, routedExperts: GLM5NextExecutionPolicy.RoutedExperts
    ) throws -> MLXArray {
        guard routedExperts == .fusedDecode || routedExperts == .fusedDecodeNAXPrefill else {
            return self(x)
        }
        guard glm5NextOnDefaultGPUStream() else {
            throw GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream
        }
        let tokens = x.size / hiddenSize
        guard tokens == 1, let plan = decodePlans.plan(self, dtype: x.dtype) else { return self(x) }
        return try GLM5NextMoEDecodeKernel.denseDecode(
            x.reshaped([tokens, hiddenSize]), plan: plan
        ).output.reshaped(x.shape)
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
        let projected = project(input, indices, sortedIndices: false).squeezed(axis: -2)

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

    /// MLX's expert-sorted quantized gather, on its NAX route (BF16/FP16 on
    /// Apple M5, K tiled by 64), writes wrong rows when K is not a multiple of
    /// 64; observed for K = 32 and K = 96. The sorted route therefore runs only
    /// when both projection input widths are multiples of 64, which every
    /// group-64 checkpoint guarantees.
    var admitsSortedGather: Bool { hiddenSize % 64 == 0 && intermediateSize % 64 == 0 }

    /// Native combine path (default GPU stream only). Unsorted, each assignment
    /// runs the reference's gathered projection and clamped GLU independently
    /// of slot order, so only the combine changes launch shape. `sortedGather`
    /// instead sorts assignments by expert and gathers one physical row per
    /// assignment, which selects MLX's tiled sorted matmul and changes
    /// accumulation order; without `admitsSortedGather` the unsorted path runs.
    func callAsFunction(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, sortedGather: Bool,
        addend: MLXArray? = nil
    ) throws -> MLXArray {
        precondition(x.ndim == 2 && x.dim(1) == hiddenSize && x.dtype.isFloatingPoint)
        precondition(indices.ndim == 2 && indices.dim(0) == x.dim(0))
        precondition(indices.dim(1) > 0 && indices.dim(1) <= numExperts)
        precondition(indices.shape == weights.shape && weights.dtype == .float32)
        let assignments = indices.size
        guard sortedGather, admitsSortedGather else {
            let projected = project(
                x.expandedDimensions(axes: [-2, -3]), indices, sortedIndices: false)
            return try GLM5NextMoEKernel.ascendingCombine(
                projected: projected.reshaped(assignments, hiddenSize), indices: indices,
                weights: weights, addend: addend)
        }
        let flattened = indices.flattened()
        let order = argSort(flattened)
        let sorted = flattened[order]
        // One row per assignment keeps the sorted hint index-aligned.
        let rows = x[order.floorDivide(indices.dim(1))].expandedDimensions(axis: -2)
        let projected = project(rows, sorted, sortedIndices: true)
        return try GLM5NextMoEKernel.ascendingCombine(
            projected: projected.reshaped(assignments, hiddenSize), rowMap: argSort(order),
            indices: indices, weights: weights, addend: addend)
    }

    private func project(_ input: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let gate = gateProj(input, indices, sortedIndices: sortedIndices)
        let up = upProj(input, indices, sortedIndices: sortedIndices)
        return downProj(
            glm5NextClampedGLU(gate, up, limit: swigluLimit), indices,
            sortedIndices: sortedIndices)
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
    /// Set once at target construction by `GLM5NextExecutionPolicy.routing`;
    /// short steps then use `GLM5NextRouterKernel`.
    var usesNativeKernel = false
    /// Set with `usesNativeKernel` by `Routing.nativeKernelPrefill`: longer
    /// steps use `GLM5NextRouterKernel.routePrefill` where it applies.
    var usesPrefillKernel = false

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

    /// The load-time routing choice. Native routing refuses CPU and foreign
    /// streams; steps longer than the kernel's bound keep the reference graph
    /// unless `usesPrefillKernel` and `GLM5NextRouterKernel.supportsPrefill`.
    func route(_ x: MLXArray) throws
        -> (logits: MLXArray, weights: MLXArray, indices: MLXArray)
    {
        let rows = x.size / max(1, hiddenSize)
        guard usesNativeKernel else { return self(x) }
        if GLM5NextRouterKernel.supports(tokens: rows) {
            return try GLM5NextRouterKernel.route(
                x.reshaped([rows, hiddenSize]), weight: weight, bias: correctionBias, topK: topK,
                normTopkProb: normTopkProb, routedScalingFactor: routedScalingFactor)
        }
        if usesPrefillKernel, nGroup == topkGroup,
            GLM5NextRouterKernel.supportsPrefill(tokens: rows, x: x, weight: weight, topK: topK)
        {
            return try GLM5NextRouterKernel.routePrefill(
                x.reshaped([rows, hiddenSize]), weight: weight, bias: correctionBias, topK: topK,
                normTopkProb: normTopkProb, routedScalingFactor: routedScalingFactor)
        }
        return self(x)
    }
}

final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLM5NextRouter
    @ModuleInfo(key: "experts") var experts: GLM5NextExperts
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP
    /// `RoutedExperts.fusedDecode` plans; not parameters.
    let decodePlans = GLM5NextMoEDecodeKernel.PlanCache()

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

    /// The owner's load-time routed-expert choice; `.reference` is exactly
    /// `callAsFunction(_:)`. Native paths refuse CPU and foreign streams before
    /// constructing any graph; there is no fallback.
    func callAsFunction(
        _ x: MLXArray, routedExperts: GLM5NextExecutionPolicy.RoutedExperts
    ) throws -> MLXArray {
        guard routedExperts != .reference || gate.usesNativeKernel else { return self(x) }
        guard glm5NextOnDefaultGPUStream() else {
            throw GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream
        }
        let fused = routedExperts == .fusedDecode || routedExperts == .fusedDecodeNAXPrefill
        if fused, GLM5NextMoEDecodeKernel.supports(tokens: x.size / experts.hiddenSize),
            let plan = decodePlans.plan(self, dtype: x.dtype)
        {
            return try GLM5NextMoEDecodeKernel.decode(
                x.reshaped([-1, experts.hiddenSize]), plan: plan
            ).output.reshaped(x.shape)
        }
        let routing = try gate.route(x)
        if routedExperts == .reference {
            let routed = experts(
                x.reshaped([-1, experts.hiddenSize]), indices: routing.indices,
                weights: routing.weights)
            return routed.reshaped(x.shape) + sharedExperts(x)
        }
        // Assignment count alone cannot tell prefill from a large decode batch.
        let sequenceLength = x.ndim >= 3 ? x.dim(-2) : 1
        let sortedGather =
            (routedExperts == .nativeCombineSortedPrefill || fused)
            && sequenceLength > 1
            && routing.indices.size >= GLM5NextExecutionPolicy.sortedPrefillMinimumAssignments
        let flattened = x.reshaped([-1, experts.hiddenSize])
        if sortedGather, routedExperts == .fusedDecodeNAXPrefill,
            let plan = decodePlans.plan(self, dtype: x.dtype),
            GLM5NextMoEPrefillKernel.admits(plan)
        {
            // The shared experts run inside the routed launches when their
            // banks allow and that keeps MLX's bits (enough rows that MLX
            // would not split K); otherwise their output is the combine's addend.
            let addend =
                GLM5NextMoEPrefillKernel.admitsSharedExperts(plan)
                    && sharedExperts.runsPlainQuantizedMatmul(rows: flattened.dim(0))
                ? nil : sharedExperts(x).reshaped(flattened.shape)
            return try GLM5NextMoEPrefillKernel.routedExperts(
                flattened, indices: routing.indices, weights: routing.weights,
                addend: addend, plan: plan, limit: experts.swigluLimit,
                sharedLimit: sharedExperts.swigluLimit
            ).reshaped(x.shape)
        }
        // The combine launch adds the shared experts with the reference's
        // single rounding of `routed + shared`.
        return try experts(
            flattened, indices: routing.indices, weights: routing.weights,
            sortedGather: sortedGather,
            addend: sharedExperts(x).reshaped(flattened.shape)
        ).reshaped(x.shape)
    }
}
