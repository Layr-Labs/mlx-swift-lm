// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Modified from Transformers f0705f991a71f05dad9569336b181119f5546f93,
// src/transformers/models/glm5_next/modeling_glm5_next.py: Swift/MLX translation,
// strict source-layout loading and bounded frame/query execution with checked evaluation.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum GLM5NextVisionError: Error, Equatable, Sendable {
    case invalidInput(String)
    case invalidWeights(String)
    case weightsNotLoaded
    case budgetExceeded(String)
    case unsupported(String)
}

/// Vision attention kernel. `.tiled` is the eager reference-ordered graph:
/// post-QK scaling, query tiles bounded by the score caps. `.fused` is MLX's
/// fused attention, which never materializes a score tensor (about 20x faster
/// for an 8000-token image) and stays within the official BF16 tower's error
/// against FP32. `.automatic` (the default) selects `.fused` on the GPU for
/// head dimensions it fuses (64, 80, 128) and `.tiled` otherwise.
public enum GLM5NextVisionAttentionKernel: Sendable, Equatable {
    case automatic, tiled, fused

    func usesFused(headDimension: Int) -> Bool {
        switch self {
        case .tiled: return false
        case .fused: return true
        case .automatic:
            return Device.defaultDevice().deviceType == .gpu
                && [64, 80, 128].contains(headDimension)
        }
    }
}

/// These are execution caps, not a claim of model or provider memory admission.
public struct GLM5NextVisionLimits: Sendable {
    public let maximumPatches: Int
    public let maximumFramePatches: Int
    public let maximumQueryTokens: Int
    public let maximumScoreElements: Int
    public let attention: GLM5NextVisionAttentionKernel

    public init(
        maximumPatches: Int = 960_000, maximumFramePatches: Int = 32_000,
        maximumQueryTokens: Int = 128, maximumScoreElements: Int = 32_000_000,
        attention: GLM5NextVisionAttentionKernel = .automatic
    ) {
        self.maximumPatches = maximumPatches
        self.maximumFramePatches = maximumFramePatches
        self.maximumQueryTokens = maximumQueryTokens
        self.maximumScoreElements = maximumScoreElements
        self.attention = attention
    }
}

final class GLM5NextVisionPatchEmbedding: Module {
    @ModuleInfo var proj: Conv3d
    let configuration: GLM5NextVisionConfiguration

    init(_ configuration: GLM5NextVisionConfiguration) {
        self.configuration = configuration
        let kernel = IntOrTriple([
            configuration.temporalPatchSize, configuration.patchSize, configuration.patchSize,
        ])
        _proj.wrappedValue = Conv3d(
            inputChannels: configuration.inChannels,
            outputChannels: configuration.hiddenSize, kernelSize: kernel, stride: kernel, bias: true
        )
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let c = configuration
        // Torch's patch-sized Conv3d has one output voxel per sample: its
        // columns and weight rows contract in C,T,H,W order.
        let dtype = proj.weight.dtype
        let weights = proj.weight
            .transposed(0, 4, 1, 2, 3)
            .reshaped(1, c.hiddenSize, -1)
            .asType(.float32)
        let vectors = patches.asType(dtype).asType(.float32).expandedDimensions(axis: -1)
        // Keep each patch a separate matrix-vector product. Weights on the
        // left prevent MLX from folding the patch batch into a large GEMM.
        var output = matmul(weights, vectors).squeezed(axis: -1)
        if let bias = proj.bias {
            // Apply channel bias after removing the singleton vector axis.
            // Batched column-vector addMM misindexes its broadcast bias on Metal.
            output = output + bias.asType(.float32)
        }
        return output.asType(dtype)
    }
}

/// Preserve the reference's normalization-before-weighting and dtype boundary.
final class GLM5NextVisionRMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        // Vision widths are validated multiples of four. Preserve the pinned
        // CPU reference's four-lane reduction and sqrt-then-reciprocal boundary.
        let lanes = value.square().reshaped(Array(x.shape.dropLast()) + [-1, 4]).sum(axis: -2)
        let total =
            ((lanes[.ellipsis, 0] + lanes[.ellipsis, 1]) + lanes[.ellipsis, 2])
            + lanes[.ellipsis, 3]
        let variance = total.expandedDimensions(axis: -1) / Float(x.dim(-1))
        let normalized = value * (1 / sqrt(variance + eps))
        return weight * normalized.asType(x.dtype)
    }
}

private final class GLM5NextVisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    let limit: Float

    init(input: Int, intermediate: Int, bias: Bool, limit: Float) {
        _gate.wrappedValue = Linear(input, intermediate, bias: bias)
        _up.wrappedValue = Linear(input, intermediate, bias: bias)
        _down.wrappedValue = Linear(intermediate, input, bias: bias)
        self.limit = limit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(clip(gate(x), max: limit)) * clip(up(x), min: -limit, max: limit))
    }
}

private final class GLM5NextVisionAttention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int
    let dimension: Int

    init(_ configuration: GLM5NextVisionConfiguration) {
        heads = configuration.numHeads
        dimension = configuration.hiddenSize / heads
        _qkv.wrappedValue = Linear(
            configuration.hiddenSize, 3 * configuration.hiddenSize,
            bias: configuration.attentionBias)
        _proj.wrappedValue = Linear(
            configuration.hiddenSize, configuration.hiddenSize,
            bias: configuration.attentionBias)
        _qNorm.wrappedValue = GLM5NextVisionRMSNorm(
            dimensions: dimension, eps: Float(configuration.rmsNormEps))
        _kNorm.wrappedValue = GLM5NextVisionRMSNorm(
            dimensions: dimension, eps: Float(configuration.rmsNormEps))
    }

    func forward(
        _ x: MLXArray, angles: MLXArray, limits: GLM5NextVisionLimits,
        evaluate: ([MLXArray]) throws -> Void
    ) throws -> MLXArray {
        let count = x.dim(0)
        let values = qkv(x).reshaped(count, 3, heads, dimension)
        let q = GLM5NextVisionTower.rotary(qNorm(values[0..., 0]), angles: angles)
            .transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = GLM5NextVisionTower.rotary(kNorm(values[0..., 1]), angles: angles)
            .transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = values[0..., 2].transposed(1, 0, 2).expandedDimensions(axis: 0)
        try evaluate([q, k, v])
        if limits.attention.usesFused(headDimension: dimension) {
            // One fused launch per block over this frame: no score tensor.
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1 / sqrt(Float(dimension)), mask: nil)
            try evaluate([attended])
            let result = proj(attended.transposed(0, 2, 1, 3).reshaped(count, heads * dimension))
            try evaluate([result])
            return result
        }
        // Explicit eager attention preserves the reference's post-QK scaling;
        // MLX SDPA's CPU fallback scales Q before matmul. Query tiling bounds
        // both score/probability tensors without a frame-square allocation.
        let width = min(limits.maximumQueryTokens, limits.maximumScoreElements / heads / count)
        guard width > 0 else { throw GLM5NextVisionError.budgetExceeded("attention query tile") }
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: count, by: width) {
            let end = min(count, start + width)
            let scores =
                (matmul(q[0..., 0..., start ..< end], k.swappedAxes(-1, -2))
                * (1 / sqrt(Float(dimension)))).asType(q.dtype)
            try evaluate([q, k, v] + outputs + [scores])
            let probabilities = softmax(scores.asType(.float32), axis: -1, precise: true).asType(
                q.dtype)
            try evaluate([q, k, v] + outputs + [probabilities])
            let output = matmul(probabilities, v)
            try evaluate([q, k, v] + outputs + [output])
            outputs.append(output)
        }
        let result = proj(
            concatenated(outputs, axis: 2).transposed(0, 2, 1, 3)
                .reshaped(count, heads * dimension))
        try evaluate([result])
        return result
    }
}

private final class GLM5NextVisionBlock: Module {
    let norm1: RMSNorm
    let norm2: RMSNorm
    let attn: GLM5NextVisionAttention
    let mlp: GLM5NextVisionMLP

    init(_ configuration: GLM5NextVisionConfiguration) {
        norm1 = GLM5NextVisionRMSNorm(
            dimensions: configuration.hiddenSize, eps: Float(configuration.rmsNormEps))
        norm2 = GLM5NextVisionRMSNorm(
            dimensions: configuration.hiddenSize, eps: Float(configuration.rmsNormEps))
        attn = GLM5NextVisionAttention(configuration)
        mlp = GLM5NextVisionMLP(
            input: configuration.hiddenSize,
            intermediate: configuration.intermediateSize, bias: configuration.attentionBias,
            limit: Float(configuration.swigluLimit))
    }

    func forward(
        _ x: MLXArray, angles: MLXArray, limits: GLM5NextVisionLimits,
        evaluate: ([MLXArray]) throws -> Void
    ) throws -> MLXArray {
        let residual =
            x + (try attn.forward(norm1(x), angles: angles, limits: limits, evaluate: evaluate))
        try evaluate([residual])
        let result = residual + mlp(norm2(residual))
        try evaluate([result])
        return result
    }
}

private final class GLM5NextVisionMerger: Module {
    let proj: Linear
    @ModuleInfo(key: "post_projection_norm") var norm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    let limit: Float

    init(_ c: GLM5NextVisionConfiguration) {
        proj = Linear(c.outHiddenSize, c.outHiddenSize, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: c.outHiddenSize)
        _gate.wrappedValue = Linear(c.outHiddenSize, c.projectionIntermediateSize, bias: false)
        _up.wrappedValue = Linear(c.outHiddenSize, c.projectionIntermediateSize, bias: false)
        _down.wrappedValue = Linear(c.projectionIntermediateSize, c.outHiddenSize, bias: false)
        limit = Float(c.swigluLimit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gelu(norm(proj(x)))
        return down(silu(clip(gate(h), max: limit)) * clip(up(h), min: -limit, max: limit))
    }
}

/// Official GLM5 Next vision-only component. No factory or serving registration.
public final class GLM5NextVisionTower: Module {
    public let configuration: GLM5NextVisionConfiguration
    @ModuleInfo(key: "patch_embed") private var patchEmbedding: GLM5NextVisionPatchEmbedding
    private let blocks: [GLM5NextVisionBlock]
    private let downsample: Conv2d
    @ModuleInfo(key: "post_layernorm") private var postNorm: RMSNorm
    private let merger: GLM5NextVisionMerger
    private var loadedDType: DType?

    public init(configuration c: GLM5NextVisionConfiguration) throws {
        guard c.inChannels == 3, c.temporalPatchSize == 2, c.spatialMergeSize == 2,
            c.attentionDropout == 0
        else {
            throw GLM5NextVisionError.unsupported(
                "requires RGB, temporal/spatial merge 2, inference dropout 0")
        }
        configuration = c
        _patchEmbedding.wrappedValue = GLM5NextVisionPatchEmbedding(c)
        blocks = (0 ..< c.depth).map { _ in GLM5NextVisionBlock(c) }
        downsample = Conv2d(
            inputChannels: c.hiddenSize, outputChannels: c.outHiddenSize,
            kernelSize: IntOrPair(c.spatialMergeSize), stride: IntOrPair(c.spatialMergeSize),
            bias: true)
        _postNorm.wrappedValue = GLM5NextVisionRMSNorm(
            dimensions: c.hiddenSize, eps: Float(c.rmsNormEps))
        merger = GLM5NextVisionMerger(c)
    }

    public enum WeightLayout: Sendable, Equatable { case pytorch, mlx }

    /// Names are relative to `model.visual.`. The two convolution layouts are explicit.
    public static func expectedTensorShapes(
        configuration c: GLM5NextVisionConfiguration,
        layout: WeightLayout = .pytorch
    ) -> [String: [Int]] {
        let h = c.hiddenSize
        let d = h / c.numHeads
        let o = c.outHiddenSize
        var result: [String: [Int]] = [
            "patch_embed.proj.weight": layout == .pytorch
                ? [h, c.inChannels, c.temporalPatchSize, c.patchSize, c.patchSize]
                : [h, c.temporalPatchSize, c.patchSize, c.patchSize, c.inChannels],
            "patch_embed.proj.bias": [h], "post_layernorm.weight": [h],
            "downsample.weight": layout == .pytorch
                ? [o, h, c.spatialMergeSize, c.spatialMergeSize]
                : [o, c.spatialMergeSize, c.spatialMergeSize, h],
            "downsample.bias": [o], "merger.proj.weight": [o, o],
            "merger.post_projection_norm.weight": [o], "merger.post_projection_norm.bias": [o],
            "merger.gate_proj.weight": [c.projectionIntermediateSize, o],
            "merger.up_proj.weight": [c.projectionIntermediateSize, o],
            "merger.down_proj.weight": [o, c.projectionIntermediateSize],
        ]
        for i in 0 ..< c.depth {
            let p = "blocks.\(i)."
            result[p + "norm1.weight"] = [h]
            result[p + "norm2.weight"] = [h]
            result[p + "attn.q_norm.weight"] = [d]
            result[p + "attn.k_norm.weight"] = [d]
            for (name, shape) in [
                ("attn.qkv", [3 * h, h]), ("attn.proj", [h, h]),
                ("mlp.gate_proj", [c.intermediateSize, h]),
                ("mlp.up_proj", [c.intermediateSize, h]),
                ("mlp.down_proj", [h, c.intermediateSize]),
            ] {
                result[p + name + ".weight"] = shape
                if c.attentionBias { result[p + name + ".bias"] = [shape[0]] }
            }
        }
        return result
    }

    /// Pass only the vision inventory; no silently discarded unexpected tensors.
    public func loadNativeWeights(
        _ weights: [String: MLXArray], prefix: String = "model.visual.",
        layout: WeightLayout = .pytorch, dtype: DType = .bfloat16
    ) throws {
        loadedDType = nil
        guard [.bfloat16, .float16, .float32].contains(dtype) else {
            throw GLM5NextVisionError.invalidWeights("unquantized vision dtype")
        }
        let shapes = Self.expectedTensorShapes(configuration: configuration, layout: layout)
        guard Set(weights.keys) == Set(shapes.keys.map { prefix + $0 }) else {
            throw GLM5NextVisionError.invalidWeights("missing or unexpected vision tensors")
        }
        var mapped: [(String, MLXArray)] = []
        for (name, shape) in shapes {
            guard let value = weights[prefix + name], value.shape == shape, value.dtype == dtype
            else {
                throw GLM5NextVisionError.invalidWeights(name)
            }
            let converted: MLXArray
            switch (layout, name) {
            case (.pytorch, "patch_embed.proj.weight"): converted = value.transposed(0, 2, 3, 4, 1)
            case (.pytorch, "downsample.weight"): converted = value.transposed(0, 2, 3, 1)
            default: converted = value
            }
            mapped.append((name, converted))
        }
        try update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        loadedDType = dtype
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        loadedDType = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    @discardableResult
    public override func update(
        modules: ModuleChildren, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        loadedDType = nil
        return try super.update(
            modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    /// FP32 axial RoPE, in the same merge-major order as flattened source patches.
    static func angles(height: Int, width: Int, dimension: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(height * width * dimension)
        for row in stride(from: 0, to: height, by: 2) {
            for column in stride(from: 0, to: width, by: 2) {
                for dy in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        var half: [Float] = []
                        for position in [row + dy, column + dx] {
                            for frequency in 0 ..< (dimension / 4) {
                                // The reference rounds inverse frequency to FP32,
                                // then multiplies positions; division reassociates it.
                                let inverse =
                                    1
                                    / pow(
                                        Float(10_000), Float(2 * frequency) / Float(dimension / 2))
                                half.append(Float(position) * inverse)
                            }
                        }
                        values.append(contentsOf: half + half)
                    }
                }
            }
        }
        return MLXArray(values).reshaped(height * width, 1, dimension)
    }

    static func rotary(_ x: MLXArray, angles: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return (x.asType(.float32) * cos(angles) + rotated.asType(.float32) * sin(angles)).asType(
            x.dtype)
    }

    /// The source Conv2d consumes each merge group in C,H,W order, unlike
    /// MLX's channel-last convolution. Its bias follows the completed dot.
    static func downsamplePatches(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray?, merge: Int
    ) -> MLXArray {
        let channels = x.dim(-1)
        var patches = x.reshaped(-1, merge, merge, channels).transposed(0, 3, 1, 2)
            .reshaped(-1, channels * merge * merge).asType(.float32)
        let count = patches.dim(0)
        // Preserve the source Conv2d GEMM accumulation path for small frames,
        // rather than a GEMV fallback. Only at most 15 zero rows are temporary.
        let padding = (16 - count % 16) % 16
        if padding > 0 {
            patches = concatenated([patches, MLXArray.zeros([padding, patches.dim(1)])], axis: 0)
        }
        let matrix = weight.transposed(0, 3, 1, 2).reshaped(weight.dim(0), -1).asType(.float32)
        var output = matmul(patches, matrix.T)[..<count]
        if let bias { output = output + bias.asType(.float32) }
        return output.asType(x.dtype)
    }

    /// `checkpoint` may enforce the caller's ownership/admission fence. Every eval
    /// also checks the local scoped MLX fault box before continuing to another stage.
    public func forward(
        patches: MLXArray, grids: [GLM5NextVisionGrid],
        limits: GLM5NextVisionLimits = .init(), checkpoint: ([MLXArray]) throws -> Void = { _ in }
    ) throws -> MLXArray {
        guard let dtype = loadedDType else { throw GLM5NextVisionError.weightsNotLoaded }
        guard limits.maximumPatches > 0, limits.maximumFramePatches > 0,
            limits.maximumQueryTokens > 0, limits.maximumScoreElements > 0, !grids.isEmpty
        else {
            throw GLM5NextVisionError.invalidInput("vision limits or empty grids")
        }
        var total = 0
        for grid in grids {
            let count = try grid.patchCount()
            guard grid.height.isMultiple(of: 2), grid.width.isMultiple(of: 2),
                grid.height * grid.width <= limits.maximumFramePatches,
                count <= limits.maximumPatches - total
            else {
                throw GLM5NextVisionError.budgetExceeded("frame or aggregate patches")
            }
            total += count
        }
        let c = configuration
        let patchWidth = c.inChannels * c.temporalPatchSize * c.patchSize * c.patchSize
        guard patches.ndim == 2, patches.shape == [total, patchWidth],
            [.float32, .float16, .bfloat16].contains(patches.dtype)
        else {
            throw GLM5NextVisionError.invalidInput("flattened C,T,H,W patch tensor")
        }
        return try MLX.withError { error in
            var outputs: [MLXArray] = []
            func evaluate(_ roots: [MLXArray]) throws {
                try error.check()
                try Task.checkCancellation()
                let retained = [patches] + outputs + roots
                try checkpoint(retained)
                try error.check()
                eval(retained)
                try error.check()
            }
            var offset = 0
            for grid in grids {
                let count = grid.height * grid.width
                let angles = Self.angles(
                    height: grid.height, width: grid.width,
                    dimension: c.hiddenSize / c.numHeads)
                for _ in 0 ..< grid.temporal {
                    var x = patchEmbedding(patches[offset ..< (offset + count)].asType(dtype))
                    try evaluate([angles, x])
                    for block in blocks {
                        x = try block.forward(x, angles: angles, limits: limits, evaluate: evaluate)
                    }
                    x = postNorm(x)
                    try evaluate([x])
                    x = Self.downsamplePatches(
                        x, weight: downsample.weight, bias: downsample.bias,
                        merge: c.spatialMergeSize)
                    try evaluate([x])
                    x = merger(x)
                    try evaluate([x])
                    outputs.append(x)
                    offset += count
                }
            }
            let result = concatenated(outputs, axis: 0)
            try evaluate([result])
            return result
        }
    }
}
