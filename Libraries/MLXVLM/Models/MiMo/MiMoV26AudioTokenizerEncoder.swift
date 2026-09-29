// Copyright © 2026 Eigen Labs.
// Original source copyright 2026 Xiaomi Corporation and HuggingFace Inc.
// SPDX-License-Identifier: Apache-2.0
// Adapted from Xiaomi/HuggingFace HF5711 modeling_mimo_v2.py (Apache-2.0).
import Foundation
import MLX
import MLXLLM
import MLXNN

private final class MiMoV26CodecAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads, headDim, lookBack: Int
    let local: Bool
    init(_ c: MiMoV26AudioInputConfiguration, layer: Int) {
        heads = c.heads
        headDim = c.headDim
        lookBack = c.localLookBack
        local = c.usesLocalAttention(layer: layer)
        _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }
    func callAsFunction(_ x: MLXArray, lengths: [Int], cosine: MLXArray, sine: MLXArray) -> MLXArray
    {
        let count = x.dim(0)
        let hidden = heads * headDim
        func rotate(_ value: MLXArray) -> MLXArray {
            let first = value[0..., 0..., 0 ..< (headDim / 2)]
            let second = value[0..., 0..., (headDim / 2)...]
            let half = concatenated([-second, first], axis: -1)
            // Keep source BF16 intermediate multiplication/addition boundaries.
            return value * cosine.expandedDimensions(axis: 1) + half
                * sine.expandedDimensions(axis: 1)
        }
        let query = rotate(q(x).reshaped(count, heads, headDim))
        let key = rotate(k(x).reshaped(count, heads, headDim))
        let value = v(x).reshaped(count, heads, headDim)
        var offset = 0
        var rows: [MLXArray] = []
        for length in lengths {
            let end = offset + length
            let positions = MLXArray(0 ..< length)
            let delta =
                positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
            var allowed = delta .>= Int32(0)
            if local { allowed = allowed .&& (delta .<= Int32(lookBack)) }
            let mask = MLX.where(allowed, Float(0), -Float.infinity).asType(.bfloat16)
            let attended = MLXFast.scaledDotProductAttention(
                queries: query[offset ..< end].transposed(1, 0, 2).expandedDimensions(axis: 0),
                keys: key[offset ..< end].transposed(1, 0, 2).expandedDimensions(axis: 0),
                values: value[offset ..< end].transposed(1, 0, 2).expandedDimensions(axis: 0),
                scale: 1 / sqrt(Float(headDim)), mask: .array(mask))
            rows.append(attended.squeezed(axis: 0).transposed(1, 0, 2).reshaped(length, hidden))
            offset = end
        }
        return output(rows.count == 1 ? rows[0] : concatenated(rows, axis: 0))
    }
}

private final class MiMoV26CodecBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoV26CodecAttention
    @ModuleInfo(key: "self_attn_layer_norm") var beforeAttention: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var beforeFFN: LayerNorm
    @ModuleInfo(key: "fc1") var up: Linear
    @ModuleInfo(key: "fc2") var down: Linear
    init(_ c: MiMoV26AudioInputConfiguration, layer: Int) {
        _attention.wrappedValue = MiMoV26CodecAttention(c, layer: layer)
        _beforeAttention.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-5)
        _beforeFFN.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-5)
        _up.wrappedValue = Linear(c.hiddenSize, c.ffnSize, bias: true)
        _down.wrappedValue = Linear(c.ffnSize, c.hiddenSize, bias: true)
    }
    func callAsFunction(_ x: MLXArray, lengths: [Int], cosine: MLXArray, sine: MLXArray) -> MLXArray
    {
        let h = x + attention(beforeAttention(x), lengths: lengths, cosine: cosine, sine: sine)
        return h + down(gelu(up(beforeFFN(h))))
    }
}

private final class MiMoV26CodecBody: Module {
    @ModuleInfo(key: "conv1") var first: Conv1d
    @ModuleInfo(key: "conv2") var second: Conv1d
    @ModuleInfo(key: "layers") var layers: [MiMoV26CodecBlock]
    @ModuleInfo(key: "layer_norm") var finalNorm: LayerNorm
    // The checkpoint's down_sample_layer.0.weight unflattens as an array.
    // Register the same one-convolution list, not a module with a numeric key.
    @ModuleInfo(key: "down_sample_layer") var pool: [Conv1d]
    @ModuleInfo(key: "down_sample_norm") var poolNorm: LayerNorm
    init(_ c: MiMoV26AudioInputConfiguration) {
        _first.wrappedValue = Conv1d(
            inputChannels: c.melBands, outputChannels: c.hiddenSize, kernelSize: 3, padding: 1,
            bias: true)
        _second.wrappedValue = Conv1d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize, kernelSize: 3, stride: 2,
            padding: 1, bias: true)
        _layers.wrappedValue = (0 ..< c.layers).map { MiMoV26CodecBlock(c, layer: $0) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-5)
        _pool.wrappedValue = [
            Conv1d(
                inputChannels: c.hiddenSize, outputChannels: c.hiddenSize, kernelSize: 2, stride: 2,
                bias: false)
        ]
        _poolNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-5)
    }
}

public struct MiMoV26AudioEncoderOutput {
    public let features: MLXArray
    public let inputPlanIdentity: Data
    public let generation: UUID
    fileprivate init(features: MLXArray, inputPlanIdentity: Data, generation: UUID) {
        self.features = features
        self.inputPlanIdentity = inputPlanIdentity
        self.generation = generation
    }
}

public final class MiMoV26AudioTokenizerEncoder: Module {
    public let configuration: MiMoV26AudioInputConfiguration
    @ModuleInfo(key: "encoder") private var body: MiMoV26CodecBody
    public private(set) var loadedGeneration: UUID?
    public init(configuration: MiMoV26AudioInputConfiguration) {
        self.configuration = configuration
        _body.wrappedValue = MiMoV26CodecBody(configuration)
    }
    /// Metadata validation and graph installation only; no default evaluation.
    public func loadNativeNetworkWeights(_ source: [String: MLXArray]) throws {
        loadedGeneration = nil
        let weights = try MiMoV26AudioTokenizerWeights.prepareNetwork(
            source, configuration: configuration)
        try update(
            parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }),
            verify: .all)
        loadedGeneration = UUID()
    }
    @discardableResult public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        loadedGeneration = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
    @discardableResult public override func update(
        modules: ModuleChildren, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        loadedGeneration = nil
        return try super.update(
            modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    public func encodeFeatures(
        mels: [MiMoV26PreparedMel], plan: MiMoV26AudioInputPlan,
        isCancelled: () -> Bool = { false }
    ) throws -> MiMoV26AudioEncoderOutput {
        let identity = try plan.preparationIdentityData()
        guard plan.pcmDescriptors != nil, mels.count == plan.melFrameCounts.count,
            mels.enumerated().allSatisfy({
                $0.element.clipIndex == $0.offset && $0.element.inputPlanIdentity == identity
            })
        else {
            throw MiMoV26AudioInputError.input("prepared mel origin/order differs from plan")
        }
        return try encode(mels.map(\.values), plan: plan, isCancelled: isCancelled, trace: nil)
    }
    public func encodeMelFeatures(
        mels: [MLXArray], plan: MiMoV26AudioInputPlan,
        isCancelled: () -> Bool = { false }
    ) throws -> MiMoV26AudioEncoderOutput {
        guard plan.pcmDescriptors == nil else {
            throw MiMoV26AudioInputError.input("precomputed mels cannot claim PCM preparation")
        }
        return try encode(mels, plan: plan, isCancelled: isCancelled, trace: nil)
    }

    static func rotary(configuration c: MiMoV26AudioInputConfiguration, positions: [Int]) -> (
        inverse: MLXArray, cosine: MLXArray, sine: MLXArray
    ) {
        let exponent =
            MLXArray(Array(stride(from: 0, to: c.headDim, by: 2))).asType(.float32)
            / Float(c.headDim)
        let inverse = (Float(1) / pow(c.ropeTheta, exponent)).asType(.bfloat16).asType(.float32)
        let half =
            MLXArray(positions).asType(.float32).expandedDimensions(axis: 1)
            * inverse.expandedDimensions(axis: 0)
        let phase = concatenated([half, half], axis: 1)
        return (inverse, cos(phase).asType(.bfloat16), sin(phase).asType(.bfloat16))
    }

    /// Trace is internal/testing only: retaining intermediates is not a free
    /// production diagnostic and must be separately budgeted by its caller.
    func encode(
        _ mels: [MLXArray], plan: MiMoV26AudioInputPlan, isCancelled: () -> Bool,
        trace: ((String, MLXArray) -> Void)?
    ) throws -> MiMoV26AudioEncoderOutput {
        guard let generation = loadedGeneration else {
            throw MiMoV26AudioInputError.weightsNotLoaded
        }
        guard plan.configuration == configuration, mels.count == plan.melFrameCounts.count,
            zip(mels, plan.melFrameCounts).allSatisfy({
                $0.0.shape == [$0.1, configuration.melBands] && $0.0.dtype == .float32
            })
        else {
            throw MiMoV26AudioInputError.input("mel geometry/dtype/configuration")
        }
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        let identity = try plan.preparationIdentityData()
        let c = configuration
        var encodedGroups: [MLXArray] = []
        for (groupIndex, group) in plan.groups.enumerated() {
            if isCancelled() { throw MiMoV26AudioInputError.cancelled }
            let segments = group.segmentIndices.map { plan.segments[$0] }
            let rows = segments.map { segment -> MLXArray in
                let x = mels[segment.clipIndex][
                    segment.melStart ..< (segment.melStart + segment.melFrames)]
                return padded(
                    x, widths: [IntOrPair((0, group.maximumMelFrames - segment.melFrames)), 0]
                ).expandedDimensions(axis: 0)
            }
            let input = (rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)).asType(.bfloat16)
            let conv1 = gelu(body.first(input))
            let conv2 = gelu(body.second(conv1))
            trace?("group.\(groupIndex).conv1", conv1)
            trace?("group.\(groupIndex).conv2", conv2)
            let lengths = segments.map(\.convFrames)
            var hidden = concatenated(
                segments.enumerated().map { conv2[$0.offset, 0 ..< $0.element.convFrames, 0...] },
                axis: 0)
            let rotary = Self.rotary(
                configuration: c, positions: lengths.flatMap { Array(0 ..< $0) })
            trace?("group.\(groupIndex).inverse", rotary.inverse)
            trace?("group.\(groupIndex).cosine", rotary.cosine)
            trace?("group.\(groupIndex).sine", rotary.sine)
            var saved: MLXArray?
            for (i, layer) in body.layers.enumerated() {
                if isCancelled() { throw MiMoV26AudioInputError.cancelled }
                hidden = layer(hidden, lengths: lengths, cosine: rotary.cosine, sine: rotary.sine)
                if i == c.skipLayerIndex { saved = hidden }
                trace?("group.\(groupIndex).layer.\(i)", hidden)
            }
            hidden = body.finalNorm(hidden + saved!)
            trace?("group.\(groupIndex).normalized", hidden)
            var offset = 0
            var paddedHidden: [MLXArray] = []
            for length in lengths {
                let valid = hidden[offset ..< (offset + length)]
                var row = valid
                if length < group.maximumConvFrames {
                    let tail = broadcast(
                        valid[(length - 1) ..< length],
                        to: [group.maximumConvFrames - length, c.hiddenSize])
                    row = concatenated([valid, tail], axis: 0)
                }
                if group.maximumConvFrames % 2 != 0 {
                    row = padded(row, widths: [IntOrPair((0, 1)), 0])
                }
                paddedHidden.append(row.expandedDimensions(axis: 0))
                offset += length
            }
            let poolInput =
                paddedHidden.count == 1 ? paddedHidden[0] : concatenated(paddedHidden, axis: 0)
            trace?("group.\(groupIndex).pool_input", poolInput)
            let pooled = gelu(body.pool[0](poolInput))
            let packed = concatenated(
                segments.enumerated().map { pooled[$0.offset, 0 ..< $0.element.codeFrames, 0...] },
                axis: 0)
            let output = body.poolNorm(packed)
            trace?("group.\(groupIndex).features", output)
            encodedGroups.append(output)
        }
        let features =
            encodedGroups.isEmpty
            ? MLXArray.zeros([0, c.hiddenSize], dtype: .bfloat16)
            : encodedGroups.count == 1 ? encodedGroups[0] : concatenated(encodedGroups, axis: 0)
        guard features.shape == [plan.totalCodeFrames, c.hiddenSize] else {
            throw MiMoV26AudioInputError.input("encoder frame accounting")
        }
        return .init(features: features, inputPlanIdentity: identity, generation: generation)
    }
}
