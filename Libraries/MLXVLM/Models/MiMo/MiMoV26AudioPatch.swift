// Copyright © 2026 Eigen Labs.
// Derived from SGLang67bb6a58 mimo_audio.py (Apache-2.0), consuming Qwen2
// from Transformers5.12.1/ddb849abe (Apache-2.0). Discrete codes only.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum MiMoV26AudioPatchError: Error, Equatable, Sendable {
    case invalidConfiguration(String), invalidInput(String), executionLimit(String)
    case invalidWeights(String), weightsNotLoaded
}

/// One source code_list item/segment, in frame-major then channel-major order.
public struct MiMoV26AudioCodeClip: Equatable, Sendable {
    public let codes: [Int32]
    public let frameCount: Int
    public init(codes: [Int32], frameCount: Int) { self.codes = codes; self.frameCount = frameCount }
}

/// Caller-owned execution bounds; these do not declare model/API capacity.
public struct MiMoV26AudioPatchLimits: Equatable, Sendable {
    public let maximumClips, maximumFrames, maximumPatches, maximumWorkingElements: Int
    public init(maximumClips: Int, maximumFrames: Int, maximumPatches: Int, maximumWorkingElements: Int) {
        self.maximumClips = maximumClips; self.maximumFrames = maximumFrames
        self.maximumPatches = maximumPatches; self.maximumWorkingElements = maximumWorkingElements
    }
}

private func audioProduct(_ values: [Int], _ name: String) throws -> Int {
    var result = 1
    for value in values {
        guard value >= 0 else { throw MiMoV26AudioPatchError.invalidConfiguration(name) }
        let next = result.multipliedReportingOverflow(by: value)
        guard !next.overflow else { throw MiMoV26AudioPatchError.executionLimit(name + " overflow") }
        result = next.partialValue
    }
    return result
}
private func audioSum(_ values: [Int], _ name: String) throws -> Int {
    var result = 0
    for value in values {
        let next = result.addingReportingOverflow(value)
        guard value >= 0 && !next.overflow else { throw MiMoV26AudioPatchError.executionLimit(name + " overflow") }
        result = next.partialValue
    }
    return result
}

private struct MiMoV26AudioShape {
    let config: MiMoV26AudioConfiguration
    let intermediate, projectionInput, projectionHidden: Int
    let ropeTheta: Float
    init(_ c: MiMoV26AudioConfiguration) throws {
        config = c
        func number(_ key: String) throws -> Decimal {
            guard case .number(let value) = c.rawFields[key] else {
                throw MiMoV26AudioPatchError.invalidConfiguration(key)
            }
            return value
        }
        func integer(_ key: String) throws -> Int {
            let value = try number(key), i = NSDecimalNumber(decimal: value).int64Value
            guard value == Decimal(i), let exact = Int(exactly: i), exact > 0, exact <= Int(Int32.max) else {
                throw MiMoV26AudioPatchError.invalidConfiguration(key)
            }
            return exact
        }
        let headWidth = try audioProduct([c.queryHeads, c.headDim], "head geometry")
        guard c.rawFields["input_full_attention"] == .bool(true),
              c.rawFields["add_post_norm"] == .bool(true),
              try integer("projection_layers") == 2,
              try number("partial_rotary_factor") == 1,
              try number("input_local_hidden_dropout") == 0,
              c.groupSize == 4, c.headDim.isMultiple(of: 2),
              c.hiddenSize == headWidth else {
            throw MiMoV26AudioPatchError.invalidConfiguration("requires full attention/full RoPE/post-norm/group4/two projections")
        }
        intermediate = try integer("input_local_intermediate_size")
        projectionInput = try audioProduct([c.groupSize, c.hiddenSize], "projection input")
        projectionHidden = try audioProduct([projectionInput, 4], "projection hidden")
        ropeTheta = NSDecimalNumber(decimal: try number("rope_theta")).floatValue
        let dimensions = [c.channels, c.hiddenSize, c.queryHeads, c.headDim, c.layers,
                          c.outputHiddenSize, c.speechVocabularySize, projectionInput, projectionHidden]
        guard ropeTheta.isFinite && ropeTheta > 0,
              dimensions.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }),
              c.channels == 20, c.hiddenSize <= 1024, c.queryHeads <= 16, c.headDim <= 64,
              c.layers <= 6, intermediate <= 4096, c.outputHiddenSize <= 4096,
              c.speechVocabularySize <= 1280 else {
            throw MiMoV26AudioPatchError.invalidConfiguration("dimensions or RoPE theta")
        }
        // This component supports the authenticated audio-patch geometry and
        // smaller diagnostic fixtures. Reject unknown larger architectures
        // before constructing a parameter inventory or allocating modules.
        for shape in tensorShapes().values {
            guard try audioProduct(shape, "parameter geometry") <= Int(Int32.max) else {
                throw MiMoV26AudioPatchError.invalidConfiguration("parameter exceeds MLX indexing range")
            }
        }
    }
    func tensorShapes() -> [String: [Int]] {
        let c = config, h = c.hiddenSize
        var result: [String: [Int]] = [:]
        for channel in 0..<c.channels { result["speech_embeddings.\(channel).weight"] = [c.speechVocabularySize, h] }
        let prefix = "audio_encoder.input_local_transformer."
        for layer in 0..<c.layers {
            let p = prefix + "layers.\(layer)."
            result[p + "input_layernorm.weight"] = [h]
            result[p + "post_attention_layernorm.weight"] = [h]
            for qkv in ["q_proj", "k_proj", "v_proj"] {
                result[p + "self_attn.\(qkv).weight"] = [h, h]
                result[p + "self_attn.\(qkv).bias"] = [h]
            }
            result[p + "self_attn.o_proj.weight"] = [h, h]
            for projection in ["gate_proj", "up_proj"] { result[p + "mlp.\(projection).weight"] = [intermediate, h] }
            result[p + "mlp.down_proj.weight"] = [h, intermediate]
        }
        result[prefix + "norm.weight"] = [h]
        result["audio_encoder.projection.mlp.0.weight"] = [projectionHidden, projectionInput]
        result["audio_encoder.projection.mlp.2.weight"] = [c.outputHiddenSize, projectionHidden]
        return result
    }
}

/// Pure CPU preflight and repeat-last padding. Empty list is valid; an empty
/// item is invalid because the source requires audio[-1] for that item.
public struct MiMoV26AudioPatchPlan: Equatable, Sendable {
    public let groupedCodes: [Int32]
    public let clipPatchRanges: [Range<Int>]
    public let patchCount, originalFrameCount, workingElements: Int
    public static func make(clips: [MiMoV26AudioCodeClip], configuration c: MiMoV26AudioConfiguration,
                            limits: MiMoV26AudioPatchLimits) throws -> Self {
        let shape = try MiMoV26AudioShape(c)
        guard [limits.maximumClips, limits.maximumFrames, limits.maximumPatches, limits.maximumWorkingElements].allSatisfy({ $0 >= 0 }),
              clips.count <= limits.maximumClips else { throw MiMoV26AudioPatchError.executionLimit("clip limits") }
        var frames = 0, patches = 0
        // Validate the complete request before allocating any derived arrays.
        for clip in clips {
            guard clip.frameCount > 0 else { throw MiMoV26AudioPatchError.invalidInput("zero-frame item") }
            let elements = try audioProduct([clip.frameCount, c.channels], "code shape")
            guard elements == clip.codes.count else { throw MiMoV26AudioPatchError.invalidInput("exact frame/channel geometry") }
            guard clip.codes.allSatisfy({ $0 >= 0 && Int($0) < c.speechVocabularySize }) else {
                throw MiMoV26AudioPatchError.invalidInput("speech code outside its embedding vocabulary")
            }
            frames = try audioSum([frames, clip.frameCount], "total frames")
            let rounded = try audioSum([clip.frameCount, c.groupSize - 1], "padded frames")
            patches = try audioSum([patches, rounded / c.groupSize], "patch count")
        }
        guard frames <= limits.maximumFrames, patches <= limits.maximumPatches,
              patches <= Int(Int32.max) else { throw MiMoV26AudioPatchError.executionLimit("frame/patch limit") }
        let paddedFrames = try audioProduct([patches, c.groupSize], "padded frames")
        let codeElements = try audioProduct([paddedFrames, c.channels], "padded codes")
        let layerElements = try audioSum([audioProduct([5, c.hiddenSize], "attention work"),
                                         audioProduct([3, shape.intermediate], "MLP work")], "layer work")
        let allHidden = try audioProduct([paddedFrames, audioSum([2 * c.hiddenSize,
            audioProduct([c.layers, layerElements], "layer work")], "hidden work")], "hidden work")
        let scores = try audioProduct([patches, c.queryHeads, c.groupSize, c.groupSize, c.layers], "attention scores")
        let projection = try audioProduct([patches, audioSum([shape.projectionHidden, c.outputHiddenSize], "projection work")], "projection work")
        let working = try audioSum([codeElements, allHidden, scores, projection], "working elements")
        guard working <= limits.maximumWorkingElements, codeElements <= Int(Int32.max) else {
            throw MiMoV26AudioPatchError.executionLimit("working element limit")
        }
        var grouped: [Int32] = [], ranges: [Range<Int>] = []
        grouped.reserveCapacity(codeElements); ranges.reserveCapacity(clips.count)
        var offset = 0
        for clip in clips {
            let n = (clip.frameCount + c.groupSize - 1) / c.groupSize
            ranges.append(offset..<(offset + n)); offset += n
            for frame in 0..<(n * c.groupSize) {
                let start = min(frame, clip.frameCount - 1) * c.channels
                grouped.append(contentsOf: clip.codes[start..<(start + c.channels)])
            }
        }
        return .init(groupedCodes: grouped, clipPatchRanges: ranges, patchCount: patches,
                     originalFrameCount: frames, workingElements: working)
    }
}

/// Preserve Qwen2's FP32 variance followed by input-dtype rounding BEFORE the
/// learned scale, rather than silently substituting a fused rounding policy.
private final class MiMoV26AudioRMSNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    init(_ hidden: Int) { _weight.wrappedValue = MLXArray.ones([hidden]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt((f * f).mean(axis: -1, keepDims: true) + Float(1e-6))).asType(x.dtype) * weight
    }
}
private final class MiMoV26AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let c: MiMoV26AudioConfiguration
    init(_ c: MiMoV26AudioConfiguration) {
        self.c = c
        _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = x.dim(0), half = c.headDim / 2
        func rotary(_ projected: MLXArray) -> MLXArray {
            let t = projected.reshaped(batch, c.groupSize, c.queryHeads, c.headDim).transposed(0, 2, 1, 3)
            let rotated = concatenated([-t[0..., 0..., 0..., half...], t[0..., 0..., 0..., ..<half]], axis: -1)
            return t * cosine + rotated * sine
        }
        let values = v(x).reshaped(batch, c.groupSize, c.queryHeads, c.headDim).transposed(0, 2, 1, 3)
        // Explicit bidirectional mask, one independent length4 sequence per
        // patch. No KV cache, causal fallback, cross-patch or cross-clip state.
        let attended = MLXFast.scaledDotProductAttention(queries: rotary(q(x)), keys: rotary(k(x)),
            values: values, scale: 1 / Float(c.headDim).squareRoot(), mask: .array(mask))
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, c.groupSize, c.hiddenSize))
    }
}
private final class MiMoV26AudioMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ s: MiMoV26AudioShape) {
        _gate.wrappedValue = Linear(s.config.hiddenSize, s.intermediate, bias: false)
        _up.wrappedValue = Linear(s.config.hiddenSize, s.intermediate, bias: false)
        _down.wrappedValue = Linear(s.intermediate, s.config.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}
private final class MiMoV26AudioBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: MiMoV26AudioRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: MiMoV26AudioRMSNorm
    @ModuleInfo(key: "self_attn") var attention: MiMoV26AudioAttention
    let mlp: MiMoV26AudioMLP
    init(_ s: MiMoV26AudioShape) {
        _inputNorm.wrappedValue = MiMoV26AudioRMSNorm(s.config.hiddenSize)
        _postNorm.wrappedValue = MiMoV26AudioRMSNorm(s.config.hiddenSize)
        _attention.wrappedValue = MiMoV26AudioAttention(s.config); mlp = MiMoV26AudioMLP(s)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), cosine: cosine, sine: sine, mask: mask)
        return h + mlp(postNorm(h))
    }
}
private final class MiMoV26AudioLocalTransformer: Module {
    let layers: [MiMoV26AudioBlock]
    let norm: MiMoV26AudioRMSNorm
    init(_ s: MiMoV26AudioShape) {
        layers = (0..<s.config.layers).map { _ in MiMoV26AudioBlock(s) }; norm = MiMoV26AudioRMSNorm(s.config.hiddenSize)
    }
    func callAsFunction(_ input: MLXArray, cosine: MLXArray, sine: MLXArray, mask: MLXArray) -> MLXArray {
        var x = input
        for layer in layers { x = layer(x, cosine: cosine, sine: sine, mask: mask) }
        return norm(x)
    }
}
private final class MiMoV26AudioProjection: Module, UnaryLayer {
    @ModuleInfo var mlp: (Linear, GELU, Linear)
    init(_ s: MiMoV26AudioShape) {
        _mlp.wrappedValue = (Linear(s.projectionInput, s.projectionHidden, bias: false), GELU(approximation: .none),
               Linear(s.projectionHidden, s.config.outputHiddenSize, bias: false))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp.2(mlp.1(mlp.0(x))) }
}
private final class MiMoV26AudioEncoderBody: Module {
    @ModuleInfo(key: "input_local_transformer") var transformer: MiMoV26AudioLocalTransformer
    let projection: MiMoV26AudioProjection
    init(_ s: MiMoV26AudioShape) { _transformer.wrappedValue = MiMoV26AudioLocalTransformer(s); projection = MiMoV26AudioProjection(s) }
}

public struct MiMoV26AudioPatchOutput {
    public let features: MLXArray
    public let clipPatchRanges: [Range<Int>]
}

public final class MiMoV26AudioPatchEncoder: Module {
    public let configuration: MiMoV26AudioConfiguration
    private let shape: MiMoV26AudioShape
    @ModuleInfo(key: "speech_embeddings") private var embeddings: [Embedding]
    @ModuleInfo(key: "audio_encoder") private var encoder: MiMoV26AudioEncoderBody
    private var validatedWeightDType: DType?

    public init(configuration: MiMoV26AudioConfiguration) throws {
        let shape = try MiMoV26AudioShape(configuration)
        self.shape = shape; self.configuration = configuration
        _embeddings.wrappedValue = (0..<configuration.channels).map { _ in Embedding(embeddingCount: configuration.speechVocabularySize, dimensions: configuration.hiddenSize) }
        _encoder.wrappedValue = MiMoV26AudioEncoderBody(shape)
    }
    public static func expectedTensorShapes(configuration: MiMoV26AudioConfiguration) throws -> [String: [Int]] {
        try MiMoV26AudioShape(configuration).tensorShapes()
    }
    public func loadNativeWeights(_ weights: [String: MLXArray], expectedDType: DType = .bfloat16) throws {
        validatedWeightDType = nil
        guard [.bfloat16, .float16, .float32].contains(expectedDType) else { throw MiMoV26AudioPatchError.invalidWeights("unsupported dtype") }
        let expected = shape.tensorShapes()
        guard Set(weights.keys) == Set(expected.keys) else { throw MiMoV26AudioPatchError.invalidWeights("missing/unmapped audio tensors") }
        for (key, dimensions) in expected {
            guard let value = weights[key], value.shape == dimensions, value.dtype == expectedDType else { throw MiMoV26AudioPatchError.invalidWeights(key) }
        }
        try update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }), verify: .all)
        validatedWeightDType = expectedDType
    }

    /// Called only after the complete published-layout plan has checked every
    /// tensor and installed its declared packed module shells.
    func loadPackedWeights(_ weights: [String: MLXArray], expectedDType: DType) throws {
        validatedWeightDType = nil
        guard expectedDType == .bfloat16 else { throw MiMoV26AudioPatchError.invalidWeights("packed audio activation dtype") }
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        guard Set(expected.keys) == Set(weights.keys) else { throw MiMoV26AudioPatchError.invalidWeights("packed audio parameter closure") }
        for (key, parameter) in expected {
            let value = weights[key]!
            guard value.shape == parameter.shape,
                  value.dtype == (parameter.dtype == .uint32 ? .uint32 : expectedDType) else {
                throw MiMoV26AudioPatchError.invalidWeights(key)
            }
        }
        try update(parameters: .unflattened(weights), verify: .all)
        validatedWeightDType = expectedDType
    }
    @discardableResult public override func update(parameters: ModuleParameters, verify: VerifyUpdate,
                                                   path: [String] = [], modulePath: [String] = []) throws -> Self {
        validatedWeightDType = nil
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
    @discardableResult public override func update(modules: ModuleChildren, verify: VerifyUpdate,
                                                   path: [String] = [], modulePath: [String] = []) throws -> Self {
        validatedWeightDType = nil
        return try super.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
    }
    public func forward(clips: [MiMoV26AudioCodeClip], limits: MiMoV26AudioPatchLimits) throws -> MiMoV26AudioPatchOutput {
        guard let dtype = validatedWeightDType else { throw MiMoV26AudioPatchError.weightsNotLoaded }
        let plan = try MiMoV26AudioPatchPlan.make(clips: clips, configuration: configuration, limits: limits)
        let c = configuration
        guard plan.patchCount > 0 else { return .init(features: MLXArray.zeros([0, c.outputHiddenSize], dtype: dtype), clipPatchRanges: []) }
        let codes = MLXArray(plan.groupedCodes).reshaped(plan.patchCount, c.groupSize, c.channels)
        var x = MLXArray.zeros([plan.patchCount, c.groupSize, c.hiddenSize], dtype: dtype)
        for channel in 0..<c.channels { x = x + embeddings[channel](codes[0..., 0..., channel]) }
        var angles: [Float] = []; angles.reserveCapacity(c.groupSize * c.headDim / 2)
        for position in 0..<c.groupSize {
            for pair in 0..<(c.headDim / 2) { angles.append(Float(position) / pow(shape.ropeTheta, Float(2 * pair) / Float(c.headDim))) }
        }
        let half = MLXArray(angles).reshaped(c.groupSize, c.headDim / 2)
        let phase = concatenated([half, half], axis: -1).reshaped(1, 1, c.groupSize, c.headDim)
        let cosine = cos(phase).asType(dtype), sine = sin(phase).asType(dtype)
        let mask = MLXArray.ones([c.groupSize, c.groupSize], dtype: .bool)
        x = encoder.transformer(x, cosine: cosine, sine: sine, mask: mask)
        let features = encoder.projection(x.reshaped(plan.patchCount, shape.projectionInput))
        return .init(features: features, clipPatchRanges: plan.clipPatchRanges)
    }
}
