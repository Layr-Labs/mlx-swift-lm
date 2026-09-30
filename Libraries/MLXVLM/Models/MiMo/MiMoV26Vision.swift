// Copyright © 2026 Eigen Labs.
// Native contracts derived from SGLang (Apache-2.0), commit
// 67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b: models/mimo_vl.py,
// layers/attention/vision.py and models/qwen2_5_vl.py.
// Preprocessed-patch component only: no pixels, media ingestion or factory alias.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum MiMoV26VisionError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidInput(String)
    case executionLimit(String)
    case invalidWeights(String)
    case weightsNotLoaded
}

public struct MiMoV26VisionGrid: Equatable, Sendable {
    public let temporal, height, width: Int
    public init(temporal: Int, height: Int, width: Int) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }
}

/// Caller-owned component allocation limits, distinct from model capability.
public struct MiMoV26VisionLimits: Equatable, Sendable {
    public let maximumPatches, maximumAttentionScoreElements: Int
    public init(maximumPatches: Int, maximumAttentionScoreElements: Int) {
        self.maximumPatches = maximumPatches
        self.maximumAttentionScoreElements = maximumAttentionScoreElements
    }
}

/// CPU layout metadata. All arrays are request-owned. Sequence ranges are one
/// temporal grid each: neither global nor local attention crosses a frame.
public struct MiMoV26VisionLayout: Equatable, Sendable {
    public let patchCount: Int
    public let frames: [Range<Int>]
    public let rowPositions, columnPositions: [Int]
    public let columnPermutation, inverseColumnPermutation: [Int]

    public static func make(
        grids: [MiMoV26VisionGrid], mergeSize: Int,
        queryHeads: Int, limits: MiMoV26VisionLimits
    ) throws -> Self {
        guard !grids.isEmpty, mergeSize > 0, queryHeads > 0,
            limits.maximumPatches > 0, limits.maximumAttentionScoreElements > 0
        else {
            throw MiMoV26VisionError.invalidInput("empty grids or invalid layout limits")
        }
        let unit = try visionProduct([mergeSize, mergeSize], "merge area")
        var total = 0
        for g in grids {
            guard g.temporal > 0, g.height > 0, g.width > 0,
                g.height.isMultiple(of: mergeSize), g.width.isMultiple(of: mergeSize)
            else {
                throw MiMoV26VisionError.invalidInput(
                    "grid must contain complete spatial merge groups")
            }
            let perFrame = try visionProduct([g.height, g.width], "frame patches")
            let scores = try visionProduct([queryHeads, perFrame, perFrame], "attention scores")
            guard scores <= limits.maximumAttentionScoreElements else {
                throw MiMoV26VisionError.executionLimit("per-frame attention score bound")
            }
            let count = try visionProduct([g.temporal, perFrame], "grid patches")
            let sum = total.addingReportingOverflow(count)
            guard !sum.overflow, sum.partialValue <= limits.maximumPatches,
                sum.partialValue <= Int(Int32.max)
            else {
                throw MiMoV26VisionError.executionLimit("total patch bound")
            }
            total = sum.partialValue
        }
        var rows: [Int] = []
        var columns: [Int] = []
        var permutation: [Int] = []
        var frames: [Range<Int>] = []
        rows.reserveCapacity(total)
        columns.reserveCapacity(total)
        permutation.reserveCapacity(total)
        var start = 0
        for g in grids {
            let groupsH = g.height / mergeSize
            let groupsW = g.width / mergeSize
            let perFrame = g.height * g.width  // Checked above.
            for _ in 0 ..< g.temporal {
                frames.append(start ..< (start + perFrame))
                for groupH in 0 ..< groupsH {
                    for groupW in 0 ..< groupsW {
                        for localH in 0 ..< mergeSize {
                            for localW in 0 ..< mergeSize {
                                rows.append(groupH * mergeSize + localH)
                                columns.append(groupW * mergeSize + localW)
                            }
                        }
                    }
                }
                for groupW in 0 ..< groupsW {
                    for groupH in 0 ..< groupsH {
                        let group = (groupH * groupsW + groupW) * unit
                        permutation.append(contentsOf: (0 ..< unit).map { start + group + $0 })
                    }
                }
                start += perFrame
            }
        }
        var inverse = [Int](repeating: 0, count: total)
        for (destination, source) in permutation.enumerated() { inverse[source] = destination }
        return Self(
            patchCount: total, frames: frames, rowPositions: rows, columnPositions: columns,
            columnPermutation: permutation, inverseColumnPermutation: inverse)
    }
}

private func visionProduct(_ values: [Int], _ name: String) throws -> Int {
    var result = 1
    for value in values {
        guard value > 0 else { throw MiMoV26VisionError.invalidConfiguration(name) }
        let next = result.multipliedReportingOverflow(by: value)
        guard !next.overflow else {
            throw MiMoV26VisionError.invalidConfiguration("overflow: " + name)
        }
        result = next.partialValue
    }
    return result
}

private struct MiMoV26VisionShape {
    let config: MiMoV26VisionConfiguration
    let channels, headDim, qWidth, kvWidth, fusedWidth, patchWidth, mergeWidth: Int

    init(_ config: MiMoV26VisionConfiguration) throws {
        self.config = config
        func integer(_ key: String, default fallback: Int) throws -> Int {
            guard let field = config.rawFields[key] else { return fallback }
            guard case .number(let number) = field else {
                throw MiMoV26VisionError.invalidConfiguration(key)
            }
            let value = NSDecimalNumber(decimal: number).int64Value
            guard number == Decimal(value), let exact = Int(exactly: value), exact > 0,
                exact <= Int(Int32.max)
            else { throw MiMoV26VisionError.invalidConfiguration(key) }
            return exact
        }
        channels = try integer("in_channels", default: 3)
        if let declared = config.rawFields["in_chans"], declared != .number(Decimal(channels)) {
            throw MiMoV26VisionError.invalidConfiguration(
                "in_chans disagrees with consumed in_channels")
        }
        // MiMo's defaults are 64 even though hidden1280 / heads32 equals40.
        headDim = try integer("qk_channels", default: 64)
        let valueDim = try integer("kv_channels", default: 64)
        guard headDim == valueDim, headDim.isMultiple(of: 4),
            config.queryHeads.isMultiple(of: config.keyValueHeads),
            config.windowAttentionTypes.last != 1
        else {
            throw MiMoV26VisionError.invalidConfiguration("head geometry or final column ordering")
        }
        qWidth = try visionProduct([config.queryHeads, headDim], "query projection")
        kvWidth = try visionProduct([config.keyValueHeads, headDim], "KV projection")
        let kvPair = try visionProduct([2, kvWidth], "KV pair")
        let fused = qWidth.addingReportingOverflow(kvPair)
        guard !fused.overflow else {
            throw MiMoV26VisionError.invalidConfiguration("QKV width overflow")
        }
        fusedWidth = fused.partialValue
        patchWidth = try visionProduct(
            [channels, config.temporalPatchSize, config.patchSize, config.patchSize], "patch width")
        mergeWidth = try visionProduct(
            [config.hiddenSize, config.spatialMergeSize, config.spatialMergeSize], "merger width")
        let dimensions = [
            config.depth, config.hiddenSize, config.intermediateSize, config.outputHiddenSize,
            qWidth, kvWidth, fusedWidth, patchWidth, mergeWidth,
        ]
        guard dimensions.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else {
            throw MiMoV26VisionError.invalidConfiguration("MLX dimension exceeds Int32")
        }
        _ = try visionProduct([config.hiddenSize, fusedWidth], "QKV tensor")
        _ = try visionProduct([mergeWidth, mergeWidth], "merger tensor")
        _ = try integer("visual_token_window_size", default: 64)
    }

    var window: Int {
        // Validated by integer() above; source config supplies an integer.
        if case .number(let value) = config.rawFields["visual_token_window_size"] {
            return NSDecimalNumber(decimal: value).intValue
        }
        return 64
    }

    func sourceShapes() -> [String: [Int]] {
        let c = config
        var shapes = [
            "visual.patch_embed.proj.weight": [
                c.hiddenSize, channels, c.temporalPatchSize, c.patchSize, c.patchSize,
            ],
            "visual.merger.ln_q.weight": [c.hiddenSize],
            "visual.merger.mlp.0.weight": [mergeWidth, mergeWidth],
            "visual.merger.mlp.2.weight": [c.outputHiddenSize, mergeWidth],
        ]
        for i in 0 ..< c.depth {
            let p = "visual.blocks.\(i)."
            shapes[p + "norm1.weight"] = [c.hiddenSize]
            shapes[p + "norm2.weight"] = [c.hiddenSize]
            shapes[p + "attn.qkv.weight"] = [fusedWidth, c.hiddenSize]
            shapes[p + "attn.qkv.bias"] = [fusedWidth]
            shapes[p + "attn.proj.weight"] = [c.hiddenSize, qWidth]
            shapes[p + "attn.proj.bias"] = [c.hiddenSize]
            for name in ["gate_proj", "up_proj"] {
                shapes[p + "mlp.\(name).weight"] = [c.intermediateSize, c.hiddenSize]
                shapes[p + "mlp.\(name).bias"] = [c.intermediateSize]
            }
            shapes[p + "mlp.down_proj.weight"] = [c.hiddenSize, c.intermediateSize]
            shapes[p + "mlp.down_proj.bias"] = [c.hiddenSize]
            if c.usesSinks && !c.fullAttentionBlocks.contains(i) {
                shapes[p + "attn.sinks"] = [c.queryHeads]
            }
        }
        return shapes
    }
}

private final class MiMoV26VisionPatchEmbedding: Module, UnaryLayer {
    @ModuleInfo var proj: Linear
    init(_ shape: MiMoV26VisionShape) {
        _proj.wrappedValue = Linear(shape.patchWidth, shape.config.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

private final class MiMoV26VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: MiMoV26VisionConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

private final class MiMoV26VisionAttention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ParameterInfo var sinks: MLXArray?
    let shape: MiMoV26VisionShape
    init(_ shape: MiMoV26VisionShape, full: Bool) {
        self.shape = shape
        _qkv.wrappedValue = Linear(shape.config.hiddenSize, shape.fusedWidth, bias: true)
        _proj.wrappedValue = Linear(shape.qWidth, shape.config.hiddenSize, bias: true)
        _sinks.wrappedValue =
            shape.config.usesSinks && !full ? MLXArray.zeros([shape.config.queryHeads]) : nil
    }

    func callAsFunction(_ x: MLXArray, angles: MLXArray, frames: [Range<Int>]) -> MLXArray {
        let c = shape.config
        let n = x.dim(0)
        let fused = qkv(x)
        var q = fused[0..., ..<shape.qWidth].reshaped(n, c.queryHeads, shape.headDim)
        var k = fused[0..., shape.qWidth ..< (shape.qWidth + shape.kvWidth)].reshaped(
            n, c.keyValueHeads, shape.headDim)
        let v = fused[0..., (shape.qWidth + shape.kvWidth)...].reshaped(
            n, c.keyValueHeads, shape.headDim)
        let phase = concatenated([angles, angles], axis: -1)[0..., .newAxis, 0...]
        let cosine = cos(phase)
        let sine = sin(phase)
        func rotary(_ input: MLXArray) -> MLXArray {
            let half = shape.headDim / 2
            let fp32 = input.asType(.float32)
            let rotated = concatenated(
                [-fp32[0..., 0..., half...], fp32[0..., 0..., ..<half]], axis: -1)
            return (fp32 * cosine + rotated * sine).asType(input.dtype)
        }
        q = rotary(q)
        k = rotary(k)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(frames.count)
        for frame in frames {
            let length = frame.count
            // Local queries see exactly [i-window, i+window], clipped to this
            // frame. Tiling avoids a dense frame² local mask; it is execution
            // geometry, not a model window or reduced native context limit.
            // Numerical equivalence to the selected backend remains a gate.
            // Pinned SGLang67bb6a58 VisionAttention.forward selects an unlimited
            // window when full_attn OR sinks is None. A no-sink local block
            // therefore deliberately uses full-frame attention as well.
            let tile = sinks == nil ? length : 128
            for start in stride(from: 0, to: length, by: tile) {
                let end = min(start + tile, length)
                let keyStart = sinks == nil ? 0 : max(0, start - shape.window)
                let keyEnd = sinks == nil ? length : min(length, end + shape.window)
                let queryRange = (frame.lowerBound + start) ..< (frame.lowerBound + end)
                let keyRange = (frame.lowerBound + keyStart) ..< (frame.lowerBound + keyEnd)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode
                if sinks != nil {
                    let queryPosition = MLXArray((start ..< end).map(Int32.init))
                    let keyPosition = MLXArray((keyStart ..< keyEnd).map(Int32.init))
                    let distance = queryPosition[0..., .newAxis] - keyPosition[.newAxis, 0...]
                    mask = .array((distance .>= -shape.window) .&& (distance .<= shape.window))
                } else {
                    mask = .none
                }
                let result = MLXFast.scaledDotProductAttention(
                    queries: q[queryRange].transposed(1, 0, 2).expandedDimensions(axis: 0),
                    keys: k[keyRange].transposed(1, 0, 2).expandedDimensions(axis: 0),
                    values: v[keyRange].transposed(1, 0, 2).expandedDimensions(axis: 0),
                    scale: 1 / Float(shape.headDim).squareRoot(), mask: mask, sinks: sinks)
                outputs.append(result[0].transposed(1, 0, 2).reshaped(end - start, shape.qWidth))
            }
        }
        return proj(concatenated(outputs, axis: 0))
    }
}

private final class MiMoV26VisionBlock: Module {
    @ModuleInfo var norm1: RMSNorm
    @ModuleInfo var norm2: RMSNorm
    let attn: MiMoV26VisionAttention
    let mlp: MiMoV26VisionMLP
    init(_ shape: MiMoV26VisionShape, index: Int, epsilon: Float) {
        _norm1.wrappedValue = RMSNorm(dimensions: shape.config.hiddenSize, eps: epsilon)
        _norm2.wrappedValue = RMSNorm(dimensions: shape.config.hiddenSize, eps: epsilon)
        attn = MiMoV26VisionAttention(shape, full: shape.config.fullAttentionBlocks.contains(index))
        mlp = MiMoV26VisionMLP(shape.config)
    }
    func callAsFunction(_ x: MLXArray, angles: MLXArray, frames: [Range<Int>]) -> MLXArray {
        let residual = x + attn(norm1(x), angles: angles, frames: frames)
        return residual + mlp(norm2(residual))
    }
}

private final class MiMoV26VisionMerger: Module, UnaryLayer {
    @ModuleInfo(key: "ln_q") var norm: RMSNorm
    @ModuleInfo var mlp: (Linear, GELU, Linear)
    let width: Int
    init(_ shape: MiMoV26VisionShape) {
        width = shape.mergeWidth
        _norm.wrappedValue = RMSNorm(dimensions: shape.config.hiddenSize, eps: 1e-6)
        // The 364-tensor checkpoint contains no merger biases. SGLang's
        // _post_init sets those runtime biases to zero; no trained value is
        // synthesized here and strict loading requires only actual parameters.
        _mlp.wrappedValue = (
            Linear(width, width, bias: false), GELU(approximation: .none),
            Linear(width, shape.config.outputHiddenSize, bias: false)
        )
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp.2(mlp.1(mlp.0(norm(x).reshaped(-1, width))))
    }
}

public final class MiMoV26VisionTower: Module {
    public let configuration: MiMoV26VisionConfiguration
    private let shape: MiMoV26VisionShape
    @ModuleInfo(key: "patch_embed") private var patchEmbedding: MiMoV26VisionPatchEmbedding
    private let blocks: [MiMoV26VisionBlock]
    private let merger: MiMoV26VisionMerger
    private var validatedWeightDType: DType?

    public init(configuration: MiMoV26VisionConfiguration, normEpsilon: Float = 1e-6) throws {
        guard normEpsilon.isFinite && normEpsilon > 0 else {
            throw MiMoV26VisionError.invalidConfiguration("normalization epsilon")
        }
        let shape = try MiMoV26VisionShape(configuration)
        self.shape = shape
        self.configuration = configuration
        _patchEmbedding.wrappedValue = MiMoV26VisionPatchEmbedding(shape)
        blocks = (0 ..< configuration.depth).map {
            MiMoV26VisionBlock(shape, index: $0, epsilon: normEpsilon)
        }
        merger = MiMoV26VisionMerger(shape)
    }

    public static func expectedTensorShapes(configuration: MiMoV26VisionConfiguration) throws
        -> [String: [Int]]
    {
        try MiMoV26VisionShape(configuration).sourceShapes()
    }

    /// Exact visual-only source/converted dictionary. Patch storage is O,C,T,H,W;
    /// reshape preserves that byte order as the flattened Linear input. Every
    /// other trained tensor retains its name/shape after removing `visual.`.
    public func loadNativeWeights(
        _ weights: [String: MLXArray], expectedDType: DType = .bfloat16,
        flattenedPatchStorage: Bool = false
    ) throws {
        guard [.bfloat16, .float16, .float32].contains(expectedDType) else {
            throw MiMoV26VisionError.invalidWeights("unsupported dtype")
        }
        var expected = shape.sourceShapes()
        if flattenedPatchStorage {
            expected["visual.patch_embed.proj.weight"] = [
                configuration.hiddenSize, shape.patchWidth,
            ]
        }
        guard Set(weights.keys) == Set(expected.keys) else {
            throw MiMoV26VisionError.invalidWeights("missing or unmapped visual tensors")
        }
        for (name, dimensions) in expected {
            guard let value = weights[name], value.shape == dimensions, value.dtype == expectedDType
            else {
                throw MiMoV26VisionError.invalidWeights(name)
            }
        }
        var mapped: [(String, MLXArray)] = []
        for (name, value) in weights {
            let tensor =
                name == "visual.patch_embed.proj.weight"
                ? value.reshaped(configuration.hiddenSize, shape.patchWidth) : value
            mapped.append((String(name.dropFirst("visual.".count)), tensor))
        }
        try update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        validatedWeightDType = expectedDType
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        validatedWeightDType = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    @discardableResult
    public override func update(
        modules: ModuleChildren, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        validatedWeightDType = nil
        return try super.update(
            modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    public func forward(
        patches: MLXArray, grids: [MiMoV26VisionGrid],
        limits: MiMoV26VisionLimits
    ) throws -> MLXArray {
        try forwardWithCheckpoints(
            patches: patches, grids: grids, limits: limits, checkpoint: { _ in })
    }

    func forwardWithCheckpoints(
        patches: MLXArray, grids: [MiMoV26VisionGrid], limits: MiMoV26VisionLimits,
        checkpoint: ([MLXArray]) throws -> Void
    ) throws -> MLXArray {
        guard let dtype = validatedWeightDType else { throw MiMoV26VisionError.weightsNotLoaded }
        guard patches.ndim == 2, patches.dim(1) == shape.patchWidth,
            [.bfloat16, .float16, .float32].contains(patches.dtype)
        else {
            throw MiMoV26VisionError.invalidInput("native flattened patch geometry/dtype")
        }
        let layout = try MiMoV26VisionLayout.make(
            grids: grids, mergeSize: configuration.spatialMergeSize,
            queryHeads: configuration.queryHeads, limits: limits)
        guard patches.dim(0) == layout.patchCount else {
            throw MiMoV26VisionError.invalidInput("patch count does not match grids")
        }
        let axisFrequencies = shape.headDim / 4
        let frequencyCount = try visionProduct(
            [layout.patchCount, axisFrequencies, 2], "rotary table")
        var frequencies: [Float] = []
        frequencies.reserveCapacity(frequencyCount)
        for i in 0 ..< layout.patchCount {
            for position in [layout.rowPositions[i], layout.columnPositions[i]] {
                for j in 0 ..< axisFrequencies {
                    let inverse = 1 / pow(Float(10000), Float(j * 2) / Float(shape.headDim / 2))
                    frequencies.append(Float(position) * inverse)
                }
            }
        }
        let rowAngles = MLXArray(frequencies).reshaped(layout.patchCount, shape.headDim / 2)
        let columnIndex = MLXArray(layout.columnPermutation.map(Int32.init))
        let inverseIndex = MLXArray(layout.inverseColumnPermutation.map(Int32.init))
        let columnAngles = rowAngles[columnIndex]
        var x = patchEmbedding(patches.asType(dtype))
        let retained = [patches, rowAngles, columnIndex, inverseIndex, columnAngles]
        try checkpoint(retained + [x])
        var columnOrder = false
        for (i, block) in blocks.enumerated() {
            let nextColumnOrder = configuration.windowAttentionTypes[i] == 1
            if nextColumnOrder != columnOrder {
                x = x[nextColumnOrder ? columnIndex : inverseIndex]
            }
            columnOrder = nextColumnOrder
            x = block(x, angles: columnOrder ? columnAngles : rowAngles, frames: layout.frames)
            try checkpoint(retained + [x])
        }
        let result = merger(x)
        try checkpoint(retained + [result])
        return result
    }
}
