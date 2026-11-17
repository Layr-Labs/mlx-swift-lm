import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// `Gemma4FusedNorms` writes the bits of the unfused MLX ops it replaces: the
/// kernels alone, one decoder layer, and a whole model's logits.
@Suite("Gemma 4 fused norm chains", .tags(.kernel), .serialized)
struct Gemma4FusedNormsTests {
    private static let eps: Float = 1e-6

    /// Byte parity is promised only where the mean form is measured
    /// (`RMSNormMeanForm.measured`); elsewhere the product keeps the unfused ops.
    private var verified: Bool {
        Device.defaultDevice().deviceType == .gpu
            && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
    }

    @Test func meanFormIsMeasuredOnRowsThatSeparateTheForms() throws {
        guard verified else {
            #expect(RMSNormMeanForm.measured(width: 2816, dtype: .bfloat16) == nil)
            return
        }
        for dtype in [DType.bfloat16, .float16, .float32] {
            // Gemma 4 26B's width, and power-of-two widths where the forms coincide.
            for width in [2816, 4096, 128] {
                let m = try #require(
                    RMSNormMeanForm.Calibration.measurement(width: width, dtype: dtype))
                if m.formsCoincide {
                    #expect(m.divideMatches && m.fusedMultiplyAddMatches, "width \(width) \(dtype)")
                } else {
                    // The rows tell the two compiled forms apart, and exactly one
                    // of them writes MLX's prebuilt bytes.
                    #expect(m.formsDiffer, "width \(width) \(dtype)")
                    #expect(m.divideMatches != m.fusedMultiplyAddMatches, "width \(width) \(dtype)")
                }
                #expect(RMSNormMeanForm.measured(width: width, dtype: dtype) != nil)
            }
        }
    }

    private func random(_ shape: [Int], _ seed: UInt64, _ dtype: DType, scale: Float = 1)
        -> MLXArray
    {
        let x = (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(dtype)
        eval(x)
        return x
    }

    private func weight(_ width: Int, _ seed: UInt64, _ dtype: DType) -> MLXArray {
        let w = (MLXRandom.normal([width], key: MLXRandom.key(seed)) / 4 + 1).asType(dtype)
        eval(w)
        return w
    }

    private func bytes(_ x: MLXArray) -> Data {
        eval(x)
        return x.asData(access: .copy).data
    }

    private func expectExact(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == expected.dtype)
        #expect(bytes(actual) == bytes(expected))
    }

    @Test func kernelsWriteTheUnfusedBits() throws {
        guard verified else {
            #expect(!Gemma4FusedNorms.supports(width: 128, dtype: .bfloat16))
            return
        }
        let eps = Self.eps
        for dtype in [DType.bfloat16, .float16, .float32] {
            // 2816 is Gemma 4 26B's width; 1000 leaves a partial last read.
            for width in [2816, 4096, 128, 1000] {
                try #require(Gemma4FusedNorms.supports(width: width, dtype: dtype))
                for rows in [1, 3, 8, 64] {
                    let seed = UInt64(width * 31 + rows)
                    let shape = [rows, 1, width]
                    let attention = random(shape, seed, dtype, scale: 3)
                    let residual = random(shape, seed + 1, dtype, scale: 5)
                    let post = weight(width, seed + 2, dtype)
                    let norms = (0 ..< 3).map { weight(width, seed + 3 + UInt64($0), dtype) }

                    let sum = residual + MLXFast.rmsNorm(attention, weight: post, eps: eps)
                    for count in [1, 3] {
                        let fused = Gemma4FusedNorms.attentionTail(
                            attention, residual: residual, postAttention: post,
                            sumNorms: Array(norms.prefix(count)), eps: eps)
                        expectExact(fused.sum, sum)
                        for (index, w) in norms.prefix(count).enumerated() {
                            let expected = MLXFast.rmsNorm(sum, weight: w, eps: eps)
                            expectExact(fused.norms[index], expected)
                        }
                    }

                    let dense = random(shape, seed + 7, dtype, scale: 2)
                    let experts = random(shape, seed + 8, dtype, scale: 0.5)
                    let postDense = weight(width, seed + 9, dtype)
                    let postExperts = weight(width, seed + 10, dtype)
                    let postFeedforward = weight(width, seed + 11, dtype)
                    let next = weight(width, seed + 12, dtype)
                    let scalar = (MLXArray([Float(0.75)]) + Float(rows) / 64).asType(dtype)
                    for withExperts in [true, false] {
                        let s =
                            withExperts
                            ? MLXFast.rmsNorm(dense, weight: postDense, eps: eps)
                                + MLXFast.rmsNorm(experts, weight: postExperts, eps: eps)
                            : dense
                        let out =
                            (residual + MLXFast.rmsNorm(s, weight: postFeedforward, eps: eps))
                            * scalar
                        for withNext in [true, false] {
                            let fused = Gemma4FusedNorms.feedforwardTail(
                                dense: dense,
                                experts: withExperts ? (experts, postDense, postExperts) : nil,
                                residual: residual, post: postFeedforward, layerScalar: scalar,
                                next: withNext ? next : nil, eps: eps)
                            expectExact(fused.out, out)
                            if withNext {
                                let fusedNext = try #require(fused.next)
                                expectExact(fusedNext, MLXFast.rmsNorm(out, weight: next, eps: eps))
                            } else {
                                #expect(fused.next == nil)
                            }
                        }
                    }
                }
            }
        }
    }

    private func configuration(moe: Bool, layers: Int) throws -> Gemma4TextConfiguration {
        let types = (0 ..< layers).map {
            $0 % 2 == 1 ? "\"full_attention\"" : "\"sliding_attention\""
        }
        let experts =
            moe
            ? "\"enable_moe_block\": true, \"num_experts\": 16, \"top_k_experts\": 4, \"moe_intermediate_size\": 64,"
            : "\"enable_moe_block\": false,"
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gemma4_text", "vocab_size": 64,
                    "hidden_size": 128, "intermediate_size": 256,
                    "num_hidden_layers": \(layers), "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 64,
                    "layer_types": [\(types.joined(separator: ", "))], "num_kv_shared_layers": 0,
                    "sliding_window": 16, "rms_norm_eps": 1e-6,
                    "final_logit_softcapping": 30.0, "tie_word_embeddings": true,
                    \(experts)
                    "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0
                }
                """.utf8))
    }

    /// Random bf16 weights, norm weights near one, a layer scalar below one.
    private func randomize(_ module: Module, seed: UInt64) throws {
        var weights = [String: MLXArray]()
        for (index, (name, parameter)) in module.parameters().flattened().sorted(by: {
            $0.0 < $1.0
        }).enumerated() {
            let key = MLXRandom.key(seed + UInt64(index))
            if parameter.ndim < 2 {
                let base: Float = name.hasSuffix("layer_scalar") ? 0.5 : 1
                weights[name] = (MLXRandom.normal(parameter.shape, key: key) / 8 + base)
                    .asType(.bfloat16)
            } else {
                weights[name] = (MLXRandom.normal(parameter.shape, key: key) / 16).asType(.bfloat16)
            }
        }
        try module.update(parameters: .unflattened(weights), verify: .all)
        module.train(false)
        eval(module)
    }

    @Test(arguments: [true, false])
    func decoderLayerKeepsItsBits(moe: Bool) throws {
        guard verified else { return }
        let layer = Gemma4DecoderLayer(try configuration(moe: moe, layers: 1), layerIdx: 0)
        try randomize(layer, seed: moe ? 9_600 : 9_700)
        for (rows, width) in [(1, 1), (3, 1), (2, 2), (1, 16)] {
            let seed = UInt64(9_800 + rows * 10 + width)
            let x = random([rows, width, 128], seed, .bfloat16, scale: 4)
            try #require(layer.supportsFusedNorms(x, outputTailRows: nil))
            let next = weight(128, 9_900, .bfloat16)
            let (expected, _, _) = Gemma4FusedNorms.$enabledOverride.withValue(false) {
                layer(x)
            }
            let (out, _, _, fusedNext) = layer.forwardFusedNorms(
                x, normalizedInput: nil, nextNorm: next, mask: nil, cache: nil, sharedKV: nil,
                positionOffset: nil, v2SharedSource: nil, useLastQueryPrefill: false,
                isExpertPrefill: false)
            expectExact(out, expected)
            let expectedNext = MLXFast.rmsNorm(expected, weight: next, eps: Self.eps)
            expectExact(try #require(fusedNext), expectedNext)
            // A given input norm is used as is (the previous layer's tail wrote it).
            let normalized = layer.inputLayernorm(x)
            let (again, _, _, _) = layer.forwardFusedNorms(
                x, normalizedInput: normalized, nextNorm: nil, mask: nil, cache: nil,
                sharedKV: nil, positionOffset: nil, v2SharedSource: nil,
                useLastQueryPrefill: false, isExpertPrefill: false)
            expectExact(again, expected)
        }
    }

    @Test(arguments: [true, false])
    func modelLogitsKeepTheirBits(moe: Bool) throws {
        guard verified else { return }
        try #require(Gemma4FusedNorms.supports(width: 128, dtype: .bfloat16))
        let model = Gemma4TextModel(try configuration(moe: moe, layers: 4))
        try randomize(model, seed: moe ? 10_000 : 11_000)
        let prompt = MLXArray((0 ..< 12).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 12)
        func run(_ fused: Bool) -> [Data] {
            Gemma4FusedNorms.$enabledOverride.withValue(fused) {
                let cache = model.newCache(parameters: nil)
                var outputs = [bytes(model(prompt, cache: cache))]
                var token = MLXArray([Int32(5)]).reshaped(1, 1)
                for step in 0 ..< 6 {
                    let logits = model(token, cache: cache)
                    outputs.append(bytes(logits))
                    token = MLXArray([Int32((step * 11 + 1) % 64)]).reshaped(1, 1)
                }
                return outputs
            }
        }
        let unfused = run(false)
        let fused = run(true)
        #expect(fused.count == unfused.count)
        for (a, b) in zip(fused, unfused) { #expect(a == b) }
    }

    // MARK: - Attention prologue

    /// A proportional rotation as Gemma 4's global layers build it
    /// (`partial` of the pairs rotate; the rest have infinite frequencies).
    private static func proportional(_ dims: Int, partial: Float) -> MLXArray {
        let rope = ProportionalRoPE(
            dims: dims, traditional: false, base: 1_000_000,
            scalingConfig: ["partial_rotary_factor": .float(partial)])
        let frequencies = rope.exactRotation!.frequencies
        eval(frequencies)
        return frequencies
    }

    @Test func ropeFormIsMeasuredForGemmaRotations() throws {
        guard verified else {
            #expect(
                RoPEForm.measured(dimensions: 256, base: 10_000, frequencies: nil, dtype: .bfloat16)
                    == nil)
            return
        }
        let rotations: [(Int, Float?, MLXArray?)] = [
            (256, 10_000, nil), (64, 10_000, nil),
            (512, nil, Self.proportional(512, partial: 0.25)),
            (128, nil, Self.proportional(128, partial: 1)),
        ]
        for dtype in [DType.bfloat16, .float16, .float32] {
            for (dims, base, frequencies) in rotations {
                let source: RoPEForm.Calibration.Source =
                    frequencies.map { .frequencies($0.asArray(Float.self).map(\.bitPattern)) }
                    ?? .base(base!.bitPattern)
                let m = try #require(
                    RoPEForm.Calibration.measurement(
                        .init(dimensions: dims, source: source, dtype: dtype)))
                // Random rows tell the multiply-add forms apart, and a form
                // matches MLX on every row (unit, random, product dtype).
                #expect(m.productFormsDiffer, "\(dims) \(dtype)")
                #expect(!m.matches.isEmpty, "\(dims) \(dtype)")
                #expect(
                    RoPEForm.measured(
                        dimensions: dims, base: base, frequencies: frequencies, dtype: dtype)
                        != nil, "\(dims) \(dtype)")
            }
        }
    }

    @Test func prologueWritesTheUnfusedBits() throws {
        guard verified else { return }
        let eps = Self.eps
        // (head dim, query heads, key/value heads, base, frequencies)
        let layers: [(Int, Int, Int, Float?, MLXArray?)] = [
            (256, 8, 2, 10_000, nil),
            (512, 4, 1, nil, Self.proportional(512, partial: 0.25)),
            (128, 4, 2, nil, Self.proportional(128, partial: 1)),
            (64, 2, 1, 10_000, nil),
        ]
        for dtype in [DType.bfloat16, .float16, .float32] {
            for (dims, queryHeads, kvHeads, base, frequencies) in layers {
                let rotation = try #require(
                    Gemma4FusedNorms.RotationCache().rotation(
                        headDim: dims, base: base, frequencies: frequencies, dtype: dtype))
                for (batch, tokens) in [(1, 1), (3, 1), (8, 1), (2, 4)] {
                    let seed = UInt64(dims * 97 + batch * 7 + tokens)
                    let q = random([batch, tokens, queryHeads * dims], seed, dtype, scale: 4)
                    let k = random([batch, tokens, kvHeads * dims], seed + 1, dtype, scale: 3)
                    let v = random([batch, tokens, kvHeads * dims], seed + 2, dtype, scale: 2)
                    let wq = weight(dims, seed + 3, dtype)
                    let wk = weight(dims, seed + 4, dtype)
                    // Positions from the start of a request to deep into one.
                    let offsets = MLXArray(
                        (0 ..< batch).map { Int32(($0 * 104_729 + Int(seed) * 7_919) % 300_000) })
                    func rope(_ x: MLXArray) -> MLXArray {
                        MLXFast.RoPE(
                            x, dimensions: dims, traditional: false, base: base, scale: 1,
                            offset: offsets, freqs: frequencies)
                    }
                    let shapeQ = [batch, tokens, queryHeads, dims]
                    let shapeK = [batch, tokens, kvHeads, dims]
                    let expectedQ = rope(
                        MLXFast.rmsNorm(q.reshaped(shapeQ), weight: wq, eps: eps)
                            .transposed(0, 2, 1, 3))
                    let expectedK = rope(
                        MLXFast.rmsNorm(k.reshaped(shapeK), weight: wk, eps: eps)
                            .transposed(0, 2, 1, 3))
                    for values in [v, k] {
                        let expectedV = MLXFast.rmsNorm(
                            values.reshaped(shapeK), weight: MLXArray.mlxNone, eps: eps
                        ).transposed(0, 2, 1, 3)
                        let fused = Gemma4FusedNorms.attentionPrologue(
                            queries: q, keys: k, values: values, queryHeads: queryHeads,
                            keyValueHeads: kvHeads, headDim: dims, queryWeight: wq,
                            keyWeight: wk, offsets: offsets, rotation: rotation, eps: eps)
                        expectExact(fused.queries, expectedQ)
                        expectExact(fused.keys, expectedK)
                        expectExact(fused.values, expectedV)
                    }
                }
            }
        }
    }

    /// Records what the attention hands the paged cache. (The trunk carries
    /// caches as `[KVCache]`; the legacy requirements pass to a plain cache.)
    private final class RecordingCache: CBv2AttendingLayerCache, KVCache {
        let layerIndex: Int
        let kind: CBv2LayerKind
        let positionOffsets: MLXArray
        var recorded: [MLXArray] = []
        private let legacy = KVCacheSimple()

        var offset: Int { legacy.offset }
        var maxSize: Int? { legacy.maxSize }
        func innerState() -> [MLXArray] { legacy.innerState() }
        func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            legacy.update(keys: keys, values: values)
        }
        var state: [MLXArray] {
            get { legacy.state }
            set { legacy.state = newValue }
        }
        var metaState: [String] {
            get { legacy.metaState }
            set { legacy.metaState = newValue }
        }
        var isTrimmable: Bool { legacy.isTrimmable }
        @discardableResult func trim(_ n: Int) -> Int { legacy.trim(n) }
        func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
            -> MLXFast.ScaledDotProductAttentionMaskMode
        {
            legacy.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }
        func copy() -> any KVCache { legacy.copy() }

        init(layerIndex: Int, kind: CBv2LayerKind, offsets: MLXArray) {
            self.layerIndex = layerIndex
            self.kind = kind
            self.positionOffsets = offsets
        }

        var rows: [CBv2SequenceKV] { [] }
        func setRows(_ rows: [CBv2SequenceKV]) {}

        func updateAndAttend(
            queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
        ) -> MLXArray {
            recorded = [queries, keys, values]
            // Something of the attention's shape that depends on all three.
            let keySum = keys.sum(axis: 2, keepDims: true)
            let valueSum = values.sum(axis: 2, keepDims: true)
            let repeats = queries.dim(1) / keys.dim(1)
            return queries
                + MLX.repeated(keySum + valueSum, count: repeats, axis: 1).asType(queries.dtype)
        }

        func attendBorrowing(
            source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
        ) -> MLXArray { queries }
    }

    private func attentionConfiguration() throws -> Gemma4TextConfiguration {
        try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gemma4_text", "vocab_size": 64,
                    "hidden_size": 128, "intermediate_size": 256,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "num_global_key_value_heads": 1,
                    "head_dim": 64, "global_head_dim": 128, "attention_k_eq_v": true,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_kv_shared_layers": 0, "sliding_window": 16, "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "sliding_attention": {"rope_theta": 10000.0},
                        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
                    },
                    "final_logit_softcapping": 30.0, "tie_word_embeddings": true,
                    "enable_moe_block": false,
                    "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0
                }
                """.utf8))
    }

    /// One attention layer (sliding: own values, base rotation; global:
    /// K-eq-V, a quarter rotated) hands the paged cache the same queries, keys
    /// and values, and returns the same output, fused and unfused.
    @Test(arguments: [0, 1])
    func attentionKeepsItsBits(layer: Int) throws {
        guard verified else { return }
        let config = try attentionConfiguration()
        let attention = Gemma4Attention(config, layerIdx: layer)
        try randomize(attention, seed: UInt64(12_000 + layer))
        let heads = (attention.nHeads, attention.nKvHeads, attention.effectiveHeadDim)
        #expect(layer == 0 ? heads == (4, 2, 64) : heads == (4, 1, 128))
        #expect(attention.useKeqV == (layer == 1))
        for (batch, tokens) in [(1, 1), (3, 1), (2, 3)] {
            let x = random([batch, tokens, 128], UInt64(12_100 + batch * 10 + tokens), .bfloat16)
            let offsets = MLXArray((0 ..< batch).map { Int32(17 + $0 * 40_003) })
            #expect(attention.fusedPrologueRotation(attention.qProj(x)) != nil)
            func run(_ fused: Bool) -> (MLXArray, [MLXArray]) {
                Gemma4FusedNorms.$prologueOverride.withValue(fused) {
                    let cache = RecordingCache(
                        layerIndex: layer,
                        kind: CBv2LayerKind(
                            attention: .full, headDim: heads.2, kvHeads: heads.1,
                            queryHeads: heads.0),
                        offsets: offsets)
                    let (out, _, _) = attention(x, cache: cache)
                    return (out, cache.recorded)
                }
            }
            let (expected, expectedRecorded) = run(false)
            let (out, recorded) = run(true)
            expectExact(out, expected)
            #expect(recorded.count == 3 && expectedRecorded.count == 3)
            for (a, b) in zip(recorded, expectedRecorded) { expectExact(a, b) }
        }
    }

    // MARK: - Expert routing

    @Test func softmaxFormIsMeasured() throws {
        guard verified else {
            #expect(SoftmaxForm.measured(axis: 8, dtype: .bfloat16) == nil)
            return
        }
        for dtype in [DType.bfloat16, .float16, .float32] {
            for axis in [8, 4, 3, 32, 128] {
                let m = try #require(SoftmaxForm.Calibration.measurement(axis: axis, dtype: dtype))
                #expect(m.preciseMatches || m.fastMatches, "\(axis) \(dtype)")
                if m.formsDiffer {
                    #expect(m.preciseMatches != m.fastMatches, "\(axis) \(dtype)")
                }
                #expect(SoftmaxForm.measured(axis: axis, dtype: dtype) != nil, "\(axis) \(dtype)")
            }
        }
    }

    /// The unfused routing after the projection.
    private func unfusedRoute(_ scores: MLXArray, _ scale: MLXArray, topK: Int)
        -> (MLXArray, MLXArray)
    {
        let kth = scores.dim(-1) - topK
        let indices = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        var weights = MLX.takeAlong(scores, indices, axis: -1)
        weights = MLX.softmax(weights, axis: -1, precise: true)
        return (indices, weights * scale[indices])
    }

    @Test func routeWritesTheUnfusedBits() throws {
        guard verified else { return }
        for dtype in [DType.bfloat16, .float16, .float32] {
            for (experts, topK) in [(128, 8), (16, 4), (64, 6), (32, 32)] {
                let routing = try #require(
                    Gemma4FusedNorms.routing(experts: experts, topK: topK, dtype: dtype))
                let scale = weight(experts, UInt64(experts * 13 + topK), dtype)
                for rows in [1, 3, 8, 64] {
                    let seed = UInt64(experts * 1_000 + topK * 10 + rows)
                    let random = random([rows, 1, experts], seed, dtype, scale: 2)
                    // Few distinct values: ties inside and at the edge of the top k.
                    let tied =
                        (MLXRandom.randInt(
                            0 ..< 6, [rows, 1, experts], key: MLXRandom.key(seed + 1)
                        )
                        .asType(.float32) / 4).asType(dtype)
                    let equal = MLXArray.zeros([rows, 1, experts], dtype: dtype) + 1.5
                    // Infinities, and a NaN, in the first columns.
                    func column(_ value: Float) -> MLXArray {
                        MLXArray.full([rows, 1, 1], values: MLXArray(value)).asType(dtype)
                    }
                    let rest = random[0..., 0..., 3...]
                    let special = concatenated(
                        [column(.infinity), column(-.infinity), random[0..., 0..., 2 ..< 3], rest],
                        axis: -1)
                    let nan = concatenated(
                        [random[0..., 0..., 0 ..< 2], column(.nan), rest], axis: -1)
                    let cases = [random, tied, equal, special, nan]
                    for (index, scores) in cases.enumerated() {
                        eval(scores)
                        let (expectedIndices, expectedWeights) = unfusedRoute(
                            scores, scale, topK: topK)
                        let fused = Gemma4FusedNorms.route(
                            scores: scores, perExpertScale: scale, topK: topK, routing: routing)
                        #expect(
                            bytes(fused.indices) == bytes(expectedIndices),
                            "indices \(dtype) \(experts)/\(topK) rows \(rows) case \(index)")
                        #expect(
                            bytes(fused.weights) == bytes(expectedWeights),
                            "weights \(dtype) \(experts)/\(topK) rows \(rows) case \(index)")
                        #expect(fused.indices.shape == expectedIndices.shape)
                        #expect(fused.weights.dtype == expectedWeights.dtype)
                    }
                }
            }
        }
    }

    // MARK: - GeGLU

    /// `ExactGeGLU` matches `compiledGeGLU` on every bf16 and f16 gate value
    /// (its own check), and on decode-shaped expert and dense rows.
    @Test func exactGeGLUWritesTheCompiledBits() throws {
        guard verified else {
            #expect(!ExactGeGLU.prepare(.bfloat16))
            return
        }
        // Outside a caller's scope the compiled ops keep their contract.
        let outside = MLXArray.zeros([1, 1, 8, 704], dtype: .bfloat16)
        #expect(!ExactGeGLU.supports(outside, outside))
        try ExactGeGLU.$inScope.withValue(true) { try checkExactGeGLU() }
    }

    private func checkExactGeGLU() throws {
        for dtype in [DType.bfloat16, .float16] {
            #expect(ExactGeGLU.prepare(dtype), "\(dtype)")
            for shape in [[1, 1, 8, 704], [3, 1, 8, 704], [2, 4, 2112], [1, 1, 2112]] {
                let seed = UInt64(shape.reduce(1, *))
                let gate = random(shape, seed, dtype, scale: 3)
                let up = random(shape, seed + 1, dtype, scale: 2)
                try #require(ExactGeGLU.supports(gate, up))
                expectExact(ExactGeGLU.apply(gate, up), compiledGeGLU(gate, up))
            }
        }
        // Prompt-sized calls keep the compiled ops.
        let wide = MLXArray.zeros([64, 1, 8, 704], dtype: .bfloat16)
        #expect(!ExactGeGLU.supports(wide, wide))
    }

    // MARK: - Experts' weighted sum

    /// The feed-forward tail given the experts' rows and weights writes what it
    /// writes given `weightedExpertSum(rows, weights)`.
    @Test func feedforwardTailSumsExpertRowsAsWeightedExpertSum() throws {
        guard verified else { return }
        let eps = Self.eps
        for dtype in [DType.bfloat16, .float16, .float32] {
            for width in [2816, 128, 1000] {
                try #require(Gemma4FusedNorms.supports(width: width, dtype: dtype))
                for (rows, experts) in [(1, 8), (3, 8), (8, 4), (2, 6)] {
                    let seed = UInt64(width * 13 + rows * 3 + experts)
                    var expertRows = random([rows, experts, width], seed, dtype, scale: 2)
                    // Negative zeros: MLX's reduction adds +0 to each product.
                    let mask =
                        MLXRandom.uniform(
                            0 ..< 1, [rows, experts, width], key: MLXRandom.key(seed + 9)) .< 0.05
                    expertRows = MLX.where(mask, MLXArray(-0.0).asType(dtype), expertRows)
                    let weights =
                        (MLXRandom.uniform(0 ..< 1, [rows, experts], key: MLXRandom.key(seed + 1)))
                        .asType(dtype)
                    let dense = random([rows, 1, width], seed + 2, dtype, scale: 2)
                    let residual = random([rows, 1, width], seed + 3, dtype, scale: 4)
                    let (postDense, postExperts, post, next) = (
                        weight(width, seed + 4, dtype), weight(width, seed + 5, dtype),
                        weight(width, seed + 6, dtype), weight(width, seed + 7, dtype)
                    )
                    let scalar = MLXArray([Float(0.875)]).asType(dtype)
                    let summed = weightedExpertSum(expertRows, weights).reshaped(rows, 1, width)
                    let expected = Gemma4FusedNorms.feedforwardTail(
                        dense: dense, experts: (summed, postDense, postExperts), residual: residual,
                        post: post, layerScalar: scalar, next: next, eps: eps)
                    let fused = Gemma4FusedNorms.feedforwardTail(
                        dense: dense, experts: (expertRows, postDense, postExperts),
                        expertWeights: weights, residual: residual, post: post, layerScalar: scalar,
                        next: next, eps: eps)
                    expectExact(fused.out, expected.out)
                    expectExact(try #require(fused.next), try #require(expected.next))
                }
            }
        }
    }
}
