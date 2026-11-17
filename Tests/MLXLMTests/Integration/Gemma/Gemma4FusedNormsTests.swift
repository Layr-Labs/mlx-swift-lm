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
    /// (`RMSNormMeanForm.measured`) on a family whose kernels were checked
    /// (`Gemma4FusedNorms.verifiedFamilies`); elsewhere the unfused ops run.
    private var verified: Bool {
        Device.defaultDevice().deviceType == .gpu
            && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
            && Gemma4FusedNorms.verifiedFamilies.contains(AffineQuantizedVectors.family())
    }

    @Test func onlyCheckedFamiliesFuse() {
        #expect(Gemma4FusedNorms.verifiedFamilies == [17])
        if !Gemma4FusedNorms.verifiedFamilies.contains(AffineQuantizedVectors.family()) {
            #expect(!Gemma4FusedNorms.supports(width: 2816, dtype: .bfloat16))
        }
    }

    /// The kernels' own check runs on the rows that separate the two mean
    /// forms, and it sees a kernel compiled with the other form.
    @Test func readinessCheckSeesTheWrongMeanForm() throws {
        guard verified else { return }
        for dtype in [DType.bfloat16, .float16] {
            let width = 2816
            let form = try #require(RMSNormMeanForm.measured(width: width, dtype: dtype))
            #expect(Gemma4FusedNorms.Readiness.check(width: width, dtype: dtype) == nil)
            let (separating, _) = RMSNormMeanForm.Calibration.rows(width: width, dtype: dtype)
            try #require(!separating.isEmpty)
            let x = MLXArray(separating.flatMap { $0 }, [separating.count, 1, width])
                .asType(dtype)
            let other: RMSNormMeanForm = form == .divide ? .fusedMultiplyAdd : .divide
            let caught = try Gemma4FusedNorms.$meanFormOverride.withValue(other) {
                try Gemma4FusedNorms.Readiness.mismatch(x, x, width: width, dtype: dtype)
            }
            #expect(caught != nil, "\(dtype)")
            // The feed-forward tails see it on their own too (dense input on the rows).
            let one = MLXArray.ones([width], dtype: dtype)
            let zero = MLXArray.zeros(x.shape, dtype: dtype)
            let scalar = MLXArray([Float(1)]).asType(dtype)
            for experts in [true, false] {
                func tail() -> MLXArray {
                    Gemma4FusedNorms.feedforwardTail(
                        dense: x, experts: experts ? (zero, one, one) : nil, residual: zero,
                        post: one, layerScalar: scalar, next: nil, eps: 1e-6
                    ).out
                }
                let right = bytes(tail())
                let wrong = Gemma4FusedNorms.$meanFormOverride.withValue(other) { bytes(tail()) }
                #expect(right != wrong, "feed-forward tail, experts \(experts), \(dtype)")
            }
            #expect(
                try Gemma4FusedNorms.Readiness.mismatch(x, x, width: width, dtype: dtype) == nil)
        }
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

    private func configuration(moe: Bool, layers: Int, hidden: Int = 128) throws
        -> Gemma4TextConfiguration
    {
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
                    "hidden_size": \(hidden), "intermediate_size": 256,
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

    /// Gemma 4 26B's width with its quantized projections (8-bit dense and
    /// router, 4-bit experts): the fused path, with the compiled middle where
    /// the shape takes it, keeps the bits.
    @Test func quantizedLayerAtModelWidthKeepsItsBits() throws {
        guard verified else { return }
        let layer = Gemma4DecoderLayer(
            try configuration(moe: true, layers: 1, hidden: 2816), layerIdx: 0)
        try randomize(layer, seed: 12_000)
        quantize(model: layer) { path, _ in
            path.hasPrefix("experts.")
                ? (groupSize: 64, bits: 4, mode: .affine) : (groupSize: 64, bits: 8, mode: .affine)
        }
        layer.train(false)
        eval(layer)
        #expect(layer.feedforwardCompilationState() != nil)
        for (rows, width) in [(1, 1), (3, 1), (1, 2), (2, 2)] {
            let x = random(
                [rows, width, 2816], UInt64(12_100 + rows * 10 + width), .bfloat16, scale: 4)
            try #require(layer.supportsFusedNorms(x, outputTailRows: nil))
            let (expected, _, _) = Gemma4FusedNorms.$enabledOverride.withValue(false) {
                layer(x)
            }
            let next = weight(2816, 12_200, .bfloat16)
            let (out, _, _, fusedNext) = layer.forwardFusedNorms(
                x, normalizedInput: nil, nextNorm: next, mask: nil, cache: nil, sharedKV: nil,
                positionOffset: nil, v2SharedSource: nil, useLastQueryPrefill: false,
                isExpertPrefill: false)
            expectExact(out, expected)
            expectExact(
                try #require(fusedNext), MLXFast.rmsNorm(expected, weight: next, eps: Self.eps))
        }
    }

    /// An attention output in another dtype (a float32 output projection)
    /// keeps the unfused tail and its promotion.
    @Test func anotherOutputDtypeKeepsTheUnfusedOps() throws {
        guard verified else { return }
        let layer = Gemma4DecoderLayer(try configuration(moe: true, layers: 1), layerIdx: 0)
        try randomize(layer, seed: 13_000)
        let o = try #require(
            layer.namedModules().first { $0.0 == "self_attn.o_proj" }?.1 as? Linear)
        o.update(parameters: .unflattened(["weight": o.weight.asType(.float32)]))
        eval(layer)
        let x = random([2, 1, 128], 13_100, .bfloat16, scale: 4)
        try #require(layer.supportsFusedNorms(x, outputTailRows: nil))
        let (expected, _, _) = Gemma4FusedNorms.$enabledOverride.withValue(false) { layer(x) }
        #expect(expected.dtype == .float32)
        let next = weight(128, 13_200, .bfloat16)
        let (out, _, _, fusedNext) = layer.forwardFusedNorms(
            x, normalizedInput: nil, nextNorm: next, mask: nil, cache: nil, sharedKV: nil,
            positionOffset: nil, v2SharedSource: nil, useLastQueryPrefill: false,
            isExpertPrefill: false)
        expectExact(out, expected)
        expectExact(
            try #require(fusedNext), MLXFast.rmsNorm(expected, weight: next, eps: Self.eps))
    }

    /// A layer whose output is promoted (a float32 output projection) hands
    /// the next layer float32 rows: that layer takes the unfused ops and its
    /// own input norm, and the logits keep their bits.
    @Test func promotedRowsKeepTheUnfusedOpsInTheNextLayer() throws {
        guard verified else { return }
        try #require(Gemma4FusedNorms.supports(width: 128, dtype: .bfloat16))
        let model = Gemma4TextModel(try configuration(moe: true, layers: 2))
        try randomize(model, seed: 16_000)
        let o = try #require(
            model.namedModules().first { $0.0.hasSuffix("layers.0.self_attn.o_proj") }?.1
                as? Linear)
        o.update(parameters: .unflattened(["weight": o.weight.asType(.float32)]))
        eval(model)
        let tokens = MLXArray((0 ..< 4).map { Int32(($0 * 9 + 2) % 64) }).reshaped(2, 2)
        let unfused = Gemma4FusedNorms.$enabledOverride.withValue(false) {
            bytes(model(tokens, cache: nil))
        }
        let fused = Gemma4FusedNorms.$enabledOverride.withValue(true) {
            bytes(model(tokens, cache: nil))
        }
        #expect(fused == unfused)
    }

    /// Two rows of verification width (2 tokens each) after a two-row prompt.
    @Test(arguments: [true, false])
    func verificationRowsKeepTheirBits(moe: Bool) throws {
        guard verified else { return }
        try #require(Gemma4FusedNorms.supports(width: 128, dtype: .bfloat16))
        let model = Gemma4TextModel(try configuration(moe: moe, layers: 4))
        try randomize(model, seed: moe ? 14_000 : 15_000)
        let prompt = MLXArray((0 ..< 24).map { Int32(($0 * 5 + 1) % 64) }).reshaped(2, 12)
        func run(_ fused: Bool) -> [Data] {
            Gemma4FusedNorms.$enabledOverride.withValue(fused) {
                let cache = model.newCache(parameters: nil)
                var outputs = [bytes(model(prompt, cache: cache))]
                for step in 0 ..< 4 {
                    let tokens = MLXArray((0 ..< 4).map { Int32((step * 13 + $0 * 7 + 2) % 64) })
                        .reshaped(2, 2)
                    outputs.append(bytes(model(tokens, cache: cache)))
                }
                return outputs
            }
        }
        let unfused = run(false)
        let fused = run(true)
        #expect(fused.count == unfused.count)
        for (a, b) in zip(fused, unfused) { #expect(a == b) }
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
}
