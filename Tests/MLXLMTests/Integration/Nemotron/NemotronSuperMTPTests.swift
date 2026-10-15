import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

struct NemotronSuperMTPTests {
    private struct Tensor { let dtype: String; let shape: [Int]; let data: Data }
    private func configuration() -> NemotronHConfiguration {
        NemotronHConfiguration(vocabSize: 32, hiddenSize: 16, numHiddenLayers: 2,
            numAttentionHeads: 1, numKeyValueHeads: 1, mambaNumHeads: 1, mambaHeadDim: 16,
            ssmStateSize: 4, convKernel: 4, nGroups: 1, intermediateSize: 16,
            moeIntermediateSize: 16, moeSharedExpertIntermediateSize: 16,
            nRoutedExperts: 2, numExpertsPerTok: 1, hybridOverridePattern: "*E", headDim: 16,
            nSharedExperts: 1, numNextnPredictLayers: 1, mtpLayersBlockType: ["attention", "moe"],
            moeLatentSize: 16)
    }
    private func scalar(_ value: Float) -> Tensor {
        var bits = value.bitPattern.littleEndian
        return Tensor(dtype: "F32", shape: [1], data: withUnsafeBytes(of: &bits) { Data($0) })
    }
    private func tensors(dtype: DType = .float32) throws -> [String: Tensor] {
        let model = try NemotronSuperMTPModule(configuration: configuration(),
            calibration: .init(keyScale: 2, valueScale: 3), dtype: .float32)
        let prefix = "language_model.mtp."
        var result: [String: Tensor] = [:]
        for (name, value) in model.parameters().flattened() {
            if name.contains(".shared_experts.") || name.contains(".switch_mlp.") { continue }
            let storedDType: DType = name.hasSuffix("e_score_correction_bias") ? .float32 : dtype
            let bytes = value.asType(storedDType).asData()
            result[prefix + name] = Tensor(dtype: storedDType == .float32 ? "F32" : "BF16",
                shape: bytes.shape, data: bytes.data)
        }
        for projection in ["up_proj", "down_proj"] {
            let stems = [prefix + "layers.1.mixer.shared_experts." + projection]
                + (0..<2).map { prefix + "layers.1.mixer.experts.\($0)." + projection }
            for stem in stems {
                result[stem + ".weight"] = Tensor(dtype: "U8", shape: [16, 8], data: Data(repeating: 0, count: 128))
                result[stem + ".weight_scale"] = Tensor(dtype: "F8_E4M3", shape: [16, 1], data: Data(repeating: 56, count: 16))
                result[stem + ".input_scale"] = scalar(1)
                result[stem + ".weight_scale_2"] = scalar(1)
            }
        }
        result[prefix + "layers.0.mixer.k_proj.k_scale"] = scalar(2)
        result[prefix + "layers.0.mixer.v_proj.v_scale"] = scalar(3)
        return result
    }
    private func withSource(_ tensors: [String: Tensor], body: (ModelOptCheckpointSource) throws -> Void) throws {
        var files: [String: ModelOptTensorFile] = [:], map: [String: String] = [:]
        var urls: [URL] = []
        defer { for url in urls { try? FileManager.default.removeItem(at: url) } }
        for shard in 0..<2 {
            let name = "fixture-\(shard).safetensors"
            var header: [String: Any] = [:], payload = Data()
            for (index, key) in tensors.keys.sorted().enumerated() where index % 2 == shard {
                let tensor = tensors[key]!, start = payload.count
                payload.append(tensor.data)
                header[key] = ["dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [start, payload.count]]
                map[key] = name
            }
            let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(headerData.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(headerData); data.append(payload)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            urls.append(url); try data.write(to: url)
            files[name] = try ModelOptTensorFile(url: url,
                sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
        try body(ModelOptCheckpointSource(indexData: index,
            indexSHA256: SHA256.hash(data: index).map { String(format: "%02x", $0) }.joined(), files: files))
    }
    @Test(arguments: [DType.float32, .bfloat16])
    func loadsMixedMTPAndMatchesFusionResidualEquation(dtype: DType) throws {
        try withSource(tensors(dtype: dtype)) { source in
            let scope = NativeConstructionScope()
            try scope.withPhase(.filesystemLoad) {
                try scope.capture(StreamOrDevice.default.stream); try scope.capture(StreamOrDevice.cpu.stream)
                try scope.willSubmit()
                let model: NemotronSuperMTPModule
                if dtype == .bfloat16 {
                    // Exercise the public BF16-only loader rather than the
                    // internal fixture overload used by the FP32 reference.
                    model = try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                        maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20,
                        construction: scope, cachePolicy: .unquantized)
                } else {
                    model = try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                        maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20,
                        construction: scope, cachePolicy: .unquantized, dtype: dtype)
                }
                let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                #expect(parameters["layers.0.eh_proj.weight"]?.dtype == dtype)
                #expect(parameters["layers.1.mixer.gate.e_score_correction_bias"]?.dtype == .float32)
                #expect(model.calibration.keyScale == 2 && model.calibration.valueScale == 3)
                #expect(source.tensors.count == 40)
                let hidden = MLXArray.ones([1, 1, 16], dtype: dtype)
                let embedding = (MLXArray.ones([1, 1, 16]) * Float(2)).asType(dtype)
                let attention = try #require(model.layers[0] as? NemotronH35MTPAttentionBlock)
                let experts = try #require(model.layers[1] as? NemotronH35MTPExpertBlock)
                let fused = attention.projection(concatenated([attention.embeddingNorm(embedding), attention.hiddenNorm(hidden)], axis: -1))
                let expectedCache = KVCacheSimple()
                let attended = fused + attention.mixer(attention.norm(fused),
                    attentionMask: createAttentionMask(h: fused, cache: expectedCache), ssmMask: nil, cache: expectedCache)
                let expected = experts.finalNorm(attended + experts.mixer(experts.norm(attended)))
                let actual = try model.forward(postNormHidden: hidden, nextEmbedding: embedding,
                    cache: KVCacheSimple(), policy: .unquantized, maximumCacheTokens: 256)
                try scope.retain(actual); try scope.retain(expected)
                try MLX.withError { errors in eval(actual, expected); try errors.check() }
                #expect(actual.dtype == dtype && expected.dtype == dtype)
                #expect(isFinite(actual).all().item(Bool.self))
                #expect(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self) < 1e-5)
                #expect(throws: NemotronSuperMTPModule.Failure.self) {
                    try model.forward(postNormHidden: hidden, nextEmbedding: embedding, cache: KVCacheSimple(), policy: .fp8, maximumCacheTokens: 256)
                }
            }
        }
    }
    @Test func fp8HeadMatchesExplicitUnitScaleAttentionAndReplays() throws {
        try withSource(tensors(dtype: .bfloat16)) { source in
            let scope = NativeConstructionScope()
            try scope.withPhase(.filesystemLoad) {
                try scope.capture(StreamOrDevice.default.stream); try scope.capture(StreamOrDevice.cpu.stream)
                try scope.willSubmit()
                let model = try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                    maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20,
                    construction: scope, cachePolicy: .fp8)
                try scope.retainOwner(model)
                let attention = try #require(model.layers[0] as? NemotronH35MTPAttentionBlock)
                let experts = try #require(model.layers[1] as? NemotronH35MTPExpertBlock)
                let cache = try NemotronSuperFP8KVCache(heads: 1, headDimension: 16, capacityTokens: 2)
                try scope.retainOwner(cache)
                var keys: [MLXArray] = [], values: [MLXArray] = []
                var second: MLXArray?
                for step in 0..<2 {
                    let hidden = MLXArray((0..<16).map { Float($0 + step + 1) }).reshaped(1, 1, 16).asType(.bfloat16)
                    let embedding = MLXArray((0..<16).map { Float(17 - $0 + step) }).reshaped(1, 1, 16).asType(.bfloat16)
                    let fused = attention.projection(concatenated([attention.embeddingNorm(embedding), attention.hiddenNorm(hidden)], axis: -1))
                    let normalized = attention.norm(fused)
                    let q = attention.mixer.wq(normalized).reshaped(1, 1, 1, 16).transposed(0, 2, 1, 3)
                    let k = attention.mixer.wk(normalized).reshaped(1, 1, 1, 16).transposed(0, 2, 1, 3)
                    let v = attention.mixer.wv(normalized).reshaped(1, 1, 1, 16).transposed(0, 2, 1, 3)
                    keys.append(try NemotronSuperFP8KVCodec.decode(NemotronSuperFP8KVCodec.encode(k)))
                    values.append(try NemotronSuperFP8KVCodec.decode(NemotronSuperFP8KVCodec.encode(v)))
                    let attended = MLXFast.scaledDotProductAttention(queries: q, keys: concatenated(keys, axis: 2),
                        values: concatenated(values, axis: 2), scale: 0.25, mask: .none)
                    let residual = fused + attention.mixer.wo(attended.transposed(0, 2, 1, 3).reshaped(1, 1, 16))
                    let expected = experts.finalNorm(residual + experts.mixer(experts.norm(residual)))
                    let actual = try model.forward(postNormHidden: hidden, nextEmbedding: embedding,
                        cache: cache, policy: .fp8, maximumCacheTokens: 2)
                    try scope.retain(actual); try scope.retain(expected)
                    try MLX.withError { errors in eval(actual, expected); try errors.check() }
                    #expect(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self) < 0.001)
                    #expect(cache.offset == step + 1 && cache.innerState().allSatisfy { $0.dtype == .uint8 })
                    if step == 1 {
                        second = actual
                        #expect(cache.trim(1) == 1)
                        let replay = try model.forward(postNormHidden: hidden, nextEmbedding: embedding,
                            cache: cache, policy: .fp8, maximumCacheTokens: 2)
                        try scope.retain(replay); eval(replay)
                        #expect(arrayEqual(replay, actual).item(Bool.self))
                    }
                }
                #expect(second != nil)
                #expect(throws: NemotronSuperMTPModule.Failure.self) {
                    try model.forward(postNormHidden: MLXArray.ones([1, 1, 16], dtype: .bfloat16),
                        nextEmbedding: MLXArray.ones([1, 1, 16], dtype: .bfloat16),
                        cache: cache, policy: .fp8, maximumCacheTokens: 2)
                }
                #expect(cache.offset == 2)
            }
        }
    }
    @Test func rejectsMalformedCacheBeforeAttention() throws {
        let model = try NemotronSuperMTPModule(configuration: configuration(),
            calibration: .init(keyScale: 2, valueScale: 3), dtype: .float32)
        let hidden = MLXArray.ones([1, 1, 16])
        for shape in [[2, 1, 1, 16], [1, 2, 1, 16], [1, 1, 1, 8], [1, 1, 16]] {
            let cache = KVCacheSimple()
            cache.state = [MLXArray.zeros(shape), MLXArray.zeros(shape)]
            #expect(throws: NemotronSuperMTPModule.Failure.self) {
                try model.forward(postNormHidden: hidden, nextEmbedding: hidden, cache: cache, policy: .unquantized, maximumCacheTokens: 256)
            }
        }
        for offset in [-1, 2] {
            let cache = KVCacheSimple()
            cache.state = [MLXArray.zeros([1, 1, 1, 16]), MLXArray.zeros([1, 1, 1, 16])]
            cache.offset = offset
            #expect(throws: NemotronSuperMTPModule.Failure.self) {
                try model.forward(postNormHidden: hidden, nextEmbedding: hidden, cache: cache, policy: .unquantized, maximumCacheTokens: 256)
            }
        }
        let invalidStep = KVCacheSimple()
        invalidStep.step = 0
        #expect(throws: NemotronSuperMTPModule.Failure.self) {
            try model.forward(postNormHidden: hidden, nextEmbedding: hidden, cache: invalidStep, policy: .unquantized, maximumCacheTokens: 256)
        }
        let oversized = KVCacheSimple()
        oversized.step = Int.max / 2
        #expect(throws: NemotronSuperMTPModule.Failure.self) {
            try model.forward(postNormHidden: hidden, nextEmbedding: hidden,
                cache: oversized, policy: .unquantized, maximumCacheTokens: 256)
        }
        #expect(oversized.offset == 0 && oversized.innerState().isEmpty)
        let growth = KVCacheSimple()
        growth.step = 256
        growth.state = [MLXArray.zeros([1, 1, 1, 16]), MLXArray.zeros([1, 1, 1, 16])]
        growth.offset = 1
        #expect(throws: NemotronSuperMTPModule.Failure.self) {
            try model.forward(postNormHidden: hidden, nextEmbedding: hidden,
                cache: growth, policy: .unquantized, maximumCacheTokens: 256)
        }
        #expect(growth.offset == 1)
    }
    @Test func rejectsMissingMetadataForeignKeysAndUnsupportedPolicy() throws {
        var missing = try tensors()
        missing.removeValue(forKey: "language_model.mtp.layers.0.mixer.k_proj.k_scale")
        var extra = try tensors()
        extra["language_model.mtp.unexpected.weight"] = scalar(1)
        for tensors in [missing, extra] {
            try withSource(tensors) { source in
                let scope = NativeConstructionScope()
                _ = try scope.withPhase(.filesystemLoad) {
                    #expect(throws: (any Error).self) {
                        try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                            maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20, construction: scope, cachePolicy: .unquantized, dtype: .float32)
                    }
                }
            }
        }
        try withSource(tensors()) { source in
            let scope = NativeConstructionScope()
            #expect(throws: NemotronSuperMTPModule.Failure.self) {
                try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                    maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20, construction: scope, cachePolicy: .unquantized, dtype: .float32)
            }
            try scope.withPhase(.filesystemLoad) {
                #expect(throws: NemotronSuperMTPModule.Failure.self) {
                    try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                        maximumBytes: 1 << 20, constructionAllowanceBytes: 1 << 20, construction: scope, cachePolicy: .paged, dtype: .float32)
                }
                #expect(throws: NemotronSuperMTPModule.Failure.self) {
                    try NemotronSuperMTPModule.load(configuration: configuration(), source: source,
                        maximumBytes: 1, constructionAllowanceBytes: 1 << 20, construction: scope, cachePolicy: .unquantized, dtype: .float32)
                }
            }
        }
    }
}
