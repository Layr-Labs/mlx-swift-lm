import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite struct GLM5NextTextModelTests {
        private func makeModel(layers: Int = 4, zeroBranches: Bool = false) throws
            -> GLM5NextTextModel
        {
            let config = try GLM5NextInventoryFixture.configuration(layers: layers)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            // Resolver generates explicit deterministic tensors, not model random
            // initialization. MTP/vision metadata remains accounted but is never
            // requested by this text owner.
            return try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                name in
                let entry = try #require(inventory.entries[name])
                #expect(entry.part == .text)
                let shape = entry.tensor.shape
                let count = shape.reduce(1, *)
                let values: [Float]
                if name == "model.language_model.embed_tokens.weight" {
                    values = (0 ..< count).map {
                        Float((($0 / 4 + 1) * ($0 % 4 + 2)) % 11 - 5) * 0.1
                    }
                } else if name == "lm_head.weight" {
                    values = (0 ..< count).map { Float(($0 / 4 + $0 % 4) % 7 - 3) * 0.07 }
                } else if name.hasSuffix("_scale") || name.contains("norm.weight") {
                    values = Array(repeating: 1, count: count)
                } else if name.hasSuffix("A_log") {
                    values = Array(repeating: -0.7, count: count)
                } else if name.hasSuffix("dt_bias") {
                    values = Array(repeating: -0.2, count: count)
                } else if zeroBranches || name.hasSuffix("_base") || name.hasSuffix(".bias") {
                    values = Array(repeating: 0, count: count)
                } else {
                    let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                    values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
                }
                return MLXArray(values, shape)
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 0.0001) {
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                    < tolerance)
        }

        @Test func all45LayersProduceExpectedLogitsAndFinalNormHidden() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel(layers: 45, zeroBranches: true)
                let cache = model.newTextCache()
                let result = try model.forward(tokens: MLXArray([Int32(7)], [1, 1]), cache: cache)
                #expect(model.model.layers.count == 45)
                #expect(model.model.layers[0].sparseAttention == nil)
                #expect(model.model.layers[3].sparseAttention?.modelLayerIndex == 3)
                #expect(cache.layers.count == 45 && cache.offset == 1)
                #expect(cache.layers.filter { $0 is GLM5NextKDACache }.count == 34)
                #expect(cache.layers.filter { $0 is GLM5NextDSACache }.count == 11)
                #expect(result.hidden.shape == [1, 1, 4] && result.logits.shape == [1, 1, 32])
                // Independent scalar oracle: zero sublayers leave only the two
                // uniform Sinkhorn residual mixers per layer, then final RMS.
                let eps = 1e-6
                var coefficient = (0.25 + eps) / (4 * (0.25 + eps) + eps)
                for _ in 1 ..< 20 {
                    coefficient /= 4 * coefficient + eps
                    coefficient /= 4 * coefficient + eps
                }
                let factor = pow(4 * coefficient, 90)
                let embedded = (0 ..< 4).map { (index: Int) -> Double in
                    Double(((8 * (index + 2)) % 11) - 5) * 0.1 * factor
                }
                let norm = sqrt(embedded.map { $0 * $0 }.reduce(0, +) / 4 + 1e-5)
                let hidden = embedded.map { $0 / norm }
                let expectedLogits = (0 ..< 32).map { row in
                    (0 ..< 4).reduce(0.0) { $0 + hidden[$1] * Double((row + $1) % 7 - 3) * 0.07 }
                }
                for (value, expected) in zip(result.hidden.asArray(Float.self), hidden) {
                    #expect(abs(Double(value) - expected) < 0.0001)
                }
                for (value, expected) in zip(result.logits.asArray(Float.self), expectedLogits) {
                    #expect(abs(Double(value) - expected) < 0.0001)
                }
                close(result.logits, model.lmHead(result.hidden))
            }
        }

        @Test func nonzeroFullForwardMatchesChunkedAndTokenDecoding() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let tokens = MLXArray([Int32(2), 5, 3, 9, 7, 1, 4], [1, 7])
                let full = try model.forward(tokens: tokens, cache: nil)
                let chunkCache = model.newTextCache()
                var chunks: [MLXArray] = []
                var start = 0
                for end in [3, 4, 5, 7] {
                    chunks.append(
                        try model.forward(tokens: tokens[0..., start ..< end], cache: chunkCache)
                            .logits)
                    start = end
                }
                close(concatenated(chunks, axis: 1), full.logits)
                #expect(chunkCache.layers.allSatisfy { $0.offset == 7 })
                let tokenCache = model.newTextCache()
                let singles = try (0 ..< 7).map {
                    try model.forward(tokens: tokens[0..., $0 ..< $0 + 1], cache: tokenCache).hidden
                }
                close(concatenated(singles, axis: 1), full.hidden)
                #expect(full.logits.asArray(Float.self).allSatisfy { $0.isFinite })
                let viaEmbeddings = try model.forward(embeddings: model.embed(tokens), cache: nil)
                close(viaEmbeddings.logits, full.logits)
            }
        }

        @Test func paddedBatchPreservesLogicalKDAHistoryAndSparsePoolOrigin() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let batch = MLXArray([Int32(2), 3, 4, 5, 6, 0, 0, 2, 3, 4], [2, 5])
                let mask = MLXArray(
                    [true, true, true, true, true, false, false, true, true, true], [2, 5])
                let cache = model.newTextCache()
                let result = try model.forward(tokens: batch, cache: cache, validity: mask)
                let single = try model.forward(
                    tokens: MLXArray([Int32(2), 3, 4], [1, 3]), cache: nil)
                close(result.logits[1 ..< 2, 2...], single.logits)
                #expect(result.hidden[1, ..<2].asArray(Float.self).allSatisfy { $0 == 0 })
                let allPadding = try model.forward(
                    tokens: MLXArray.zeros([1, 5], dtype: .int32),
                    cache: model.newTextCache(), validity: MLXArray.zeros([1, 5], dtype: .bool))
                #expect(allPadding.hidden.asArray(Float.self).allSatisfy { $0 == 0 })
                #expect(allPadding.logits.asArray(Float.self).allSatisfy { $0.isFinite })
            }
        }

        @Test func cacheCopyRollbackAndFailedRewindAreAtomic() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let cache = model.newTextCache()
                _ = try model.forward(tokens: MLXArray([Int32(2), 3, 4], [1, 3]), cache: cache)
                let saved = cache.copy()
                let savedArrays = saved.layers.map {
                    $0.state.map { $0.asType(.float32).asArray(Float.self) }
                }
                let suffix = MLXArray([Int32(5), 6], [1, 2])
                let first = try model.forward(tokens: suffix, cache: cache)
                #expect(saved.offset == 3 && cache.offset == 5)
                for (layer, arrays) in zip(saved.layers, savedArrays) {
                    #expect(layer.state.map { $0.asType(.float32).asArray(Float.self) } == arrays)
                }
                #expect(throws: GLM5NextLoadError.self) { try cache.truncate(to: 4) }
                #expect(cache.layers.allSatisfy { $0.offset == 5 })
                try cache.truncate(to: 3)
                let replay = try model.forward(tokens: suffix, cache: cache)
                close(first.logits, replay.logits)
                let fromCopy = try model.forward(tokens: suffix, cache: saved)
                close(first.hidden, fromCopy.hidden)
                let standard = model.newCache(parameters: nil)
                close(
                    model(MLXArray([Int32(2)], [1, 1]), cache: standard),
                    try model.forward(tokens: MLXArray([Int32(2)], [1, 1]), cache: nil).logits)
            }
        }

        @Test func payloadMismatchAndForeignCachesFailExplicitly() throws {
            try Device.withDefaultDevice(.cpu) {
                let config = try GLM5NextInventoryFixture.configuration()
                let headers = try GLM5NextWeightInventory.expectedTensors(
                    configuration: config, storage: .float32, shard: "tiny.safetensors")
                let inventory = try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: headers, storage: .float32)
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                        _ in MLXArray.zeros([1])
                    }
                }
                let model = try makeModel()
                let foreign = GLM5NextTextCache(layers: (0 ..< 4).map { _ in KVCacheSimple() })
                #expect(throws: GLM5NextLoadError.self) {
                    try model.forward(tokens: MLXArray([Int32(1)], [1, 1]), cache: foreign)
                }
                #expect(foreign.layers.allSatisfy { $0.offset == 0 })
            }
        }

        @Test func packedQ4LoadsOriginalWordsAndRequiresBoundedMaterialization() throws {
            try Device.withDefaultDevice(.cpu) {
                let config = try GLM5NextInventoryFixture.configuration(q4: true)
                let headers = try GLM5NextWeightInventory.expectedTensors(
                    configuration: config, storage: .vontraQ4, shard: "tiny-q4.safetensors")
                let inventory = try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: headers, storage: .vontraQ4)
                var prematureReads = 0
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                        _ in
                        prematureReads += 1
                        return MLXArray.zeros([1])
                    }
                }
                #expect(prematureReads == 0)
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextModel(
                        configuration: config.textConfig, inventory: inventory,
                        tensor: { _ in
                            prematureReads += 1
                            return MLXArray.zeros([1])
                        }, materializeExpertStack: { _ in })
                }
                #expect(prematureReads == 0)
                var failedAliases: [String] = []
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextModel(
                        configuration: config.textConfig, inventory: inventory,
                        tensor: { _ in MLXArray.zeros([1]) }, materializeExpertStack: { _ in },
                        releaseSourceTensors: { failedAliases += $0 })
                }
                #expect(failedAliases == ["model.language_model.embed_tokens.weight"])
                var materializedStacks = 0
                var sourceCache: [String: MLXArray] = [:]
                var peakSourceAliases = 0
                var released: Set<String> = []
                let model = try GLM5NextTextModel(
                    configuration: config.textConfig, inventory: inventory,
                    tensor: { name in
                        let tensor = try #require(inventory.entries[name]?.tensor)
                        let result: MLXArray
                        if tensor.dtype == "U32" {
                            result = MLXArray(
                                [UInt32](repeating: 0x1111_1111, count: tensor.shape.reduce(1, *)),
                                tensor.shape)
                        } else {
                            let value: Float
                            if name.hasSuffix(".scales") {
                                value = 0.001
                            } else if name.contains("norm.weight") || name.hasSuffix("_scale") {
                                value = 1
                            } else {
                                value = 0
                            }
                            result = (MLXArray.ones(tensor.shape) * value).asType(
                                tensor.dtype == "F32" ? .float32 : .bfloat16)
                        }
                        sourceCache[name] = result
                        peakSourceAliases = max(peakSourceAliases, sourceCache.count)
                        return result
                    },
                    materializeExpertStack: { stack in
                        #expect(sourceCache.count == config.textConfig.nRoutedExperts)
                        materializedStacks += 1
                        eval(stack)
                    },
                    releaseSourceTensors: { names in
                        for name in names {
                            sourceCache.removeValue(forKey: name)
                            released.insert(name)
                        }
                    })
                // One routed layer: three projections times weight/scales/offsets.
                #expect(materializedStacks == 9)
                #expect(sourceCache.isEmpty)
                #expect(peakSourceAliases == config.textConfig.nRoutedExperts)
                #expect(released == Set(inventory.entries(in: .text).map { $0.tensor.name }))
                #expect(model.lmHead.weight.dtype == .uint32)
                #expect(model.lmHead.weight.asArray(UInt32.self).allSatisfy { $0 == 0x1111_1111 })
                let output = try model.forward(
                    tokens: MLXArray([Int32(2), 3, 4, 5, 6], [1, 5]), cache: model.newTextCache())
                #expect(output.logits.shape == [1, 5, 32])
                #expect(
                    output.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
                #expect(output.logits.asType(.float32).max().item(Float.self) > 0)
            }
        }

        @Test func candidatePublishesOnlyAfterSuccessfulFenceAndRejectsStaleCommits() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let cache = model.newTextCache()
                let token = MLXArray([Int32(2)], [1, 1])
                let first = try model.prepareForward(tokens: token, cache: cache)
                #expect(cache.layers.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
                try first.commit { roots in
                    #expect(cache.offset == 0)
                    #expect(roots.count > 2)
                    eval(roots)
                }
                #expect(cache.layers.allSatisfy { $0.offset == 1 })
                #expect(throws: GLM5NextLoadError.self) { try first.commit { _ in } }

                let stale = try model.prepareForward(tokens: token, cache: cache)
                let newer = try model.prepareForward(tokens: token, cache: cache)
                try newer.commit { eval($0) }
                var staleFenceRan = false
                #expect(throws: GLM5NextLoadError.self) {
                    try stale.commit { _ in staleFenceRan = true }
                }
                #expect(!staleFenceRan && cache.offset == 2)

                enum Refused: Error { case fence }
                let before = cache.layers.map {
                    $0.state.map { $0.asType(.float32).asArray(Float.self) }
                }
                let failed = try model.prepareForward(tokens: token, cache: cache)
                #expect(throws: Refused.self) { try failed.commit { _ in throw Refused.fence } }
                #expect(cache.offset == 2)
                #expect(
                    cache.layers.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
                        == before)
                #expect(throws: GLM5NextLoadError.self) { try failed.commit { _ in } }
            }
        }

        @Test func candidateDetectsRewindToSameEmptyFrontier() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let cache = model.newTextCache()
                let token = MLXArray([Int32(3)], [1, 1])
                let stale = try model.prepareForward(tokens: token, cache: cache)
                _ = try model.forward(tokens: token, cache: cache)
                try cache.truncate(to: 0)
                #expect(cache.offset == 0)
                #expect(throws: GLM5NextLoadError.self) { try stale.commit { _ in } }
                let discarded = try model.prepareForward(tokens: token, cache: cache)
                discarded.discard()
                #expect(throws: GLM5NextLoadError.self) { try discarded.commit { _ in } }
                #expect(cache.layers.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
            }
        }
    }
}
