import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite(.serialized, .tags(.integration))
struct FlintTests {
    struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
    }
    struct Fixture: Decodable {
        let config: FlintConfiguration
        let weights: [String: Tensor]
        let ids: [[Int32]]
        let logits: [[[Float]]]
    }
    func fixture() throws -> Fixture {
        let url = try #require(
            Bundle.module.url(forResource: "flint-oracle", withExtension: "json"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }
    func model(_ fixture: Fixture) throws -> FlintModel {
        let model = FlintModel(fixture.config)
        try model.installFlintWeights(
            fixture.weights.mapValues { MLXArray($0.values).reshaped($0.shape) })
        return model
    }
    @Test func matchesIndependentNumpyOracle() throws {
        let f = try fixture()
        let m = try model(f)
        let ids = MLXArray(f.ids.flatMap { $0 }).reshaped(f.ids.count, -1)
        let actual = m(ids, cache: nil)
        let expected = MLXArray(f.logits.flatMap { $0.flatMap { $0 } }).reshaped(actual.shape)
        let error = abs(actual - expected)
        #expect(all(error .<= (1e-3 + 1e-3 * abs(expected))).item(Bool.self))
        #expect(actual.dtype == .float32)
    }
    @Test func hybridCacheMatchesAcrossWindowAndChunkBoundaries() throws {
        let m = try model(fixture())
        let tokens = (MLXArray(0 ..< 1031) * 7 + 3) % 32
        let expected = m(tokens.reshaped(1, -1), cache: nil)
        for size in [1, 127, 512, 513, 514] {
            let cache = m.newCache(parameters: nil)
            var outputs = [MLXArray]()
            for start in stride(from: 0, to: 1031, by: size) {
                let out = m(tokens[start ..< min(start + size, 1031)].reshaped(1, -1), cache: cache)
                eval(out)
                outputs.append(out)
            }
            let actual = concatenated(outputs, axis: 1)
            #expect(all(abs(actual - expected) .<= (1e-3 + 1e-3 * abs(expected))).item(Bool.self))
            #expect(cache.allSatisfy { $0.offset == 1031 })
            // RotatingKVCache keeps window + current chunk - 1 during prefill.
            #expect(cache[1].state[0].dim(2) <= 513 + size - 1)
            let next = m(MLXArray([Int32(9)]).reshaped(1, 1), cache: cache)
            let reference = m(
                concatenated([tokens, MLXArray([Int32(9)])]).reshaped(1, -1), cache: nil)[
                    0..., (-1)..., 0...]
            #expect(all(abs(next - reference) .<= (1e-3 + 1e-3 * abs(reference))).item(Bool.self))
            #expect(cache[1].state[0].dim(2) == 513)
        }
    }
    @Test func narrowedPrefillPreservesKVAndContinuationAcrossSlidingWrap() throws {
        let m = try model(fixture())
        let ids = ((MLXArray(0 ..< 1031) * 7 + 3) % 32).reshaped(1, -1)
        for size in [127, 512, 513, 514, 1031] {
            let full = m.newCache(parameters: nil)
            let narrow = m.newCache(parameters: nil)
            for start in stride(from: 0, to: 1031, by: size) {
                let end = min(start + size, 1031)
                let chunk = ids[0..., start ..< end]
                let expected = m(chunk, cache: full)[0..., -1, 0...]
                let requirement: CBv2PrefillRequirement =
                    end == 1031
                    ? .lastPositionLogits : .evaluationOnly
                let actual = m.cbv2Prefill(
                    chunk, inputEmbedding: nil, cache: narrow,
                    requirement: requirement)
                eval(expected, actual)
                if end == 1031 {
                    #expect(actual.shape == expected.shape && actual.dtype == .float32)
                    #expect(
                        all(abs(actual - expected) .<= (1e-3 + 1e-3 * abs(expected))).item(
                            Bool.self))
                    #expect(
                        arrayEqual(argMax(actual, axis: -1), argMax(expected, axis: -1)).item(
                            Bool.self))
                }
                for (a, b) in zip(full, narrow) {
                    #expect(a.offset == end && b.offset == end)
                    for (x, y) in zip(a.state, b.state) {
                        #expect(arrayEqual(x, y).item(Bool.self))
                    }
                }
            }
            let next = MLXArray([Int32(9)]).reshaped(1, 1)
            #expect(arrayEqual(m(next, cache: full), m(next, cache: narrow)).item(Bool.self))
        }
    }
    @Test func continuousBatchRowsKeepIndependentPositionsAndSurviveRemoval() throws {
        let m = try model(fixture())
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        let kinds = m.cbv2LayerKinds
        let first: [Int32] = (0 ..< 521).map { (i: Int) -> Int32 in Int32((i * 7 + 3) % 32) }
        let second: [Int32] = (0 ..< 507).map { (i: Int) -> Int32 in Int32((i * 5 + 2) % 32) }
        let histories: [[Int32]] = [first, second]
        let rows = try histories.map {
            try backend.makeSequenceState(
                layerKinds: kinds, promptLength: $0.count, maxLength: 2048)
        }
        let caches = kinds.enumerated().map {
            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
        }
        for lane in 0 ..< 2 {
            for i in caches.indices { caches[i].setRows([rows[lane][i]!]) }
            for start in stride(from: 0, to: histories[lane].count, by: 127) {
                eval(
                    m(
                        MLXArray(
                            Array(
                                histories[lane][start ..< min(start + 127, histories[lane].count)])
                        ).reshaped(1, -1), cache: caches))
            }
        }
        // Reverse membership order: offsets must follow the row, not its batch index.
        for i in caches.indices { caches[i].setRows([rows[1][i]!, rows[0][i]!]) }
        let actual = m(MLXArray([Int32(9), 11]).reshaped(2, 1), cache: caches)
        for (batch, lane) in [1, 0].enumerated() {
            let expected = m(
                MLXArray(histories[lane] + [batch == 0 ? 9 : 11]).reshaped(1, -1), cache: nil)[
                    0..., (-1)..., 0...]
            let row = actual[batch ..< (batch + 1)]
            #expect(all(abs(row - expected) .<= (1e-3 + 1e-3 * abs(expected))).item(Bool.self))
            #expect(
                argMax(row, axis: -1).item(Int.self) == argMax(expected, axis: -1).item(Int.self))
        }
        eval(actual)
        for i in caches.indices { caches[i].setRows([rows[0][i]!]) }
        backend.release(rows[1])
        let continued = m(MLXArray([Int32(5)]).reshaped(1, 1), cache: caches)
        let expected = m(MLXArray(histories[0] + [11, 5]).reshaped(1, -1), cache: nil)[
            0..., (-1)..., 0...]
        #expect(all(abs(continued - expected) .<= (1e-3 + 1e-3 * abs(expected))).item(Bool.self))
        for cache in caches { cache.setRows([]) }
        backend.release(rows[0])
        #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
    }
    @Test func rejectsMissingExtraAndConflictingTensors() throws {
        let f = try fixture()
        let weights = f.weights.mapValues { MLXArray($0.values).reshaped($0.shape) }
        var missing = weights
        missing.removeValue(forKey: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        #expect(throws: FlintCheckpointError.self) {
            try FlintModel(f.config).installFlintWeights(missing)
        }
        var extra = weights
        extra["foreign.weight"] = MLXArray.zeros([2, 2])
        #expect(throws: FlintCheckpointError.self) {
            try FlintModel(f.config).installFlintWeights(extra)
        }
        var alias = weights
        alias["language_model.lm_head.weight"] = weights["lm_head.weight"]
        #expect(throws: FlintCheckpointError.self) {
            try FlintModel(f.config).installFlintWeights(alias)
        }
    }
    @Test func derivesHybridLayerMetadataWithoutAliasingAnotherFamily() throws {
        let f = try fixture()
        #expect(f.config.cbv2LayerKinds.map(\.attention) == [.full, .slidingWindow(513), .full])
        #expect(f.config.cbv2LayerKinds.allSatisfy { $0.hasSinks })
    }

    @Test func rejectsMalformedPerExpertTensorsBeforeStacking() throws {
        let f = try fixture()
        let model = FlintModel(f.config)
        let prefix = "model.layers.1.mlp."
        let destination = prefix + "switch_mlp.gate_proj.weight"
        var raw = f.weights.mapValues { MLXArray($0.values).reshaped($0.shape) }
        let removed = raw.removeValue(forKey: destination)
        let stacked = try #require(removed)
        for expert in 0 ..< f.config.numExperts {
            raw[prefix + "experts.\(expert).gate_proj.weight"] = stacked[expert]
        }
        let normalized = try model.normalizedFlintWeights(raw)
        #expect(normalized[destination]?.shape == stacked.shape)
        let first = prefix + "experts.0.gate_proj.weight"
        for invalid in [MLXArray.zeros([1, 1]), MLXArray.zeros(stacked[0].shape, dtype: .int32)] {
            var malformed = raw
            malformed[first] = invalid
            #expect(throws: FlintCheckpointError.self) {
                try model.normalizedFlintWeights(malformed)
            }
        }
        raw.removeValue(forKey: first)
        #expect(throws: FlintCheckpointError.self) { try model.normalizedFlintWeights(raw) }
    }

    @Test func rejectedExportLeavesModulesUnchangedAndAllowsRetry() throws {
        let url = try #require(
            Bundle.module.url(forResource: "flint-oracle", withExtension: "json"))
        let fixtureJSON = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var config = try #require(fixtureJSON["config"] as? [String: Any])
        for key in [
            "hidden_size", "intermediate_size", "head_dim", "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ] {
            config[key] = 64
        }
        config["mlx_export"] = ["bits": 4, "embedding_bits": 8, "routed_expert_bits": 4]
        config["quantization"] = ["bits": 4, "group_size": 64]
        let configuration = try JSONDecoder().decode(
            FlintConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        let model = FlintModel(configuration)
        let original = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var packed = [String: MLXArray]()
        for (name, parameter) in original {
            let shape = parameter.shape
            if name.hasSuffix(".weight"), shape.count >= 2, !name.hasSuffix(".mlp.gate.weight") {
                let path = String(name.dropLast(".weight".count))
                let bits = path == "model.embed_tokens" ? 8 : 4
                var packedShape = shape
                packedShape[shape.count - 1] /= 32 / bits
                var scaleShape = shape
                scaleShape[shape.count - 1] /= 64
                packed[name] = MLXArray.zeros(packedShape, dtype: .uint32).view(dtype: .uint8)
                packed[path + ".scales"] = MLXArray.ones(scaleShape, dtype: .bfloat16)
                packed[path + ".biases"] = MLXArray.zeros(scaleShape, dtype: .bfloat16)
            } else {
                packed[name] = MLXArray.zeros(shape, dtype: .bfloat16)
            }
        }
        var extra = packed
        extra["foreign.weight"] = MLXArray.zeros([1])
        #expect(throws: FlintCheckpointError.self) { try model.installFlintWeights(extra) }
        for (name, parameter) in model.parameters().flattened() {
            #expect(parameter.shape == original[name]?.shape)
            #expect(parameter.dtype == original[name]?.dtype)
        }
        try model.installFlintWeights(packed)
        let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(loaded["model.embed_tokens.weight"]?.shape == [configuration.vocabularySize, 16])
        #expect(loaded["lm_head.weight"]?.shape == [configuration.vocabularySize, 8])
        #expect(loaded["model.layers.1.mlp.gate.weight"]?.dtype == .bfloat16)
        #expect(loaded["model.layers.1.self_attn.softmax_offset"]?.dtype == .float32)
    }
}
