import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextOfficialReferenceTests {
        @Test func fullTargetAndEachLayerMatchOfficialCPU() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let test = try fixture.caseObject("text_full")
                let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                let valid = MLXArray.ones(tokens.shape, dtype: .bool)
                let cache = model.newTextCache()
                var streams = broadcast(
                    model.embed(tokens).expandedDimensions(axis: 2),
                    to: [
                        tokens.dim(0), tokens.dim(1), model.configuration.hcMult,
                        model.configuration.hiddenSize,
                    ])
                let outputs = try #require(test["layer_outputs"] as? [String])
                guard outputs.count == model.model.layers.count else {
                    throw GLM5NextReferenceFixture.Failure("target layer count differs from oracle")
                }
                for (index, layer) in model.model.layers.enumerated() {
                    streams = try layer.forward(
                        streams, cache: cache.layers[index], validity: valid)
                    // Stop at the first divergent layer rather than reporting only final logits.
                    try fixture.compare(streams, key: outputs[index])
                }
                let hidden = model.model.norm(streams.mean(axis: 2))
                try fixture.compare(hidden, key: fixture.key(test, "final_hidden"))
                try fixture.compare(model.lmHead(hidden), key: fixture.key(test, "logits"))
                try compareCache(cache, description: test, fixture: fixture)

                let whole = try model.forward(tokens: tokens, cache: nil)
                try fixture.compare(whole.hidden, key: fixture.key(test, "final_hidden"))
                try fixture.compare(whole.logits, key: fixture.key(test, "logits"))
            }
        }

        @Test func tokenAndChunkContinuationMatchRecordedOfficialOutputs() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let full = try fixture.caseObject("text_full")
                let tokenwise = try fixture.caseObject("text_tokenwise")
                let tokens = try fixture.tensor(fixture.key(full, "input_ids"))
                let steps = try #require(tokenwise["steps"] as? [[String: Any]])
                #expect(steps.count == tokens.dim(1))
                let cache = model.newTextCache()
                var pieces: [MLXArray] = []
                for (index, step) in steps.enumerated() {
                    let result = try model.forward(
                        tokens: tokens[0..., index ..< index + 1], cache: cache)
                    try fixture.compare(result.logits, key: fixture.key(step, "logits"))
                    try fixture.compare(result.hidden, key: fixture.key(step, "final_hidden"))
                    try compareCache(cache, description: step, fixture: fixture)
                    pieces.append(result.logits)
                }
                try fixture.compare(
                    concatenated(pieces, axis: 1), key: fixture.key(tokenwise, "combined_logits"))
                let chunks = model.newTextCache()
                var start = 0
                var logits: [MLXArray] = []
                for end in [3, 4, 5, 11, tokens.dim(1)] {
                    let result = try model.forward(
                        tokens: tokens[0..., start ..< end], cache: chunks)
                    logits.append(result.logits)
                    start = end
                }
                try fixture.compare(concatenated(logits, axis: 1), key: fixture.key(full, "logits"))
                try compareCache(chunks, description: full, fixture: fixture)
            }
        }

        @Test func leftPaddingComparesOnlyOfficialValidTokenRows() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let test = try fixture.caseObject("text_left_padding")
                let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                let valid = try fixture.tensor(fixture.key(test, "attention_mask")).asType(.bool)
                let cache = model.newTextCache()
                var streams = broadcast(
                    model.embed(tokens).expandedDimensions(axis: 2),
                    to: [
                        tokens.dim(0), tokens.dim(1), model.configuration.hcMult,
                        model.configuration.hiddenSize,
                    ])
                let outputs = try #require(test["layer_outputs"] as? [String])
                for (index, layer) in model.model.layers.enumerated() {
                    streams = try layer.forward(
                        streams, cache: cache.layers[index], validity: valid)
                    try fixture.compare(streams, key: outputs[index], mask: valid)
                }
                let result = try model.forward(tokens: tokens, cache: nil, validity: valid)
                // HF records meaningful/nonzero padded outputs too. The native
                // API zeroes its padded final hidden rows: do not invent a padded
                // equality contract or alter the official expected tensors.
                try fixture.compare(
                    result.hidden, key: fixture.key(test, "final_hidden"), mask: valid)
                try fixture.compare(result.logits, key: fixture.key(test, "logits"), mask: valid)
            }
        }

        @Test func isolatedKDAProjectionsOutputsAndFinalStatesMatchOfficial() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let full = try fixture.caseObject("text_full")
                let descriptions = try #require(full["cache"] as? [String: Any])
                let conv = try #require(descriptions["conv_states"] as? [String])
                let recurrent = try #require(descriptions["kda_states"] as? [String])
                for (slot, layerIndex) in model.configuration.linearAttention.kdaLayers.enumerated()
                {
                    let layer = try #require(
                        model.model.layers[layerIndex].attention as? GLM5NextKDASublayer)
                    let prefix = "text.full.model.language_model.layers.\(layerIndex)."
                    let input = try fixture.tensor(prefix + "input_layernorm.call0.output")
                    try fixture.compare(
                        layer.q(input), key: prefix + "self_attn.q_proj.call0.output")
                    try fixture.compare(
                        layer.k(input), key: prefix + "self_attn.k_proj.call0.output")
                    try fixture.compare(
                        layer.v(input), key: prefix + "self_attn.v_proj.call0.output")
                    try fixture.compare(
                        layer.fB(layer.fA(input)),
                        key: prefix + "self_attn.forget_gate.f_b_proj.call0.output")
                    let cache = GLM5NextKDACache()
                    let result = try layer.forward(
                        input, cache: cache,
                        validity: MLXArray.ones(Array(input.shape.prefix(2)), dtype: .bool))
                    try fixture.compare(result, key: prefix + "self_attn.o_proj.call0.output")
                    try fixture.compare(cache.state[0], key: conv[slot])
                    try fixture.compare(cache.state[1], key: recurrent[slot])
                }
            }
        }

        @Test func learnedPoolSelectionAndRouterMatchOfficialSelectedSets() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let attention = try #require(model.model.layers[3].sparseAttention)
                let test = try fixture.caseObject("dsa_left_padding")
                let inputs = try #require(test["inputs"] as? [String: Any])
                let hidden = try fixture.tensor(fixture.key(inputs, "hidden_states"))
                let residual = try fixture.tensor(fixture.key(inputs, "q_resid"))
                let valid = try fixture.tensor(fixture.key(inputs, "attention_mask")).asType(.bool)
                let projected = attention.indexer.project(hidden, residual: residual)
                let prefix =
                    "components.left_padded_indexer.model.language_model.layers.3.self_attn.indexer."
                try fixture.compare(projected.keys, key: prefix + "k_norm.call0.output")
                try fixture.compare(
                    projected.queries.reshaped(hidden.dim(0), hidden.dim(1), -1),
                    key: prefix + "wq_b.call0.output")
                try fixture.compare(projected.weights, key: prefix + "weights_proj.call0.output")
                let selected = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: projected.queries,
                    indexKeys: projected.keys, poolGates: projected.gates,
                    poolPositionBias: attention.indexer.positionBias,
                    headWeights: projected.weights,
                    keyValidity: valid, queryValidity: valid, queryStart: 0,
                    tokenBudget: model.configuration.indexTopK)
                try fixture.compareSelections(selected.indices, key: fixture.key(test, "indices"))

                let full = try fixture.caseObject("text_full")
                let routerDescription = try #require(full["router"] as? [String: Any])
                let logitKey = try fixture.key(routerDescription, "logits")
                let inputKey = logitKey.replacingOccurrences(of: ".output.0", with: ".input.0")
                let moe = try #require(model.model.layers[3].mlp as? GLM5NextMoE)
                let route = moe.gate(try fixture.tensor(inputKey))
                try fixture.compare(route.logits, key: logitKey)
                let expectedIDs = try fixture.tensor(fixture.key(routerDescription, "indices"))
                    .asType(.int64).asArray(Int64.self)
                let expectedWeights = try fixture.tensor(fixture.key(routerDescription, "weights"))
                    .asArray(Float.self)
                let actualIDs = route.indices.asType(.int64).asArray(Int64.self)
                let actualWeights = route.weights.asArray(Float.self)
                guard actualWeights.allSatisfy(\.isFinite), expectedWeights.allSatisfy(\.isFinite)
                else {
                    throw GLM5NextReferenceFixture.Failure("router produced nonfinite weights")
                }
                let width = model.configuration.numExpertsPerTok
                guard actualIDs.count == expectedIDs.count,
                    actualWeights.count == expectedWeights.count
                else {
                    throw GLM5NextReferenceFixture.Failure(
                        "router output shape differs from oracle")
                }
                for row in 0 ..< (actualIDs.count / width) {
                    let range = row * width ..< (row + 1) * width
                    guard Set(actualIDs[range]) == Set(expectedIDs[range]) else {
                        throw GLM5NextReferenceFixture.Failure(
                            "router selected set differs at row \(row)")
                    }
                    for index in range {
                        let match = try #require(
                            range.first { expectedIDs[$0] == actualIDs[index] })
                        guard
                            abs(actualWeights[index] - expectedWeights[match]) <= 3e-5 + 3e-5
                                * abs(expectedWeights[match])
                        else {
                            throw GLM5NextReferenceFixture.Failure(
                                "router weight differs at row \(row), expert \(actualIDs[index])")
                        }
                    }
                }
            }
        }

        private func compareCache(
            _ cache: GLM5NextTextCache, description: [String: Any],
            fixture: GLM5NextReferenceFixture
        ) throws {
            let expected = try #require(description["cache"] as? [String: Any])
            let conv = try #require(expected["conv_states"] as? [String])
            let recurrent = try #require(expected["kda_states"] as? [String])
            let linearLayers = fixture.configuration.textConfig.linearAttention.kdaLayers
            guard conv.count == linearLayers.count, recurrent.count == linearLayers.count else {
                throw GLM5NextReferenceFixture.Failure("KDA cache count differs from oracle")
            }
            for (slot, index) in linearLayers.enumerated() {
                let state = cache.layers[index].state
                guard state.count == 2 else {
                    throw GLM5NextReferenceFixture.Failure("missing KDA state layer \(index)")
                }
                try fixture.compare(state[0], key: conv[slot])
                try fixture.compare(state[1], key: recurrent[slot])
            }
            let dsa = try #require(cache.layers[3] as? GLM5NextDSACache)
            let latent = try #require(dsa.latents)
            let latentKey = try fixture.key(expected, "latent_kv")
            let referenceLatent = try fixture.tensor(latentKey)
            guard referenceLatent.ndim == 4 && referenceLatent.dim(1) == 1 else {
                throw GLM5NextReferenceFixture.Failure("unexpected official latent layout")
            }
            try fixture.compare(
                latent, expected: referenceLatent.squeezed(axis: 1), label: latentKey)
            let keys = try #require(dsa.indexKeys)
            let gates = try #require(dsa.poolGates)
            let valid = try #require(dsa.validity)
            let packed = concatenated(
                [keys, gates, valid.asType(keys.dtype).expandedDimensions(axis: -1)], axis: -1)
            try fixture.compare(
                packed, key: fixture.key(expected, "packed_indexer_keys_gates_validity"))
        }
    }
}
