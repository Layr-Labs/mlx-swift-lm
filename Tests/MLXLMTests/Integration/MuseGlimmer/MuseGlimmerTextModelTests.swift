// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The native tower against the MLXVLM text tower on the same weights. Both
/// run the eager caches; every comparison is of raw bytes.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerTextModelTests {
    @Test(arguments: [DType.bfloat16, DType.float32])
    func eagerForwardMatchesTheUpstreamTowerBitForBit(dtype: DType) throws {
        let (upstream, native) = try MuseGlimmerFixture.upstreamPair(dtype: dtype)
        // 21 tokens against a window of 8: the sliding mask is in force.
        let prompt = MuseGlimmerFixture.tokens(21, vocabulary: 64)
        let whole = MLXArray(prompt).reshaped(1, -1)
        let expected = upstream(whole, cache: nil)
        let actual = native(whole, cache: nil)
        #expect(actual.dtype == dtype && actual.shape == [1, 21, 64])
        #expect(all(isFinite(actual.asType(.float32))).item(Bool.self))
        #expect(MuseGlimmerFixture.sameBits(actual, expected))

        // Chunked prefill across the window boundary, then greedy decode
        // until both sliding caches have wrapped more than once.
        let upstreamCache = try upstream.newCache(parameters: nil)
        let nativeCache = try native.newCache(parameters: nil)
        var start = 0
        var next = 0
        for count in [5, 9, 7] {
            let chunk = MLXArray(Array(prompt[start ..< start + count])).reshaped(1, -1)
            let a = upstream(chunk, cache: upstreamCache)
            let b = native(chunk, cache: nativeCache)
            #expect(MuseGlimmerFixture.sameBits(b, a))
            next = argMax(b[0, -1], axis: -1).item(Int.self)
            start += count
        }
        var generated = [Int]()
        for _ in 0 ..< 24 {
            generated.append(next)
            let token = MLXArray([Int32(next)]).reshaped(1, 1)
            let a = upstream(token, cache: upstreamCache)
            let b = native(token, cache: nativeCache)
            #expect(MuseGlimmerFixture.sameBits(b, a))
            #expect(
                argMax(a[0, -1], axis: -1).item(Int.self)
                    == argMax(b[0, -1], axis: -1).item(Int.self))
            next = argMax(b[0, -1], axis: -1).item(Int.self)
        }
        // The fixture is only a test if decode moves through the vocabulary.
        #expect(Set(generated).count > 3)
    }

    @Test func bothCheckpointLayoutsMapOntoTheTowerAndVisionIsNotDropped() throws {
        let native = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: .float32)
        let own = Dictionary(uniqueKeysWithValues: native.parameters().flattened())
        var transformers = [String: MLXArray]()
        var mlx = [String: MLXArray]()
        for (key, value) in own {
            if key.hasPrefix("model.") {
                transformers["model.language_model." + key.dropFirst("model.".count)] = value
                mlx["language_model." + key] = value
            } else {
                transformers[key] = value
                mlx["language_model." + key] = value
            }
        }
        for layout in [transformers, mlx] {
            let target = MuseGlimmerTextModel(native.configuration)
            let mapped = target.sanitize(weights: layout)
            #expect(Set(mapped.keys) == Set(own.keys))
            try target.update(parameters: .unflattened(mapped), verify: [.all])
        }
        // A vision tensor keeps its key, so a strict update refuses it.
        var withVision = transformers
        withVision["model.vision_projection.weight"] = MLXArray.zeros([32, 32])
        let target = MuseGlimmerTextModel(native.configuration)
        let mapped = target.sanitize(weights: withVision)
        #expect(mapped["model.vision_projection.weight"] != nil)
        #expect(throws: (any Error).self) {
            try target.update(parameters: .unflattened(mapped), verify: [.all])
        }
    }

    @Test func parameterTreeMatchesThePublishedTextKeys() throws {
        let native = MuseGlimmerTextModel(
            try MuseGlimmerFixture.configuration(MuseGlimmerFixture.eagerText))
        let keys = Set(native.parameters().flattened().map(\.0))
        var expected: Set<String> = [
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ]
        for layer in 0 ..< 6 {
            let prefix = "model.layers.\(layer)."
            for norm in [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] { expected.insert(prefix + norm + ".weight") }
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] {
                expected.insert(prefix + "self_attn." + projection + ".weight")
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert(prefix + "mlp." + projection + ".weight")
            }
        }
        #expect(keys == expected)
    }
}
