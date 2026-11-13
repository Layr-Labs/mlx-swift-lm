// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Tiny Muse Glimmer towers with deterministic host-generated weights. No
/// checkpoint or downloaded data.
enum MuseGlimmerFixture {
    /// Deterministic values in [-1, 1).
    static func hostTensor(_ shape: [Int], seed: UInt64) -> MLXArray {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
        let values: [Float] = (0 ..< shape.reduce(1, *)).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int((state >> 33) & 0xffff) - 32_768) / 32_768
        }
        return MLXArray(values).reshaped(shape)
    }

    /// The MLXVLM tiny text shape: head dim 8, window 8, two full layers.
    static var eagerText: [String: Any] {
        [
            "hidden_size": 32, "num_hidden_layers": 6, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
            "sliding_window": 8, "vocab_size": 64,
            "rope_parameters": ["rope_theta": 500000.0],
        ]
    }

    /// The production attention shape (32 query heads over 2 KV heads of 128)
    /// on a small trunk; the native paged kernel needs head dim >= 64.
    static func pagedText(window: Int, layers: Int = 6) -> [String: Any] {
        [
            "hidden_size": 64, "num_hidden_layers": layers, "num_attention_heads": 32,
            "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 128,
            "sliding_window": window, "vocab_size": 96,
            "rope_parameters": ["rope_theta": 500000.0],
        ]
    }

    static func configuration(_ text: [String: Any]) throws -> MuseGlimmerTextModelConfiguration {
        try JSONDecoder().decode(
            MuseGlimmerTextModelConfiguration.self,
            from: JSONSerialization.data(withJSONObject: text))
    }

    /// Weights for every parameter of `module`, by sorted name. Block norm
    /// scales are centered at zero; the final norm's scale is near one.
    static func weights(for module: Module, dtype: DType, seed: UInt64) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        let parameters = module.parameters().flattened().sorted { $0.0 < $1.0 }
        for (index, (name, value)) in parameters.enumerated() {
            let noise = hostTensor(value.shape, seed: seed &+ UInt64(index + 1))
            let scaled: MLXArray
            if value.ndim <= 1 {
                scaled = name.contains("layernorm") ? noise * 0.1 : 1 + noise * 0.1
            } else {
                scaled = noise * (1.7 / Float(value.shape.last ?? 1).squareRoot())
            }
            weights[name] = scaled.asType(dtype)
        }
        return weights
    }

    static func model(
        _ text: [String: Any], dtype: DType = .bfloat16, seed: UInt64 = 20_261_113
    ) throws -> MuseGlimmerTextModel {
        let model = MuseGlimmerTextModel(try configuration(text))
        try model.update(
            parameters: .unflattened(weights(for: model, dtype: dtype, seed: seed)),
            verify: [.all])
        eval(model)
        return model
    }

    /// The MLXVLM model and the native tower holding the same text weights.
    static func upstreamPair(dtype: DType) throws -> (MuseGlimmer, MuseGlimmerTextModel) {
        let root: [String: Any] = [
            "model_type": "muse_glimmer", "image_token_id": 7, "video_token_id": 8,
            "out_hidden_size": 64, "projector_hidden_size": 32, "text_config": eagerText,
            "vision_config": [
                "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
                "num_hidden_layers": 2, "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
                "pos_emb_height": 32, "pos_emb_width": 32,
            ],
        ]
        let upstream = MuseGlimmer(
            try JSONDecoder().decode(
                MuseGlimmerConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root)))
        let all = weights(for: upstream, dtype: dtype, seed: 7)
        try upstream.update(parameters: .unflattened(all), verify: [.all])
        eval(upstream)
        let native = MuseGlimmerTextModel(try configuration(eagerText))
        let text = all.filter { $0.key.hasPrefix("language_model.") }
        try native.update(
            parameters: .unflattened(native.sanitize(weights: text)), verify: [.all])
        eval(native)
        return (upstream, native)
    }

    static func tokens(_ count: Int, vocabulary: Int, salt: Int = 0) -> [Int32] {
        (0 ..< count).map { Int32(($0 * 7 + 3 + salt * 5 + ($0 * $0) % 11) % vocabulary) }
    }

    static func sameBits(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape
            && a.asData(access: .copy).data == b.asData(access: .copy).data
    }

    /// Prefills `prompt` in chunks on the eager caches, then decodes greedily.
    static func eagerGreedy(
        _ model: MuseGlimmerTextModel, prompt: [Int32], chunk: Int, steps: Int
    ) throws -> [Int] {
        try eagerGreedyWithMargins(model, prompt: prompt, chunk: chunk, steps: steps).tokens
    }

    /// The greedy tokens with each step's gap between the two largest logits.
    static func eagerGreedyWithMargins(
        _ model: MuseGlimmerTextModel, prompt: [Int32], chunk: Int, steps: Int
    ) throws -> (tokens: [Int], margins: [Float]) {
        let cache = try model.newCache(parameters: nil)
        var logits = MLXArray.zeros([1])
        for start in stride(from: 0, to: prompt.count, by: chunk) {
            let piece = Array(prompt[start ..< min(start + chunk, prompt.count)])
            logits = model(MLXArray(piece).reshaped(1, -1), cache: cache)
            eval([logits] + cache.flatMap { $0.innerState() })
        }
        var generated = [Int]()
        var margins = [Float]()
        for _ in 0 ..< steps {
            let row = logits[0, -1].asType(.float32)
            let top = sorted(row)[(-2)...].asArray(Float.self)
            margins.append(top[1] - top[0])
            let next = argMax(row, axis: -1).item(Int.self)
            generated.append(next)
            logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
            eval([logits] + cache.flatMap { $0.innerState() })
        }
        return (generated, margins)
    }

    /// Token equality for BF16 towers, whose logits sit on a coarse grid: the
    /// two paths may part only where the reference's top two logits are within
    /// `tie` of each other. Float32 towers are compared with `==` instead.
    static func expectSameUntilNearTie(
        _ actual: [Int], _ expected: (tokens: [Int], margins: [Float]), tie: Float = 0.13,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.tokens.count, sourceLocation: sourceLocation)
        guard
            let first = zip(actual, expected.tokens).enumerated().first(where: {
                $0.element.0 != $0.element.1
            })?.offset
        else { return }
        #expect(
            expected.margins[first] <= tie,
            "tokens part at step \(first) with reference margin \(expected.margins[first])",
            sourceLocation: sourceLocation)
    }
}
