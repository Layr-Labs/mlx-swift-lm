// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// A tiny Q4 target whose KDA layers have the serving per-head geometry: head
/// width 128, low-rank f/g gates of width 128 with affine 4-bit group-64 f_b
/// and g_b rows (so the fused step applies them in-kernel), and a fused input
/// projection of whole 8-row tiles over whole 512-value blocks. Deterministic
/// synthetic weights; not real-checkpoint evidence.
struct GLM5NextKDALayerFixture {
    let configuration: GLM5NextConfiguration
    let inventory: GLM5NextWeightInventory
    var text: GLM5NextTextConfiguration { configuration.textConfig }
    static let prefix = "model.language_model.layers.0.self_attn"

    init(heads: Int = 8, hidden: Int = 512) throws {
        let base = try GLM5NextInventoryFixture.configuration(layers: 4, q4: true)
        var root = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        var text = try #require(root["text_config"] as? [String: Any])
        var linear = try #require(text["linear_attn_config"] as? [String: Any])
        linear["num_heads"] = heads
        linear["head_dim"] = 128
        text["linear_attn_config"] = linear
        text["hidden_size"] = hidden
        root["text_config"] = text
        var vision = try #require(root["vision_config"] as? [String: Any])
        vision["out_hidden_size"] = hidden
        root["vision_config"] = vision
        configuration = try JSONDecoder().decode(
            GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        inventory = try GLM5NextWeightInventory.validate(
            configuration: configuration,
            tensors: GLM5NextWeightInventory.expectedTensors(
                configuration: configuration, storage: .vontraQ4, shard: "kda.safetensors"),
            storage: .vontraQ4)
    }

    /// 2^-5 (1 + odd/128): exact in BF16.
    private static func scale(_ index: Int) -> Float {
        (1 + Float(2 * (index % 64) + 1) / 128) / 32
    }

    func tensor(_ name: String) throws -> MLXArray {
        let header = try #require(inventory.entries[name]).tensor
        let count = header.shape.reduce(1, *)
        let salt = name.utf8.reduce(0) { ($0 * 31 + Int($1)) % 1009 }
        if header.dtype == "U32" {
            return MLXArray(
                (0 ..< count).map {
                    UInt32(
                        truncatingIfNeeded: (UInt64($0 + salt * 7919) &* 0x9E37_79B9_7F4A_7C15)
                            >> 29)
                }, header.shape)
        }
        let values: [Float] = (0 ..< count).map { index in
            let wave = Float(sin(Double(index) * 0.731 + Double(salt)))
            if name.hasSuffix(".scales") { return Self.scale(index) }
            if name.hasSuffix(".biases") { return -8 * Self.scale(index) }
            if name.contains("norm.weight") { return 1 + 0.25 * wave }
            if name.hasSuffix("A_log") { return -0.7 + 0.1 * wave }
            if name.hasSuffix("dt_bias") { return -0.2 + 0.1 * wave }
            return 0.25 * wave
        }
        let array = MLXArray(values, header.shape)
        return header.dtype == "BF16" ? array.asType(.bfloat16) : array
    }

    /// Layer 0's KDA sublayer under the given load-time members.
    func sublayer(
        decodeStep: GLM5NextExecutionPolicy.LinearDecodeStep = .fusedEveryStep,
        projections: GLM5NextExecutionPolicy.Projections = .fusedInputs
    ) throws -> GLM5NextKDASublayer {
        try GLM5NextKDASublayer(
            text, prefix: Self.prefix,
            loader: GLM5NextTensorLoader(inventory, tensor, releaseSourceTensors: { _ in }),
            linearAttention: .nativeRecurrenceChunkedPrefill, decodeStep: decodeStep,
            projections: projections)
    }

    /// Deterministic `[1, tokens, hidden]` activations.
    func input(tokens: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, tokens, text.hiddenSize], key: MLXRandom.key(seed))
            .asType(.bfloat16)
    }
}

/// A Q4 linear that counts how often its matmul is built.
final class GLM5NextCountingQuantizedLinear: QuantizedLinear {
    private(set) var calls = 0

    init(_ other: QuantizedLinear) {
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales, biases: other.biases,
            groupSize: other.groupSize, bits: other.bits, mode: other.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        calls += 1
        return super.callAsFunction(x)
    }
}
