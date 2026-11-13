// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

@testable import MLXLLM

/// Tiny random drafters and a reference that shares no code with the module:
/// it recomputes a block from every context row with plain matrix products,
/// an explicit mask and hand-written RoPE.
enum MuseGlimmerDFlashFixture {
    static func configuration(
        window: Int = 12, blockSize: Int = 4, layers: Int = 2
    ) throws -> MuseGlimmerDFlashConfiguration {
        try MuseGlimmerDFlashConfiguration(
            hiddenSize: 32, intermediateSize: 64, hiddenLayers: layers, attentionHeads: 4,
            kvHeads: 2, headDimensions: 8, blockSize: blockSize, maskTokenID: 7,
            targetLayerIDs: [0, 2], slidingWindow: window, rmsNormEps: 1e-5, ropeTheta: 500_000)
    }

    /// Seeded values for every parameter; norm weights stay away from zero.
    static func parameters(
        of model: MuseGlimmerDFlashDraftModel, seed: UInt64, dtype: DType
    ) -> [(String, MLXArray)] {
        model.parameters().flattened().sorted { $0.0 < $1.0 }.enumerated().map { index, entry in
            let key = MLXRandom.key(seed &+ UInt64(index) &* 7919)
            let shape = entry.1.shape
            let value =
                shape.count == 1
                ? MLXRandom.uniform(low: Float(0.5), high: Float(1.5), shape, key: key)
                : MLXRandom.normal(shape, key: key) * (1 / Float(shape[1]).squareRoot())
            return (entry.0, value.asType(dtype))
        }
    }

    static func model(
        _ configuration: MuseGlimmerDFlashConfiguration, seed: UInt64 = 11,
        dtype: DType = .float32
    ) throws -> MuseGlimmerDFlashDraftModel {
        let model = try MuseGlimmerDFlashDraftModel(configuration, dtype: dtype)
        try model.update(
            parameters: .unflattened(parameters(of: model, seed: seed, dtype: dtype)),
            verify: .all)
        eval(model)
        return model
    }

    static func features(
        rows: Int, configuration c: MuseGlimmerDFlashConfiguration, seed: UInt64
    ) -> MLXArray {
        let value = MLXRandom.normal([rows, c.contextWidth], key: MLXRandom.key(seed))
        eval(value)
        return value
    }

    static func embeddings(
        configuration c: MuseGlimmerDFlashConfiguration, seed: UInt64
    ) -> MLXArray {
        let value = MLXRandom.normal([c.blockSize, c.hiddenSize], key: MLXRandom.key(seed))
        eval(value)
        return value
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

enum MuseGlimmerDFlashReference {
    static func rms(_ x: MLXArray, _ weight: MLXArray, _ eps: Float) -> MLXArray {
        let value = x.asType(.float32)
        let meanSquare = mean(value * value, axis: -1, keepDims: true)
        return weight.asType(.float32) * (value / sqrt(meanSquare + eps))
    }

    /// Half-split RoPE of `[heads, rows, d]` at explicit absolute positions.
    static func rotate(_ x: MLXArray, positions: [Int], theta: Float) -> MLXArray {
        let d = x.dim(-1)
        let half = d / 2
        let frequencies = (0 ..< half).map { powf(theta, -Float(2 * $0) / Float(d)) }
        let angles =
            MLXArray(positions.map { Float($0) }).reshaped([positions.count, 1])
            * MLXArray(frequencies).reshaped([1, half])
        let (c, s) = (cos(angles), sin(angles))
        let first = x[.ellipsis, ..<half]
        let second = x[.ellipsis, half...]
        return concatenated([first * c - second * s, second * c + first * s], axis: -1)
    }

    /// The block's final hidden states `[blockSize, hiddenSize]`, from every
    /// context feature row at its absolute position. `strictWindow` models
    /// the one-row-shorter window of other implementations.
    static func block(
        model: MuseGlimmerDFlashDraftModel, features: MLXArray?, positions: [Int],
        embeddings: MLXArray, anchor: Int, strictWindow: Bool = false
    ) -> MLXArray {
        let c = model.configuration
        let p = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        func w(_ name: String) -> MLXArray { p[name]!.asType(.float32) }
        func linear(_ x: MLXArray, _ name: String) -> MLXArray {
            matmul(x, w(name + ".weight").T)
        }
        let (block, d, eps) = (c.blockSize, c.headDimensions, c.rmsNormEps)
        let rows = positions.count
        precondition((features?.dim(0) ?? 0) == rows)
        let context = features.map {
            rms(linear($0.asType(.float32), "encoder.fc"), w("encoder.output_norm_enc.weight"), eps)
        }
        let blockPositions = (0 ..< block).map { anchor + $0 }
        let keyPositions = positions + blockPositions
        var bias: [Float] = []
        for query in blockPositions {
            for key in keyPositions {
                let distance = abs(query - key)
                let visible =
                    strictWindow ? distance < c.slidingWindow : distance <= c.slidingWindow
                bias.append(visible ? 0 : -Float.infinity)
            }
        }
        let additive = MLXArray(bias, [1, block, rows + block])
        let groups = c.attentionHeads / c.kvHeads
        let scale = 1 / Float(d).squareRoot()
        var hidden = embeddings.asType(.float32)
        for layer in 0 ..< c.hiddenLayers {
            let root = "layers.\(layer)."
            let x = rms(hidden, w(root + "input_layernorm.weight"), eps)
            let source = context.map { concatenated([$0, x], axis: 0) } ?? x
            var q = linear(x, root + "self_attn.q_proj")
                .reshaped([block, c.attentionHeads, d]).transposed(1, 0, 2)
            var k = linear(source, root + "self_attn.k_proj")
                .reshaped([rows + block, c.kvHeads, d]).transposed(1, 0, 2)
            let v = linear(source, root + "self_attn.v_proj")
                .reshaped([rows + block, c.kvHeads, d]).transposed(1, 0, 2)
            q = rotate(
                rms(q, w(root + "self_attn.q_norm.weight"), eps), positions: blockPositions,
                theta: c.ropeTheta)
            k = rotate(
                rms(k, w(root + "self_attn.k_norm.weight"), eps), positions: keyPositions,
                theta: c.ropeTheta)
            let keys = repeated(k, count: groups, axis: 0)
            let values = repeated(v, count: groups, axis: 0)
            let scores = matmul(q, keys.transposed(0, 2, 1)) * scale + additive
            let attended = matmul(softmax(scores, axis: -1), values)
                .transposed(1, 0, 2).reshaped([block, c.attentionHeads * d])
            hidden = hidden + linear(attended, root + "self_attn.o_proj")
            let m = rms(hidden, w(root + "post_attention_layernorm.weight"), eps)
            let gate = linear(m, root + "mlp.gate_proj")
            hidden =
                hidden
                + linear(
                    gate * sigmoid(gate) * linear(m, root + "mlp.up_proj"), root + "mlp.down_proj")
        }
        return rms(hidden, w("norm.weight"), eps)
    }
}
