import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite(.tags(.integration))
struct NemotronSuperVisionProjectorTests {
    @Test func rectangularPixelOrder() {
        let input = MLXArray((0 ..< 24).map(Float.init)).reshaped(1, 4, 6, 1)
        let output = NemotronSuperVisionProjector.pixelShuffleV2(input)
        #expect(output.shape == [1, 2, 3, 4])
        #expect(
            output.asArray(Float.self) == [
                0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11,
                12, 13, 18, 19, 14, 15, 20, 21, 16, 17, 22, 23,
            ])
        let multi = MLXArray((0 ..< 96).map(Float.init)).reshaped(2, 4, 6, 2)
        let shuffled = NemotronSuperVisionProjector.pixelShuffleV2(multi).asArray(Float.self)
        var expected: [Float] = []
        for batch in 0 ..< 2 {
            for row in 0 ..< 2 {
                for column in 0 ..< 3 {
                    for dy in 0 ..< 2 {
                        for dx in 0 ..< 2 {
                            for channel in 0 ..< 2 {
                                expected.append(
                                    Float(
                                        ((batch * 4 + row * 2 + dy) * 6 + column * 2 + dx) * 2
                                            + channel))
                            }
                        }
                    }
                }
            }
        }
        #expect(shuffled == expected)
    }

    private func weights() -> [String: MLXArray] {
        [
            "vision_projector.vision_final_layernorm.weight": MLXArray([Float(1.5), 0.5]),
            "vision_projector.vision_final_layernorm.bias": MLXArray([Float(0.25), -0.75]),
            "mlp1.0.weight": MLXArray([Float(1), 2, 0.5, 1.5, 0.75, 1.25, 2, 0.25]),
            "mlp1.1.weight": MLXArray((0 ..< 24).map { Float($0 - 12) / 16 }).reshaped(3, 8),
            "mlp1.3.weight": MLXArray([Float(0.5), -1, 2, -0.25, 1.5, 0.75]).reshaped(2, 3),
        ]
    }

    @Test func strictWeightExtraction() throws {
        let model = NemotronSuperVisionProjector(visionWidth: 2, hiddenWidth: 3, outputWidth: 2)
        var complete = weights()
        complete["vision_model.unrelated.weight"] = MLXArray([Float(99)])
        try model.loadProjectorWeights(from: complete)
        #expect(
            model.projection.norm.weight.asArray(Float.self) == [
                1, 2, 0.5, 1.5, 0.75, 1.25, 2, 0.25,
            ])
        var duplicate = complete
        duplicate["mlp1.norm.weight"] = duplicate["mlp1.0.weight"]
        #expect(throws: NemotronSuperVisionProjector.WeightError.self) {
            try model.loadProjectorWeights(from: duplicate)
        }
        var missing = complete
        missing.removeValue(forKey: "mlp1.3.weight")
        #expect(throws: NemotronSuperVisionProjector.WeightError.self) {
            try model.loadProjectorWeights(from: missing)
        }
        var malformed = complete
        malformed["mlp1.1.weight"] = MLXArray.zeros([8, 3])
        #expect(throws: NemotronSuperVisionProjector.WeightError.self) {
            try model.loadProjectorWeights(from: malformed)
        }
        var quantized = complete
        quantized["mlp1.1.weight"] = MLXArray.zeros([3, 8], dtype: .uint8)
        #expect(throws: NemotronSuperVisionProjector.WeightError.self) {
            try model.loadProjectorWeights(from: quantized)
        }
        var sidecar = complete
        sidecar["mlp1.1.weight_scale"] = MLXArray(Float(1))
        #expect(throws: NemotronSuperVisionProjector.WeightError.self) {
            try model.loadProjectorWeights(from: sidecar)
        }
        #expect(
            model.projection.norm.weight.asArray(Float.self) == [
                1, 2, 0.5, 1.5, 0.75, 1.25, 2, 0.25,
            ])
    }

    @Test func bfloat16NormAndProjectionPreserveDtype() throws {
        let model = NemotronSuperVisionProjector(visionWidth: 2, hiddenWidth: 3, outputWidth: 2)
        try model.loadProjectorWeights(from: weights().mapValues { $0.asType(.bfloat16) })
        let values = MLXArray([Float(0.3), -1.7, 2.1, 0.4, -0.2, 3.3, 0.9, -2.5])
            .reshaped(1, 1, 8).asType(.bfloat16)
        let actual = model.projection.norm(values)
        let promoted = values.asType(.float32)
        let normalized =
            promoted * rsqrt(mean(promoted * promoted, axis: -1, keepDims: true) + 1e-5)
        let expected = (model.projection.norm.weight.asType(.float32) * normalized).asType(
            .bfloat16)
        #expect(actual.dtype == .bfloat16)
        #expect(
            actual.asType(.float32).asArray(Float.self)
                == expected.asType(.float32).asArray(Float.self))
        let output = model(values.reshaped(1, 4, 2), height: 2, width: 2)
        #expect(output.dtype == .bfloat16)
        #expect(output.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
        // Independent PyTorch 2.12 CPU BF16 oracle, pinned NVIDIA7d4bc747
        // LayerNorm -> v2 shuffle -> FP32 RMSNorm/cast -> ReLU² MLP sequence.
        // One BF16 ULP at these output magnitudes bounds backend rounding.
        let reference: [Float] = [0.193359375, 0.205078125]
        for (actual, expected) in zip(output.asType(.float32).asArray(Float.self), reference) {
            #expect(abs(actual - expected) <= 1.0 / 1024)
        }
    }

    @Test func fullFlowMatchesScalarOracle() throws {
        let model = NemotronSuperVisionProjector(visionWidth: 2, hiddenWidth: 3, outputWidth: 2)
        try model.loadProjectorWeights(from: weights())
        let values: [Float] = [1, 3, 4, -2, 0, 8, -3, -1]
        let output = model(MLXArray(values).reshaped(1, 4, 2), height: 2, width: 2)
        let w1 = model.projection.linear1.weight.asArray(Float.self)
        let w2 = model.projection.linear2.weight.asArray(Float.self)
        var normalized: [Float] = []
        for token in 0 ..< 4 {
            let a = values[token * 2]
            let b = values[token * 2 + 1]
            let average = (a + b) / 2
            let variance = ((a - average) * (a - average) + (b - average) * (b - average)) / 2
            normalized += [
                1.5 * (a - average) / sqrt(variance + 1e-6) + 0.25,
                0.5 * (b - average) / sqrt(variance + 1e-6) - 0.75,
            ]
        }
        let rms = sqrt(normalized.reduce(Float(0)) { $0 + $1 * $1 } / 8 + 1e-5)
        let normWeights: [Float] = [1, 2, 0.5, 1.5, 0.75, 1.25, 2, 0.25]
        normalized = zip(normalized, normWeights).map { $0 * $1 / rms }
        let hidden = (0 ..< 3).map { row -> Float in
            let value = (0 ..< 8).reduce(Float(0)) { $0 + w1[row * 8 + $1] * normalized[$1] }
            return max(value, 0) * max(value, 0)
        }
        let expected = (0 ..< 2).map { row in
            (0 ..< 3).reduce(Float(0)) { $0 + w2[row * 3 + $1] * hidden[$1] }
        }
        #expect(output.shape == [1, 1, 2])
        for (actual, reference) in zip(output.asArray(Float.self), expected) {
            #expect(abs(actual - reference) < 1e-5)
        }
        #expect(model.projection.linear1.bias == nil)
        #expect(model.projection.linear2.bias == nil)
    }
}
