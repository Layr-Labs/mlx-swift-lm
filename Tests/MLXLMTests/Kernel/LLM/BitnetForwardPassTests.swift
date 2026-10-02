import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `BitnetModel` and of its `BitLinear` Metal
    /// kernel.
    ///
    /// A `BitLinear` weight packs 4 ternary values in each byte: bits
    /// `2k ..< 2k + 2` of byte `[r, i]` hold `w + 1` for output row
    /// `r + k * (out / 4)` and input `i`. The tests pack random ternary
    /// weights, and compare the kernel with a float matrix product of the
    /// unpacked weights.
    @Suite
    struct BitnetForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": vocabularySize, "rope_theta": 10000,
            ]
        }

        /// Random ternary values in [-1, 1], as a packed `[out / 4, in]`
        /// uint8 array and as the unpacked `[out, in]` float matrix.
        static func ternary(out: Int, in inputs: Int, seed: Int) -> (MLXArray, MLXArray) {
            let values = SyntheticModel.tokens(count: out * inputs, vocabularySize: 3, seed: seed)
            var packed = [UInt8](repeating: 0, count: out / 4 * inputs)
            var dense = [Float](repeating: 0, count: out * inputs)
            for row in 0 ..< out {
                for column in 0 ..< inputs {
                    let value = values[row * inputs + column]
                    dense[row * inputs + column] = Float(value - 1)
                    let byte = (row % (out / 4)) * inputs + column
                    packed[byte] |= UInt8(value) << (2 * (row / (out / 4)))
                }
            }
            return (
                MLXArray(packed).reshaped(out / 4, inputs),
                MLXArray(dense).reshaped(out, inputs)
            )
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> BitnetModel
        {
            let configuration = try SyntheticModel.configuration(
                BitnetConfiguration.self, base, overrides: overrides)
            let model = BitnetModel(configuration)
            SyntheticModel.randomize(model, seed: seed)
            // Replace every packed weight with random ternary values and use
            // 1 / sqrt(in) as the weight scale, so activations stay near 1.
            var updates: [(String, MLXArray)] = []
            for (index, (key, value)) in SyntheticModel.flatParameters(model)
                .sorted(by: { $0.key < $1.key }).enumerated() where value.dtype == .uint8
            {
                let prefix = String(key.dropLast(".weight".count))
                let (packed, _) = ternary(
                    out: value.dim(0) * 4, in: value.dim(1), seed: Int(seed) * 100 + index)
                updates.append((key, packed))
                updates.append(
                    ("\(prefix).weight_scale", MLXArray([1 / Float(value.dim(1)).squareRoot()])))
            }
            model.update(parameters: ModuleParameters.unflattened(updates))
            eval(model)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the kernel sums in float32
        // in another order than a matrix product, and the cached path sums
        // attention in another order. The differences are near 1e-6.
        static let tolerance: Float = 1e-4

        @Test(arguments: [false, true])
        func bitLinearKernelMatchesTheUnpackedMatrixProduct(invertScale: Bool) {
            let layer = BitLinear(64, 32, bias: true, invertWeightScales: invertScale)
            let (packed, dense) = Self.ternary(out: 32, in: 64, seed: 11)
            let bias = MLXRandom.normal([32], key: MLXRandom.key(12))
            layer.update(
                parameters: ModuleParameters.unflattened([
                    "weight": packed, "bias": bias, "weight_scale": MLXArray([0.5] as [Float]),
                ]))
            let x = MLXRandom.normal([2, 3, 64], key: MLXRandom.key(13))
            eval(layer, x)

            let scale: Float = invertScale ? 2 : 0.5
            let expected = matmul(x, dense.T) * scale + bias
            let output = layer(x)
            #expect(output.shape == [2, 3, 32])
            #expect(SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance)
        }

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            ForwardPassChecks.checkShapeDTypeAndFinite(
                try Self.makeModel(), vocabularySize: Self.vocabularySize, length: 7)
        }

        @Test func sameSeedGivesTheSameLogits() throws {
            try ForwardPassChecks.checkDeterminism(
                make: { try Self.makeModel() }, seed: 3, vocabularySize: Self.vocabularySize)
        }

        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1), Self.row(2)], chunks: [4, 4, 1, 1, 1],
                tolerance: Self.tolerance)
        }

        @Test func eachRowOfABatchMatchesTheRowAlone() throws {
            ForwardPassChecks.checkBatchInvariance(
                try Self.makeModel(), rowA: Self.row(1), rowB: Self.row(2),
                tolerance: Self.tolerance)
        }

        @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
            ForwardPassChecks.checkCausality(
                try Self.makeModel(), row: Self.row(1), position: 6,
                vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            let expected: [String: [Int]] = [
                "model.layers.0.self_attn.q_proj.weight": [8, 32],
                "model.layers.0.self_attn.q_proj.weight_scale": [1],
                "model.layers.0.self_attn.k_proj.weight": [4, 32],
                "model.layers.0.self_attn.attn_sub_norm.weight": [32],
                "model.layers.0.mlp.gate_proj.weight": [16, 32],
                "model.layers.0.mlp.down_proj.weight": [8, 64],
                "model.layers.0.mlp.ffn_sub_norm.weight": [64],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
            #expect(parameters["model.layers.0.self_attn.q_proj.weight"]?.dtype == .uint8)
            // Tied embeddings are the default.
            #expect(parameters["lm_head.weight"] == nil)
        }

        @Test func loaderAcceptsACheckpointAndGivesTheSameLogits() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            checkpoint["model.layers.0.self_attn.rotary_emb.inv_freq"] = MLXArray.ones([4])
            checkpoint["lm_head.weight"] = MLXArray.zeros([64, 32])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [Self.row(3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)

            checkpoint["model.layers.1.mlp.up_proj.weight"] = MLXArray.zeros(
                [8, 32], dtype: .uint8)
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel(seed: 6))
            }
        }
    }
}
