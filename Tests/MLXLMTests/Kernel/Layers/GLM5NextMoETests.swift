import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Fixed-weight primitive tests, not a random model or generation qualification.
    /// Scalar equations follow the f0705f991a71f05dad9569336b181119f5546f93 reference
    /// pinned in GLM5NextMoE.swift. No downloaded weights or network are used.
    @Suite
    struct GLM5NextMoETests {

        /// The shared experts run inside the routed prefill launches only
        /// where MLX itself would run their products as plain `qmm` (no split
        /// K, not `qmv`); GLM-5.3's 4,096-wide fused gate/up and down need 65
        /// rows, the unfused 2,048-wide gate/up 129 (measured bitwise from 65
        /// with the fused projection, 2^-6 apart below).
        @Test func sharedExpertsJoinTheRoutedLaunchesOnlyWhereMLXRunsPlainQMM() {
            let rule = GLM5NextMLP.mlxRunsPlainQuantizedMatmul
            #expect(!rule(64, 4096) && rule(65, 4096) && rule(4096, 4096))
            #expect(!rule(128, 2048) && rule(129, 2048))
            #expect(!rule(33, 1 << 20) && rule(34, 1 << 20))
            let mlp = GLM5NextMLP(hiddenSize: 4096, intermediateSize: 2048)
            #expect(!mlp.runsPlainQuantizedMatmul(rows: 128))
            #expect(mlp.runsPlainQuantizedMatmul(rows: 129))
        }

        @Test func denseMLPClampsBeforeSiluAndDoesNotLowerClampGate() throws {
            let weights = DenseWeights(
                hidden: 2, intermediate: 4,
                gate: [12, 0, -12, 0, 1, 0, -1, 0],
                up: [25, 0, -25, 0, 2, 0, -3, 0],
                down: [0.5, -2, 1, 0, 1, 1, 0, -0.25])
            let mlp = GLM5NextMLP(hiddenSize: 2, intermediateSize: 4)
            try weights.install(into: mlp)
            let rows: [[Float]] = [[1, 0], [-1, 0], [0.5, 1]]
            let actual = mlp(MLXArray(rows.flatMap { $0 }).reshaped([1, 3, 2]))
            expectClose(actual, rows.flatMap { weights.output($0) })
            #expect(actual.shape == [1, 3, 2])
            #expect(
                mlp.parameters().flattened().map(\.0).sorted() == [
                    "down_proj.weight", "gate_proj.weight", "up_proj.weight",
                ])
        }

        @Test func correctionBiasSelectsExpertsButNeverChangesTheirWeights() throws {
            let router = try makeRouter()
            let result = router(MLXArray([Float(1), 0, -1, 0]).reshaped([1, 2, 2]))
            #expect(result.logits.dtype == .float32 && result.weights.dtype == .float32)
            expectClose(result.logits, [2, 0, -2, -3, -2, 0, 2, 3])
            let probability = 1 / (1 + exp(-2.0))
            let other = 1 / (1 + exp(-3.0))
            expectRoutes(
                result,
                [
                    [0: probability * 2.5, 2: (1 - probability) * 2.5],
                    [
                        2: probability / (probability + other) * 2.5,
                        3: other / (probability + other) * 2.5,
                    ],
                ])
            let trainable = router.trainableParameters().flattened().map(\.0)
            #expect(trainable == ["weight"])
        }

        @Test func routerPromotesBeforeMatmulAndNormalizesEvenOneSelection() throws {
            let router = GLM5NextRouter(hiddenSize: 2, numExperts: 2, topK: 1)
            try router.update(
                parameters: .unflattened([
                    "weight": MLXArray([Float(256), -256, 256, 256]).reshaped([2, 2]).asType(
                        .float16),
                    "e_score_correction_bias": MLXArray.zeros([2]),
                ]), verify: .all)
            let result = router(MLXArray([Float(256), 256]).reshaped([1, 2]).asType(.float16))
            // 131072 would overflow a float16 matmul before a late cast to float32.
            expectClose(result.logits, [0, 131072])
            expectRoutes(result, [[1: 2.5]])
        }

        @Test func normalizationUsesTheSpecifiedEpsilon() throws {
            let router = GLM5NextRouter(hiddenSize: 1, numExperts: 2, topK: 1)
            try router.update(
                parameters: .unflattened([
                    "weight": MLXArray([Float(-60), -61]).reshaped([2, 1]),
                    "e_score_correction_bias": MLXArray.zeros([2]),
                ]), verify: .all)
            let result = router(MLXArray([Float(1)]).reshaped([1, 1]))
            let score = 1 / (1 + exp(60.0))
            expectClose(result.weights, [score / (score + 1e-20) * 2.5], absolute: 1e-10)
        }

        @Test func disabledNormalizationRetainsUncorrectedSigmoidScores() throws {
            let result = try makeRouter(normalize: false)(
                MLXArray([Float(1), 0]).reshaped([1, 2]))
            let high = 1 / (1 + exp(-2.0))
            expectRoutes(result, [[0: high * 2.5, 2: (1 - high) * 2.5]])
            // Use a pair whose uncorrected scores do not sum to one as well.
            let second = try makeRouter(normalize: false)(
                MLXArray([Float(-1), 0]).reshaped([1, 2]))
            expectRoutes(second, [[2: high * 2.5, 3: 2.5 / (1 + exp(-3.0))]])
        }

        @Test func groupedRoutingMasksWithNegativeInfinityNotZero() throws {
            let router = GLM5NextRouter(
                hiddenSize: 1, numExperts: 4, topK: 2, nGroup: 2, topkGroup: 1)
            try router.update(
                parameters: .unflattened([
                    "weight": MLXArray([Float(0), 0, 2, -2]).reshaped([4, 1]),
                    "e_score_correction_bias": MLXArray([Float(-3), -3, -2, -2]),
                ]), verify: .all)
            let result = router(MLXArray([Float(1)]).reshaped([1, 1]))
            let high = 1 / (1 + exp(-2.0))
            // Every corrected score is negative. A zeroed excluded group wins incorrectly.
            expectRoutes(result, [[2: high * 2.5, 3: (1 - high) * 2.5]])
        }

        @Test func routedExpertsMatchExplicitPerExpertMLPs() throws {
            let fixture = expertWeights
            let experts = GLM5NextExperts(hiddenSize: 2, intermediateSize: 2, numExperts: 4)
            try install(fixture, into: experts)
            let input: [[Float]] = [[1, 0], [-1, 0.5]]
            let indices = [[2, 0], [3, 1]]
            let weights: [[Float]] = [[0.25, 0.75], [0.625, 0.375]]
            let actual = experts(
                MLXArray(input.flatMap { $0 }).reshaped([2, 2]),
                indices: MLXArray(indices.flatMap { $0.map(UInt32.init) }).reshaped([2, 2]),
                weights: MLXArray(weights.flatMap { $0 }).reshaped([2, 2]))
            expectClose(actual, weightedOutputs(input, indices, weights, experts: fixture))
        }

        @Test func moeAddsUnscaledSharedMLPAndPreservesBatchShape() throws {
            let experts = GLM5NextExperts(hiddenSize: 2, intermediateSize: 2, numExperts: 4)
            try install(expertWeights, into: experts)
            let sharedWeights = DenseWeights(
                hidden: 2, intermediate: 2,
                gate: [0.5, 0, 0, -0.25], up: [2, 0, 0, 1], down: [0.5, 1, -1, 0.25])
            let shared = GLM5NextMLP(hiddenSize: 2, intermediateSize: 2)
            try sharedWeights.install(into: shared)
            let model = GLM5NextMoE(gate: try makeRouter(), experts: experts, sharedExperts: shared)
            let rows: [[Float]] = [[1, 0], [-1, 0]]
            let high = 1 / (1 + exp(-2.0))
            let other = 1 / (1 + exp(-3.0))
            let weights: [[Float]] = [
                [Float(high * 2.5), Float((1 - high) * 2.5)],
                [Float(high / (high + other) * 2.5), Float(other / (high + other) * 2.5)],
            ]
            let routed = weightedOutputs(rows, [[0, 2], [2, 3]], weights, experts: expertWeights)
            let sharedOutput = rows.flatMap { sharedWeights.output($0) }
            let expected = zip(routed, sharedOutput).map(+)
            let actual = model(MLXArray(rows.flatMap { $0 }).reshaped([2, 1, 2]))
            #expect(actual.shape == [2, 1, 2] && actual.dtype == .float32)
            expectClose(actual, expected)
        }

        @Test func mixedFourFiveSixBitProjectionsStayPackedAndMatchScalarOracle() throws {
            let width = 64
            let count = 2
            var fixtures = [DenseWeights]()
            for expert in 0 ..< count {
                var gate = [Float]()
                var up = [Float]()
                var down = [Float]()
                for row in 0 ..< width {
                    for column in 0 ..< width {
                        let sign: Float = (row + column).isMultiple(of: 2) ? 1 : -1
                        gate.append(row == column ? Float(expert + 1) / 4 : sign / 128)
                        up.append(row == column ? Float(expert + 1) / 2 : -sign / 64)
                        down.append(row == column ? Float(expert + 1) / 2 : sign / 256)
                    }
                }
                fixtures.append(
                    DenseWeights(
                        hidden: width, intermediate: width, gate: gate, up: up, down: down))
            }
            let experts = GLM5NextExperts(
                hiddenSize: width, intermediateSize: width, numExperts: count)
            try install(fixtures, into: experts)
            quantize(
                model: experts,
                filter: { path, _ in
                    switch path {
                    case "gate_proj": (32, 4, .affine)
                    case "up_proj": (32, 5, .affine)
                    case "down_proj": (32, 6, .affine)
                    default: nil
                    }
                })
            let gate = try #require(experts.gateProj as? QuantizedSwitchLinear)
            let up = try #require(experts.upProj as? QuantizedSwitchLinear)
            let down = try #require(experts.downProj as? QuantizedSwitchLinear)
            for (layer, bits) in [(gate, 4), (up, 5), (down, 6)] {
                #expect(layer.bits == bits && layer.groupSize == 32 && layer.mode == .affine)
                #expect(layer.weight.dtype == .uint32)
                #expect(layer.weight.shape == [count, width, width * bits / 32])
                #expect(layer.scales.shape == [count, width, 2])
            }
            #expect(
                experts.leafModules().flattened().map(\.0).sorted() == [
                    "down_proj", "gate_proj", "up_proj",
                ])

            // Dequantization is ONLY a tiny independent test oracle. The module above
            // executes QuantizedSwitchLinear/gatherQuantizedMM and keeps packed banks.
            func unpack(_ layer: QuantizedSwitchLinear) -> [Float] {
                dequantized(
                    layer.weight, scales: layer.scales, biases: layer.biases,
                    groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode,
                    dtype: .float32
                ).asArray(Float.self)
            }
            let gateValues = unpack(gate)
            let upValues = unpack(up)
            let downValues = unpack(down)
            let stride = width * width
            let decoded = (0 ..< count).map { index in
                let range = (index * stride) ..< ((index + 1) * stride)
                return DenseWeights(
                    hidden: width, intermediate: width,
                    gate: Array(gateValues[range]), up: Array(upValues[range]),
                    down: Array(downValues[range]))
            }
            let rows: [[Float]] = [
                (0 ..< width).map { $0.isMultiple(of: 2) ? 0.5 : -0.25 },
                (0 ..< width).map { $0.isMultiple(of: 3) ? -0.75 : 0.125 },
            ]
            let indices = [[1, 0], [0, 1]]
            let weights: [[Float]] = [[0.25, 0.75], [0.625, 0.375]]
            let actual = experts(
                MLXArray(rows.flatMap { $0 }).reshaped([2, width]),
                indices: MLXArray(indices.flatMap { $0.map(UInt32.init) }).reshaped([2, 2]),
                weights: MLXArray(weights.flatMap { $0 }).reshaped([2, 2]))
            expectClose(actual, weightedOutputs(rows, indices, weights, experts: decoded))
        }

        private func makeRouter(normalize: Bool = true) throws -> GLM5NextRouter {
            let router = GLM5NextRouter(
                hiddenSize: 2, numExperts: 4, topK: 2, normTopkProb: normalize)
            try router.update(
                parameters: .unflattened([
                    "weight": MLXArray([Float(2), 0, 0, 0, -2, 0, -3, 0]).reshaped([4, 2]),
                    "e_score_correction_bias": MLXArray([Float(0), 0, 1, 0]),
                ]), verify: .all)
            return router
        }

        private var expertWeights: [DenseWeights] {
            [
                DenseWeights(
                    hidden: 2, intermediate: 2,
                    gate: [12, 0, -12, 0], up: [25, 0, -25, 0], down: [0.5, -2, 1, 1]),
                DenseWeights(
                    hidden: 2, intermediate: 2,
                    gate: [1, 0, 0.5, 0], up: [2, 0, 3, 0], down: [1, 0, 0, 1]),
                DenseWeights(
                    hidden: 2, intermediate: 2,
                    gate: [2, 0, -1, 0], up: [-3, 0, 4, 0], down: [-0.5, 1, 0.25, -2]),
                DenseWeights(
                    hidden: 2, intermediate: 2,
                    gate: [-2, 0, 1.5, 0], up: [-1, 0, 2, 0], down: [0.5, -1, -1, 0.5]),
            ]
        }

        private func install(_ weights: [DenseWeights], into experts: GLM5NextExperts) throws {
            let first = weights[0]
            try experts.update(
                parameters: .unflattened([
                    "gate_proj.weight": MLXArray(weights.flatMap(\.gate))
                        .reshaped([weights.count, first.intermediate, first.hidden]),
                    "up_proj.weight": MLXArray(weights.flatMap(\.up))
                        .reshaped([weights.count, first.intermediate, first.hidden]),
                    "down_proj.weight": MLXArray(weights.flatMap(\.down))
                        .reshaped([weights.count, first.hidden, first.intermediate]),
                ]), verify: .all)
        }

        private func weightedOutputs(
            _ input: [[Float]], _ indices: [[Int]], _ weights: [[Float]], experts: [DenseWeights]
        ) -> [Double] {
            var output = [Double]()
            for row in input.indices {
                var sum = [Float](repeating: 0, count: experts[0].hidden)
                for slot in indices[row].indices.sorted(by: { indices[row][$0] < indices[row][$1] })
                {
                    let current = experts[indices[row][slot]].output(input[row])
                    for channel in sum.indices {
                        // FP32 cast per contribution, before ascending-expert addition.
                        sum[channel] += Float(current[channel]) * weights[row][slot]
                    }
                }
                output += sum.map(Double.init)
            }
            return output
        }

        private func expectRoutes(
            _ result: (logits: MLXArray, weights: MLXArray, indices: MLXArray),
            _ expected: [[Int: Double]]
        ) {
            let ids = result.indices.asType(.int32).asArray(Int32.self)
            let weights = result.weights.asArray(Float.self)
            let topK = result.indices.dim(1)
            #expect(result.indices.dim(0) == expected.count)
            for row in expected.indices {
                let rowIDs = ids[(row * topK) ..< ((row + 1) * topK)].map(Int.init)
                #expect(Set(rowIDs) == Set(expected[row].keys))
                for slot in 0 ..< topK {
                    if let value = expected[row][rowIDs[slot]] {
                        #expect(abs(Double(weights[row * topK + slot]) - value) < 1e-6)
                    }
                }
            }
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: [Double], absolute: Double = 2e-6,
            relative: Double = 2e-5
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == expected.count)
            for (index, pair) in zip(values, expected).enumerated() {
                #expect(
                    pair.0.isFinite
                        && abs(Double(pair.0) - pair.1) <= absolute + relative * abs(pair.1),
                    "element \(index): actual \(pair.0), expected \(pair.1)")
            }
        }

        private struct DenseWeights {
            let hidden: Int
            let intermediate: Int
            let gate: [Float]
            let up: [Float]
            let down: [Float]

            func install(into module: GLM5NextMLP) throws {
                try module.update(
                    parameters: .unflattened([
                        "gate_proj.weight": MLXArray(gate).reshaped([intermediate, hidden]),
                        "up_proj.weight": MLXArray(up).reshaped([intermediate, hidden]),
                        "down_proj.weight": MLXArray(down).reshaped([hidden, intermediate]),
                    ]), verify: .all)
            }

            /// Direct scalar MLP, with distinct gate and up bounds before multiplication.
            func output(_ input: [Float]) -> [Double] {
                var activated = [Double](repeating: 0, count: intermediate)
                for row in 0 ..< intermediate {
                    var g = 0.0
                    var u = 0.0
                    for column in 0 ..< hidden {
                        g += Double(gate[row * hidden + column]) * Double(input[column])
                        u += Double(up[row * hidden + column]) * Double(input[column])
                    }
                    g = min(g, 10)
                    u = min(max(u, -10), 10)
                    activated[row] = g / (1 + exp(-g)) * u
                }
                return (0 ..< hidden).map { row in
                    (0 ..< intermediate).reduce(0) {
                        $0 + Double(down[row * intermediate + $1]) * activated[$1]
                    }
                }
            }
        }
    }
}
