import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextLinearAttentionTests {
        private typealias Primitive = GLM5NextLinearAttention

        /// Independent scalar FP32 implementation of the pinned Transformers
        /// recurrent KDA + causal conv + sigmoid-gated norm, not the MLX helpers.
        private struct Fixture {
            let batch = 2, tokens = 5, heads = 2, dimension = 3, kernel = 4
            var width: Int { heads * dimension }
            var count: Int { batch * tokens * width }
            var q: [Float] { values(count, multiplier: 7, modulus: 23, divisor: 13) }
            var k: [Float] { values(count, multiplier: 11, modulus: 29, divisor: 17) }
            var v: [Float] { values(count, multiplier: 3, modulus: 19, divisor: 11) }
            var forget: [Float] { values(count, multiplier: 5, modulus: 31, divisor: 19) }
            var beta: [Float] {
                values(batch * tokens * heads, multiplier: 3, modulus: 11, divisor: 7)
            }
            var gate: [Float] { values(count, multiplier: 13, modulus: 17, divisor: 9) }
            var conv: [Float] {
                values(3 * width * kernel, multiplier: 5, modulus: 13, divisor: 11)
            }
            let aLog: [Float] = [-0.7, 0.4]
            var dtBias: [Float] { [-0.6, 0.2, 0.7, -0.1, 0.5, -0.8] }
            let norm: [Float] = [0.7, 1.3, -0.9]

            private func values(_ count: Int, multiplier: Int, modulus: Int, divisor: Float)
                -> [Float]
            {
                (0 ..< count).map { Float(($0 * multiplier) % modulus - modulus / 2) / divisor }
            }

            func run(
                _ range: Range<Int>? = nil, state: Primitive.State? = nil,
                dtype: DType = .float32, lowerBound: Float? = -5
            ) throws -> Primitive.Output {
                let range = range ?? 0 ..< tokens
                func tensor(_ data: [Float]) -> MLXArray {
                    MLXArray(data).reshaped(batch, tokens, heads, dimension)
                        .asType(dtype)[0..., range, 0..., 0...]
                }
                return try Primitive.apply(
                    query: tensor(q), key: tensor(k), value: tensor(v),
                    forgetLogits: tensor(forget),
                    betaLogits: MLXArray(beta).reshaped(batch, tokens, heads)
                        .asType(dtype)[0..., range, 0...],
                    outputGate: tensor(gate),
                    weights: .init(
                        convolution: MLXArray(conv).reshaped(3 * width, kernel),
                        aLog: MLXArray(aLog), dtBias: MLXArray(dtBias), outputNorm: MLXArray(norm)),
                    state: state, gateLowerBound: lowerBound)
            }

            func oracle(
                dtype: DType = .float32, lowerBound: Float? = -5
            ) -> (
                values: [Float], convolution: [Float], recurrent: [Float], convolved: [Float],
                beta: [Float]
            ) {
                var history = Array(repeating: Float(0), count: batch * 3 * width * kernel)
                var recurrent = Array(
                    repeating: Float(0), count: batch * heads * dimension * dimension)
                var output = Array(repeating: Float(0), count: count)
                var convolutionOutput = Array(
                    repeating: Float(0), count: batch * tokens * 3 * width)
                var betaOutput = Array(repeating: Float(0), count: batch * tokens * heads)
                let projected = [q, k, v].map { $0.map { Self.round($0, to: dtype) } }
                for t in 0 ..< tokens {
                    for b in 0 ..< batch {
                        var convolved = Array(repeating: Float(0), count: 3 * width)
                        for channel in 0 ..< (3 * width) {
                            let historyStart = (b * 3 * width + channel) * kernel
                            for tap in 0 ..< (kernel - 1) {
                                history[historyStart + tap] = history[historyStart + tap + 1]
                            }
                            history[historyStart + kernel - 1] =
                                projected[channel / width][
                                    (b * tokens + t) * width + channel % width]
                            var sum: Float = 0
                            for tap in 0 ..< kernel {
                                sum += history[historyStart + tap] * conv[channel * kernel + tap]
                            }
                            convolved[channel] = Self.round(sum * Self.sigmoid(sum), to: dtype)
                            convolutionOutput[(b * tokens + t) * 3 * width + channel] =
                                convolved[channel]
                        }
                        for h in 0 ..< heads {
                            let offset = (b * tokens + t) * width + h * dimension
                            let stateStart = (b * heads + h) * dimension * dimension
                            var q = Array(convolved[(h * dimension) ..< ((h + 1) * dimension)])
                            var k = Array(
                                convolved[(width + h * dimension) ..< (width + (h + 1) * dimension)]
                            )
                            var qSquared: Float = 0
                            var kSquared: Float = 0
                            for d in 0 ..< dimension {
                                qSquared += q[d] * q[d]
                                kSquared += k[d] * k[d]
                            }
                            for d in 0 ..< dimension {
                                q[d] =
                                    q[d] / Foundation.sqrt(qSquared + 1e-6)
                                    * (1 / Foundation.sqrt(Float(dimension)))
                                k[d] = k[d] / Foundation.sqrt(kSquared + 1e-6)
                                let f =
                                    Self.round(forget[offset + d], to: dtype)
                                    + dtBias[h * dimension + d]
                                let rate = Foundation.exp(aLog[h])
                                let logDecay =
                                    lowerBound.map { $0 * Self.sigmoid(rate * f) }
                                    ?? (-rate
                                        * (f > 20 ? f : Foundation.log(1 + Foundation.exp(f))))
                                for v in 0 ..< dimension {
                                    recurrent[stateStart + d * dimension + v] *= Foundation.exp(
                                        logDecay)
                                }
                            }
                            let betaInput = Self.round(
                                self.beta[(b * tokens + t) * heads + h], to: dtype)
                            let beta = Self.round(Self.sigmoid(betaInput), to: dtype)
                            betaOutput[(b * tokens + t) * heads + h] = beta
                            for v in 0 ..< dimension {
                                var memory: Float = 0
                                for d in 0 ..< dimension {
                                    memory += recurrent[stateStart + d * dimension + v] * k[d]
                                }
                                let delta =
                                    (convolved[2 * width + h * dimension + v] - memory) * beta
                                for d in 0 ..< dimension {
                                    recurrent[stateStart + d * dimension + v] += k[d] * delta
                                }
                                var result: Float = 0
                                for d in 0 ..< dimension {
                                    result += recurrent[stateStart + d * dimension + v] * q[d]
                                }
                                output[offset + v] = Self.round(result, to: dtype)
                            }
                            var squared: Float = 0
                            for v in 0 ..< dimension {
                                squared += output[offset + v] * output[offset + v]
                            }
                            let denominator = Foundation.sqrt(squared / Float(dimension) + 1e-5)
                            for v in 0 ..< dimension {
                                output[offset + v] = Self.round(
                                    output[offset + v] / denominator * norm[v]
                                        * Self.sigmoid(Self.round(gate[offset + v], to: dtype)),
                                    to: dtype)
                            }
                        }
                    }
                }
                return (output, history, recurrent, convolutionOutput, betaOutput)
            }

            private static func sigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }

            /// Round-to-nearest-even at the reference's activation boundaries.
            /// Fixture values are finite, so NaN payload handling is not needed.
            private static func round(_ x: Float, to dtype: DType) -> Float {
                switch dtype {
                case .float16:
                    return Float(Float16(x))
                case .bfloat16:
                    let bits = x.bitPattern
                    return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff_0000)
                default:
                    return x
                }
            }
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: [Float], tolerance: Float = 2e-5,
            context: String = "", sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == expected.count, sourceLocation: sourceLocation)
            for (index, pair) in zip(values, expected).enumerated() {
                #expect(
                    pair.0.isFinite, "\(context) nonfinite element \(index)",
                    sourceLocation: sourceLocation)
                #expect(
                    abs(pair.0 - pair.1) <= tolerance,
                    "\(context) element \(index): actual \(pair.0), expected \(pair.1)",
                    sourceLocation: sourceLocation)
            }
        }

        @Test func testNativeConvolutionAndBetaPrecisionMatchScalarStages() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = Fixture()
                for dtype: DType in [.float32, .bfloat16, .float16] {
                    let expected = fixture.oracle(dtype: dtype)
                    let projected = [fixture.q, fixture.k, fixture.v].map {
                        MLXArray($0).reshaped(fixture.batch, fixture.tokens, fixture.width).asType(
                            dtype)
                    }
                    let raw = concatenated(projected, axis: -1)
                    let history = concatenated(
                        [
                            MLXArray.zeros(
                                [fixture.batch, fixture.kernel, 3 * fixture.width], dtype: dtype),
                            raw,
                        ], axis: 1)
                    let convolution = MLX.conv1d(
                        history[0..., 1..., 0...].asType(.float32),
                        MLXArray(fixture.conv).reshaped(3 * fixture.width, fixture.kernel, 1),
                        groups: 3 * fixture.width)
                    expectClose(
                        silu(convolution).asType(dtype), expected.convolved,
                        context: "\(dtype) convolved QKV")
                    let betaInput = MLXArray(fixture.beta)
                        .reshaped(fixture.batch, fixture.tokens, fixture.heads).asType(dtype)
                    let beta = Primitive.betaGate(betaInput)
                    #expect(beta.dtype == dtype)
                    expectClose(
                        beta, expected.beta,
                        context: "\(dtype) beta sigmoid")
                }
            }
        }

        @Test func testBetaGatePreservesSignAndRoundsBackToProjectionDType() {
            Device.withDefaultDevice(.cpu) {
                // Positive logits exposed a sign reversal in native FP16 sigmoid.
                // The non-dyadic outputs also distinguish FP32 retention from the
                // reference's required rounding back to the projection dtype.
                let input = MLXArray([Float(1) / 7, Float(4) / 7, Float(2) / 7, -Float(1) / 7])
                let halfGate = Primitive.betaGate(input.asType(.float16))
                #expect(halfGate.dtype == .float16)
                expectClose(
                    halfGate, [0.53564453125, 0.63916015625, 0.57080078125, 0.46435546875],
                    tolerance: 0, context: "FP16 beta rounding")
                let bfloatGate = Primitive.betaGate(input.asType(.bfloat16))
                #expect(bfloatGate.dtype == .bfloat16)
                expectClose(
                    bfloatGate, [0.53515625, 0.640625, 0.5703125, 0.46484375],
                    tolerance: 0, context: "BF16 beta rounding")
            }
        }

        @Test func testMatchesIndependentScalarKDAAndConvolutionOracle() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = Fixture()
                for dtype: DType in [.float32, .bfloat16, .float16] {
                    for lowerBound: Float? in [-5, nil] {
                        let actual = try fixture.run(dtype: dtype, lowerBound: lowerBound)
                        let expected = fixture.oracle(dtype: dtype, lowerBound: lowerBound)
                        expectClose(
                            actual.values, expected.values,
                            tolerance: dtype == .float32 ? 2e-5 : 0.015,
                            context: "\(dtype) lower=\(String(describing: lowerBound)) output")
                        expectClose(actual.state.convolution, expected.convolution, tolerance: 0)
                        expectClose(
                            actual.state.recurrent, expected.recurrent,
                            context: "\(dtype) lower=\(String(describing: lowerBound)) recurrent")
                    }
                }
            }
        }

        @Test func testChunkAndSingleTokenPartitionsPreserveState() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = Fixture()
                for dtype: DType in [.float32, .bfloat16, .float16] {
                    let whole = try fixture.run(dtype: dtype)
                    for ranges in [[0 ..< 2, 2 ..< 5], (0 ..< 5).map { $0 ..< ($0 + 1) }] {
                        var state: Primitive.State?
                        var values: [MLXArray] = []
                        for range in ranges {
                            let part = try fixture.run(range, state: state, dtype: dtype)
                            state = part.state
                            values.append(part.values)
                        }
                        let final = try #require(state)
                        expectClose(
                            concatenated(values, axis: 1),
                            whole.values.asType(.float32).asArray(Float.self),
                            tolerance: dtype == .float32 ? 2e-5 : 0.01)
                        expectClose(
                            final.convolution,
                            whole.state.convolution.asType(.float32).asArray(Float.self),
                            tolerance: 0)
                        expectClose(final.recurrent, whole.state.recurrent.asArray(Float.self))
                        #expect(final.recurrent.dtype == .float32)
                        #expect(final.convolution.dtype == dtype)
                        #expect(values[0].dtype == dtype)
                    }
                }
            }
        }

        @Test func testTwoContinuationsDoNotMutateTheirSharedInputState() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = Fixture()
                let prefix = try fixture.run(0 ..< 2)
                let originalConv = prefix.state.convolution.asArray(Float.self)
                let originalRecurrent = prefix.state.recurrent.asArray(Float.self)
                let first = try fixture.run(2 ..< 5, state: prefix.state)
                _ = first.values.asArray(Float.self)
                let other = try fixture.run(0 ..< 1, state: prefix.state)
                _ = other.values.asArray(Float.self)
                let repeated = try fixture.run(2 ..< 5, state: prefix.state)
                expectClose(repeated.values, first.values.asArray(Float.self), tolerance: 0)
                expectClose(
                    repeated.state.recurrent, first.state.recurrent.asArray(Float.self),
                    tolerance: 0)
                expectClose(prefix.state.convolution, originalConv, tolerance: 0)
                expectClose(prefix.state.recurrent, originalRecurrent, tolerance: 0)
                expectClose(try fixture.run().values, fixture.oracle().values)
            }
        }

        @Test func testRejectsLossyOrMismatchedStateBeforeNativeWork() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = Fixture()
                let prefix = try fixture.run(0 ..< 2)
                let lossy = Primitive.State(
                    convolution: prefix.state.convolution,
                    recurrent: prefix.state.recurrent.asType(.bfloat16))
                #expect(throws: Primitive.Failure.invalidTensor("recurrent state")) {
                    try fixture.run(2 ..< 5, state: lossy)
                }
                let wrong = Primitive.State(
                    convolution: prefix.state.convolution[0 ..< 1],
                    recurrent: prefix.state.recurrent)
                #expect(throws: Primitive.Failure.invalidTensor("convolution state")) {
                    try fixture.run(2 ..< 5, state: wrong)
                }
                #expect(throws: Primitive.Failure.invalidGeometry) {
                    try fixture.run(lowerBound: 1)
                }
            }
        }
    }
}
