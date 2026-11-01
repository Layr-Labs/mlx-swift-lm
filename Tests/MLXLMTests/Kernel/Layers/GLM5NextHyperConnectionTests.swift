import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Direct primitive checks, without a model, random weights, downloads or generation.
    /// The scalar oracle independently evaluates the pinned Transformers equations cited
    /// by GLM5NextHyperConnection; it does not call another MLX HC implementation.
    @Suite
    struct GLM5NextHyperConnectionTests {

        @Test func explicitUniformFixturePinsEpsilonAndUnnormalizedPre() {
            let residual = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8]).reshaped([1, 1, 4, 2])
            let result = GLM5NextHyperConnection.collapse(
                residual, fn: MLXArray.zeros([24, 8]), base: MLXArray.zeros([24]),
                scale: MLXArray([Float(1), 1, 1]), rmsNormEpsilon: 0.25,
                epsilon: 0.125, sinkhornIterations: 1)

            // Official stream count N=4: pre = 1/2 + 1/8; post = 1;
            // comb = (1/4 + 1/8)/(1 + 5/8), not exactly doubly stochastic with epsilon.
            expectClose(result.collapsed, [10, 12.5])
            expectClose(result.post, [1, 1, 1, 1])
            expectClose(result.comb, Array(repeating: 3.0 / 13.0, count: 16))
            let expanded = GLM5NextHyperConnection.expand(
                MLXArray([Float(2), -1]).reshaped([1, 1, 2]), residual: residual,
                post: result.post, comb: result.comb)
            expectClose(
                expanded, Array(repeating: [74.0 / 13.0, 47.0 / 13.0], count: 4).flatMap { $0 })
        }

        @Test(arguments: [1, 2, 20])
        func denseProjectionAndSinkhornMatchScalarOracle(iterations: Int) {
            let fixture = Fixture()
            let residual = MLXArray(fixture.hidden).reshaped([2, 2, 3, 2])
            let result = fixture.collapse(residual, iterations: iterations)
            let expected = fixture.oracle(iterations: iterations)

            #expect(result.collapsed.shape == [2, 2, 2])
            #expect(result.post.shape == [2, 2, 3])
            #expect(result.comb.shape == [2, 2, 3, 3])
            #expect(result.collapsed.dtype == .float32)
            #expect(result.post.dtype == .float32 && result.comb.dtype == .float32)
            expectClose(result.collapsed, expected.collapsed)
            expectClose(result.post, expected.post)
            expectClose(result.comb, expected.comb)

            let output: [Float] = [0.5, -1, 2, 0.25, -0.5, 3, 1.5, -2]
            let expanded = GLM5NextHyperConnection.expand(
                MLXArray(output).reshaped([2, 2, 2]), residual: residual,
                post: result.post, comb: result.comb)
            var expectedExpanded = [Double]()
            for token in 0 ..< 4 {
                for destination in 0 ..< 3 {
                    for channel in 0 ..< 2 {
                        var value =
                            expected.post[token * 3 + destination]
                            * Double(output[token * 2 + channel])
                        for source in 0 ..< 3 {
                            value +=
                                expected.comb[token * 9 + source * 3 + destination]
                                * Double(fixture.hidden[token * 6 + source * 2 + channel])
                        }
                        expectedExpanded.append(value)
                    }
                }
            }
            expectClose(expanded, expectedExpanded)
        }

        @Test(arguments: [DType.float16, .bfloat16])
        func normalizationAndCollapseAccumulateInFloat32(dtype: DType) {
            var fixture = Fixture()
            // Half-precision squaring would overflow; token 0 also has cancellation.
            fixture.hidden = [
                8192, -8192, -8192, 8192, 0.25, -0.5,
                1024, -2048, 4096, 8192, -16384, 2048,
                2048, -1024, 8192, -4096, 1024, 16384,
                -8192, 4096, 2048, -1024, 16384, -2048,
            ]
            let residual = MLXArray(fixture.hidden).reshaped([2, 2, 3, 2]).asType(dtype)
            let result = fixture.collapse(residual, iterations: 20, parameterType: dtype)
            let expected = fixture.oracle(iterations: 20)
            #expect(result.collapsed.dtype == dtype)
            #expect(result.post.dtype == .float32 && result.comb.dtype == .float32)
            expectClose(result.post, expected.post)
            expectClose(result.comb, expected.comb)
            let rounded = expected.collapsed.map { Double(roundToDType($0, dtype: dtype)) }
            // One output ULP is allowed for float32 reduction/rsqrt differences before the final cast.
            expectClose(
                result.collapsed, rounded, absolute: 0.01,
                relative: dtype == .float16 ? 0.001 : 0.008)
        }

        @Test(arguments: [DType.float32, .float16, .bfloat16])
        func expansionTransposesCombinationAndCastsCoefficientsFirst(dtype: DType) {
            let hidden: [Float] = [2, -4, 8, -16, 32, -64]
            let output: [Float] = [3, -7]
            let post: [Float] = [0.3, 0.7, 1.1]
            // Deliberately nonsymmetric, so comb @ residual is observably wrong.
            let comb: [Float] = [0, 0.25, 0.75, 0.5, 0, 0.5, 1, 0, 0]
            let result = GLM5NextHyperConnection.expand(
                MLXArray(output).reshaped([1, 1, 2]).asType(dtype),
                residual: MLXArray(hidden).reshaped([1, 1, 3, 2]).asType(dtype),
                post: MLXArray(post).reshaped([1, 1, 3]),
                comb: MLXArray(comb).reshaped([1, 1, 3, 3]))
            var expected = [Double]()
            for destination in 0 ..< 3 {
                for channel in 0 ..< 2 {
                    let injected = roundToDType(
                        Double(roundToDType(Double(post[destination]), dtype: dtype))
                            * Double(output[channel]), dtype: dtype)
                    var mixed = 0.0
                    for source in 0 ..< 3 {
                        mixed +=
                            Double(
                                roundToDType(Double(comb[source * 3 + destination]), dtype: dtype))
                            * Double(hidden[source * 2 + channel])
                    }
                    expected.append(
                        Double(
                            roundToDType(
                                Double(injected) + Double(roundToDType(mixed, dtype: dtype)),
                                dtype: dtype)))
                }
            }
            #expect(result.dtype == dtype)
            expectClose(result, expected)
        }

        @Test(arguments: [DType.float16, .bfloat16])
        func collapseRoundsOnlyAfterWeightedStreamReduction(dtype: DType) {
            let magnitude: Float = 256
            let logit: Float = 0.001
            let epsilon: Float = 1e-6
            let residual = MLXArray([magnitude, -magnitude]).reshaped([1, 1, 2, 1]).asType(dtype)
            let result = GLM5NextHyperConnection.collapse(
                residual, fn: MLXArray.zeros([8, 2]),
                base: MLXArray([logit, 0, 0, 0, 0, 0, 0, 0]),
                scale: MLXArray([Float(1), 1, 1]), rmsNormEpsilon: 1e-5,
                epsilon: epsilon, sinkhornIterations: 20)

            // The reference's sigmoid OUTPUT, epsilon additions, products and sum
            // are FP32. Double evaluates only this scalar sigmoid, not the reduction.
            let probability = Float(1 / (1 + exp(-Double(logit))))
            let zeroProbability: Float = 0.5
            func collapse(_ probability: Float) -> Float {
                let firstPre = probability + epsilon
                let secondPre = zeroProbability + epsilon
                let firstProduct = magnitude * firstPre
                let secondProduct = -magnitude * secondPre
                return firstProduct + secondProduct
            }
            let sum = collapse(probability)
            // FP32 cancellation lands exactly on an FP16 midpoint; ties-to-even
            // rounds DOWN. Cancelling epsilon in Double incorrectly rounds UP.
            #expect(sum == Float(0.063995361328125))
            #expect(roundToDType(Double(sum), dtype: dtype) == Float(0.06396484375))

            // Allow one FP32 sigmoid-output ULP across transcendental backends.
            // For these operands, adding epsilon preserves that ULP spacing,
            // multiplying by 256 is exact, and the subtraction is exact (Sterbenz).
            // Propagate that bound THROUGH the final cast, including midpoint ties.
            #expect(sum - collapse(probability.nextDown) == magnitude * probability.ulp)
            #expect(collapse(probability.nextUp) - sum == magnitude * probability.ulp)
            let lower = roundToDType(Double(collapse(probability.nextDown)), dtype: dtype)
            let upper = roundToDType(Double(collapse(probability.nextUp)), dtype: dtype)
            let actual = result.collapsed.asType(.float32).item(Float.self)
            #expect(result.collapsed.dtype == dtype)
            #expect(actual.isFinite && actual >= lower && actual <= upper)

            // Both incorrect early-cast paths are outside the derived interval:
            // FP16 gives 0.125; BF16 gives 0. No final-output tolerance can hide them.
            let firstPre = probability + epsilon
            let secondPre = zeroProbability + epsilon
            let roundedPre =
                magnitude * roundToDType(Double(firstPre), dtype: dtype)
                - magnitude * roundToDType(Double(secondPre), dtype: dtype)
            let roundedProducts =
                roundToDType(Double(magnitude * firstPre), dtype: dtype)
                + roundToDType(Double(-magnitude * secondPre), dtype: dtype)
            let wrong: Float = dtype == .float16 ? 0.125 : 0
            #expect(roundedPre == wrong && roundedProducts == wrong)
            #expect(wrong < lower || wrong > upper)
        }

        @Test func zeroInputAndExtremeLogitsStayFinite() {
            // Row softmax must subtract its maximum; an unshifted exp(1000) overflows.
            let base: [Float] = [
                -1000, 1000, 0, 1000, -1000, 0, 1000, -1000, 0, -1000, 1000, 0, 0, -1000, 1000,
            ]
            let result = GLM5NextHyperConnection.collapse(
                MLXArray.zeros([1, 1, 3, 2]), fn: MLXArray.zeros([15, 6]), base: MLXArray(base),
                scale: MLXArray([Float(1), 1, 1]), rmsNormEpsilon: 1e-5,
                epsilon: 1e-6, sinkhornIterations: 20)
            expectClose(result.collapsed, [0, 0])
            expectClose(result.post, [2, 0, 1])
            let combination = result.comb.asArray(Float.self)
            #expect(combination.allSatisfy { $0.isFinite && $0 > 0 })
            for column in 0 ..< 3 {
                let sum = (0 ..< 3).reduce(Float(0)) { $0 + combination[$1 * 3 + column] }
                #expect(abs(sum - 1) < 2e-6)
            }
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: [Double], absolute: Double = 2e-6,
            relative: Double = 2e-5
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == expected.count)
            for (index, pair) in zip(values, expected).enumerated() {
                let error = abs(Double(pair.0) - pair.1)
                #expect(
                    pair.0.isFinite && error <= absolute + relative * abs(pair.1),
                    "element \(index): actual \(pair.0), expected \(pair.1)")
            }
        }

        private func roundToDType(_ value: Double, dtype: DType) -> Float {
            switch dtype {
            case .float16:
                return Float(Float16(value))
            case .bfloat16:
                let bits = Float(value).bitPattern
                let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
                return Float(bitPattern: rounded & 0xffff_0000)
            default:
                return Float(value)
            }
        }

        private struct Fixture {
            var hidden: [Float] = [
                0.25, -0.5, 1, 2, -3, 0.75,
                -1.5, 0.25, 0.5, -2, 1.25, 3,
                2, 1, -0.25, 0.5, -1, -2,
                -0.75, 1.5, 2.5, -1, 0.25, 0.5,
            ]
            let fn: [Float] = [
                0.5, 0, -0.25, 0, 0, 0.125,
                0, -0.5, 0, 0.25, 0.125, 0,
                0.25, 0, 0.5, 0, -0.125, 0,
                0, 0.125, 0, -0.25, 0, 0.5,
                -0.25, 0.5, 0, 0, 0.125, 0,
                0, -0.125, 0.25, 0.5, 0, 0,
                0.5, 0, 0, -0.25, 0.125, 0,
                0, 0.25, -0.5, 0, 0, 0.125,
                -0.125, 0, 0.25, 0, 0.5, 0,
                0, -0.25, 0, 0.125, 0, 0.5,
                0.25, 0, 0.125, 0, -0.5, 0,
                0, 0.5, 0, -0.125, 0.25, 0,
                -0.5, 0, 0, 0.25, 0, 0.125,
                0, 0.125, 0.5, 0, -0.25, 0,
                0.125, 0, -0.25, 0.5, 0, 0,
            ]
            let base: [Float] = [
                0.125, -0.25, 0.5, -0.125, 0.25, -0.5,
                0.5, -0.25, 0.125, -0.5, 0.25, 0, 0.125, -0.125, 0.5,
            ]
            let scale: [Float] = [0.75, -1.25, 1.5]
            let normEpsilon: Float = 0.25
            let epsilon: Float = 0.015625

            func collapse(_ residual: MLXArray, iterations: Int, parameterType: DType = .float32)
                -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
            {
                GLM5NextHyperConnection.collapse(
                    residual, fn: MLXArray(fn).reshaped([15, 6]).asType(parameterType),
                    base: MLXArray(base).asType(parameterType),
                    scale: MLXArray(scale).asType(parameterType),
                    rmsNormEpsilon: normEpsilon, epsilon: epsilon, sinkhornIterations: iterations)
            }

            /// Scalar Double equations, independently indexed over tokens, streams and channels.
            func oracle(iterations: Int) -> (post: [Double], comb: [Double], collapsed: [Double]) {
                var allPost = [Double]()
                var allComb = [Double]()
                var allCollapsed = [Double]()
                let eps = Double(epsilon)
                for token in 0 ..< 4 {
                    let x = hidden[(token * 6) ..< (token * 6 + 6)].map(Double.init)
                    let inverseRMS =
                        1 / sqrt(x.reduce(0) { $0 + $1 * $1 } / 6 + Double(normEpsilon))
                    var projected = [Double](repeating: 0, count: 15)
                    for row in 0 ..< 15 {
                        for column in 0 ..< 6 {
                            projected[row] += x[column] * inverseRMS * Double(fn[row * 6 + column])
                        }
                        projected[row] =
                            projected[row] * Double(scale[row < 3 ? 0 : row < 6 ? 1 : 2])
                            + Double(base[row])
                    }
                    let pre = projected[0 ..< 3].map { 1 / (1 + exp(-$0)) + eps }
                    allPost += projected[3 ..< 6].map { 2 / (1 + exp(-$0)) }
                    var comb = [Double](repeating: 0, count: 9)
                    for row in 0 ..< 3 {
                        let logits = Array(projected[(6 + row * 3) ..< (9 + row * 3)])
                        let maximum = logits.max()!
                        let exponentials = logits.map { exp($0 - maximum) }
                        let denominator = exponentials.reduce(0, +)
                        for column in 0 ..< 3 {
                            comb[row * 3 + column] = exponentials[column] / denominator + eps
                        }
                    }
                    for iteration in 0 ..< iterations {
                        if iteration > 0 {
                            for row in 0 ..< 3 {
                                let denominator = comb[(row * 3) ..< (row * 3 + 3)].reduce(eps, +)
                                for column in 0 ..< 3 { comb[row * 3 + column] /= denominator }
                            }
                        }
                        for column in 0 ..< 3 {
                            let denominator = (0 ..< 3).reduce(eps) { $0 + comb[$1 * 3 + column] }
                            for row in 0 ..< 3 { comb[row * 3 + column] /= denominator }
                        }
                    }
                    allComb += comb
                    for channel in 0 ..< 2 {
                        allCollapsed.append(
                            (0 ..< 3).reduce(0) { $0 + pre[$1] * x[$1 * 2 + channel] })
                    }
                }
                return (allPost, allComb, allCollapsed)
            }
        }
    }
}
