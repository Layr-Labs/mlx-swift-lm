// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

private typealias HCKernel = GLM5NextHyperConnectionKernel

private enum HCCandidateCheckError: Error {
    case nonfiniteOrMismatch
}

/// Literal coefficient portion of the unchanged baseline, not the candidate.
private func hcReference(
    projected: MLXArray, base: MLXArray, scale: MLXArray, epsilon: Float, iterations: Int
) -> HCKernel.Coefficients {
    let pre = sigmoid(projected[.ellipsis, ..<4] * scale[0] + base[..<4]) + epsilon
    let post = 2 * sigmoid(projected[.ellipsis, 4 ..< 8] * scale[1] + base[4 ..< 8])
    let logits =
        projected[.ellipsis, 8...].reshaped(projected.dim(0), projected.dim(1), 4, 4)
        * scale[2] + base[8...].reshaped(4, 4)
    var comb = softmax(logits, axis: -1, precise: true) + epsilon
    comb = comb / (comb.sum(axis: -2, keepDims: true) + epsilon)
    for _ in 1 ..< iterations {
        comb = comb / (comb.sum(axis: -1, keepDims: true) + epsilon)
        comb = comb / (comb.sum(axis: -2, keepDims: true) + epsilon)
    }
    return .init(pre: pre, post: post, comb: comb)
}

private func hcFence(_ arrays: [MLXArray]) throws {
    try withError { error in
        eval(arrays)
        StreamOrDevice.default.stream.synchronize()
        try error.check()
    }
}

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextHyperConnectionKernelTests {
        private struct Fixture {
            let batch: Int
            let tokens: Int
            let projected: [Float]
            let base: [Float]
            let scale: [Float] = [0.75, -1.25, 1.5]

            init(batch: Int, tokens: Int) {
                self.batch = batch
                self.tokens = tokens
                projected = (0 ..< batch * tokens * 24).map {
                    Float(($0 * 17 + 11) % 101 - 50) / 16
                }
                base = (0 ..< 24).map { Float(($0 * 7 + 3) % 19 - 9) / 8 }
            }

            func arrays() -> (projected: MLXArray, base: MLXArray, scale: MLXArray) {
                (MLXArray(projected, [batch, tokens, 24]), MLXArray(base), MLXArray(scale))
            }

            /// Independent scalar Double oracle after the FP32 affine boundary.
            /// Epsilon is added AFTER a completed sum, never used as its seed.
            func oracle(epsilon: Float, iterations: Int) -> [[Float]] {
                var pre: [Float] = []
                var post: [Float] = []
                var allComb: [Float] = []
                for token in 0 ..< batch * tokens {
                    var affine: [Double] = []
                    for index in 0 ..< 24 {
                        let product =
                            projected[token * 24 + index]
                            * scale[index < 4 ? 0 : index < 8 ? 1 : 2]
                        affine.append(Double(product + base[index]))
                    }
                    pre += affine[..<4].map {
                        Float(1 / (1 + Foundation.exp(-$0)) + Double(epsilon))
                    }
                    post += affine[4 ..< 8].map { Float(2 / (1 + Foundation.exp(-$0))) }
                    var comb = [Double](repeating: 0, count: 16)
                    for row in 0 ..< 4 {
                        let logits = Array(affine[(8 + row * 4) ..< (12 + row * 4)])
                        let maximum = logits.max()!
                        let exponents = logits.map { Foundation.exp($0 - maximum) }
                        let denominator = exponents.reduce(0, +)
                        for column in 0 ..< 4 {
                            comb[row * 4 + column] =
                                exponents[column] / denominator + Double(epsilon)
                        }
                    }
                    for iteration in 0 ..< iterations {
                        if iteration > 0 {
                            for row in 0 ..< 4 {
                                let sum = comb[(row * 4) ..< (row * 4 + 4)].reduce(0, +)
                                let denominator = sum + Double(epsilon)
                                for column in 0 ..< 4 { comb[row * 4 + column] /= denominator }
                            }
                        }
                        for column in 0 ..< 4 {
                            let sum = (0 ..< 4).reduce(0.0) { $0 + comb[$1 * 4 + column] }
                            let denominator = sum + Double(epsilon)
                            for row in 0 ..< 4 { comb[row * 4 + column] /= denominator }
                        }
                    }
                    allComb += comb.map(Float.init)
                }
                return [pre, post, allComb]
            }
        }

        private static func matches(
            _ actual: [Float], _ expected: [Float], absolute: Float = 2e-6,
            relative: Float = 2e-5
        ) -> Bool {
            guard actual.count == expected.count,
                actual.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite)
            else { return false }
            return zip(actual, expected).allSatisfy {
                abs($0.0 - $0.1) <= absolute + relative * abs($0.1)
            }
        }

        private func expectClose(
            _ actual: HCKernel.Coefficients, _ expected: [[Float]],
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            try hcFence(actual.evaluationTargets)
            #expect(expected.count == 3, sourceLocation: sourceLocation)
            for (array, expected) in zip(actual.evaluationTargets, expected) {
                #expect(array.dtype == .float32, sourceLocation: sourceLocation)
                #expect(
                    Self.matches(array.asArray(Float.self), expected),
                    "nonfinite value or coefficient mismatch", sourceLocation: sourceLocation)
            }
        }

        @Test func uniformFixturePinsFirstColumnStepAndEpsilonPlacement() throws {
            try Device.withDefaultDevice(.gpu) {
                let result = try HCKernel.apply(
                    projected: .zeros([1, 1, 24]), base: .zeros([24]), scale: .ones([3]),
                    epsilon: 0.125, sinkhornIterations: 1)
                try expectClose(
                    result,
                    [
                        Array(repeating: 0.625, count: 4), Array(repeating: 1, count: 4),
                        Array(repeating: 3.0 / 13.0, count: 16),
                    ])
                #expect(result.pre.shape == [1, 1, 4])
                #expect(result.post.shape == [1, 1, 4])
                #expect(result.comb.shape == [1, 1, 4, 4])
            }
        }

        @Test func metalMatchesCPUAndScalarOracleAcrossBatchesAndIterationCounts() throws {
            for (batch, tokens) in [(1, 1), (2, 3), (1, 33), (2, 128)] {
                let fixture = Fixture(batch: batch, tokens: tokens)
                for iterations in [1, 2, 20, 128] {
                    let epsilon: Float = iterations == 2 ? 0.015625 : 1e-6
                    let cpu = try Device.withDefaultDevice(.cpu) {
                        let input = fixture.arrays()
                        let result = hcReference(
                            projected: input.projected, base: input.base, scale: input.scale,
                            epsilon: epsilon, iterations: iterations)
                        try hcFence(result.evaluationTargets)
                        return result.evaluationTargets.map { $0.asArray(Float.self) }
                    }
                    try Device.withDefaultDevice(.gpu) {
                        let input = fixture.arrays()
                        let result = try HCKernel.apply(
                            projected: input.projected, base: input.base, scale: input.scale,
                            epsilon: epsilon, sinkhornIterations: iterations)
                        let baseline = hcReference(
                            projected: input.projected, base: input.base, scale: input.scale,
                            epsilon: epsilon, iterations: iterations)
                        try hcFence(baseline.evaluationTargets)
                        try expectClose(result, cpu)
                        try expectClose(
                            result, baseline.evaluationTargets.map { $0.asArray(Float.self) })
                        try expectClose(
                            result, fixture.oracle(epsilon: epsilon, iterations: iterations))
                    }
                }
            }
        }

        @Test func extremeLogitsUseStableSigmoidAndShiftedSoftmax() throws {
            try Device.withDefaultDevice(.gpu) {
                let base: [Float] = [
                    -1000, 1000, -80, 80, 1000, -1000, 80, -80,
                    1000, -1000, 0, 1000, -1000, 1000, -1000, 0,
                    0, -1000, 1000, -1000, -1000, 0, -1000, 1000,
                ]
                let projected = MLXArray.zeros([2, 3, 24])
                let b = MLXArray(base)
                let scale = MLXArray([Float(1), -1, 1])
                for epsilon: Float in [1e-6, 0.125] {
                    let result = try HCKernel.apply(
                        projected: projected, base: b, scale: scale,
                        epsilon: epsilon, sinkhornIterations: 20)
                    let baseline = hcReference(
                        projected: projected, base: b, scale: scale,
                        epsilon: epsilon, iterations: 20)
                    try hcFence(baseline.evaluationTargets)
                    try expectClose(
                        result, baseline.evaluationTargets.map { $0.asArray(Float.self) })
                    #expect(result.comb.asArray(Float.self).allSatisfy { $0 > 0 })
                }
            }
        }

        @Test func affineMultiplyRoundsBeforeBaseAddition() throws {
            try Device.withDefaultDevice(.gpu) {
                var projected = [Float](repeating: 0, count: 24)
                var base = [Float](repeating: 0, count: 24)
                projected[0] = 8193
                base[0] = -67_108_864
                // FP32 product 8193*8191 rounds to 67108864, then cancels to 0.
                // A fused multiply-add would yield -1 and a different sigmoid.
                let result = try HCKernel.apply(
                    projected: MLXArray(projected, [1, 1, 24]), base: MLXArray(base),
                    scale: MLXArray([Float(8191), 1, 1]),
                    epsilon: 0.125, sinkhornIterations: 20)
                try hcFence(result.evaluationTargets)
                #expect(result.pre.asArray(Float.self)[0] == 0.625)
            }
        }

        @Test func cancellationAndActivationRoundingRemainOutsideTheKernel() throws {
            try Device.withDefaultDevice(.gpu) {
                let epsilon: Float = 1e-6
                let logit: Float = 0.001
                var base = [Float](repeating: 0, count: 24)
                base[0] = logit
                let b = MLXArray(base)
                let scale = MLXArray.ones([3])
                let result = try HCKernel.apply(
                    projected: .zeros([1, 1, 24]), base: b, scale: scale,
                    epsilon: epsilon, sinkhornIterations: 20)
                let probability = Float(1 / (1 + Foundation.exp(-Double(logit))))
                func cancelled(_ p: Float) -> Float {
                    let first = (p + epsilon) * 256
                    let second = (Float(0.5) + epsilon) * -256
                    return first + second
                }
                let lower = cancelled(probability.nextDown)
                let upper = cancelled(probability.nextUp)
                for dtype: DType in [.float16, .bfloat16] {
                    let hidden = MLXArray([Float(256), -256, 0, 0], [1, 1, 4, 1]).asType(dtype)
                    let collapsed =
                        (result.pre.expandedDimensions(axis: -1) * hidden.asType(.float32))
                        .sum(axis: 2).asType(dtype)
                    try hcFence([collapsed])
                    let actual = collapsed.asArray(Float.self)[0]
                    #expect(actual.isFinite)
                    #expect(
                        actual >= Self.round(lower, to: dtype)
                            && actual <= Self.round(upper, to: dtype))
                    // Neither early coefficient nor early product rounding is acceptable.
                    let wrong = (result.pre.asType(dtype).expandedDimensions(axis: -1) * hidden)
                        .sum(axis: 2).asType(dtype).asArray(Float.self)[0]
                    #expect(
                        wrong < Self.round(lower, to: dtype) || wrong > Self.round(upper, to: dtype)
                    )
                    let earlyProducts =
                        (result.pre.expandedDimensions(axis: -1) * hidden.asType(.float32))
                        .asType(dtype).asType(.float32).sum(axis: 2).asType(dtype)
                    try hcFence([earlyProducts])
                    let wrongProducts = earlyProducts.asArray(Float.self)[0]
                    #expect(
                        wrongProducts < Self.round(lower, to: dtype)
                            || wrongProducts > Self.round(upper, to: dtype))
                }
            }
        }

        @Test func unchangedCollapseAndExpandKeepFP16AndBF16Boundaries() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    func tensor(_ shape: [Int], salt: Int) -> MLXArray {
                        MLXArray(
                            (0 ..< shape.reduce(1, *)).map {
                                Float(($0 * 7 + salt) % 31 - 15) / 13
                            }, shape
                        ).asType(dtype)
                    }
                    let hidden = tensor([2, 3, 4, 8], salt: 1)
                    let fn = tensor([24, 32], salt: 2)
                    let base = tensor([24], salt: 3)
                    let scale = MLXArray([Float(0.731), -1.217, 1.527]).asType(dtype)
                    let flat = hidden.reshaped(2, 3, 32).asType(.float32)
                    let normalized =
                        flat * rsqrt(flat.square().mean(axis: -1, keepDims: true) + 0.25)
                    let projected = matmul(normalized, fn.asType(.float32).T)
                    let coefficients = try HCKernel.apply(
                        projected: projected, base: base.asType(.float32),
                        scale: scale.asType(.float32),
                        epsilon: 0.015625, sinkhornIterations: 20)
                    let baseline = GLM5NextHyperConnection.collapse(
                        hidden, fn: fn, base: base, scale: scale,
                        rmsNormEpsilon: 0.25, epsilon: 0.015625, sinkhornIterations: 20)
                    let collapsed =
                        (coefficients.pre.expandedDimensions(axis: -1) * hidden.asType(.float32))
                        .sum(axis: 2).asType(dtype)
                    let output = tensor([2, 3, 8], salt: 4)
                    let expanded = GLM5NextHyperConnection.expand(
                        output, residual: hidden, post: coefficients.post, comb: coefficients.comb)
                    let expectedExpanded = GLM5NextHyperConnection.expand(
                        output, residual: hidden, post: baseline.post, comb: baseline.comb)
                    try hcFence([collapsed, baseline.collapsed, expanded, expectedExpanded])
                    #expect(collapsed.dtype == dtype && expanded.dtype == dtype)
                    #expect(
                        coefficients.pre.dtype == .float32 && coefficients.post.dtype == .float32)
                    let relative: Float =
                        dtype == .float32 ? 2e-5 : dtype == .float16 ? 0.001 : 0.008
                    #expect(
                        Self.matches(
                            collapsed.asArray(Float.self), baseline.collapsed.asArray(Float.self),
                            absolute: dtype == .float32 ? 2e-6 : 0.01, relative: relative))
                    #expect(
                        Self.matches(
                            expanded.asArray(Float.self), expectedExpanded.asArray(Float.self),
                            absolute: dtype == .float32 ? 2e-6 : 0.02, relative: relative))
                }
            }
        }

        @Test func stridedInputsAndIndependentOutputsRetainImmutableStorage() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = Fixture(batch: 2, tokens: 3)
                let packed = fixture.arrays()
                let projected = stacked([packed.projected, packed.projected], axis: -1)
                    .reshaped(2, 3, 48)[.ellipsis, .stride(by: 2)]
                let base = stacked([packed.base, packed.base], axis: -1).flattened()[.stride(by: 2)]
                let scale = stacked([packed.scale, packed.scale], axis: -1).flattened()[
                    .stride(by: 2)]
                let inputs = [projected, base, scale]
                let before = inputs.map { $0.asArray(Float.self).map(\.bitPattern) }
                let first = try HCKernel.apply(
                    projected: projected, base: base, scale: scale,
                    epsilon: 1e-6, sinkhornIterations: 20)
                let second = try HCKernel.apply(
                    projected: -projected, base: base, scale: scale,
                    epsilon: 0.125, sinkhornIterations: 2)
                try hcFence(first.evaluationTargets + second.evaluationTargets)
                let repeated = try HCKernel.apply(
                    projected: projected, base: base, scale: scale,
                    epsilon: 1e-6, sinkhornIterations: 20)
                try expectClose(first, fixture.oracle(epsilon: 1e-6, iterations: 20))
                try hcFence(repeated.evaluationTargets)
                for (a, b) in zip(first.evaluationTargets, repeated.evaluationTargets) {
                    #expect(
                        a.asArray(Float.self).map(\.bitPattern)
                            == b.asArray(Float.self).map(\.bitPattern))
                }
                for (array, snapshot) in zip(inputs, before) {
                    #expect(array.asArray(Float.self).map(\.bitPattern) == snapshot)
                }
                func temporary() throws -> HCKernel.Coefficients {
                    let inputs = fixture.arrays()
                    return try HCKernel.apply(
                        projected: inputs.projected, base: inputs.base, scale: inputs.scale,
                        epsilon: 1e-6, sinkhornIterations: 20)
                }
                try expectClose(temporary(), fixture.oracle(epsilon: 1e-6, iterations: 20))
            }
        }

        @Test func comparisonRejectsNonfiniteValuesBeforeArithmetic() {
            for value in [Float.nan, .infinity, -.infinity] {
                #expect(!Self.matches([value], [value], absolute: .infinity, relative: .infinity))
                #expect(!Self.matches([value], [0]))
                #expect(!Self.matches([0], [value]))
            }
        }

        @Test func rejectsInvalidGeometryDtypesAndStreams() throws {
            for shape in [
                [1, 24], [0, 1, 24], [1, 0, 24], [1, 1, 23], [Int.max, 1, 24],
                [65_536, 65_536, 24],
            ] {
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.Layout(
                        projectedShape: shape, hcMult: 4, epsilon: 1e-6,
                        sinkhornIterations: 20)
                }
            }
            for hcMult in [0, 1, 2, 3, 5, Int.max] {
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.Layout(
                        projectedShape: [1, 1, 24], hcMult: hcMult, epsilon: 1e-6,
                        sinkhornIterations: 20)
                }
            }
            for epsilon: Float in [0, -1, .nan, .infinity] {
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.Layout(
                        projectedShape: [1, 1, 24], hcMult: 4, epsilon: epsilon,
                        sinkhornIterations: 20)
                }
            }
            for iterations in [0, -1, 129, Int.max] {
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.Layout(
                        projectedShape: [1, 1, 24], hcMult: 4, epsilon: 1e-6,
                        sinkhornIterations: iterations)
                }
            }
            let input = Fixture(batch: 1, tokens: 1).arrays()
            for index in 0 ..< 3 {
                for dtype: DType in [.float16, .bfloat16, .int32] {
                    var arrays = [input.projected, input.base, input.scale]
                    arrays[index] = arrays[index].asType(dtype)
                    #expect(throws: HCKernel.Failure.self) {
                        try HCKernel.apply(
                            projected: arrays[0], base: arrays[1], scale: arrays[2],
                            epsilon: 1e-6, sinkhornIterations: 20)
                    }
                }
            }
            #expect(throws: HCKernel.Failure.self) {
                try HCKernel.apply(
                    projected: input.projected, base: .zeros([23]), scale: input.scale,
                    epsilon: 1e-6, sinkhornIterations: 20)
            }
            #expect(throws: HCKernel.Failure.self) {
                try HCKernel.apply(
                    projected: input.projected, base: input.base, scale: .zeros([4]),
                    epsilon: 1e-6, sinkhornIterations: 20)
            }
            for cpuStream in [false, true] {
                func run() {
                    #expect(throws: HCKernel.Failure.unsupportedDeviceOrStream) {
                        try HCKernel.apply(
                            projected: input.projected, base: input.base, scale: input.scale,
                            epsilon: 1e-6, sinkhornIterations: 20)
                    }
                }
                if cpuStream {
                    Stream.withNewDefaultStream(device: .cpu) { run() }
                } else {
                    Device.withDefaultDevice(.cpu) { run() }
                }
            }
        }

        private static func round(_ value: Float, to dtype: DType) -> Float {
            if dtype == .float16 { return Float(Float16(value)) }
            let bits = value.bitPattern
            return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff_0000)
        }
    }

    /// Prepared only: never used to assert the full-model throughput floors.
    @Suite(
        .serialized,
        .enabled(
            if:
                ProcessInfo.processInfo.environment["GLM53_RUN_HC_BENCHMARK"] == "1"
                && ProcessInfo.processInfo.environment["GLM53_GPU_LANE_LOCKED"] == "1"))
    struct GLM5NextHyperConnectionKernelBenchmark {
        @Test func interleavedFourStreamCoefficientStage() throws {
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 128, 512] {
                    let projected = sin(MLXArray(0 ..< tokens * 24).asType(.float32) * 0.037)
                        .reshaped(1, tokens, 24)
                    let base = cos(MLXArray(0 ..< 24).asType(.float32) * 0.17)
                    let scale = MLXArray([Float(0.75), -1.25, 1.5])
                    try hcFence([projected, base, scale])
                    func run(_ native: Bool) throws -> HCKernel.Coefficients {
                        if native {
                            return try HCKernel.apply(
                                projected: projected, base: base, scale: scale,
                                epsilon: 1e-6, sinkhornIterations: 20)
                        }
                        return hcReference(
                            projected: projected, base: base, scale: scale,
                            epsilon: 1e-6, iterations: 20)
                    }
                    do {
                        let reference = try run(false)
                        let native = try run(true)
                        try hcFence(reference.evaluationTargets + native.evaluationTargets)
                        for (a, b) in zip(reference.evaluationTargets, native.evaluationTargets) {
                            let expected = a.asArray(Float.self)
                            let actual = b.asArray(Float.self)
                            guard expected.count == actual.count,
                                expected.allSatisfy(\.isFinite), actual.allSatisfy(\.isFinite),
                                zip(expected, actual).allSatisfy({
                                    abs($0.0 - $0.1) <= 2e-6 + 2e-5 * abs($0.0)
                                })
                            else { throw HCCandidateCheckError.nonfiniteOrMismatch }
                        }
                    }
                    // Warm compilation and allocation separately from timing.
                    for _ in 0 ..< 3 {
                        for native in [false, true] { try hcFence(run(native).evaluationTargets) }
                    }
                    let clock = ContinuousClock()
                    for sample in 0 ..< 12 {
                        for native in (sample.isMultiple(of: 2) ? [false, true] : [true, false]) {
                            Stream.gpu.synchronize()
                            Memory.peakMemory = 0
                            let active = Memory.activeMemory
                            let start = clock.now
                            try hcFence(run(native).evaluationTargets)
                            let elapsed = start.duration(to: clock.now).components
                            let milliseconds =
                                Double(elapsed.seconds) * 1000
                                + Double(elapsed.attoseconds) / 1e15
                            let record: [String: Any] = [
                                "scope": "hc-coefficients-only-not-full-model-throughput",
                                "backend": native ? "native" : "reference", "sample": sample,
                                "batch": 1, "tokens": tokens, "hcMult": 4, "iterations": 20,
                                "milliseconds": milliseconds, "activeBeforeBytes": active,
                                "peakBytes": Memory.peakMemory,
                            ]
                            print(
                                String(
                                    decoding: try JSONSerialization.data(
                                        withJSONObject: record,
                                        options: [.sortedKeys]), as: UTF8.self))
                        }
                    }
                }
            }
        }
    }
}
