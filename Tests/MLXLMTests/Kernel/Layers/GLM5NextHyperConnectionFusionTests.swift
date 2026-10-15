// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

private typealias HCKernel = GLM5NextHyperConnectionKernel

private func hcFusionFence(_ arrays: [MLXArray]) throws {
    try withError { error in
        eval(arrays)
        StreamOrDevice.default.stream.synchronize()
        try error.check()
    }
}

extension KernelTests {
    /// Fused mHC collapse/expand against the unchanged reference graph. The
    /// fused FP32 reductions use a fixed tree, so cross-implementation checks
    /// are tolerance-based while cross-shape checks of the fused path are exact.
    @Suite(.serialized)
    struct GLM5NextHyperConnectionFusionTests {
        private struct Fixture {
            let batch: Int
            let tokens: Int
            let width: Int
            let dtype: DType
            var channels: Int { 4 * width }

            func streams() -> MLXArray {
                let values: [Float] = (0 ..< batch * tokens * channels).map {
                    Float(($0 * 37 + 11) % 97 - 48) / 24
                }
                return MLXArray(values, [batch, tokens, 4, width]).asType(dtype)
            }

            func fn(dtype: DType) -> MLXArray {
                let scale = 1 / Float(channels).squareRoot()
                let values: [Float] = (0 ..< 24 * channels).map {
                    Float(($0 * 53 + 7) % 89 - 44) / 44 * scale
                }
                return MLXArray(values, [24, channels]).asType(dtype)
            }

            func base(dtype: DType) -> MLXArray {
                let values: [Float] = (0 ..< 24).map { Float(($0 * 7 + 3) % 19 - 9) / 8 }
                return MLXArray(values).asType(dtype)
            }

            func scale(dtype: DType) -> MLXArray {
                MLXArray([Float(0.731), -1.217, 1.527]).asType(dtype)
            }

            /// Q4 artifacts store fn in BF16 and base/scale in FP32.
            func reference(iterations: Int, fnType: DType)
                -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
            {
                GLM5NextHyperConnection.collapse(
                    streams(), fn: fn(dtype: fnType), base: base(dtype: .float32),
                    scale: scale(dtype: .float32), rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                    sinkhornIterations: iterations)
            }

            func fused(iterations: Int, fnType: DType) throws
                -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
            {
                try HCKernel.collapse(
                    streams(), fn: fn(dtype: fnType), base: base(dtype: .float32),
                    scale: scale(dtype: .float32), rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                    sinkhornIterations: iterations)
            }
        }

        /// One activation-dtype output ULP on top of FP32 reduction-order noise.
        private static func tolerance(_ dtype: DType) -> (absolute: Float, relative: Float) {
            switch dtype {
            case .float16: (0.01, 0.001)
            case .bfloat16: (0.02, 0.008)
            default: (1e-5, 1e-4)
            }
        }

        /// MLX_ENABLE_TF32 defaults on in this MLX, so an M>1 FP32 matmul in the
        /// GPU reference may run on NAX at TF32 precision. The fused path is full
        /// FP32; tight parity is asserted against the CPU (FP32) reference.
        private static let gpuReferenceSlack: Float = 5e-3

        private static func expectClose(
            _ actual: MLXArray, _ expected: [Float], absolute: Float, relative: Float,
            _ label: String, sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == expected.count, "\(label) count", sourceLocation: sourceLocation)
            let worst = zip(values, expected).map { pair -> Float in
                guard pair.0.isFinite, pair.1.isFinite else { return .infinity }
                return abs(pair.0 - pair.1) - (absolute + relative * abs(pair.1))
            }.max() ?? 0
            #expect(worst <= 0, "\(label) exceeds tolerance by \(worst)", sourceLocation: sourceLocation)
        }

        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        @Test func collapseMatchesGPUAndCPUReferenceAcrossShapesAndDTypes() throws {
            let shapes = [(1, 1, 8), (2, 3, 8), (1, 33, 70), (2, 2, 300), (1, 2, 1030)]
            for (batch, tokens, width) in shapes {
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    for iterations in [1, 2, 20] {
                        let fnType: DType = dtype == .float32 ? .float32 : .bfloat16
                        let fixture = Fixture(
                            batch: batch, tokens: tokens, width: width, dtype: dtype)
                        let cpu = try Device.withDefaultDevice(.cpu) {
                            let result = fixture.reference(iterations: iterations, fnType: fnType)
                            try hcFusionFence([result.post, result.comb, result.collapsed])
                            return [result.post, result.comb, result.collapsed].map {
                                $0.asType(.float32).asArray(Float.self)
                            }
                        }
                        try Device.withDefaultDevice(.gpu) {
                            let reference = fixture.reference(
                                iterations: iterations, fnType: fnType)
                            let fused = try fixture.fused(iterations: iterations, fnType: fnType)
                            try hcFusionFence([
                                reference.post, reference.comb, reference.collapsed, fused.post,
                                fused.comb, fused.collapsed,
                            ])
                            #expect(fused.post.dtype == .float32 && fused.comb.dtype == .float32)
                            #expect(fused.collapsed.dtype == dtype)
                            #expect(fused.post.shape == reference.post.shape)
                            #expect(fused.comb.shape == reference.comb.shape)
                            #expect(fused.collapsed.shape == reference.collapsed.shape)
                            let label = "B\(batch) T\(tokens) D\(width) \(dtype) it\(iterations)"
                            let collapsed = Self.tolerance(dtype)
                            let gpu: [[Float]] = [
                                reference.post, reference.comb, reference.collapsed,
                            ].map { $0.asType(.float32).asArray(Float.self) }
                            for (name, expected, slack) in [
                                ("cpu", cpu, Float(0)), ("gpu", gpu, Self.gpuReferenceSlack),
                            ] {
                                Self.expectClose(
                                    fused.post, expected[0], absolute: 2e-5 + slack,
                                    relative: 2e-4 + slack, "\(label) post vs \(name)")
                                Self.expectClose(
                                    fused.comb, expected[1], absolute: 2e-5 + slack,
                                    relative: 2e-4 + slack, "\(label) comb vs \(name)")
                                Self.expectClose(
                                    fused.collapsed, expected[2],
                                    absolute: collapsed.absolute + 4 * slack,
                                    relative: collapsed.relative + slack,
                                    "\(label) collapsed vs \(name)")
                            }
                        }
                    }
                }
            }
        }

        @Test func expandMatchesReferenceAndKeepsActivationBoundaries() throws {
            for dtype: DType in [.float32, .float16, .bfloat16] {
                for (batch, tokens, width) in [(1, 1, 8), (2, 3, 300), (1, 2, 1030)] {
                    let fixture = Fixture(batch: batch, tokens: tokens, width: width, dtype: dtype)
                    let outputValues: [Float] = (0 ..< batch * tokens * width).map {
                        Float(($0 * 29 + 5) % 83 - 41) / 16
                    }
                    func expandReference() throws -> (MLXArray, MLXArray, MLXArray) {
                        let coefficients = fixture.reference(iterations: 20, fnType: dtype)
                        let output = MLXArray(outputValues, [batch, tokens, width]).asType(dtype)
                        let reference = GLM5NextHyperConnection.expand(
                            output, residual: fixture.streams(), post: coefficients.post,
                            comb: coefficients.comb)
                        try hcFusionFence([coefficients.post, coefficients.comb, reference])
                        return (coefficients.post, coefficients.comb, reference)
                    }
                    // CPU coefficients feed both expansions; only the expand differs.
                    let (post, comb, cpu) = try Device.withDefaultDevice(.cpu) {
                        try expandReference()
                    }
                    try Device.withDefaultDevice(.gpu) {
                        let output = MLXArray(outputValues, [batch, tokens, width]).asType(dtype)
                        let gpu = GLM5NextHyperConnection.expand(
                            output, residual: fixture.streams(), post: post, comb: comb)
                        let fused = try HCKernel.expand(
                            output, residual: fixture.streams(), post: post, comb: comb)
                        try hcFusionFence([gpu, fused])
                        #expect(fused.dtype == dtype && fused.shape == cpu.shape)
                        let tolerance = Self.tolerance(dtype)
                        let label = "expand B\(batch) T\(tokens) D\(width) \(dtype)"
                        Self.expectClose(
                            fused, cpu.asType(.float32).asArray(Float.self),
                            absolute: tolerance.absolute, relative: tolerance.relative,
                            "\(label) vs cpu")
                        Self.expectClose(
                            fused, gpu.asType(.float32).asArray(Float.self),
                            absolute: tolerance.absolute + Self.gpuReferenceSlack,
                            relative: tolerance.relative + Self.gpuReferenceSlack,
                            "\(label) vs gpu")
                    }
                }
            }
        }

        /// The fixed reduction tree makes each row independent of batch, sequence
        /// and grid mapping, so token-by-token and chunked forwards agree exactly.
        @Test func fusedPathIsBitwiseInvariantToBatchAndSequenceShape() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .bfloat16] {
                    let fixture = Fixture(batch: 2, tokens: 9, width: 300, dtype: dtype)
                    let streams = fixture.streams()
                    let fn = fixture.fn(dtype: .bfloat16)
                    let base = fixture.base(dtype: .float32)
                    let scale = fixture.scale(dtype: .float32)
                    let output = streams[0..., 0..., 1, 0...] * 0.5
                    let full = try HCKernel.collapse(
                        streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                        epsilon: 1e-6, sinkhornIterations: 20)
                    let fullExpand = try HCKernel.expand(
                        output, residual: streams, post: full.post, comb: full.comb)
                    try hcFusionFence([full.post, full.comb, full.collapsed, fullExpand])
                    for b in 0 ..< 2 {
                        for t in 0 ..< 9 {
                            let row = streams[b ..< b + 1, t ..< t + 1]
                            let single = try HCKernel.collapse(
                                row, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                                epsilon: 1e-6, sinkhornIterations: 20)
                            let singleExpand = try HCKernel.expand(
                                output[b ..< b + 1, t ..< t + 1], residual: row,
                                post: single.post, comb: single.comb)
                            try hcFusionFence([
                                single.post, single.comb, single.collapsed, singleExpand,
                            ])
                            let slice = { (array: MLXArray) in array[b ..< b + 1, t ..< t + 1] }
                            #expect(Self.bits(single.post) == Self.bits(slice(full.post)))
                            #expect(Self.bits(single.comb) == Self.bits(slice(full.comb)))
                            #expect(Self.bits(single.collapsed) == Self.bits(slice(full.collapsed)))
                            #expect(Self.bits(singleExpand) == Self.bits(slice(fullExpand)))
                        }
                    }
                }
            }
        }

        /// Same derivation as GLM5NextHyperConnectionTests: the FP32 cancellation
        /// lands on an FP16 tie, so any early coefficient or product rounding,
        /// or epsilon cancelled in wider precision, leaves the allowed interval.
        @Test(arguments: [DType.float16, .bfloat16])
        func collapseRoundsOnlyAfterWeightedStreamReduction(dtype: DType) throws {
            try Device.withDefaultDevice(.gpu) {
                let magnitude: Float = 256
                let logit: Float = 0.001
                let epsilon: Float = 1e-6
                var base = [Float](repeating: 0, count: 24)
                base[0] = logit
                let result = try HCKernel.collapse(
                    MLXArray([magnitude, -magnitude, 0, 0], [1, 1, 4, 1]).asType(dtype),
                    fn: .zeros([24, 4]), base: MLXArray(base), scale: .ones([3]),
                    rmsNormEpsilon: 1e-5, epsilon: epsilon, sinkhornIterations: 20)
                try hcFusionFence([result.collapsed])
                let probability = Float(1 / (1 + Foundation.exp(-Double(logit))))
                func collapse(_ probability: Float) -> Float {
                    let first = magnitude * (probability + epsilon)
                    let second = -magnitude * (Float(0.5) + epsilon)
                    return first + second
                }
                func round(_ value: Float) -> Float {
                    if dtype == .float16 { return Float(Float16(value)) }
                    let bits = value.bitPattern
                    return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff_0000)
                }
                let lower = round(collapse(probability.nextDown))
                let upper = round(collapse(probability.nextUp))
                let actual = result.collapsed.asType(.float32).item(Float.self)
                #expect(result.collapsed.dtype == dtype)
                #expect(actual.isFinite && actual >= lower && actual <= upper)
                let wrong: Float = dtype == .float16 ? 0.125 : 0
                #expect(wrong < lower || wrong > upper)
            }
        }

        @Test func refusesInvalidInputsCPUAndForeignStreams() throws {
            let fixture = Fixture(batch: 1, tokens: 2, width: 8, dtype: .float32)
            func collapse(
                streams: MLXArray? = nil, fn: MLXArray? = nil, base: MLXArray? = nil,
                scale: MLXArray? = nil, iterations: Int = 20, rmsEpsilon: Float = 1e-5
            ) throws {
                _ = try HCKernel.collapse(
                    streams ?? fixture.streams(), fn: fn ?? fixture.fn(dtype: .float32),
                    base: base ?? fixture.base(dtype: .float32),
                    scale: scale ?? fixture.scale(dtype: .float32), rmsNormEpsilon: rmsEpsilon,
                    epsilon: 1e-6, sinkhornIterations: iterations)
            }
            Device.withDefaultDevice(.gpu) {
                for invalid in [
                    { try collapse(streams: MLXArray.zeros([1, 2, 3, 8])) },
                    { try collapse(streams: MLXArray.zeros([1, 2, 4, 8], dtype: .int32)) },
                    { try collapse(fn: MLXArray.zeros([24, 31])) },
                    { try collapse(base: MLXArray.zeros([23])) },
                    { try collapse(scale: MLXArray.zeros([3], dtype: .int32)) },
                    { try collapse(iterations: 0) }, { try collapse(iterations: 129) },
                    { try collapse(rmsEpsilon: 0) },
                ] as [() throws -> Void] {
                    #expect(throws: HCKernel.Failure.self) { try invalid() }
                }
                let post = MLXArray.zeros([1, 2, 4])
                let comb = MLXArray.zeros([1, 2, 4, 4])
                let output = MLXArray.zeros([1, 2, 8])
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.expand(
                        output, residual: fixture.streams(), post: post.asType(.float16),
                        comb: comb)
                }
                #expect(throws: HCKernel.Failure.self) {
                    try HCKernel.expand(
                        output.asType(.bfloat16), residual: fixture.streams(), post: post,
                        comb: comb)
                }
            }
            for cpuStream in [false, true] {
                func run() {
                    #expect(throws: HCKernel.Failure.unsupportedDeviceOrStream) {
                        try collapse()
                    }
                    #expect(throws: HCKernel.Failure.unsupportedDeviceOrStream) {
                        try HCKernel.expand(
                            .zeros([1, 2, 8]), residual: fixture.streams(),
                            post: .zeros([1, 2, 4]), comb: .zeros([1, 2, 4, 4]))
                    }
                }
                if cpuStream {
                    Stream.withNewDefaultStream(device: .cpu) { run() }
                } else {
                    Device.withDefaultDevice(.cpu) { run() }
                }
            }
        }
    }

    /// Prepared only: one mHC site (collapse + expand) at the real GLM-5.3
    /// geometry, never used to assert the full-model throughput floors.
    @Suite(
        .serialized,
        .enabled(
            if:
                ProcessInfo.processInfo.environment["GLM53_RUN_HC_BENCHMARK"] == "1"
                && ProcessInfo.processInfo.environment["GLM53_GPU_LANE_LOCKED"] == "1"))
    struct GLM5NextHyperConnectionFusionBenchmark {
        @Test func referenceVsFusedSiteAtD4096() throws {
            try Device.withDefaultDevice(.gpu) {
                let width = 4096
                let fn = (sin(MLXArray(0 ..< 24 * 4 * width).asType(.float32) * 0.013) * 0.01)
                    .reshaped(24, 4 * width).asType(.bfloat16)
                let base = cos(MLXArray(0 ..< 24).asType(.float32) * 0.17)
                let scale = MLXArray([Float(0.75), -1.25, 1.5])
                for tokens in [1, 128, 512] {
                    let streams = sin(MLXArray(0 ..< tokens * 4 * width).asType(.float32) * 0.037)
                        .reshaped(1, tokens, 4, width).asType(.bfloat16)
                    let output = cos(MLXArray(0 ..< tokens * width).asType(.float32) * 0.029)
                        .reshaped(1, tokens, width).asType(.bfloat16)
                    try hcFusionFence([fn, base, scale, streams, output])
                    func run(_ fused: Bool) throws -> [MLXArray] {
                        if fused {
                            let site = try HCKernel.collapse(
                                streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                                epsilon: 1e-6, sinkhornIterations: 20)
                            let expanded = try HCKernel.expand(
                                output, residual: streams, post: site.post, comb: site.comb)
                            return [site.collapsed, expanded]
                        }
                        let site = GLM5NextHyperConnection.collapse(
                            streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                            epsilon: 1e-6, sinkhornIterations: 20)
                        let expanded = GLM5NextHyperConnection.expand(
                            output, residual: streams, post: site.post, comb: site.comb)
                        return [site.collapsed, expanded]
                    }
                    for _ in 0 ..< 3 {
                        for fused in [false, true] { try hcFusionFence(run(fused)) }
                    }
                    let clock = ContinuousClock()
                    for sample in 0 ..< 12 {
                        for fused in (sample.isMultiple(of: 2) ? [false, true] : [true, false]) {
                            Stream.gpu.synchronize()
                            Memory.peakMemory = 0
                            let start = clock.now
                            try hcFusionFence(run(fused))
                            let elapsed = start.duration(to: clock.now).components
                            let record: [String: Any] = [
                                "scope": "hc-site-collapse-expand-not-full-model-throughput",
                                "backend": fused ? "fused" : "reference", "sample": sample,
                                "batch": 1, "tokens": tokens, "width": width, "iterations": 20,
                                "milliseconds": Double(elapsed.seconds) * 1000
                                    + Double(elapsed.attoseconds) / 1e15,
                                "peakBytes": Memory.peakMemory,
                            ]
                            print(
                                String(
                                    decoding: try JSONSerialization.data(
                                        withJSONObject: record, options: [.sortedKeys]),
                                    as: UTF8.self))
                        }
                    }
                }
            }
        }
    }
}
