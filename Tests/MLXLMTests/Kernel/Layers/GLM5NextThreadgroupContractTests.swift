// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// These are the production mHC kernels, not a launch-size mock.
    /// Run on the CI Mac that rejected 1024 threads at limits of 832/896.
    @Suite(.serialized)
    struct GLM5NextThreadgroupContractTests {
        private typealias Kernel = GLM5NextHyperConnectionKernel

        /// (B,T,D): split, the CI CBv2 first step, prefill tail, and both tile heights.
        private static let geometries = [
            [1, 1, 4096], [2, 3, 1000], [1, 8, 33],
            [2, 6, 64], [1, 9, 4], [1, 17, 4096], [1, 4100, 96],
        ]

        private func evaluate(_ arrays: [MLXArray], label: String) throws {
            print("GLM native launch: \(label)")
            try withError { error in
                eval(arrays)
                try error.check()
            }
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float, label: String
        ) {
            #expect(actual.shape == expected.shape && actual.dtype == expected.dtype, "\(label)")
            #expect(isFinite(actual).all().item(Bool.self), "\(label) finite")
            #expect(isFinite(expected).all().item(Bool.self), "\(label) reference finite")
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                .item(Float.self)
            #expect(error < tolerance, "\(label) max error \(error)")
        }

        private func capture(_ arrays: [MLXArray], key: String) throws {
            guard let directory = ProcessInfo.processInfo.environment["GLM_TG_CAPTURE_DIR"] else {
                return
            }
            let root = URL(fileURLWithPath: directory, isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            for (index, array) in arrays.enumerated() {
                let bits = array.asType(.float32).asArray(Float.self).map(\.bitPattern)
                let data = bits.withUnsafeBytes { Data($0) }
                try data.write(to: root.appendingPathComponent("\(key)-\(index).bits"))
            }
        }

        private func measure(_ operation: () throws -> [MLXArray]) throws -> [Double] {
            var samples = [Double]()
            for iteration in 0 ..< 60 {
                let start = DispatchTime.now().uptimeNanoseconds
                let arrays = try operation()
                try withError { error in
                    eval(arrays)
                    try error.check()
                }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                if iteration >= 10 { samples.append(elapsed) }
            }
            return samples
        }

        @Test(arguments: [DType.float32, .bfloat16, .float16], geometries)
        func actualNativeCollapseKeepsItsFullThreadgroup(
            dtype: DType, geometry: [Int]
        ) throws {
            try Device.withDefaultDevice(.gpu) {
                let batch = geometry[0]
                let tokens = geometry[1]
                let width = geometry[2]
                let split = batch * tokens <= Kernel.splitCollapseMaximumRows
                let label = "\(dtype) B\(batch) T\(tokens) D\(width)"
                let keys = MLXRandom.split(key: MLXRandom.key(UInt64(width + tokens)), into: 8)
                let streams = MLXRandom.normal([batch, tokens, 4, width], key: keys[0])
                    .asType(dtype)
                let fn = (MLXRandom.normal([24, 4 * width], key: keys[1]) * 0.05).asType(dtype)
                let base = (MLXRandom.normal([24], key: keys[2]) * 0.5).asType(dtype)
                let scale = MLXRandom.uniform(
                    low: Float(0.5), high: Float(1.5), [3], key: keys[3]
                ).asType(dtype)
                let normWeight = MLXRandom.uniform(
                    low: Float(0.5), high: Float(1.5), [width], key: keys[4]
                ).asType(dtype)
                let norm = GLM5NextRMSNorm(weight: normWeight, epsilon: 1e-5)
                // Same bounds as GLM5NextHyperConnectionNormFusionTests.
                let normTolerance: Float = dtype == .float32 ? 1e-4 : 0.05

                let reference = try Kernel.collapse(
                    streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                    epsilon: 1e-6, sinkhornIterations: 20)
                let normalized = norm(reference.collapsed)
                try evaluate(
                    [reference.post, reference.comb, normalized], label: "reference \(label)")
                try evaluate([streams, fn, base, scale, normWeight], label: "inputs \(label)")
                let captureKey = "\(dtype)-\(batch)-\(tokens)-\(width)"
                try capture([streams, fn, base, scale, normWeight], key: "\(captureKey)-inputs")
                let coldStart = DispatchTime.now().uptimeNanoseconds
                let actual = try Kernel.collapseNormalized(
                    .streams(streams), fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                    epsilon: 1e-6, sinkhornIterations: 20, normWeight: normWeight, normEpsilon: 1e-5
                )
                let plainName =
                    split
                    ? "glm5_next_hc4_post_comb" : "glm5_next_hc4_prefill_collapse_norm"
                try evaluate([actual.post, actual.comb], label: "\(plainName) \(label)")
                try evaluate(
                    [actual.normalized],
                    label: "\(split ? "glm5_next_hc4_pre_collapse_norm" : plainName) \(label)")
                let firstPlainMS = Double(DispatchTime.now().uptimeNanoseconds - coldStart) / 1e6
                try capture(
                    [actual.post, actual.comb, actual.normalized], key: "\(captureKey)-plain")
                expectClose(actual.post, reference.post, tolerance: 1e-5, label: "\(label) post")
                expectClose(actual.comb, reference.comb, tolerance: 1e-5, label: "\(label) comb")
                expectClose(actual.normalized, normalized, tolerance: normTolerance, label: label)

                let output = MLXRandom.normal([batch, tokens, width], key: keys[5]).asType(dtype)
                let expanded = try Kernel.expand(
                    output, residual: streams, post: reference.post, comb: reference.comb)
                let next = try Kernel.collapse(
                    expanded, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                    epsilon: 1e-6, sinkhornIterations: 20)
                let nextNormalized = norm(next.collapsed)
                try evaluate(
                    [expanded, next.post, next.comb, nextNormalized],
                    label: "expanded reference \(label)")
                try capture(
                    [output, reference.post, reference.comb], key: "\(captureKey)-folded-inputs")
                let foldedStart = DispatchTime.now().uptimeNanoseconds
                let folded = try Kernel.collapseNormalized(
                    .expand(
                        output: output, residual: streams, post: reference.post,
                        comb: reference.comb),
                    fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                    sinkhornIterations: 20, normWeight: normWeight, normEpsilon: 1e-5)
                let foldedName =
                    split
                    ? "glm5_next_hc4_expand_pre_collapse_norm"
                    : "glm5_next_hc4_expand_prefill_collapse_norm"
                // For split collapse, `post` first depends on the folded expand.
                try evaluate(
                    [folded.normalized, folded.streams], label: "\(foldedName) \(label)")
                try evaluate([folded.post, folded.comb], label: "folded coefficients \(label)")
                let firstFoldedMS = Double(DispatchTime.now().uptimeNanoseconds - foldedStart) / 1e6
                try capture(
                    [folded.post, folded.comb, folded.normalized, folded.streams],
                    key: "\(captureKey)-folded")
                #expect(
                    folded.streams.asType(.float32).asArray(Float.self).map(\.bitPattern)
                        == expanded.asType(.float32).asArray(Float.self).map(\.bitPattern),
                    "\(label) folded expand bit patterns")
                expectClose(folded.post, next.post, tolerance: 1e-5, label: "\(label) folded post")
                expectClose(folded.comb, next.comb, tolerance: 1e-5, label: "\(label) folded comb")
                expectClose(
                    folded.normalized, nextNormalized, tolerance: normTolerance,
                    label: "\(label) folded normalized")
                if let directory = ProcessInfo.processInfo.environment["GLM_TG_CAPTURE_DIR"] {
                    let plainMS = try measure {
                        let value = try Kernel.collapseNormalized(
                            .streams(streams), fn: fn, base: base, scale: scale,
                            rmsNormEpsilon: 1e-5, epsilon: 1e-6, sinkhornIterations: 20,
                            normWeight: normWeight, normEpsilon: 1e-5)
                        return [value.post, value.comb, value.normalized]
                    }
                    let foldedMS = try measure {
                        let value = try Kernel.collapseNormalized(
                            .expand(
                                output: output, residual: streams, post: reference.post,
                                comb: reference.comb),
                            fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                            epsilon: 1e-6, sinkhornIterations: 20,
                            normWeight: normWeight, normEpsilon: 1e-5)
                        return [value.post, value.comb, value.normalized, value.streams]
                    }
                    let report: [String: Any] = [
                        "case": captureKey,
                        "firstPlainMS": firstPlainMS, "firstFoldedMS": firstFoldedMS,
                        "warmPlainMS": plainMS, "warmFoldedMS": foldedMS,
                    ]
                    let data = try JSONSerialization.data(
                        withJSONObject: report, options: [.sortedKeys])
                    try data.write(
                        to: URL(fileURLWithPath: directory)
                            .appendingPathComponent("\(captureKey)-timings.json"))
                }
            }
        }
    }
}
