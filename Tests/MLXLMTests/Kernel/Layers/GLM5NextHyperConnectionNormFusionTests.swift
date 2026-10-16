// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The normalized collapse must match `collapse` followed by the reference
    /// RMSNorm, and its folded expand must store exactly `expand`'s streams.
    @Suite(.serialized)
    struct GLM5NextHyperConnectionNormFusionTests {
        private typealias Kernel = GLM5NextHyperConnectionKernel

        private func maximumError(_ a: MLXArray, _ b: MLXArray) -> Float {
            abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        }

        private func run(dtype: DType, batch: Int, tokens: Int, width: Int) throws {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(width + tokens)), into: 8)
            let streams = MLXRandom.normal([batch, tokens, 4, width], key: keys[0]).asType(dtype)
            let fn = (MLXRandom.normal([24, 4 * width], key: keys[1]) * 0.05).asType(dtype)
            let base = (MLXRandom.normal([24], key: keys[2]) * 0.5).asType(dtype)
            let scale = (MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [3], key: keys[3]))
                .asType(dtype)
            let normWeight = MLXRandom.uniform(
                low: Float(0.5), high: Float(1.5), [width], key: keys[4]
            )
            .asType(dtype)
            let norm = GLM5NextRMSNorm(weight: normWeight, epsilon: 1e-5)
            let label = "\(dtype) B\(batch) T\(tokens) D\(width)"

            let expected = try Kernel.collapse(
                streams, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                sinkhornIterations: 20)
            let actual = try Kernel.collapseNormalized(
                .streams(streams), fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                epsilon: 1e-6, sinkhornIterations: 20, normWeight: normWeight, normEpsilon: 1e-5)
            let normalized = norm(expected.collapsed)
            eval(
                expected.post, expected.comb, normalized, actual.post, actual.comb,
                actual.normalized)
            #expect(actual.normalized.dtype == dtype && actual.normalized.shape == normalized.shape)
            #expect(maximumError(actual.post, expected.post) < 1e-5, "\(label) post")
            #expect(maximumError(actual.comb, expected.comb) < 1e-5, "\(label) comb")
            let tolerance: Float = dtype == .float32 ? 1e-4 : 0.05
            let normError = maximumError(actual.normalized, normalized)
            #expect(normError < tolerance, "\(label) normalized \(normError)")

            let output = MLXRandom.normal([batch, tokens, width], key: keys[5]).asType(dtype)
            let expanded = try Kernel.expand(
                output, residual: streams, post: expected.post, comb: expected.comb)
            let next = try Kernel.collapse(
                expanded, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                sinkhornIterations: 20)
            let folded = try Kernel.collapseNormalized(
                .expand(
                    output: output, residual: streams, post: expected.post, comb: expected.comb),
                fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                sinkhornIterations: 20, normWeight: normWeight, normEpsilon: 1e-5)
            let nextNormalized = norm(next.collapsed)
            eval(
                expanded, next.post, nextNormalized, folded.streams, folded.post, folded.normalized)
            #expect(
                folded.streams.asType(.float32).asArray(Float.self).map(\.bitPattern)
                    == expanded.asType(.float32).asArray(Float.self).map(\.bitPattern),
                "\(label) folded expand")
            #expect(maximumError(folded.post, next.post) < 1e-5, "\(label) folded post")
            let foldedError = maximumError(folded.normalized, nextNormalized)
            #expect(foldedError < tolerance, "\(label) folded normalized \(foldedError)")
        }

        @Test func normalizedCollapseMatchesCollapseThenRMSNormOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .bfloat16] {
                    try run(dtype: dtype, batch: 1, tokens: 1, width: 4096)
                    try run(dtype: dtype, batch: 2, tokens: 3, width: 1000)
                    try run(dtype: dtype, batch: 1, tokens: 9, width: 4)
                    try run(dtype: dtype, batch: 2, tokens: 11, width: 1000)
                    try run(dtype: dtype, batch: 1, tokens: 17, width: 4096)
                }
            }
        }

        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        /// Prefill rows do not depend on the step's row count, offset or tile
        /// height, and the folded expand equals collapsing the expanded streams.
        @Test func prefillRowsAreBitwiseInvariantToChunkingAndFolding() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .bfloat16] {
                    let width = 96
                    let rows = 4100
                    let keys = MLXRandom.split(key: MLXRandom.key(41), into: 7)
                    let streams = MLXRandom.normal([1, rows, 4, width], key: keys[0]).asType(dtype)
                    let output = MLXRandom.normal([1, rows, width], key: keys[1]).asType(dtype)
                    let fn = (MLXRandom.normal([24, 4 * width], key: keys[2]) * 0.05)
                        .asType(.bfloat16)
                    let base = MLXRandom.normal([24], key: keys[3]) * 0.5
                    let scale = MLXArray([Float(0.75), 1.25, 1])
                    let weight = MLXRandom.uniform(
                        low: Float(0.5), high: 1.5, [width], key: keys[4]
                    )
                    .asType(dtype)
                    let post = MLXRandom.uniform(
                        low: Float(0), high: 2, [1, rows, 4], key: keys[5])
                    let comb = MLXRandom.uniform(
                        low: Float(0), high: 0.5, [1, rows, 4, 4], key: keys[6])
                    func normalized(_ source: Kernel.StreamsSource) throws -> [MLXArray] {
                        let result = try Kernel.collapseNormalized(
                            source, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                            epsilon: 1e-6, sinkhornIterations: 20, normWeight: weight,
                            normEpsilon: 1e-5)
                        return [result.streams, result.post, result.comb, result.normalized]
                    }
                    let whole = try normalized(.streams(streams))
                    let folded = try normalized(
                        .expand(output: output, residual: streams, post: post, comb: comb))
                    let expanded = try Kernel.expand(
                        output, residual: streams, post: post, comb: comb)
                    let unfolded = try normalized(.streams(expanded))
                    eval(whole + folded + unfolded)
                    #expect(rows > 8 && Kernel.prefillTileRows(rows: rows) == 8)
                    for (name, index) in [("post", 1), ("comb", 2), ("normalized", 3)] {
                        #expect(
                            Self.bits(folded[index]) == Self.bits(unfolded[index]),
                            "\(dtype) folded \(name)")
                    }
                    #expect(Self.bits(folded[0]) == Self.bits(expanded), "\(dtype) folded streams")
                    for range in [0 ..< 9, 5 ..< 30, 1031 ..< 1300, 4087 ..< 4100] {
                        #expect(range.count > 8 && Kernel.prefillTileRows(rows: range.count) == 16)
                        let chunk = try normalized(.streams(streams[0..., range]))
                        let chunkFolded = try normalized(
                            .expand(
                                output: output[0..., range], residual: streams[0..., range],
                                post: post[0..., range], comb: comb[0..., range]))
                        eval(chunk + chunkFolded)
                        for index in 1 ..< 4 {
                            #expect(
                                Self.bits(chunk[index]) == Self.bits(whole[index][0..., range]),
                                "\(dtype) rows \(range) output \(index)")
                            #expect(
                                Self.bits(chunkFolded[index])
                                    == Self.bits(folded[index][0..., range]),
                                "\(dtype) folded rows \(range) output \(index)")
                        }
                    }
                }
            }
        }

        @Test func refusesCPUAndMismatchedNormDtype() throws {
            let streams = MLXArray.zeros([1, 1, 4, 8])
            func call(_ weight: MLXArray) throws {
                _ = try Kernel.collapseNormalized(
                    .streams(streams), fn: MLXArray.zeros([24, 32]), base: MLXArray.zeros([24]),
                    scale: MLXArray.ones([3]), rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                    sinkhornIterations: 2, normWeight: weight, normEpsilon: 1e-5)
            }
            try Device.withDefaultDevice(.gpu) {
                #expect(throws: Kernel.Failure.self) {
                    try call(MLXArray.ones([8]).asType(.bfloat16))
                }
            }
            try Device.withDefaultDevice(.cpu) {
                #expect(throws: Kernel.Failure.unsupportedDeviceOrStream) {
                    try call(MLXArray.ones([8]))
                }
            }
        }
    }
}

extension KernelTests {
    /// The one-launch indexer LayerNorm matches the reference operations.
    @Suite(.serialized)
    struct GLM5NextIndexNormKernelTests {
        @Test func indexNormKernelMatchesReferenceOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .bfloat16] {
                    for (shape, width) in [
                        ([1, 1], 128), ([2, 7], 128), ([3], 5), ([1, 4], 300),
                    ] {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(width)), into: 3)
                        let weight = MLXRandom.normal([width], key: keys[0]).asType(dtype)
                        let bias = MLXRandom.normal([width], key: keys[1]).asType(dtype)
                        let x = (MLXRandom.normal(shape + [width], key: keys[2]) * 3 + 1)
                            .asType(dtype)
                        let norm = GLM5NextIndexNorm(weight: weight, bias: bias)
                        let expected = norm(x)
                        norm.usesFusedKernel = true
                        let actual = norm(x)
                        eval(expected, actual)
                        #expect(actual.dtype == dtype && actual.shape == expected.shape)
                        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                            .item(Float.self)
                        #expect(
                            error <= (dtype == .float32 ? 1e-5 : 0.02), "\(dtype) \(shape) \(width)"
                        )
                    }
                }
            }
        }
    }
}
