// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private func moeFusionFence(_ arrays: [MLXArray]) throws {
    try withError { error in
        eval(arrays)
        StreamOrDevice.default.stream.synchronize()
        try error.check()
    }
}

private func ramp(_ shape: [Int], salt: Int, scale: Float, modulus: Int = 97) -> MLXArray {
    let half = modulus / 2
    let values: [Float] = (0 ..< shape.reduce(1, *)).map {
        Float(($0 * 37 + salt * 13) % modulus - half) / Float(half) * scale
    }
    return MLXArray(values, shape)
}

private func bits(_ array: MLXArray) -> [UInt32] {
    array.asType(.float32).asArray(Float.self).map(\.bitPattern)
}

/// Distinct expert IDs per token, deliberately not in ascending slot order.
private func routes(tokens: Int, topK: Int, experts: Int) -> MLXArray {
    var values: [UInt32] = []
    for token in 0 ..< tokens {
        for slot in 0 ..< topK {
            values.append(UInt32((token * 5 + slot * 7 + (slot * slot) % 3 * 11) % experts))
        }
    }
    return MLXArray(values, [tokens, topK])
}

extension KernelTests {
    /// Fused routed-expert combine and native MoE policies against the
    /// unchanged reference. Unsorted paths share the reference projections and
    /// clamped GLU and are compared bitwise; the expert-sorted prefill gather
    /// changes accumulation order and is compared with a tolerance.
    @Suite(.serialized)
    struct GLM5NextMoEFusionTests {
        /// Expert widths. `aligned` is the group-64 layout of every checkpoint
        /// and admits the expert-sorted gather. `unaligned` (K = 64 and 32 at
        /// group 32) is refused, because MLX's sorted quantized gather writes
        /// wrong rows for a K remainder of a 64-wide tile.
        struct Geometry: Sendable, Equatable, CustomStringConvertible {
            let hidden: Int
            let intermediate: Int
            let group: Int
            var description: String { "h\(hidden) i\(intermediate) g\(group)" }
            static let aligned = Geometry(hidden: 128, intermediate: 64, group: 64)
            static let unaligned = Geometry(hidden: 64, intermediate: 32, group: 32)
        }

        private static let experts = 16
        private static let topK = 4

        /// Sorted tiles change FP32 accumulation order, so each BF16 projection
        /// output may round one unit (2^-8 relative) differently before the GLU,
        /// down projection and four-way BF16 combine. On an M5 Ultra the aligned
        /// Q4 geometry differed by at most 1.2% of the output peak (3 to 256
        /// tokens): 2^-5. FP32 differs only by accumulation order (6e-7 of the
        /// peak), or by TF32 operands when MLX_ENABLE_TF32 is on: 2^-8.
        private static func sortedTolerance(_ dtype: DType) -> Float {
            dtype == .float32 ? 1.0 / 256 : 1.0 / 32
        }

        private static func referenceCombine(
            _ projected: MLXArray, indices: MLXArray, weights: MLXArray
        ) -> MLXArray {
            let order = argSort(indices, axis: -1)
            let sortedProjected = takeAlong(projected, order.expandedDimensions(axis: -1), axis: 1)
            let sortedWeights = takeAlong(weights, order, axis: -1)
            let contributions =
                (sortedProjected.asType(.float32) * sortedWeights.expandedDimensions(axis: -1))
                .asType(projected.dtype)
            var output = MLXArray.zeros(
                [projected.dim(0), projected.dim(2)], dtype: projected.dtype)
            for slot in 0 ..< indices.dim(1) {
                output = output + contributions[0..., slot, 0...]
            }
            return output
        }

        private static func makeExperts(_ geometry: Geometry, dtype: DType, quantized: Bool)
            throws -> GLM5NextExperts
        {
            let (hidden, intermediate) = (geometry.hidden, geometry.intermediate)
            let module = GLM5NextExperts(
                hiddenSize: hidden, intermediateSize: intermediate, numExperts: experts)
            try module.update(
                parameters: .unflattened([
                    "gate_proj.weight": ramp([experts, intermediate, hidden], salt: 1, scale: 0.6)
                        .asType(dtype),
                    "up_proj.weight": ramp([experts, intermediate, hidden], salt: 2, scale: 0.6)
                        .asType(dtype),
                    "down_proj.weight": ramp([experts, hidden, intermediate], salt: 3, scale: 0.4)
                        .asType(dtype),
                ]), verify: .all)
            if quantized {
                quantize(model: module, filter: { _, _ in (geometry.group, 4, .affine) })
            }
            return module
        }

        private static func makeMoE(_ geometry: Geometry, dtype: DType) throws -> GLM5NextMoE {
            let (hidden, intermediate) = (geometry.hidden, geometry.intermediate)
            let shared = GLM5NextMLP(hiddenSize: hidden, intermediateSize: intermediate)
            try shared.update(
                parameters: .unflattened([
                    "gate_proj.weight": ramp([intermediate, hidden], salt: 4, scale: 0.7)
                        .asType(dtype),
                    "up_proj.weight": ramp([intermediate, hidden], salt: 5, scale: 0.7)
                        .asType(dtype),
                    "down_proj.weight": ramp([hidden, intermediate], salt: 6, scale: 0.3)
                        .asType(dtype),
                ]), verify: .all)
            let router = try GLM5NextRouter(
                hiddenSize: hidden, numExperts: experts, topK: topK,
                weight: ramp([experts, hidden], salt: 7, scale: 0.2).asType(dtype),
                correctionBias: ramp([experts], salt: 8, scale: 0.05).asType(dtype))
            return GLM5NextMoE(
                gate: router, experts: try makeExperts(geometry, dtype: dtype, quantized: true),
                sharedExperts: shared)
        }

        /// max |actual - expected| <= tolerance * max |expected|.
        private static func expectClose(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let left = actual.asType(.float32).asArray(Float.self)
            let right = expected.asType(.float32).asArray(Float.self)
            #expect(
                actual.shape == expected.shape, "\(label) shape", sourceLocation: sourceLocation)
            guard left.allSatisfy(\.isFinite), right.allSatisfy(\.isFinite) else {
                Issue.record("\(label): nonfinite output", sourceLocation: sourceLocation)
                return
            }
            let peak = right.map(abs).max() ?? 0
            let difference = zip(left, right).map { abs($0 - $1) }.max() ?? 0
            #expect(
                difference <= tolerance * peak,
                "\(label): max difference \(difference) = \(difference / peak) of peak \(peak)",
                sourceLocation: sourceLocation)
        }

        @Test func ascendingCombineIsBitwiseToReferenceCombine() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    for (tokens, topK, width) in [(1, 8, 96), (3, 8, 300), (70, 4, 64), (2, 1, 8)] {
                        let indices = routes(tokens: tokens, topK: topK, experts: 32)
                        let weights = ramp([tokens, topK], salt: 9, scale: 1.7)
                        let projected = ramp([tokens, topK, width], salt: 8, scale: 3)
                            .asType(dtype)
                        let reference = Self.referenceCombine(
                            projected, indices: indices, weights: weights)
                        let direct = try GLM5NextMoEKernel.ascendingCombine(
                            projected: projected.reshaped(tokens * topK, width), indices: indices,
                            weights: weights)
                        // The sorted gather's layout: rows in expert order plus
                        // the inverse permutation back to (token, slot).
                        let order = argSort(indices.flattened())
                        let mapped = try GLM5NextMoEKernel.ascendingCombine(
                            projected: projected.reshaped(tokens * topK, width)[order],
                            rowMap: argSort(order), indices: indices, weights: weights)
                        let signed = try GLM5NextMoEKernel.ascendingCombine(
                            projected: projected.reshaped(tokens * topK, width),
                            indices: indices.asType(.int32), weights: weights)
                        try moeFusionFence([reference, direct, mapped, signed])
                        #expect(direct.dtype == dtype && direct.shape == [tokens, width])
                        let label = "\(dtype) tokens \(tokens) K \(topK) D \(width)"
                        #expect(bits(direct) == bits(reference), "direct \(label)")
                        #expect(bits(mapped) == bits(reference), "mapped \(label)")
                        #expect(bits(signed) == bits(reference), "int32 \(label)")
                    }
                }
            }
        }

        @Test(arguments: [Geometry.aligned, .unaligned], [false, true])
        func nativeExpertsMatchReference(geometry: Geometry, quantized: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let dtype: DType = quantized ? .bfloat16 : .float32
                let experts = try Self.makeExperts(geometry, dtype: dtype, quantized: quantized)
                #expect(experts.admitsSortedGather == (geometry == .aligned))
                for tokens in [1, 3, 20, 64] {
                    let x = ramp([tokens, geometry.hidden], salt: 11, scale: 1).asType(dtype)
                    let indices = routes(tokens: tokens, topK: Self.topK, experts: Self.experts)
                    let weights = ramp([tokens, Self.topK], salt: 12, scale: 1.2)
                    let reference = experts(x, indices: indices, weights: weights)
                    let unsorted = try experts(
                        x, indices: indices, weights: weights, sortedGather: false)
                    let sorted = try experts(
                        x, indices: indices, weights: weights, sortedGather: true)
                    try moeFusionFence([reference, unsorted, sorted])
                    let label = "\(quantized ? "Q4" : "dense") \(geometry) tokens \(tokens)"
                    #expect(bits(unsorted) == bits(reference), "unsorted \(label)")
                    if experts.admitsSortedGather {
                        Self.expectClose(
                            sorted, reference, tolerance: Self.sortedTolerance(dtype),
                            "sorted \(label)")
                    } else {
                        #expect(bits(sorted) == bits(reference), "refused sort \(label)")
                    }
                }
            }
        }

        /// The reference and native-combine paths share explicit activation
        /// operations on the caller's current device, including after a CPU
        /// invocation. Neither path may inherit a cached CPU SiLU trace.
        @Test func nativeCombineSharesTheReferenceActivationAfterACPUTrace() throws {
            let geometry = Geometry.aligned
            let moe = try Device.withDefaultDevice(.gpu) {
                let moe = try Self.makeMoE(geometry, dtype: .bfloat16)
                try moeFusionFence(moe.parameters().flattened().map(\.1))
                return moe
            }
            let x = ramp([2, 7, geometry.hidden], salt: 15, scale: 4).asType(.bfloat16)
            try Device.withDefaultDevice(.cpu) { try moeFusionFence([moe(x)]) }
            try Device.withDefaultDevice(.gpu) {
                let reference = moe(x)
                let fused = try moe(x, routedExperts: .nativeCombine)
                try moeFusionFence([reference, fused])
                #expect(bits(fused) == bits(reference))
            }
        }

        @Test(arguments: [Geometry.aligned, .unaligned])
        func moePolicySelectsPrefillSortingAndRefusesCPU(geometry: Geometry) throws {
            let dtype = DType.bfloat16
            let moe = try Device.withDefaultDevice(.gpu) {
                let moe = try Self.makeMoE(geometry, dtype: dtype)
                try moeFusionFence(moe.parameters().flattened().map(\.1))
                return moe
            }
            try Device.withDefaultDevice(.gpu) {
                // [1,20,D] prefill has 80 >= 64 assignments; [3,1,D] is decode.
                for shape in [[1, 20, geometry.hidden], [3, 1, geometry.hidden]] {
                    let x = ramp(shape, salt: 13, scale: 1).asType(dtype)
                    let reference = moe(x)
                    let explicit = try moe(x, routedExperts: .reference)
                    let fused = try moe(x, routedExperts: .nativeCombine)
                    let prefill = try moe(x, routedExperts: .nativeCombineSortedPrefill)
                    try moeFusionFence([reference, explicit, fused, prefill])
                    let label = "\(geometry) \(shape)"
                    #expect(bits(explicit) == bits(reference), "\(label) explicit reference")
                    #expect(bits(fused) == bits(reference), "\(label) native combine")
                    if shape[1] > 1, moe.experts.admitsSortedGather {
                        Self.expectClose(
                            prefill, reference, tolerance: Self.sortedTolerance(dtype),
                            "\(label) sorted prefill")
                    } else {
                        #expect(
                            bits(prefill) == bits(reference),
                            "\(label) decode or refused widths never sort")
                    }
                }
            }
            try Device.withDefaultDevice(.cpu) {
                let x = ramp([1, 4, geometry.hidden], salt: 14, scale: 1).asType(dtype)
                for policy: GLM5NextExecutionPolicy.RoutedExperts in [
                    .nativeCombine, .nativeCombineSortedPrefill,
                ] {
                    #expect(throws: GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream) {
                        try moe(x, routedExperts: policy)
                    }
                }
                let reference = moe(x)
                let explicit = try moe(x, routedExperts: .reference)
                try moeFusionFence([reference, explicit])
                #expect(bits(explicit) == bits(reference))
            }
        }
    }
}
