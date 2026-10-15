// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// The three-launch MoE decode (`RoutedExperts.fusedDecode`) against the
    /// existing native combine path on random affine Q4 group-64 banks.
    @Suite(.serialized)
    struct GLM5NextMoEDecodeKernelTests {
        struct Geometry: Sendable, CustomStringConvertible {
            let experts: Int
            let hidden: Int
            let intermediate: Int
            let sharedIntermediate: Int
            let topK: Int
            var description: String {
                "E\(experts) D\(hidden) I\(intermediate) S\(sharedIntermediate) K\(topK)"
            }
            /// GLM-5.3-Flash.
            static let real = Geometry(
                experts: 288, hidden: 4096, intermediate: 2048, sharedIntermediate: 2048, topK: 8)
            /// A wider shared expert exercises tiles past the routed width.
            static let small = Geometry(
                experts: 32, hidden: 512, intermediate: 512, sharedIntermediate: 1024, topK: 4)
        }

        private static let group = 64

        /// Uniform packed nibbles; the scale gives unit-variance outputs for
        /// unit-variance inputs and the offset centers the codes.
        private static func bank(
            _ leading: [Int], rows: Int, inputs: Int, seed: UInt64, outputScale: Float = 1
        ) -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 4)
            let packedShape = leading + [rows, inputs / 8]
            let high = MLXRandom.randInt(Int32(0) ..< 65536, packedShape, key: keys[0])
                .asType(.uint32)
            let low = MLXRandom.randInt(Int32(0) ..< 65536, packedShape, key: keys[1])
                .asType(.uint32)
            let weight = bitwiseOr(leftShift(high, MLXArray(UInt32(16))), low)
            let step = outputScale / (Float(inputs) * 21.25).squareRoot()
            let groupShape = leading + [rows, inputs / group]
            let scales = MLXRandom.uniform(low: 0.5 * step, high: 1.5 * step, groupShape, key: keys[2])
            let biases = -7.5 * scales + MLXRandom.normal(groupShape, key: keys[3]) * step
            let parts = (weight, scales.asType(.bfloat16), biases.asType(.bfloat16))
            // Bound the transient generation memory of the real-width banks.
            eval(parts.0, parts.1, parts.2)
            return parts
        }

        private static func linear(rows: Int, inputs: Int, seed: UInt64) -> QuantizedLinear {
            let parts = bank([], rows: rows, inputs: inputs, seed: seed)
            return QuantizedLinear(
                weight: parts.weight, scales: parts.scales, biases: parts.biases,
                groupSize: group, bits: 4)
        }

        private static func stack(_ geometry: Geometry, rows: Int, inputs: Int, seed: UInt64)
            throws -> QuantizedSwitchLinear
        {
            let parts = bank([geometry.experts], rows: rows, inputs: inputs, seed: seed)
            return try QuantizedSwitchLinear(
                inputDims: inputs, outputDims: rows, numExperts: geometry.experts,
                weight: parts.weight, scales: parts.scales, biases: parts.biases,
                groupSize: group, bits: 4)
        }

        private static func shared(_ geometry: Geometry, fusedGateUp: Bool) throws -> GLM5NextMLP {
            var gate: Linear = linear(
                rows: geometry.sharedIntermediate, inputs: geometry.hidden, seed: 21)
            var up: Linear = linear(
                rows: geometry.sharedIntermediate, inputs: geometry.hidden, seed: 22)
            var fused: GLM5NextFusedProjection?
            if fusedGateUp {
                let result = try #require(try GLM5NextFusedProjection.make([gate, up]))
                (gate, up, fused) = (result.members[0], result.members[1], result.fused)
            }
            let mlp = try GLM5NextMLP(
                hiddenSize: geometry.hidden, intermediateSize: geometry.sharedIntermediate,
                gateProj: gate, upProj: up,
                downProj: linear(
                    rows: geometry.hidden, inputs: geometry.sharedIntermediate, seed: 23),
                swigluLimit: 7)
            mlp.fusedGateUp = fused
            return mlp
        }

        /// Random NVFP4 codes, E4M3 block scales in [2^-3, 2^-1) and per-matrix
        /// global scales sized for unit-variance outputs from unit inputs.
        private static func nvfp4Bank(
            _ leading: [Int], rows: Int, inputs: Int, seed: UInt64
        ) -> (weight: MLXArray, scales: MLXArray, global: MLXArray) {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 5)
            let packedShape = leading + [rows, inputs / 8]
            let high = MLXRandom.randInt(Int32(0) ..< 65536, packedShape, key: keys[0])
                .asType(.uint32)
            let low = MLXRandom.randInt(Int32(0) ..< 65536, packedShape, key: keys[1])
                .asType(.uint32)
            let weight = bitwiseOr(leftShift(high, MLXArray(UInt32(16))), low)
            let groupShape = leading + [rows, inputs / 16]
            let exponent = MLXRandom.randInt(Int32(4) ..< 7, groupShape, key: keys[2])
            let mantissa = MLXRandom.randInt(Int32(0) ..< 8, groupShape, key: keys[3])
            let scales = (exponent * 8 + mantissa).asType(.uint8)
            // E2M1 RMS is about 2.93 and the block scales about 0.25.
            let target = 1 / (Float(inputs).squareRoot() * 2.93 * 0.25)
            let global = MLXRandom.uniform(
                low: 0.75 * target, high: 1.25 * target, [max(1, leading.first ?? 1)],
                key: keys[4])
            eval(weight, scales, global)
            return (weight, scales, global)
        }

        private static func nvfp4Stack(rows: Int, inputs: Int, experts: Int, seed: UInt64)
            throws -> QuantizedSwitchLinear
        {
            let parts = nvfp4Bank([experts], rows: rows, inputs: inputs, seed: seed)
            return try QuantizedSwitchLinear(
                nvfp4InputDims: inputs, outputDims: rows, numExperts: experts,
                weight: parts.weight, scales: parts.scales, globalScales: parts.global)
        }

        private static func nvfp4Linear(rows: Int, inputs: Int, seed: UInt64) throws
            -> GLM5NextNVFP4Linear
        {
            let parts = nvfp4Bank([], rows: rows, inputs: inputs, seed: seed)
            return try GLM5NextNVFP4Linear(
                weight: parts.weight, scales: parts.scales, globalScale: parts.global)
        }

        /// `nvfp4` makes the routed banks NVFP4 (the serving layout);
        /// `sharedNVFP4` also the shared expert.
        static func moe(
            _ geometry: Geometry, fusedSharedGateUp: Bool = false, nGroup: Int = 1,
            nvfp4: Bool = false, sharedNVFP4: Bool = false
        ) throws -> GLM5NextMoE {
            let (e, d, i) = (geometry.experts, geometry.hidden, geometry.intermediate)
            let experts =
                nvfp4
                ? try GLM5NextExperts(
                    hiddenSize: d, intermediateSize: i, numExperts: e,
                    gateProj: nvfp4Stack(rows: i, inputs: d, experts: e, seed: 41),
                    upProj: nvfp4Stack(rows: i, inputs: d, experts: e, seed: 42),
                    downProj: nvfp4Stack(rows: d, inputs: i, experts: e, seed: 43),
                    swigluLimit: 7)
                : try GLM5NextExperts(
                    hiddenSize: d, intermediateSize: i, numExperts: e,
                    gateProj: stack(geometry, rows: i, inputs: d, seed: 11),
                    upProj: stack(geometry, rows: i, inputs: d, seed: 12),
                    downProj: stack(geometry, rows: d, inputs: i, seed: 13), swigluLimit: 7)
            let sharedExperts =
                sharedNVFP4
                ? try GLM5NextMLP(
                    hiddenSize: d, intermediateSize: geometry.sharedIntermediate,
                    gateProj: nvfp4Linear(rows: geometry.sharedIntermediate, inputs: d, seed: 51),
                    upProj: nvfp4Linear(rows: geometry.sharedIntermediate, inputs: d, seed: 52),
                    downProj: nvfp4Linear(rows: d, inputs: geometry.sharedIntermediate, seed: 53),
                    swigluLimit: 7)
                : try shared(geometry, fusedGateUp: fusedSharedGateUp)
            let keys = MLXRandom.split(key: MLXRandom.key(31), into: 2)
            let router = try GLM5NextRouter(
                hiddenSize: d, numExperts: e, topK: geometry.topK,
                weight: (MLXRandom.normal([e, d], key: keys[0]) * 0.02).asType(.bfloat16),
                correctionBias: MLXRandom.normal([e], key: keys[1]) * 0.05,
                nGroup: nGroup, topkGroup: 1, normTopkProb: true, routedScalingFactor: 2.5)
            return GLM5NextMoE(gate: router, experts: experts, sharedExperts: sharedExperts)
        }

        static func input(_ geometry: Geometry, tokens: Int, seed: UInt64) -> MLXArray {
            MLXRandom.normal([1, tokens, geometry.hidden], key: MLXRandom.key(seed))
                .asType(.bfloat16)
        }

        static func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        /// Returns (max |a - b| / max |b|, fraction of elements that differ).
        static func difference(_ actual: MLXArray, _ expected: MLXArray) -> (Float, Float)
        {
            let left = actual.asType(.float32).asArray(Float.self)
            let right = expected.asType(.float32).asArray(Float.self)
            let peak = right.map(abs).max() ?? 0
            let worst = zip(left, right).map { abs($0 - $1) }.max() ?? .infinity
            let differing = zip(left, right).filter { $0 != $1 }.count
            return (
                left.allSatisfy(\.isFinite) ? worst / max(peak, .leastNormalMagnitude) : .infinity,
                Float(differing) / Float(max(1, left.count))
            )
        }

        /// The fused path selects the reference router's experts (no near-ties
        /// in random data) with matching weights. Each stage is compared with
        /// the composed MLX operations on the kernel's own inputs. One token is
        /// bitwise: the qmv lanes, clamped-GLU roundings and combine order are
        /// MLX's. Several tokens can differ by one BF16 unit where MLX picks a
        /// multi-row matmul for the shared expert or a wider gather; an M5 Ultra
        /// measured at most 0.11% of the output peak, bounded here by 2^-8.
        @Test(arguments: [Geometry.small, .real])
        func fusedDecodeMatchesNativeCombineOnGPU(geometry: Geometry) throws {
            try Self.checkFusedDecode(try Self.moe(geometry), geometry, label: "affine")
        }

        /// NVFP4 routed banks (the serving layout) with the affine shared
        /// expert, and an all-NVFP4 variant; same contract as the affine banks.
        @Test(arguments: [(Geometry.small, false), (.real, false), (.small, true)])
        func nvfp4FusedDecodeMatchesNativeCombineOnGPU(geometry: Geometry, sharedNVFP4: Bool)
            throws
        {
            let moe = try Self.moe(geometry, nvfp4: true, sharedNVFP4: sharedNVFP4)
            try Self.checkFusedDecode(
                moe, geometry, label: sharedNVFP4 ? "nvfp4 all" : "nvfp4 routed")
        }

        private static func checkFusedDecode(
            _ moe: GLM5NextMoE, _ geometry: Geometry, label kind: String
        ) throws {
            try Device.withDefaultDevice(.gpu) {
                let plan = try #require(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .bfloat16))
                for tokens in geometry.experts == 288 ? [1, 2, 8] : [1, 2, 3, 5, 8] {
                    let label = "\(kind) \(geometry) tokens \(tokens)"
                    let x = Self.input(geometry, tokens: tokens, seed: UInt64(100 + tokens))
                    let flattened = x.reshaped(tokens, geometry.hidden)
                    let fused = try GLM5NextMoEDecodeKernel.decode(flattened, plan: plan)
                    let viaModule = try moe(x, routedExperts: .fusedDecode)
                    let routing = moe.gate(x)
                    let native = try moe(x, routedExperts: .nativeCombine)
                    let sameRouting = try moe.experts(
                        flattened, indices: fused.indices, weights: fused.weights,
                        sortedGather: false, addend: moe.sharedExperts(flattened))
                    try Self.fence([
                        fused.output, fused.indices, fused.weights, viaModule, routing.indices,
                        routing.weights, native, sameRouting,
                    ])
                    #expect(fused.output.shape == [tokens, geometry.hidden], "\(label)")
                    #expect(fused.output.dtype == .bfloat16 && fused.indices.dtype == .uint32)
                    #expect(
                        Self.bits(viaModule) == Self.bits(fused.output.reshaped(x.shape)),
                        "\(label) module path")

                    let expectedIndices = routing.indices.asType(.int32).asArray(Int32.self)
                    let expectedWeights = routing.weights.asArray(Float.self)
                    let actualIndices = fused.indices.asType(.int32).asArray(Int32.self)
                    let actualWeights = fused.weights.asArray(Float.self)
                    for row in 0 ..< tokens {
                        let slots = row * geometry.topK ..< (row + 1) * geometry.topK
                        let reference = Dictionary(
                            uniqueKeysWithValues: slots.map {
                                (expectedIndices[$0], expectedWeights[$0])
                            })
                        let actual = Dictionary(
                            uniqueKeysWithValues: slots.map { (actualIndices[$0], actualWeights[$0]) })
                        #expect(Set(actual.keys) == Set(reference.keys), "\(label) row \(row)")
                        for (expert, weight) in actual {
                            let error = abs(weight - (reference[expert] ?? .nan))
                            #expect(error < 1e-5, "\(label) row \(row) expert \(expert) \(error)")
                        }
                    }

                    // Stage by stage, each from the kernel's own inputs.
                    let expanded = flattened.expandedDimensions(axes: [-2, -3])
                    let routedActivations = glm5NextClampedGLUOperations(
                        moe.experts.gateProj(expanded, fused.indices),
                        moe.experts.upProj(expanded, fused.indices), limit: 7
                    ).reshaped(tokens * geometry.topK, geometry.intermediate)
                    let sharedActivations = glm5NextClampedGLUOperations(
                        moe.sharedExperts.gateProj(flattened),
                        moe.sharedExperts.upProj(flattened), limit: 7)
                    let down = moe.experts.downProj(
                        fused.activations.reshaped(
                            tokens, geometry.topK, 1, geometry.intermediate), fused.indices)
                    let combined = try GLM5NextMoEKernel.ascendingCombine(
                        projected: down.reshaped(tokens * geometry.topK, geometry.hidden),
                        indices: fused.indices, weights: fused.weights,
                        addend: moe.sharedExperts.downProj(fused.sharedActivations))
                    try Self.fence([routedActivations, sharedActivations, combined])
                    let stages = [
                        ("routed act", fused.activations, routedActivations),
                        ("shared act", fused.sharedActivations, sharedActivations),
                        ("down+combine", fused.output, combined),
                    ]
                    for (stage, actual, expected) in stages {
                        let (error, fraction) = Self.difference(actual, expected)
                        print("GLM53_MOE_DECODE \(label) \(stage) max \(error) differing \(fraction)")
                        #expect(
                            tokens == 1 ? fraction == 0 : error <= 1.0 / 256,
                            "\(label) \(stage) \(error)")
                    }

                    let (projectionError, projectionFraction) = Self.difference(
                        fused.output, sameRouting)
                    let (pathError, pathFraction) = Self.difference(
                        fused.output, native.reshaped(tokens, geometry.hidden))
                    print(
                        "GLM53_MOE_DECODE \(label) same-routing max \(projectionError) "
                            + "differing \(projectionFraction); native max \(pathError) "
                            + "differing \(pathFraction)")
                    #expect(
                        tokens == 1 ? projectionFraction == 0 : projectionError <= 1.0 / 256,
                        "\(label) same routing \(projectionError)")
                    #expect(
                        tokens == 1 ? pathFraction == 0 : pathError <= 1.0 / 256,
                        "\(label) native combine \(pathError)")
                }
            }
        }

        /// Row views of the fused shared gate/up buffer read the same bytes.
        @Test func sharedFusedGateUpViewsMatchSeparateProjectionsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let geometry = Geometry.small
                let separate = try Self.moe(geometry)
                let fused = try Self.moe(geometry, fusedSharedGateUp: true)
                #expect(fused.sharedExperts.fusedGateUp != nil)
                let x = Self.input(geometry, tokens: 2, seed: 7)
                let expected = try separate(x, routedExperts: .fusedDecode)
                let actual = try fused(x, routedExperts: .fusedDecode)
                try Self.fence([expected, actual])
                #expect(Self.bits(actual) == Self.bits(expected))
            }
        }

        /// Longer steps and layers the kernel cannot read keep the
        /// `nativeCombineSortedPrefill` path unchanged.
        @Test func ineligibleStepsAndLayersKeepTheNativeCombinePath() throws {
            try Device.withDefaultDevice(.gpu) {
                let geometry = Geometry.small
                let moe = try Self.moe(geometry)
                for tokens in [9, 20] {
                    let x = Self.input(geometry, tokens: tokens, seed: 3)
                    let expected = try moe(x, routedExperts: .nativeCombineSortedPrefill)
                    let actual = try moe(x, routedExperts: .fusedDecode)
                    try Self.fence([expected, actual])
                    #expect(Self.bits(actual) == Self.bits(expected), "tokens \(tokens)")
                }

                let grouped = try Self.moe(geometry, nGroup: 2)
                #expect(GLM5NextMoEDecodeKernel.Plan(grouped, dtype: .bfloat16) == nil)
                #expect(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .float16) == nil)

                let dense = GLM5NextMoE(
                    gate: moe.gate,
                    experts: GLM5NextExperts(
                        hiddenSize: geometry.hidden, intermediateSize: geometry.intermediate,
                        numExperts: geometry.experts),
                    sharedExperts: moe.sharedExperts)
                #expect(GLM5NextMoEDecodeKernel.Plan(dense, dtype: .bfloat16) == nil)
            }
        }

        @Test func cpuRefusesFusedDecode() throws {
            let geometry = Geometry.small
            let moe = try Self.moe(geometry)
            let x = Self.input(geometry, tokens: 1, seed: 5)
            try Device.withDefaultDevice(.cpu) {
                let plan = try #require(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .bfloat16))
                #expect(throws: GLM5NextMoEDecodeKernel.Failure.unsupportedDeviceOrStream) {
                    try GLM5NextMoEDecodeKernel.decode(x.reshaped(1, geometry.hidden), plan: plan)
                }
                #expect(throws: GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream) {
                    try moe(x, routedExperts: .fusedDecode)
                }
            }
        }
    }
}
