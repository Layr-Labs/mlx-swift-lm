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
            /// One rank of GLM-5.3-Flash under two-rank tensor parallelism.
            static let tensorParallelRank = Geometry(
                experts: 288, hidden: 4096, intermediate: 1024, sharedIntermediate: 1024, topK: 8)
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
            let scales = MLXRandom.uniform(
                low: 0.5 * step, high: 1.5 * step, groupShape, key: keys[2])
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
        /// `sharedNVFP4` also the shared expert. `routerTies` repeats each
        /// router row and correction bias over four consecutive experts, so
        /// corrected scores tie in groups of four.
        static func moe(
            _ geometry: Geometry, fusedSharedGateUp: Bool = false, nGroup: Int = 1,
            nvfp4: Bool = false, sharedNVFP4: Bool = false, routerTies: Bool = false
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
            let distinct = routerTies ? e / 4 : e
            var routerWeight = MLXRandom.normal([distinct, d], key: keys[0]) * 0.02
            var correctionBias = MLXRandom.normal([distinct], key: keys[1]) * 0.05
            if routerTies {
                routerWeight = repeated(routerWeight, count: 4, axis: 0)
                correctionBias = repeated(correctionBias, count: 4, axis: 0)
            }
            let router = try GLM5NextRouter(
                hiddenSize: d, numExperts: e, topK: geometry.topK,
                weight: routerWeight.asType(.bfloat16),
                correctionBias: correctionBias,
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
        static func difference(_ actual: MLXArray, _ expected: MLXArray) -> (Float, Float) {
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
        /// A prior CPU model can populate MLXNN's shapeless SiLU trace for
        /// gathered expert activations. GPU execution must still match the
        /// fused path's rounding, independent of that earlier model.
        @Test func cpuSiluTraceDoesNotChangeFusedDecodeParity() throws {
            try Device.withDefaultDevice(.cpu) {
                let gate = MLXArray((0 ..< 2048).map { Float($0) / 257 - 4 })
                    .asType(.bfloat16)
                for shape in [[4, 512], [1, 4, 512], [1, 4, 1, 512], [1, 1, 4, 1, 512]] {
                    try Self.fence([silu(gate.reshaped(shape))])
                }
            }
            try Self.checkFusedDecode(try Self.moe(.small), .small, label: "after CPU SiLU")
        }

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

        /// `Kernels.current` against `Kernels.reference` (the first version)
        /// for every output bit at 1...8 tokens: affine, NVFP4 routed and
        /// all-NVFP4 banks at small and one-rank widths, and a router whose
        /// corrected scores tie in groups of four with an all-NaN second token,
        /// which exercises the top-K tie order and the lowest-ID fallback.
        @Test(arguments: [
            (Geometry.small, false, false, false), (.small, true, false, false),
            (.small, true, true, false), (.small, true, false, true),
            (.small, false, false, true), (.tensorParallelRank, true, false, false),
        ])
        func currentKernelsMatchReferenceBitwiseOnGPU(
            geometry: Geometry, nvfp4: Bool, sharedNVFP4: Bool, ties: Bool
        ) throws {
            try Device.withDefaultDevice(.gpu) {
                let moe = try Self.moe(
                    geometry, fusedSharedGateUp: !sharedNVFP4, nvfp4: nvfp4,
                    sharedNVFP4: sharedNVFP4, routerTies: ties)
                let plan = try #require(moe.decodePlans.plan(moe, dtype: .bfloat16))
                for tokens in 1 ... GLM5NextMoEDecodeKernel.maximumTokens {
                    var x = Self.input(geometry, tokens: tokens, seed: UInt64(200 + tokens))
                        .reshaped(tokens, geometry.hidden)
                    if ties && tokens >= 2 {
                        let nan = MLXArray(Array(repeating: Float.nan, count: geometry.hidden))
                            .reshaped(1, geometry.hidden).asType(.bfloat16)
                        x = concatenated([x[..<1], nan, x[2...]], axis: 0)
                    }
                    let reference = try GLM5NextMoEDecodeKernel.decode(
                        x, plan: plan, kernels: .reference)
                    let current = try GLM5NextMoEDecodeKernel.decode(x, plan: plan)
                    let pairs = [
                        (current.output, reference.output),
                        (current.activations, reference.activations),
                        (current.sharedActivations, reference.sharedActivations),
                        (current.weights, reference.weights),
                    ]
                    try Self.fence(pairs.flatMap { [$0.0, $0.1] } + [current.indices, reference.indices])
                    let label = "\(geometry) nvfp4 \(nvfp4)/\(sharedNVFP4) ties \(ties) tokens \(tokens)"
                    #expect(
                        current.indices.asArray(UInt32.self) == reference.indices.asArray(UInt32.self),
                        "\(label) indices")
                    for (index, pair) in pairs.enumerated() {
                        #expect(Self.bits(pair.0) == Self.bits(pair.1), "\(label) output \(index)")
                    }
                }
            }
        }

        /// On a routed-activation mismatch, recomputes each selected expert's
        /// gate/up with the dense (non-gathered) quantized matmul to tell a
        /// changed gathered-matmul kernel from a changed fused kernel, and
        /// re-evaluates the clamped GLU on the gathered layout to tell a
        /// non-deterministic first evaluation from a layout-dependent one.
        private static func diagnoseRoutedActivations(
            _ moe: GLM5NextMoE, _ x: MLXArray, _ fused: GLM5NextMoEDecodeKernel.Decoded,
            firstGatheredActivations: MLXArray, label: String
        ) throws {
            guard let gate = moe.experts.gateProj as? QuantizedSwitchLinear,
                let up = moe.experts.upProj as? QuantizedSwitchLinear
            else { return }
            let topK = fused.indices.dim(1)
            let ids = fused.indices.asType(.int32).asArray(Int32.self)
            func dense(_ bank: QuantizedSwitchLinear, _ token: Int, _ expert: Int) -> MLXArray {
                let y = quantizedMM(
                    x[token ..< token + 1], bank.weight[expert], scales: bank.scales[expert],
                    biases: bank.biases?[expert], groupSize: bank.groupSize, bits: bank.bits,
                    mode: bank.mode)
                guard let global = bank.globalScales else { return y }
                return (y.asType(.float32) * global[expert]).asType(y.dtype)
            }
            var denseGate: [MLXArray] = []
            var denseUp: [MLXArray] = []
            for (row, id) in ids.enumerated() {
                denseGate.append(dense(gate, row / topK, Int(id)))
                denseUp.append(dense(up, row / topK, Int(id)))
            }
            let expanded = x.expandedDimensions(axes: [-2, -3])
            let gathered = gate(expanded, fused.indices).reshaped(ids.count, -1)
            let gatheredUp = up(expanded, fused.indices).reshaped(ids.count, -1)
            let denseGateAll = concatenated(denseGate, axis: 0)
            let denseUpAll = concatenated(denseUp, axis: 0)
            let denseActivations = glm5NextClampedGLUOperations(
                denseGateAll, denseUpAll, limit: moe.experts.swigluLimit)
            let gatheredActivations = glm5NextClampedGLUOperations(
                gate(expanded, fused.indices), up(expanded, fused.indices),
                limit: moe.experts.swigluLimit
            ).reshaped(ids.count, -1)
            try fence([
                gathered, gatheredUp, denseGateAll, denseUpAll, denseActivations,
                gatheredActivations,
            ])
            print(
                "GLM53_MOE_DIAG \(label) gathered-vs-dense gate differing "
                    + "\(difference(gathered, denseGateAll).1) up "
                    + "\(difference(gatheredUp, denseUpAll).1); kernel-vs-dense act differing "
                    + "\(difference(fused.activations, denseActivations).1); gathered-layout act "
                    + "vs dense \(difference(gatheredActivations, denseActivations).1), vs first "
                    + "evaluation \(difference(gatheredActivations, firstGatheredActivations).1)")
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
                            uniqueKeysWithValues: slots.map {
                                (actualIndices[$0], actualWeights[$0])
                            })
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
                    if Self.difference(fused.activations, routedActivations).1 != 0 {
                        try Self.diagnoseRoutedActivations(
                            moe, flattened, fused, firstGatheredActivations: routedActivations,
                            label: label)
                    }
                    for (stage, actual, expected) in stages {
                        let (error, fraction) = Self.difference(actual, expected)
                        print(
                            "GLM53_MOE_DECODE \(label) \(stage) max \(error) differing \(fraction)")
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
                        tokens == 1
                            ? projectionFraction == 0 : projectionError <= (1.0 as Float) / 256,
                        "\(label) same routing \(projectionError)")
                    #expect(
                        tokens == 1 ? pathFraction == 0 : pathError <= (1.0 as Float) / 256,
                        "\(label) native combine \(pathError)")
                }
            }
        }

        /// The NAX expert prefill (`fusedDecodeNAXPrefill`) against the sorted
        /// gather for affine and NVFP4 banks. Same rounding points. Where MLX
        /// also runs its NAX sorted kernel (here 37+ tokens small, 512 real)
        /// the results were bitwise on an M5 Ultra; for fewer rows per expert
        /// MLX picks another kernel and BF16 outputs differ by rounding (at
        /// most 0.6% of the peak measured), bounded here by 2^-6.
        @Test(
            .enabled(if: GLM5NextMoEPrefillKernel.naxAvailable),
            arguments: [(Geometry.small, false), (.small, true), (.real, false), (.real, true)])
        func naxPrefillMatchesSortedPrefillOnGPU(geometry: Geometry, nvfp4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let moe = try Self.moe(geometry, nvfp4: nvfp4)
                let plan = try #require(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .bfloat16))
                #expect(GLM5NextMoEPrefillKernel.admits(plan))
                for tokens in geometry.experts == 288 ? [100, 512] : [16, 37, 300] {
                    let x = Self.input(geometry, tokens: tokens, seed: UInt64(200 + tokens))
                    let expected = try moe(x, routedExperts: .nativeCombineSortedPrefill)
                    let actual = try moe(x, routedExperts: .fusedDecodeNAXPrefill)
                    try Self.fence([expected, actual])
                    let (error, fraction) = Self.difference(actual, expected)
                    let label = "\(nvfp4 ? "nvfp4" : "affine") \(geometry) tokens \(tokens)"
                    print("GLM53_MOE_NAX \(label) max \(error) differing \(fraction)")
                    #expect(error <= 1.0 / 64, "\(label) max \(error) differing \(fraction)")
                }
            }
        }

        /// The tile table at the serving shapes: one tensor-parallel rank at
        /// 4,096-token calls (114 rows per expert), its 2,048-token prefill
        /// micro-batches (57) and short steps (14).
        @Test func naxPrefillTilePlanTable() {
            typealias Plan = GLM5NextMoEPrefillKernel.TilePlan
            let tall = Plan(rows: 128, depth: 128, doubleBuffered: false, groupX: 32, pad: 8192)
            let mid = Plan(rows: 96, depth: 64, doubleBuffered: true, groupX: 32, pad: 0)
            let plain = Plan(rows: 64, depth: 64, doubleBuffered: true, groupX: 0, pad: 0)
            for input in [4096, 1024] {
                #expect(Plan.plan(rows: 32768, experts: 288, input: input) == tall)
                #expect(Plan.plan(rows: 16384, experts: 288, input: input) == mid)
                #expect(Plan.plan(rows: 4096, experts: 288, input: input) == plain)
            }
            // A short down projection keeps 64-row tiles below 120 rows.
            #expect(Plan.plan(rows: 32768, experts: 288, input: 640) == plain)
            // An input that is not a whole number of 128-value steps keeps
            // 64-deep double buffering.
            #expect(
                Plan.plan(rows: 32768, experts: 288, input: 4160)
                    == Plan(rows: 128, depth: 64, doubleBuffered: true, groupX: 32, pad: 0))
            for plan in [tall, mid, plain] { #expect(plan.admits(input: 1024)) }
            #expect(!tall.admits(input: 4160))
            #expect(!Plan(rows: 128, depth: 128, doubleBuffered: true, groupX: 0, pad: 0)
                .admits(input: 4096))
            #expect(
                !Plan(rows: 64, depth: 64, doubleBuffered: true, groupX: 0, pad: 0, prefetch: true)
                    .admits(input: 4096))
            #expect(
                Plan(rows: 128, depth: 128, doubleBuffered: false, groupX: 32, pad: 8192, prefetch: true)
                    .admits(input: 1024))
        }

        /// The one-launch counting sort against MLX's argsort, the scatter
        /// inverse and the tile lists rebuilt from the sorted keys: uniform
        /// and skewed routing, with and without shared-expert rows, one and
        /// two tile heights, GLM's 288 experts and the 512 maximum.
        @Test(.enabled(if: GLM5NextMoEPrefillKernel.naxAvailable))
        func naxPrefillRouteMatchesArgSortOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for (experts, tokens, topK, shared, skewed, heights) in [
                    (288, 2048, 8, true, false, (96, 96)), (288, 4096, 8, true, false, (128, 128)),
                    (288, 700, 8, false, true, (128, 64)), (16, 700, 4, true, false, (64, 128)),
                    (512, 300, 8, true, false, (128, 96)), (32, 3, 4, true, true, (64, 64)),
                ] {
                    let label = "E\(experts) T\(tokens) K\(topK) shared \(shared) skewed \(skewed)"
                    // Distinct experts per token: a per-row permutation prefix.
                    let pool = skewed ? min(experts, 2 * topK) : experts
                    let scores = MLXRandom.uniform(
                        0 ..< 1, [tokens, pool], key: MLXRandom.key(UInt64(experts + tokens)))
                    let indices = argSort(scores, axis: -1)[0..., ..<topK].asType(.uint32)
                    let sharedRows = shared ? tokens : 0
                    let sorted = GLM5NextMoEPrefillKernel.route(
                        indices, experts: experts, sharedRows: sharedRows,
                        gateUpRows: heights.0, downRows: heights.1)
                    let expectedOrder = argSort(indices.flattened()).asType(.uint32)
                    try Self.fence([
                        sorted.order, sorted.inverse, sorted.gateUp.list, sorted.gateUp.count,
                        sorted.down.list, sorted.down.count, expectedOrder,
                    ])
                    let rows = tokens * topK
                    let order = sorted.order.asArray(UInt32.self)
                    let inverse = sorted.inverse.asArray(UInt32.self)
                    #expect(Array(order[..<rows]) == expectedOrder.asArray(UInt32.self), "\(label)")
                    #expect(
                        Array(order[rows...]) == (0 ..< sharedRows).map { UInt32($0 * topK) },
                        "\(label) shared rows")
                    #expect(
                        (0 ..< rows).allSatisfy { Int(order[Int(inverse[$0])]) == $0 },
                        "\(label) inverse")
                    let keys = indices.flattened().asArray(UInt32.self)
                    let sortedKeys =
                        order[..<rows].map { keys[Int($0)] }
                        + Array(repeating: UInt32(experts), count: sharedRows)
                    for (tiles, height) in [(sorted.gateUp, heights.0), (sorted.down, heights.1)] {
                        var expected: [UInt32] = []
                        var start = 0
                        while start < sortedKeys.count {
                            var end = start
                            while end < sortedKeys.count, sortedKeys[end] == sortedKeys[start] {
                                end += 1
                            }
                            for first in stride(from: start, to: end, by: height) {
                                expected += [
                                    UInt32(first), sortedKeys[start], UInt32(min(height, end - first)),
                                    0,
                                ]
                            }
                            start = end
                        }
                        let count = Int(tiles.count.item(UInt32.self))
                        #expect(tiles.rows == height && count * 4 == expected.count, "\(label)")
                        #expect(
                            Array(tiles.list.asArray(UInt32.self)[..<(count * 4)]) == expected,
                            "\(label) tiles of \(height)")
                    }
                }
            }
        }

        /// Every tile plan issues the same tensor ops per output element in
        /// the same K order, so activations and projections are bitwise
        /// identical across tile heights, K depths, schedules (seg with and
        /// without prefetch) and grid layouts (`groupX` 4 leaves a partial
        /// last tile group).
        @Test(.enabled(if: GLM5NextMoEPrefillKernel.naxAvailable), arguments: [false, true])
        func naxPrefillTilePlansAreBitwiseIdenticalOnGPU(nvfp4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                // Inputs of whole 128-value steps on both projections.
                let geometry = Geometry(
                    experts: 16, hidden: 1024, intermediate: 512, sharedIntermediate: 512, topK: 4)
                let moe = try Self.moe(geometry, nvfp4: nvfp4)
                let plan = try #require(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .bfloat16))
                #expect(GLM5NextMoEPrefillKernel.admits(plan))
                // 175 rows per expert on average: every height has full,
                // repeated and partial tiles.
                let tokens = 700
                let x = Self.input(geometry, tokens: tokens, seed: 77)
                    .reshaped(tokens, geometry.hidden)
                let routing = moe.gate(x)
                typealias Plan = GLM5NextMoEPrefillKernel.TilePlan
                let baseline = Plan(rows: 64, depth: 64, doubleBuffered: true, groupX: 0, pad: 0)
                let expected = try GLM5NextMoEPrefillKernel.project(
                    x, indices: routing.indices, plan: plan, limit: moe.experts.swigluLimit,
                    gateUpPlan: baseline, downPlan: baseline)
                try Self.fence([expected.activations, expected.projected])
                let activations = Self.bits(expected.activations)
                let projected = Self.bits(expected.projected)
                // (double-buffered, prefetch)
                for shape in GLM5NextMoEPrefillKernel.TileShape.all {
                    for (doubleBuffered, prefetch) in shape.depth == 64
                        ? [(true, false), (false, false), (false, true)]
                        : [(false, false), (false, true)]
                    {
                        for groupX in [0, 4, 32] {
                            let candidate = Plan(
                                rows: shape.rows, depth: shape.depth,
                                doubleBuffered: doubleBuffered, groupX: groupX,
                                pad: shape.rows == 128 && !doubleBuffered ? 8192 : 0,
                                prefetch: prefetch)
                            let actual = try GLM5NextMoEPrefillKernel.project(
                                x, indices: routing.indices, plan: plan,
                                limit: moe.experts.swigluLimit, gateUpPlan: candidate,
                                downPlan: candidate)
                            try Self.fence([actual.activations, actual.projected])
                            let label = "\(nvfp4 ? "nvfp4" : "affine") \(candidate)"
                            #expect(Self.bits(actual.activations) == activations, "\(label) act")
                            #expect(Self.bits(actual.projected) == projected, "\(label) down")
                        }
                    }
                }
            }
        }

        /// The shared experts as expert E of the routed launches against the
        /// same launches with the shared MLP as the combine's addend, for
        /// every mix of routed and shared formats (the serving layout is NVFP4
        /// routed with an affine shared expert). Same dequantization, tensor
        /// ops in MLX's NAX qmm K order and rounding points: bitwise where MLX
        /// runs the addend's shared MLP as its NAX qmm (700 tokens). At 64
        /// tokens MLX runs it as its split-K qmm, which sums K in another
        /// order, so the two differ by bfloat16 rounding (measured on an M5
        /// Ultra: 0.0056 to 0.0103), bounded by 2^-6 as
        /// `naxPrefillMatchesSortedPrefillOnGPU` bounds the same regime.
        @Test(
            .enabled(if: GLM5NextMoEPrefillKernel.naxAvailable),
            arguments: [(false, false), (true, false), (true, true), (false, true)])
        func naxPrefillSharedExpertsInLaunchMatchAddendOnGPU(nvfp4: Bool, sharedNVFP4: Bool)
            throws
        {
            try Device.withDefaultDevice(.gpu) {
                let geometry = Geometry(
                    experts: 16, hidden: 1024, intermediate: 512, sharedIntermediate: 512, topK: 4)
                let moe = try Self.moe(
                    geometry, fusedSharedGateUp: !sharedNVFP4, nvfp4: nvfp4,
                    sharedNVFP4: sharedNVFP4)
                let plan = try #require(GLM5NextMoEDecodeKernel.Plan(moe, dtype: .bfloat16))
                #expect(GLM5NextMoEPrefillKernel.admitsSharedExperts(plan))
                for tokens in [64, 700] {
                    let x = Self.input(geometry, tokens: tokens, seed: UInt64(300 + tokens))
                        .reshaped(tokens, geometry.hidden)
                    let routing = moe.gate(x)
                    let limit = moe.experts.swigluLimit
                    let sharedLimit = moe.sharedExperts.swigluLimit
                    let separate = try GLM5NextMoEPrefillKernel.routedExperts(
                        x, indices: routing.indices, weights: routing.weights,
                        addend: moe.sharedExperts(x), plan: plan, limit: limit,
                        sharedLimit: sharedLimit)
                    let fused = try GLM5NextMoEPrefillKernel.routedExperts(
                        x, indices: routing.indices, weights: routing.weights, addend: nil,
                        plan: plan, limit: limit, sharedLimit: sharedLimit)
                    let stages = try GLM5NextMoEPrefillKernel.project(
                        x, indices: routing.indices, plan: plan, limit: limit,
                        sharedLimit: sharedLimit)
                    let rows = routing.indices.size
                    try Self.fence([separate, fused, stages.projected])
                    #expect(stages.projected.shape == [rows + tokens, geometry.hidden])
                    let (error, fraction) = Self.difference(fused, separate)
                    let label =
                        "\(nvfp4 ? "nvfp4" : "affine") routed, "
                        + "\(sharedNVFP4 ? "nvfp4" : "affine") shared, tokens \(tokens)"
                    print("GLM53_MOE_NAX_SHARED \(label) max \(error) differing \(fraction)")
                    #expect(error <= (tokens < 512 ? 1.0 / 64 : 0), "\(label) max \(error) differing \(fraction)")
                }
                // A shared expert wider than the routed experts keeps the addend.
                let wide = try Self.moe(.small, nvfp4: nvfp4)
                let widePlan = try #require(GLM5NextMoEDecodeKernel.Plan(wide, dtype: .bfloat16))
                #expect(!GLM5NextMoEPrefillKernel.admitsSharedExperts(widePlan))
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
