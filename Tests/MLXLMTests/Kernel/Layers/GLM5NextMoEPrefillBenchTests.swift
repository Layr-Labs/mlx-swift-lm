// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Stage timings of the routed-expert prefill paths on random Q4 banks of
    /// GLM-5.3 geometry. Diagnostic only; enabled by `GLM53_PROFILE_PREFILL`
    /// (comma-separated token counts).
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GLM53_PROFILE_PREFILL"] != nil))
    struct GLM5NextMoEPrefillBenchTests {
        private typealias Decode = GLM5NextMoEDecodeKernelTests

        /// Best of five evaluations, milliseconds (build included).
        private static func time(_ body: () throws -> [MLXArray]) rethrows -> Double {
            var best = Double.infinity
            for _ in 0 ..< 5 {
                let start = ContinuousClock.now
                let arrays = try body()
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                let elapsed = ContinuousClock.now - start
                best = min(
                    best,
                    Double(elapsed.components.seconds) * 1000
                        + Double(elapsed.components.attoseconds) / 1e15)
            }
            return best
        }

        /// Rows of a batched decode/verify step: the fused decode kernels, the
        /// unsorted native combine and the sorted NAX experts (router and
        /// shared expert included in every arm).
        @Test func decodeCrossoverOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let geometry = Decode.Geometry.real
                for nvfp4 in [false, true] {
                    let moe = try Decode.moe(geometry, fusedSharedGateUp: true, nvfp4: nvfp4)
                    moe.gate.usesNativeKernel = true
                    let plan = try #require(moe.decodePlans.plan(moe, dtype: .bfloat16))
                    let format = nvfp4 ? "nvfp4" : "affine"
                    for rows in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128] {
                        let x = MLXRandom.normal(
                            [rows, 1, geometry.hidden], key: MLXRandom.key(UInt64(rows))
                        )
                        .asType(.bfloat16)
                        let flattened = x.reshaped(rows, geometry.hidden)
                        eval(x)
                        var arms: [(String, Double)] = []
                        if GLM5NextMoEDecodeKernel.supports(tokens: rows) {
                            arms.append(
                                (
                                    "fused_decode",
                                    try Self.time {
                                        [
                                            try GLM5NextMoEDecodeKernel.decode(
                                                flattened, plan: plan
                                            ).output
                                        ]
                                    }
                                ))
                        }
                        arms.append(
                            (
                                "unsorted_combine",
                                try Self.time {
                                    [try moe(x, routedExperts: .nativeCombine)]
                                }
                            ))
                        if rows * geometry.topK >= 8, GLM5NextMoEPrefillKernel.admits(plan) {
                            arms.append(
                                (
                                    "sorted_nax",
                                    try Self.time {
                                        let routing = try moe.gate.route(x)
                                        return [
                                            try GLM5NextMoEPrefillKernel.routedExperts(
                                                flattened, indices: routing.indices,
                                                weights: routing.weights,
                                                addend: nil, plan: plan,
                                                limit: moe.experts.swigluLimit,
                                                sharedLimit: moe.sharedExperts.swigluLimit)
                                        ]
                                    }
                                ))
                        }
                        arms.append(
                            (
                                "sorted_mlx",
                                try Self.time {
                                    let routing = try moe.gate.route(x)
                                    return [
                                        try moe.experts(
                                            flattened, indices: routing.indices,
                                            weights: routing.weights,
                                            sortedGather: true, addend: moe.sharedExperts(flattened)
                                        )
                                    ]
                                }
                            ))
                        for (name, ms) in arms {
                            print(
                                String(
                                    format: "GLM53_MOE_CROSSOVER %@ rows=%d %@ ms=%.3f", format,
                                    rows, name, ms))
                        }
                    }
                }
            }
        }

        /// Each layer's weights cold (distinct banks visited once per sweep),
        /// as a full forward reads them; the stage timings above are warm.
        @Test func coldLayerSweepOnGPU() throws {
            let lengths = (ProcessInfo.processInfo.environment["GLM53_PROFILE_PREFILL"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                let geometry = Decode.Geometry.real
                for nvfp4 in [false, true] {
                    let moes = try (0 ..< 6).map { _ in
                        try Decode.moe(geometry, fusedSharedGateUp: true, nvfp4: nvfp4)
                    }
                    for tokens in lengths {
                        let x = Decode.input(geometry, tokens: tokens, seed: 9)
                        eval(x)
                        for policy in [
                            GLM5NextExecutionPolicy.RoutedExperts.nativeCombineSortedPrefill,
                            .fusedDecodeNAXPrefill,
                        ] {
                            var best = Double.infinity
                            for _ in 0 ..< 3 {
                                let start = ContinuousClock.now
                                for moe in moes { eval(try moe(x, routedExperts: policy)) }
                                let elapsed = start.duration(to: .now)
                                best = min(
                                    best,
                                    (Double(elapsed.components.seconds) * 1000
                                        + Double(elapsed.components.attoseconds) / 1e15) / 6)
                            }
                            print(
                                String(
                                    format: "GLM53_MOE_COLD %@ tokens=%d %@ ms_per_layer=%.3f",
                                    nvfp4 ? "nvfp4" : "affine", tokens, "\(policy)", best))
                        }
                    }
                }
            }
        }

        @Test func stageTimingsOnGPU() throws {
            let lengths = (ProcessInfo.processInfo.environment["GLM53_PROFILE_PREFILL"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                // `GLM53_PROFILE_TP` profiles one tensor-parallel rank's widths.
                let geometry =
                    ProcessInfo.processInfo.environment["GLM53_PROFILE_TP"] != nil
                    ? Decode.Geometry.tensorParallelRank : .real
                for nvfp4 in [false, true] {
                    let moe = try Decode.moe(geometry, fusedSharedGateUp: true, nvfp4: nvfp4)
                    let experts = moe.experts
                    let format = nvfp4 ? "nvfp4" : "affine"
                    for tokens in lengths {
                        let x = Decode.input(geometry, tokens: tokens, seed: 9)
                        let flattened = x.reshaped(tokens, geometry.hidden)
                        let routing = moe.gate(x)
                        let indices = routing.indices
                        let weights = routing.weights
                        let order = argSort(indices.flattened())
                        let sorted = indices.flattened()[order]
                        let rows = flattened[order.floorDivide(geometry.topK)]
                            .expandedDimensions(axis: -2)
                        let gate = experts.gateProj(rows, sorted, sortedIndices: true)
                        let up = experts.upProj(rows, sorted, sortedIndices: true)
                        let act = glm5NextClampedGLUOperations(gate, up, limit: experts.swigluLimit)
                        let down = experts.downProj(act, sorted, sortedIndices: true)
                        let shared = moe.sharedExperts(flattened)
                        try Decode.fence([
                            routing.indices, weights, order, sorted, rows, gate, up, act, down,
                            shared,
                        ])
                        var stages: [(String, Double)] = []
                        stages.append(
                            (
                                "total(sortedPrefill)",
                                try Self.time {
                                    [try moe(x, routedExperts: .nativeCombineSortedPrefill)]
                                }
                            ))
                        let plan = try #require(moe.decodePlans.plan(moe, dtype: .bfloat16))
                        if GLM5NextMoEPrefillKernel.admits(plan) {
                            stages.append(
                                (
                                    "total(naxPrefill)",
                                    try Self.time {
                                        [try moe(x, routedExperts: .fusedDecodeNAXPrefill)]
                                    }
                                ))
                            stages.append(
                                (
                                    "nax_experts+combine",
                                    try Self.time {
                                        [
                                            try GLM5NextMoEPrefillKernel.routedExperts(
                                                flattened, indices: indices, weights: weights,
                                                addend: shared,
                                                plan: plan, limit: experts.swigluLimit,
                                                sharedLimit: moe.sharedExperts.swigluLimit)
                                        ]
                                    }
                                ))
                            for schedule in [false, true] {
                                stages.append(
                                    (
                                        "nax_gate_up(+sort,scan) interleaved=\(schedule)",
                                        try Self.time {
                                            [
                                                try GLM5NextMoEPrefillKernel.project(
                                                    flattened, indices: indices, plan: plan,
                                                    limit: experts.swigluLimit,
                                                    gateUpSchedule: schedule
                                                ).activations
                                            ]
                                        }
                                    ))
                            }
                            let seg = try GLM5NextMoEPrefillKernel.project(
                                flattened, indices: indices, plan: plan, limit: experts.swigluLimit,
                                gateUpSchedule: false)
                            let db = try GLM5NextMoEPrefillKernel.project(
                                flattened, indices: indices, plan: plan, limit: experts.swigluLimit,
                                gateUpSchedule: true)
                            try Decode.fence([seg.activations, db.activations])
                            let (scheduleError, scheduleFraction) = Decode.difference(
                                db.activations, seg.activations)
                            print(
                                String(
                                    format:
                                        "GLM53_MOE_BENCH tokens=%d interleaved_vs_seg max=%.5f differing=%.6f",
                                    tokens, scheduleError, scheduleFraction))
                            // Tile-plan sweep (oMLX 97e473ef's table was measured on
                            // affine/MXFP4 full-width shapes; NVFP4 and the
                            // tensor-parallel widths need their own numbers).
                            typealias Plan = GLM5NextMoEPrefillKernel.TilePlan
                            // (rows, depth, double-buffered, groupX, pad, prefetch)
                            let candidates: [Plan] = [
                                (64, 64, true, 0, 0, false), (64, 64, true, 32, 0, false),
                                (96, 64, true, 32, 0, false), (128, 64, true, 0, 0, false),
                                (128, 64, true, 32, 0, false), (96, 128, false, 32, 0, false),
                                (128, 128, false, 32, 0, false), (128, 128, false, 32, 8192, false),
                                // Seg with the next step's weight words in flight.
                                (96, 128, false, 32, 0, true), (128, 128, false, 32, 0, true),
                                (128, 128, false, 32, 8192, true), (128, 64, false, 32, 0, true),
                            ].map {
                                Plan(
                                    rows: $0.0, depth: $0.1, doubleBuffered: $0.2, groupX: $0.3,
                                    pad: $0.4, prefetch: $0.5)
                            }
                            let baseline = try GLM5NextMoEPrefillKernel.project(
                                flattened, indices: indices, plan: plan,
                                limit: experts.swigluLimit, gateUpPlan: candidates[0],
                                downPlan: candidates[0])
                            try Decode.fence([baseline.activations, baseline.projected])
                            for candidate in candidates {
                                stages.append(
                                    (
                                        "nax_gate_up(+sort,scan) plan=\(candidate)",
                                        try Self.time {
                                            [
                                                try GLM5NextMoEPrefillKernel.project(
                                                    flattened, indices: indices, plan: plan,
                                                    limit: experts.swigluLimit,
                                                    gateUpPlan: candidate
                                                ).activations
                                            ]
                                        }
                                    ))
                                stages.append(
                                    (
                                        "nax_all_projections down plan=\(candidate)",
                                        try Self.time {
                                            let s = try GLM5NextMoEPrefillKernel.project(
                                                flattened, indices: indices, plan: plan,
                                                limit: experts.swigluLimit, downPlan: candidate)
                                            return [s.projected, s.inverse]
                                        }
                                    ))
                                let s = try GLM5NextMoEPrefillKernel.project(
                                    flattened, indices: indices, plan: plan,
                                    limit: experts.swigluLimit, gateUpPlan: candidate,
                                    downPlan: candidate)
                                try Decode.fence([s.projected])
                                #expect(
                                    all(s.projected .== baseline.projected).item(Bool.self),
                                    "plan \(candidate) differs from \(candidates[0])")
                            }
                            let native = try moe(x, routedExperts: .fusedDecodeNAXPrefill)
                            let direct = try GLM5NextMoEPrefillKernel.routedExperts(
                                flattened, indices: indices, weights: weights, addend: nil,
                                plan: plan, limit: experts.swigluLimit,
                                sharedLimit: moe.sharedExperts.swigluLimit)
                            stages.append(
                                (
                                    "nax_experts+shared_in_launch+combine",
                                    try Self.time {
                                        [
                                            try GLM5NextMoEPrefillKernel.routedExperts(
                                                flattened, indices: indices, weights: weights,
                                                addend: nil, plan: plan,
                                                limit: experts.swigluLimit,
                                                sharedLimit: moe.sharedExperts.swigluLimit)
                                        ]
                                    }
                                ))
                            let separate = try GLM5NextMoEPrefillKernel.routedExperts(
                                flattened, indices: indices, weights: weights, addend: shared,
                                plan: plan, limit: experts.swigluLimit,
                                sharedLimit: moe.sharedExperts.swigluLimit)
                            try Decode.fence([direct, separate])
                            let (sharedError, sharedFraction) = Decode.difference(direct, separate)
                            print(
                                String(
                                    format:
                                        "GLM53_MOE_BENCH tokens=%d shared_in_launch_vs_addend max=%.5f differing=%.6f",
                                    tokens, sharedError, sharedFraction))
                            try Decode.fence([native, direct])
                            #expect(
                                Decode.bits(native) == Decode.bits(direct.reshaped(native.shape)))
                            let sortedOut = try moe(x, routedExperts: .nativeCombineSortedPrefill)
                            try Decode.fence([native, sortedOut])
                            let (error, fraction) = Decode.difference(native, sortedOut)
                            print(
                                String(
                                    format:
                                        "GLM53_MOE_BENCH tokens=%d nax_vs_sorted max=%.5f differing=%.4f",
                                    tokens, error, fraction))
                        }
                        // Repeated prompts route every token to the same few experts.
                        for spread in [8, 32] {
                            let skewed = MLXArray(
                                (0 ..< tokens * geometry.topK).map {
                                    UInt32(
                                        ($0 % geometry.topK) + ($0 / geometry.topK)
                                            % (spread / geometry.topK) * geometry.topK)
                                },
                                [tokens, geometry.topK])
                            let skewedWeights = MLXArray.ones([tokens, geometry.topK]) * 0.25
                            eval(skewed, skewedWeights)
                            stages.append(
                                (
                                    "skewed\(spread)_experts",
                                    try Self.time {
                                        [
                                            try experts(
                                                flattened, indices: skewed, weights: skewedWeights,
                                                sortedGather: true, addend: shared)
                                        ]
                                    }
                                ))
                            let sOrder = argSort(skewed.flattened())
                            let sSorted = skewed.flattened()[sOrder]
                            let sRows = flattened[sOrder.floorDivide(geometry.topK)]
                                .expandedDimensions(axis: -2)
                            eval(sOrder, sSorted, sRows)
                            stages.append(
                                (
                                    "skewed\(spread)_gate",
                                    Self.time {
                                        [experts.gateProj(sRows, sSorted, sortedIndices: true)]
                                    }
                                ))
                        }
                        stages.append(
                            (
                                "router",
                                Self.time {
                                    let r = moe.gate(x)
                                    return [r.indices, r.weights]
                                }
                            ))
                        if GLM5NextMoEPrefillKernel.admits(plan) {
                            let gateUpRows = GLM5NextMoEPrefillKernel.TilePlan.plan(
                                rows: tokens * geometry.topK, experts: geometry.experts,
                                input: geometry.hidden
                            ).shape.rows
                            let downRows = GLM5NextMoEPrefillKernel.TilePlan.plan(
                                rows: tokens * geometry.topK, experts: geometry.experts,
                                input: geometry.intermediate
                            ).shape.rows
                            stages.append(
                                (
                                    "route(sort+tiles+inverse, one launch)",
                                    Self.time {
                                        let r = GLM5NextMoEPrefillKernel.route(
                                            indices, experts: geometry.experts, sharedRows: tokens,
                                            gateUpRows: gateUpRows, downRows: downRows)
                                        return [r.order, r.inverse, r.gateUp.list, r.down.list]
                                    }
                                ))
                        }
                        if GLM5NextRouterKernel.supportsPrefill(
                            tokens: tokens, x: flattened, weight: moe.gate.weight,
                            topK: geometry.topK)
                        {
                            stages.append(
                                (
                                    "router(prefill kernel)",
                                    try Self.time {
                                        let r = try GLM5NextRouterKernel.routePrefill(
                                            flattened, weight: moe.gate.weight,
                                            bias: moe.gate.correctionBias, topK: geometry.topK,
                                            normTopkProb: moe.gate.normTopkProb,
                                            routedScalingFactor: moe.gate.routedScalingFactor)
                                        return [r.indices, r.weights]
                                    }
                                ))
                        }
                        stages.append(
                            (
                                "sort+rows",
                                Self.time {
                                    let o = argSort(indices.flattened())
                                    return [
                                        o, indices.flattened()[o],
                                        flattened[o.floorDivide(geometry.topK)],
                                    ]
                                }
                            ))
                        stages.append(
                            (
                                "gate",
                                Self.time { [experts.gateProj(rows, sorted, sortedIndices: true)] }
                            ))
                        stages.append(
                            (
                                "up",
                                Self.time { [experts.upProj(rows, sorted, sortedIndices: true)] }
                            ))
                        stages.append(
                            (
                                "glu",
                                Self.time {
                                    [
                                        glm5NextClampedGLUOperations(
                                            gate, up, limit: experts.swigluLimit)
                                    ]
                                }
                            ))
                        stages.append(
                            (
                                "down",
                                Self.time { [experts.downProj(act, sorted, sortedIndices: true)] }
                            ))
                        stages.append(("inverse(argSort)", Self.time { [argSort(order)] }))
                        stages.append(
                            (
                                "combine",
                                try Self.time {
                                    [
                                        try GLM5NextMoEKernel.ascendingCombine(
                                            projected: down.reshaped(
                                                tokens * geometry.topK, geometry.hidden),
                                            rowMap: argSort(order), indices: indices,
                                            weights: weights,
                                            addend: shared)
                                    ]
                                }
                            ))
                        stages.append(("shared", Self.time { [moe.sharedExperts(flattened)] }))
                        for (name, ms) in stages {
                            print(
                                String(
                                    format: "GLM53_MOE_BENCH %@ tokens=%d %@ ms=%.3f", format,
                                    tokens, name, ms))
                        }
                    }
                }
            }
        }
    }
}
