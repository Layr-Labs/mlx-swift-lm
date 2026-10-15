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
                    let x = MLXRandom.normal([rows, 1, geometry.hidden], key: MLXRandom.key(UInt64(rows)))
                        .asType(.bfloat16)
                    let flattened = x.reshaped(rows, geometry.hidden)
                    eval(x)
                    var arms: [(String, Double)] = []
                    if GLM5NextMoEDecodeKernel.supports(tokens: rows) {
                        arms.append(("fused_decode", try Self.time {
                            [try GLM5NextMoEDecodeKernel.decode(flattened, plan: plan).output]
                        }))
                    }
                    arms.append(("unsorted_combine", try Self.time {
                        [try moe(x, routedExperts: .nativeCombine)]
                    }))
                    if rows * geometry.topK >= 8, GLM5NextMoEPrefillKernel.admits(plan) {
                        arms.append(("sorted_nax", try Self.time {
                            let routing = try moe.gate.route(x)
                            return [try GLM5NextMoEPrefillKernel.routedExperts(
                                flattened, indices: routing.indices, weights: routing.weights,
                                addend: moe.sharedExperts(flattened), plan: plan,
                                limit: moe.experts.swigluLimit)]
                        }))
                    }
                    arms.append(("sorted_mlx", try Self.time {
                        let routing = try moe.gate.route(x)
                        return [try moe.experts(
                            flattened, indices: routing.indices, weights: routing.weights,
                            sortedGather: true, addend: moe.sharedExperts(flattened))]
                    }))
                    for (name, ms) in arms {
                        print(String(format: "GLM53_MOE_CROSSOVER %@ rows=%d %@ ms=%.3f", format, rows, name, ms))
                    }
                }
                }
            }
        }

        @Test func stageTimingsOnGPU() throws {
            let lengths = (ProcessInfo.processInfo.environment["GLM53_PROFILE_PREFILL"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                let geometry = Decode.Geometry.real
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
                    try Decode.fence([routing.indices, weights, order, sorted, rows, gate, up, act, down, shared])
                    var stages: [(String, Double)] = []
                    stages.append(("total(sortedPrefill)", try Self.time {
                        [try moe(x, routedExperts: .nativeCombineSortedPrefill)]
                    }))
                    let plan = try #require(moe.decodePlans.plan(moe, dtype: .bfloat16))
                    if GLM5NextMoEPrefillKernel.admits(plan) {
                    stages.append(("total(naxPrefill)", try Self.time {
                        [try moe(x, routedExperts: .fusedDecodeNAXPrefill)]
                    }))
                    stages.append(("nax_experts+combine", try Self.time {
                        [try GLM5NextMoEPrefillKernel.routedExperts(
                            flattened, indices: indices, weights: weights, addend: shared,
                            plan: plan, limit: experts.swigluLimit)]
                    }))
                    for schedule in [false, true] {
                        stages.append(("nax_gate_up(+sort,scan) interleaved=\(schedule)", try Self.time {
                            [try GLM5NextMoEPrefillKernel.project(
                                flattened, indices: indices, plan: plan,
                                limit: experts.swigluLimit, gateUpSchedule: schedule
                            ).activations]
                        }))
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
                    print(String(format: "GLM53_MOE_BENCH tokens=%d interleaved_vs_seg max=%.5f differing=%.6f", tokens, scheduleError, scheduleFraction))
                    for schedule in [false, true] {
                        stages.append(("nax_all_projections down_db=\(schedule)", try Self.time {
                            let s = try GLM5NextMoEPrefillKernel.project(
                                flattened, indices: indices, plan: plan,
                                limit: experts.swigluLimit, downSchedule: schedule)
                            return [s.projected, s.inverse]
                        }))
                    }
                    let native = try moe(x, routedExperts: .fusedDecodeNAXPrefill)
                    let direct = try GLM5NextMoEPrefillKernel.routedExperts(
                        flattened, indices: indices, weights: weights, addend: shared,
                        plan: plan, limit: experts.swigluLimit)
                    try Decode.fence([native, direct])
                    #expect(Decode.bits(native) == Decode.bits(direct.reshaped(native.shape)))
                    let sortedOut = try moe(x, routedExperts: .nativeCombineSortedPrefill)
                    try Decode.fence([native, sortedOut])
                    let (error, fraction) = Decode.difference(native, sortedOut)
                    print(String(format: "GLM53_MOE_BENCH tokens=%d nax_vs_sorted max=%.5f differing=%.4f", tokens, error, fraction))
                    }
                    stages.append(("router", Self.time { let r = moe.gate(x); return [r.indices, r.weights] }))
                    stages.append(("sort+rows", Self.time {
                        let o = argSort(indices.flattened())
                        return [o, indices.flattened()[o], flattened[o.floorDivide(geometry.topK)]]
                    }))
                    stages.append(("gate", Self.time { [experts.gateProj(rows, sorted, sortedIndices: true)] }))
                    stages.append(("up", Self.time { [experts.upProj(rows, sorted, sortedIndices: true)] }))
                    stages.append(("glu", Self.time {
                        [glm5NextClampedGLUOperations(gate, up, limit: experts.swigluLimit)]
                    }))
                    stages.append(("down", Self.time { [experts.downProj(act, sorted, sortedIndices: true)] }))
                    stages.append(("inverse(argSort)", Self.time { [argSort(order)] }))
                    stages.append(("combine", try Self.time {
                        [try GLM5NextMoEKernel.ascendingCombine(
                            projected: down.reshaped(tokens * geometry.topK, geometry.hidden),
                            rowMap: argSort(order), indices: indices, weights: weights,
                            addend: shared)]
                    }))
                    stages.append(("shared", Self.time { [moe.sharedExperts(flattened)] }))
                    for (name, ms) in stages {
                        print(String(format: "GLM53_MOE_BENCH %@ tokens=%d %@ ms=%.3f", format, tokens, name, ms))
                    }
                }
                }
            }
        }
    }
}
