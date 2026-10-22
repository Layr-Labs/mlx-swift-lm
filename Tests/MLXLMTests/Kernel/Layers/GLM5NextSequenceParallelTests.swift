// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Sequence-parallel prefill pieces at one TP2 rank's geometry: each
    /// rank's rows routed alone and gathered give the MoE layer's own output,
    /// and an mHC site over each rank's rows gives the whole site's rows, bit
    /// for bit (`GLM5NextTextLayer.sequenceParallelSteps`). Both ranks are
    /// played here in one process; the collectives themselves are the pair's.
    @Suite(.serialized)
    struct GLM5NextSequenceParallelTests {
        static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
            let raw: DType = a.dtype.size == 2 ? .uint16 : .uint32
            return (a.view(dtype: raw) .== b.view(dtype: raw)).all().item(Bool.self)
        }

        @Test func moeRoutingGatheredFromHalvesIsTheLayersOwn() throws {
            try Device.withDefaultDevice(.gpu) {
                typealias Decode = GLM5NextMoEDecodeKernelTests
                let moe = try Decode.moe(.tensorParallelRank, fusedSharedGateUp: true, nvfp4: true)
                moe.gate.usesNativeKernel = true
                let routed = GLM5NextExecutionPolicy.serving.routedExperts
                for rows in [4096, 8192] {
                    let x = Decode.input(.tensorParallelRank, tokens: rows, seed: 41)
                    let half = rows / 2
                    // Each rank's rows routed alone, gathered in rank order.
                    let frames = try [x[0..., ..<half], x[0..., half...]].map { try moe.routingFrame($0) }
                    let gathered = concatenated(frames, axis: 1).reshaped(rows, -1)
                    let plain = try moe(x, routedExperts: routed)
                    let split = try moe(x, routedExperts: routed, gatheredRouting: gathered)
                    eval(plain, split)
                    #expect(Self.bitwise(plain, split), "rows \(rows)")
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try moe(
                        Decode.input(.tensorParallelRank, tokens: 16, seed: 1), routedExperts: routed,
                        gatheredRouting: MLXArray.zeros([8, 16]))
                }
            }
        }

        /// The fused mHC site of this rank's rows is the whole site's rows (the
        /// sequence-parallel layer runs every site on its own rows).
        @Test func hyperConnectionSiteOfHalfTheRowsIsTheWholeSitesRows() throws {
            typealias Kernel = GLM5NextHyperConnectionKernel
            try Device.withDefaultDevice(.gpu) {
                let width = 4096, rows = 4096
                let keys = MLXRandom.split(key: MLXRandom.key(43), into: 7)
                let streams = MLXRandom.normal([1, rows, 4, width], key: keys[0]).asType(.bfloat16)
                let output = MLXRandom.normal([1, rows, width], key: keys[1]).asType(.bfloat16)
                let fn = (MLXRandom.normal([24, 4 * width], key: keys[2]) * 0.05).asType(.bfloat16)
                let base = MLXRandom.normal([24], key: keys[3]) * 0.5
                let scale = MLXArray([Float(0.75), 1.25, 1])
                let weight = MLXRandom.uniform(low: Float(0.5), high: 1.5, [width], key: keys[4]).asType(.bfloat16)
                let post = MLXRandom.uniform(low: Float(0), high: 2, [1, rows, 4], key: keys[5])
                let comb = MLXRandom.uniform(low: Float(0), high: 0.5, [1, rows, 4, 4], key: keys[6])
                func site(_ range: Range<Int>) throws -> [MLXArray] {
                    let result = try Kernel.collapseNormalized(
                        .expand(
                            output: output[0..., range], residual: streams[0..., range],
                            post: post[0..., range], comb: comb[0..., range]),
                        fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                        sinkhornIterations: 20, normWeight: weight, normEpsilon: 1e-5)
                    return [result.streams, result.post, result.comb, result.normalized]
                }
                let whole = try site(0 ..< rows)
                let halves = try [site(0 ..< rows / 2), site(rows / 2 ..< rows)]
                eval(whole + halves[0] + halves[1])
                for index in 0 ..< 4 {
                    #expect(
                        Self.bitwise(whole[index], concatenated([halves[0][index], halves[1][index]], axis: 1)),
                        "output \(index)")
                }
            }
        }

        /// Opt-in probe: the router's FP32 GEMM (`[rows, 4096] x [4096, experts]`)
        /// over half the rows against the whole, by expert count. MLX picks
        /// the GEMM's form by shape, so the halves keep the K order for some
        /// expert counts only (M5 Ultra, 2026-10-22: 288, the artifact's, at
        /// 4,096 and 8,192 rows; not 16, 64, 128 or 256 at every size): the
        /// sequence-parallel router's exactness is the artifact's shape's.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_SEQUENCE_PARALLEL"] == "1"))
        func routerGEMMHalvesByExpertCount() throws {
            try Device.withDefaultDevice(.gpu) {
                for experts in [16, 64, 128, 256, 288] {
                    for rows in [4096, 8192] {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(experts)), into: 2)
                        let x = MLXRandom.normal([rows, 4096], key: keys[0]).asType(.bfloat16).asType(.float32)
                        let w = MLXRandom.normal([experts, 4096], key: keys[1]) * 0.02
                        let whole = matmul(x, w.T)
                        let halves = [matmul(x[..<(rows / 2)], w.T), matmul(x[(rows / 2)...], w.T)]
                        eval(whole, halves[0], halves[1])
                        let ok = Self.bitwise(whole, concatenated(halves, axis: 0))
                        print("GLM53_SP_ROUTER_GEMM experts=\(experts) rows=\(rows) halves_bitwise=\(ok ? 1 : 0)")
                    }
                }
            }
        }

        /// Opt-in (`GLM53_SEQUENCE_PARALLEL=1` with `GLM53_ROW_INVARIANCE_FULL=1`,
        /// GPU, the row-invariance suite's rank-shaped synthetic stage with the
        /// artifact's 288 experts and vocabulary, about 5.3 GB; with 16 experts
        /// the router GEMM's halves change order, `routerGEMMHalvesByExpertCount`,
        /// and a few rows route differently): a segmented prefill with
        /// sequence parallelism is the plain segmented prefill bit for bit
        /// (every segment's hidden rows, the logits, the cache and the
        /// in-graph snapshot), with split micro-batches, whole ones, and a
        /// segment mixing both. Both ranks are played in one process. The
        /// fixture's peer holds this rank's weights (its sum is `x + x`), so a
        /// rank's reduce-scatter is its own rows of `x + x`, and its all-gather
        /// takes the peer's rows from the peer's previous pass (its own rows
        /// twice in the first). Rows depend only on earlier rows, so each pass
        /// settles at least one more half micro-batch; the passes stop when
        /// neither rank's gathered rows change.
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["GLM53_SEQUENCE_PARALLEL"] == "1"
                    && GLM5NextBatchInvarianceTests.full))
        func sequenceParallelPrefillIsThePlainPrefillBitForBit() throws {
            func fence(_ roots: [MLXArray]) throws {
                try MLX.withError { error in
                    eval(roots)
                    try error.check()
                }
            }
            func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }
            func same(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
                a.count == b.count && zip(a, b).allSatisfy { $0.shape == $1.shape && $0.dtype == $1.dtype && bytes($0) == bytes($1) }
            }
            struct Outcome { let hidden: [MLXArray], logits: MLXArray, cache: [MLXArray], snapshot: [MLXArray] }
            try Device.withDefaultDevice(.gpu) {
                // Its own stage: the suite's shared one may serve other suites meanwhile.
                let stage = try GLM5NextBatchInvarianceTests.makeStage(layers: 5)
                let vocabulary = stage.plan.configuration.vocabularySize
                let segments: [GLM5NextPrefillSegment] = [
                    .init(microBatchRows: [4096, 4096]), .init(microBatchRows: [1024, 1024]),
                    .init(microBatchRows: [4096, 512]),
                ]
                let count = segments.reduce(0) { $0 + $1.tokens }
                let tokens = MLXArray((0 ..< count).map { Int32((($0 + 1) * 7919) % vocabulary) }, [1, count])
                // `sliced`: the frame in one-layer slices, each evaluated.
                func run(sliced: Bool = false) throws -> Outcome {
                    let cache = stage.newCache()
                    let prefill = try stage.beginForward(
                        tokens: tokens, cache: cache, segments: segments, readout: .lastToken, snapshotAfter: 1)
                    for _ in 0 ..< (sliced ? stage.layers.count : 1) {
                        try prefill.advance(layers: sliced ? 1 : stage.layers.count)
                        if sliced { try fence(prefill.frontier) }
                    }
                    let candidate = try prefill.finish()
                    try candidate.commit(afterEvaluating: fence)
                    guard case .final(let output) = candidate.output, let snapshot = candidate.snapshot else {
                        throw GLM5NextLoadError.invalid("segmented prefill returned no logits or snapshot")
                    }
                    let outcome = Outcome(
                        hidden: candidate.segmentHidden, logits: output.logits,
                        cache: cache.layers.flatMap(\.state), snapshot: snapshot.layers.flatMap(\.state))
                    try fence(outcome.hidden + [outcome.logits] + outcome.cache + outcome.snapshot)
                    return outcome
                }
                stage.sequenceParallel = nil
                let reference = try run()

                var sent: [[MLXArray]] = [[], []]
                var unchanged = [false, false]
                var outcomes: [Outcome?] = [nil, nil]
                var passes = 0
                for pass in 0 ..< 16 {
                    let share = pass % 2
                    let peer = sent[1 - share]
                    var calls: [MLXArray] = []
                    stage.sequenceParallel = .init(
                        share: share, shares: 2,
                        reduceScatter: { partial in
                            let half = partial.dim(1) / 2
                            let own = partial[0..., (share * half) ..< ((share + 1) * half)]
                            return (own.asType(.float32) + own.asType(.float32)).asType(partial.dtype)
                        },
                        allGather: { own in
                            let index = calls.count
                            calls.append(own)
                            let other = index < peer.count ? peer[index] : own
                            return concatenated(share == 0 ? [own, other] : [other, own], axis: 1)
                        })
                    // Odd passes of each rank run in layer slices.
                    let outcome = try run(sliced: pass % 4 >= 2)
                    try fence(calls)
                    if pass == 0 {
                        // Negative control: the first pass's peer rows are not the peer's.
                        #expect(!same(outcome.hidden, reference.hidden))
                    }
                    unchanged[share] = same(sent[share], calls)
                    sent[share] = calls
                    outcomes[share] = outcome
                    passes = pass + 1
                    if unchanged[0] && unchanged[1] { break }
                }
                print("GLM53_SEQUENCE_PARALLEL passes=\(passes) gathers=\(sent[0].count)")
                #expect(unchanged[0] && unchanged[1], "the passes did not settle")
                #expect(sent[0].count == sent[1].count)
                /// Where two lists of arrays differ: index, shape, rows that differ (axis 1).
                func differences(_ name: String, _ a: [MLXArray], _ b: [MLXArray]) {
                    for (index, (x, y)) in zip(a, b).enumerated() where !same([x], [y]) {
                        guard x.shape == y.shape, x.ndim >= 2 else {
                            print("GLM53_SP_DIFF \(name)[\(index)] shape \(x.shape) vs \(y.shape)")
                            continue
                        }
                        let raw: DType = x.dtype.size == 2 ? .uint16 : (x.dtype.size == 4 ? .uint32 : .uint8)
                        let unequal = x.view(dtype: raw) .!= y.view(dtype: raw)
                        let axes = Array(0 ..< x.ndim).filter { $0 != 1 }
                        let rows = unequal.any(axes: axes).asArray(Bool.self)
                        let differing = rows.indices.filter { rows[$0] }
                        let delta = abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                        print(
                            "GLM53_SP_DIFF \(name)[\(index)] shape \(x.shape) \(x.dtype) rows \(differing.count)/\(rows.count)"
                                + " first \(differing.first ?? -1) last \(differing.last ?? -1) max_abs \(delta)")
                    }
                }
                for share in 0 ..< 2 {
                    let outcome = try #require(outcomes[share])
                    differences("rank\(share).hidden", outcome.hidden, reference.hidden)
                    differences("rank\(share).cache", outcome.cache, reference.cache)
                    #expect(same(outcome.hidden, reference.hidden), "rank \(share) hidden rows")
                    #expect(same([outcome.logits], [reference.logits]), "rank \(share) logits")
                    #expect(same(outcome.cache, reference.cache), "rank \(share) cache")
                    #expect(same(outcome.snapshot, reference.snapshot), "rank \(share) snapshot")
                }
            }
        }
    }
}
