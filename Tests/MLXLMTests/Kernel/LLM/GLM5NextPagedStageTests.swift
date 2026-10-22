// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Opt-in (`GLM53_PAGED_KV=1`, GPU, the synthetic tensor-parallel rank-0
    /// serving stage of `GLM5NextBatchInvarianceTests`): a stage whose DSA
    /// histories live in a page pool (`GLM5NextTextStage.dsaPagePool`) gives
    /// every request exactly the logits, hidden rows and committed caches of
    /// the contiguous histories, bit for bit: prefill in chunks and in
    /// micro-batches, one-token steps, verification windows that keep a
    /// prefix (rollback), several lanes in one rows forward, and prefixes
    /// shared by page (two restores and the original diverging after one).
    @Suite(.serialized)
    struct GLM5NextPagedStageTests {
        static let enabled = ProcessInfo.processInfo.environment["GLM53_PAGED_KV"] == "1"
        typealias Fixture = GLM5NextBatchInvarianceTests

        private static func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private static func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        private static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype && bytes(a) == bytes(b)
        }

        private static func tokens(_ values: [Int32]) -> MLXArray { MLXArray(values, [1, values.count]) }

        private static func final(_ output: GLM5NextTextStageOutput) -> GLM5NextTextOutput? {
            if case .final(let value) = output { return value }
            return nil
        }

        /// One request's paged cache beside its contiguous twin.
        private struct Pair {
            let paged: GLM5NextTextStageCache
            let contiguous: GLM5NextTextStageCache
            func copy() -> Pair { Pair(paged: paged.copy(), contiguous: contiguous.copy()) }
        }

        private static func expectSame(_ pair: Pair, _ label: String) {
            #expect(pair.paged.offset == pair.contiguous.offset, "\(label) offset")
            let left = pair.paged.layers.flatMap(\.state)
            let right = pair.contiguous.layers.flatMap(\.state)
            #expect(left.count == right.count, "\(label) state count")
            for (index, (x, y)) in zip(left, right).enumerated() {
                #expect(identical(x, y), "\(label) state \(index)")
            }
        }

        private static func expectSame(
            _ a: GLM5NextTextStageOutput, _ b: GLM5NextTextStageOutput, _ label: String
        ) throws {
            switch (a, b) {
            case (.final(let x), .final(let y)):
                #expect(identical(x.logits, y.logits), "\(label) logits")
                #expect(identical(x.hidden, y.hidden), "\(label) hidden")
            case (.hidden(let x), .hidden(let y)):
                #expect(identical(x, y), "\(label) hidden")
            default:
                Issue.record("\(label): different outputs")
            }
        }

        /// Runs the same forward on both caches and compares.
        private static func forward(
            _ stage: GLM5NextTextStage, _ pair: Pair, _ values: [Int32],
            readout: GLM5NextTextStageReadout, _ label: String
        ) throws {
            let a = try stage.prepareForward(tokens: tokens(values), cache: pair.paged, readout: readout)
            try a.commit(afterEvaluating: fence)
            let b = try stage.prepareForward(
                tokens: tokens(values), cache: pair.contiguous, readout: readout)
            try b.commit(afterEvaluating: fence)
            try expectSame(a.output, b.output, label)
            expectSame(pair, label)
        }

        private static func verify(
            _ stage: GLM5NextTextStage, _ pair: Pair, _ window: [Int32], keep: Int, _ label: String
        ) throws {
            let a = try stage.prepareVerification(tokens: tokens(window), cache: pair.paged)
            try a.commit(afterEvaluating: fence) { _ in keep }
            let b = try stage.prepareVerification(tokens: tokens(window), cache: pair.contiguous)
            try b.commit(afterEvaluating: fence) { _ in keep }
            try expectSame(a.output, b.output, label)
            expectSame(pair, label)
        }

        @Test(.enabled(if: enabled), arguments: [1, 2])
        func pagedStageIsTheContiguousStage(microBatches: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                try #require(glm5NextOnDefaultGPUStream())
                let stage = try Fixture.stage()
                let saved = (stage.dsaPagePool, stage.tensorParallelMicroBatches)
                defer { (stage.dsaPagePool, stage.tensorParallelMicroBatches) = saved }
                stage.tensorParallelMicroBatches = microBatches
                let pool = try stage.makeDSAPagePool(pageSize: 64)
                let account = try pool.reserve(pages: 4096)
                let vocabulary = stage.plan.configuration.vocabularySize
                func open() -> Pair {
                    stage.dsaPagePool = pool
                    let paged = stage.newCache(account: account)
                    stage.dsaPagePool = nil
                    return Pair(paged: paged, contiguous: stage.newCache())
                }
                func prompt(_ length: Int, _ seed: Int) -> [Int32] {
                    (0 ..< length).map { Int32((($0 + 1) * 7919 + seed * 104_729) % vocabulary) }
                }
                // Dense, straddling index_topk, sparse (whole-history reads),
                // and long enough that decode steps and 3-token windows read
                // only their selected latents (past 2 x 3 x 2,051 rows).
                var lanes: [Pair] = []
                for (lane, length) in [700, 2_045, 2_600, 12_400].enumerated() {
                    let pair = open()
                    let values = prompt(length, lane)
                    // A 1,024-token grid (micro-batched when asked), a short tail.
                    for start in stride(from: 0, to: length, by: 1024) {
                        let end = min(length, start + 1024)
                        try Self.forward(
                            stage, pair, Array(values[start ..< end]),
                            readout: end == length ? .lastToken : .none,
                            "mb=\(microBatches) lane \(lane) prefill \(start)")
                    }
                    lanes.append(pair)
                }
                for (lane, pair) in lanes.enumerated() {
                    // One-token steps, then verification windows keeping 1...3.
                    for step in 0 ..< 3 {
                        try Self.forward(
                            stage, pair, [Int32((lane * 31 + step * 7 + 3) % vocabulary)],
                            readout: .lastToken, "mb=\(microBatches) lane \(lane) step \(step)")
                    }
                    for (round, keep) in [1, 3, 2, 1, 3].enumerated() {
                        let window = (0 ..< 3).map {
                            Int32((lane * 131 + round * 17 + $0 * 5 + 11) * 9973 % vocabulary)
                        }
                        try Self.verify(
                            stage, pair, window, keep: keep,
                            "mb=\(microBatches) lane \(lane) round \(round) keep \(keep)")
                    }
                }
                // Several lanes in one rows forward (paged) against their own
                // contiguous windows.
                for round in 0 ..< 3 {
                    let windows = (0 ..< lanes.count).map { lane in
                        (0 ..< (round + lane) % 3 + 1).map {
                            Int32((lane * 37 + round * 13 + $0 * 3 + 5) * 7919 % vocabulary)
                        }
                    }
                    let kept = windows.enumerated().map { lane, window in (lane + round) % window.count + 1 }
                    let rows = try stage.prepareVerificationRows(
                        zip(windows, lanes).map { .init(tokens: Self.tokens($0), cache: $1.paged) })
                    let outputs = try rows.commit(afterEvaluating: Self.fence) { _ in kept }
                    for (lane, pair) in lanes.enumerated() {
                        let own = try stage.prepareVerification(
                            tokens: Self.tokens(windows[lane]), cache: pair.contiguous)
                        try own.commit(afterEvaluating: Self.fence) { _ in kept[lane] }
                        let label = "mb=\(microBatches) rows round \(round) lane \(lane)"
                        try Self.expectSame(outputs[lane], own.output, label)
                        Self.expectSame(pair, label)
                    }
                }
                // A prefix entry at a boundary inside a page: two restores and
                // the original go on differently from the shared pages.
                let original = lanes[1]
                let entry = original.copy()
                let restores = [entry.copy(), entry.copy()]
                for (index, pair) in ([original] + restores).enumerated() {
                    let values = prompt(70 + index * 9, 50 + index)
                    try Self.forward(
                        stage, pair, values, readout: .lastToken,
                        "mb=\(microBatches) shared prefix branch \(index)")
                    try Self.verify(
                        stage, pair, Array(values.prefix(3)), keep: 2,
                        "mb=\(microBatches) shared prefix branch \(index) window")
                }
                Self.expectSame(entry, "mb=\(microBatches) entry after its branches")
                // Pages: the histories hold them; all come back when they go.
                #expect(pool.allocatedPages > 0)
                lanes.removeAll()
                _ = (restores, entry)
            }
        }

        /// Opt-in timing (`GLM53_PAGED_BENCH=1`, a release test build;
        /// `GLM53_PAGED_BENCH_CONTEXTS`, default 4096,32768): one request's
        /// one-token decode steps and 3-token verification windows (keeping
        /// 3, then 1) at each context, paged against contiguous, median wall
        /// milliseconds per step, each step evaluated before the next.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_PAGED_BENCH"] == "1"))
        func pagedDecodeStepTime() throws {
            try Device.withDefaultDevice(.gpu) {
                let environment = ProcessInfo.processInfo.environment
                let contexts = (environment["GLM53_PAGED_BENCH_CONTEXTS"] ?? "4096,32768")
                    .split(separator: ",").compactMap { Int($0) }
                let stage = try Fixture.stage()
                let saved = stage.dsaPagePool
                defer { stage.dsaPagePool = saved }
                let pool = try stage.makeDSAPagePool(pageSize: 64)
                let vocabulary = stage.plan.configuration.vocabularySize
                func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
                func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
                for context in contexts {
                    for paged in [false, true] {
                        let account = try pool.reserve(pages: 4 * pool.pages(tokens: context + 512))
                        defer { pool.release(account) }
                        stage.dsaPagePool = paged ? pool : nil
                        let cache = stage.newCache(account: account)
                        stage.dsaPagePool = nil
                        let prompt = (0 ..< context).map { Int32(($0 * 7919 + 13) % vocabulary) }
                        for start in stride(from: 0, to: context, by: 2048) {
                            try stage.prepareForward(
                                tokens: Self.tokens(Array(prompt[start ..< min(context, start + 2048)])),
                                cache: cache, readout: .none
                            ).commit(afterEvaluating: Self.fence)
                        }
                        var steps: [Double] = []
                        var builds: [Double] = []
                        for index in 0 ..< 40 {
                            let started = now()
                            let candidate = try stage.prepareForward(
                                tokens: Self.tokens([Int32((index * 31 + 7) % vocabulary)]), cache: cache,
                                readout: .lastToken)
                            let built = now()
                            try candidate.commit(afterEvaluating: Self.fence)
                            if index >= 8 {
                                steps.append(now() - started)
                                builds.append(built - started)
                            }
                        }
                        var windows: [Double] = []
                        for index in 0 ..< 24 {
                            let window = (0 ..< 3).map { Int32((index * 17 + $0 * 5 + 3) % vocabulary) }
                            let started = now()
                            try stage.prepareVerification(tokens: Self.tokens(window), cache: cache)
                                .commit(afterEvaluating: Self.fence) { _ in index % 2 == 0 ? 3 : 1 }
                            if index >= 4 { windows.append(now() - started) }
                        }
                        print(
                            String(
                                format: "GLM53_PAGED_BENCH ctx=%d kv=%@ step_ms=%.3f build_ms=%.3f window_ms=%.3f pages=%d",
                                context, paged ? "paged" : "contiguous", median(steps), median(builds),
                                median(windows), pool.allocatedPages))
                    }
                }
            }
        }
    }
}
