// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// GPU: the DSA prefill selection shared between two ranks by rows
    /// (`GLM5NextSparseAttention.selectTokensSplit`): each share scores and
    /// selects its own rows, the shares' partials are summed, and every share
    /// finishes the whole call's selection; it is `selectTokens`' bit for bit,
    /// with exact ties at the budget boundary, kept selections, odd row
    /// counts and a short call that takes the dense shortcut (no split).
    @Suite(.serialized)
    struct GLM5NextSelectionSplitTests {
        typealias Pinning = GLM5NextSelectionPinningTests

        static func selection(
            _ fixture: Pinning.Fixture, offset: Int, rows: Int, histories: [(rowEnd: Int, keys: Int)]?,
            split: Bool
        ) throws -> MLXArray? {
            let base = fixture.indexKeys.dim(1) - fixture.queries.dim(1)
            let first = offset - base
            let n = offset + rows
            let execution = GLM5NextExecutionPolicy.serving.sparseSelection
            let queries = fixture.queries[0..., first ..< first + rows]
            let keys = fixture.indexKeys[0..., ..<n], gates = fixture.gates[0..., ..<n]
            let weights = fixture.weights[0..., first ..< first + rows]
            let keyValidity = MLXArray.ones([1, n], dtype: .bool)
            let queryValidity = MLXArray.ones([1, rows], dtype: .bool)
            guard split else {
                let selection = try GLM5NextSparseAttention.selectTokens(
                    indexQueries: queries, indexKeys: keys, poolGates: gates, poolPositionBias: fixture.bias,
                    headWeights: weights, keyValidity: keyValidity, queryValidity: queryValidity,
                    queryStart: offset, tokenBudget: 2048, execution: execution, nativePrefillScores: true,
                    selectionHistories: histories)
                eval(selection.indices)
                return selection.indices
            }
            let shares = try (0 ..< 2).map { share in
                try GLM5NextSparseAttention.selectTokensSplit(
                    indexQueries: queries, indexKeys: keys, poolGates: gates, poolPositionBias: fixture.bias,
                    headWeights: weights, keyValidity: keyValidity, queryValidity: queryValidity,
                    queryStart: offset, tokenBudget: 2048, execution: execution, share: share, shares: 2,
                    nativePrefillScores: true, selectionHistories: histories)
            }
            guard let zero = shares[0], let one = shares[1] else { return nil }
            let sum = zero.partial + one.partial
            let a = try zero.finish(sum), b = try one.finish(sum)
            eval(a.indices, b.indices)
            #expect(Pinning.differingRows(a.indices, b.indices) == 0, "both shares finish the same selection")
            return a.indices
        }

        @Test func splitSelectionIsTheWholeCallsSelection() throws {
            try Device.withDefaultDevice(.gpu) {
                for (offset, rows, pinned) in [(4096, 2048, false), (6144, 1537, false), (4096, 1536, true), (8192, 4096, true)] {
                    let fixture = Pinning.fixture(offset: offset, rows: rows)
                    let histories = pinned ? Pinning.histories(productRows: 512, otherRows: rows, offset: offset) : nil
                    let whole = try #require(try Self.selection(fixture, offset: offset, rows: rows, histories: histories, split: false))
                    let split = try #require(try Self.selection(fixture, offset: offset, rows: rows, histories: histories, split: true))
                    let differing = Pinning.differingRows(whole, split)
                    print("GLM53_SELECTION_SPLIT offset=\(offset) rows=\(rows) pinned=\(pinned ? 1 : 0) rows_differing=\(differing)")
                    #expect(differing == 0)
                }
                // A call whose budget covers every pool takes the dense shortcut: no split.
                let fixture = Pinning.fixture(offset: 1024, rows: 512)
                #expect(try Self.selection(fixture, offset: 1024, rows: 512, histories: nil, split: true) == nil)
            }
        }
    }
}

extension KernelTests {
    /// GPU, one TP2 rank's DSA geometry (synthetic): a DSA step whose sparse
    /// rows' selection is shared between two ranks by rows
    /// (`GLM5NextDSAAttention.forwardExchanging`, each rank's partial summed)
    /// returns `forward`'s values and leaves its cache bit for bit, for
    /// contiguous and paged histories, at histories 0 to 12K, with kept
    /// selections; a step with nothing to share has no partial.
    @Suite(.serialized)
    struct GLM5NextDSAExchangeTests {
        static func bitwise(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
            a.count == b.count && zip(a, b).allSatisfy { x, y in
                guard x.shape == y.shape, x.dtype == y.dtype else { return false }
                let raw: DType = x.dtype.size == 2 ? .uint16 : (x.dtype.size == 4 ? .uint32 : .uint8)
                return (x.view(dtype: raw) .== y.view(dtype: raw)).all().item(Bool.self)
            }
        }

        @Test(arguments: [false, true])
        func exchangedStepIsTheStep(paged: Bool) throws {
            guard glm5NextNAXAvailable else { return }
            try Device.withDefaultDevice(.gpu) {
                let tokens = 4096
                let fixture = try GLM5NextDSARankFixture()
                let dsa = try fixture.attention()
                let cache: GLM5NextDSACache
                if paged {
                    let types = try dsa.historyTypes(input: fixture.input(tokens: 1, seed: 1).dtype)
                    let c = fixture.text
                    let pool = try GLM5NextDSAPagePool(
                        pageSize: 64, latentWidth: c.kvLoraRank, keyWidth: c.indexHeadDim,
                        latentType: types.latents, keyType: types.keys, gateType: types.gates)
                    cache = GLM5NextDSACache(pagePool: pool, account: try pool.reserve(pages: 4096))
                } else {
                    cache = GLM5NextDSACache()
                }
                let all = MLXArray.ones([1, tokens], dtype: .bool)
                for step in 0 ..< 4 {
                    let x = fixture.input(tokens: tokens, seed: UInt64(300 + step))
                    eval(x)
                    for kept in [false, true] {
                        // Kept selections: rows choose as two 2,048-row steps would.
                        let selections = kept
                            ? GLM5NextDSAAttention.SelectionHistories(pieces: [(rowEnd: 2048, keyEnd: 2048), (rowEnd: 4096, keyEnd: 4096)])
                            : nil
                        let reference = cache.copy() as! GLM5NextDSACache
                        let expected = try dsa.forward(x, cache: reference, validity: all, selections: selections)
                        eval([expected] + reference.state)
                        let caches = (0 ..< 2).map { _ in cache.copy() as! GLM5NextDSACache }
                        let steps = try (0 ..< 2).map { share in
                            try dsa.forwardExchanging(
                                x, cache: caches[share], validity: all, selections: selections, share: share, shares: 2)
                        }
                        let label = "history \(step * tokens) paged \(paged) kept \(kept)"
                        let sum: MLXArray? = steps[0].partial.flatMap { a in steps[1].partial.map { a + $0 } }
                        #expect((steps[0].partial == nil) == (steps[1].partial == nil), "\(label)")
                        #expect((sum == nil) == (step == 0 && false), "\(label): every step here shares its selection")
                        for share in 0 ..< 2 {
                            let values = try steps[share].resume(sum)
                            eval([values] + caches[share].state)
                            #expect(Self.bitwise([values], [expected]), "\(label) share \(share) values")
                            #expect(Self.bitwise(caches[share].state, reference.state), "\(label) share \(share) cache")
                        }
                    }
                    _ = try dsa.forward(x, cache: cache, validity: all)
                    eval(cache.state)
                }
                // A decode-sized step shares nothing.
                let one = fixture.input(tokens: 4, seed: 400)
                let copy = cache.copy() as! GLM5NextDSACache
                let expected = try dsa.forward(one, cache: copy.copy() as! GLM5NextDSACache,
                    validity: MLXArray.ones([1, 4], dtype: .bool))
                let shared = try dsa.forwardExchanging(one, cache: copy, validity: MLXArray.ones([1, 4], dtype: .bool),
                    share: 0, shares: 2)
                #expect(shared.partial == nil)
                #expect(Self.bitwise([try shared.resume(nil)], [expected]))
            }
        }
    }
}

extension KernelTests {
    /// Opt-in (`GLM53_SELECTION_SPLIT_STAGE=1`, GPU, the row-invariance
    /// suite's rank-shaped serving stage, one DSA layer of five): a segmented
    /// prefill with every DSA layer's selection shared between two ranks by
    /// rows (`installTensorParallelSelectionSplit`) is the plain segmented
    /// prefill bit for bit (hidden rows, logits, cache, in-graph snapshot),
    /// whole and in layer slices, with contiguous and paged histories and a
    /// segment that keeps other micro-batches' selections. Both ranks are
    /// played in one process: a rank's exchange adds the peer's partial from
    /// the peer's previous pass (none in the first), and the passes stop
    /// when neither rank's partials change.
    @Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["GLM53_SELECTION_SPLIT_STAGE"] == "1"))
    struct GLM5NextSelectionSplitStageTests {
        struct Outcome { let hidden: [MLXArray], logits: MLXArray, cache: [MLXArray], snapshot: [MLXArray] }

        static func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        static func same(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
            func bytes(_ x: MLXArray) -> Data { x.asData(access: .copy).data }
            return a.count == b.count
                && zip(a, b).allSatisfy { $0.shape == $1.shape && $0.dtype == $1.dtype && bytes($0) == bytes($1) }
        }

        static func run(
            _ stage: GLM5NextTextStage, tokens: MLXArray, segments: [GLM5NextPrefillSegment],
            account: GLM5NextDSAPageBook.Account?, sliced: Bool
        ) throws -> Outcome {
            let cache = stage.newCache(account: account)
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
                hidden: candidate.segmentHidden, logits: output.logits, cache: cache.layers.flatMap(\.state),
                snapshot: snapshot.layers.flatMap(\.state))
            try fence(outcome.hidden + [outcome.logits] + outcome.cache + outcome.snapshot)
            return outcome
        }

        @Test func splitPrefillIsThePlainPrefillBitForBit() throws {
            try Device.withDefaultDevice(.gpu) {
                let stage = try GLM5NextBatchInvarianceTests.makeStage(layers: 5)
                let vocabulary = stage.plan.configuration.vocabularySize
                let segments: [GLM5NextPrefillSegment] = [
                    .init(microBatchRows: [4096, 4096]), .init(microBatchRows: [2048, 2048]),
                    .init(microBatchRows: [8192], selectionRows: [4096, 4096]),
                ]
                let count = segments.reduce(0) { $0 + $1.tokens }
                let tokens = MLXArray((0 ..< count).map { Int32((($0 + 1) * 7919) % vocabulary) }, [1, count])
                for paged in [false, true] {
                    var account: GLM5NextDSAPageBook.Account?
                    if paged {
                        let pool = try stage.makeDSAPagePool(pageSize: 64)
                        account = try pool.reserve(pages: 8192)
                        stage.dsaPagePool = pool
                    }
                    defer { stage.dsaPagePool = nil }
                    stage.selectionSplit = nil
                    let reference = try Self.run(stage, tokens: tokens, segments: segments, account: account, sliced: false)
                    var sent: [[MLXArray]] = [[], []]
                    var unchanged = [false, false]
                    var outcomes: [Outcome?] = [nil, nil]
                    var passes = 0
                    for pass in 0 ..< 8 {
                        let share = pass % 2
                        let peer = sent[1 - share]
                        var calls: [MLXArray] = []
                        stage.selectionSplit = .init(share: share, shares: 2) { partial in
                            let index = calls.count
                            calls.append(partial)
                            return index < peer.count ? partial + peer[index] : partial
                        }
                        let outcome = try Self.run(
                            stage, tokens: tokens, segments: segments, account: account, sliced: pass % 4 >= 2)
                        try Self.fence(calls)
                        unchanged[share] = Self.same(sent[share], calls)
                        sent[share] = calls
                        outcomes[share] = outcome
                        passes = pass + 1
                        if unchanged[0] && unchanged[1] { break }
                    }
                    stage.selectionSplit = nil
                    print("GLM53_SELECTION_SPLIT_STAGE paged=\(paged ? 1 : 0) passes=\(passes) exchanges=\(sent[0].count)")
                    #expect(unchanged[0] && unchanged[1], "the passes did not settle")
                    #expect(sent[0].count > 0 && sent[0].count == sent[1].count)
                    for share in 0 ..< 2 {
                        let outcome = try #require(outcomes[share])
                        let label = "paged \(paged) rank \(share)"
                        #expect(Self.same(outcome.hidden, reference.hidden), "\(label) hidden rows")
                        #expect(Self.same([outcome.logits], [reference.logits]), "\(label) logits")
                        #expect(Self.same(outcome.cache, reference.cache), "\(label) cache")
                        #expect(Self.same(outcome.snapshot, reference.snapshot), "\(label) snapshot")
                    }
                }
            }
        }
    }
}
