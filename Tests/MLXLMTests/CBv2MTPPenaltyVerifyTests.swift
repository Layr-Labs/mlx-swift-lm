import MLX
import XCTest

@testable import MLXLMCommon

/// Penalty-aware target-prefix verification: every window position must be
/// the draw the ordinary pipeline (`LogitsPipelineV2.process` + `SamplerV2`)
/// makes from the same logits once the preceding drafts were sampled, with
/// presence/frequency penalties over the confirmed output plus those drafts.
final class CBv2MTPPenaltyVerifyTests: XCTestCase {
    private let ids = (0 ..< 4).map { CBv2RequestID(UInt64(301 + $0 * 13)) }

    private func params(seed: UInt64?) -> [CBv2SamplingParams] {
        [
            // The Qwen 3.5/3.6 card defaults.
            .init(temperature: 1, topP: 0.95, topK: 20, presencePenalty: 1.5, seed: seed),
            .init(temperature: 0, frequencyPenalty: 0.3, presencePenalty: 1.5, seed: seed),
            .init(temperature: 0.7, topP: 0.9, frequencyPenalty: 0.5, seed: seed),
            .init(temperature: 0.8, topK: 12, seed: seed),
        ]
    }

    private func histories(vocab: Int) -> [[Int]] {
        [
            [5, 9, 5, 17, 2, 2, 31],
            (0 ..< 40).map { ($0 * 7) % vocab },
            [],
            [1, 2, 3],
        ]
    }

    /// Ordinary sequential sampling: one `sample` per column, committing
    /// each draw (the pipeline's own incremental penalty state).
    private func ordinaryDraws(
        logits: MLXArray, params: [CBv2SamplingParams], histories: [[Int]], seed: UInt64
    ) -> [[Int32]] {
        let sampler = CBv2DefaultSampler(fallbackSeed: seed)
        let rows = (0 ..< ids.count).map { r in
            CBv2SamplerRow(
                id: ids[r], params: params[r], promptTokens: [1, 2, 3],
                outputTokens: histories[r])
        }
        var columns: [[Int32]] = []
        for column in 0 ..< logits.dim(1) {
            let sampled = sampler.sample(
                logits: logits[0..., column, 0...], params: params, requestIDs: ids,
                stepIndex: column, pendingSampledTokens: nil, rowContext: { rows })
            eval(sampled)
            columns.append(sampled.asArray(Int32.self))
        }
        // [B][W]
        return (0 ..< ids.count).map { r in columns.map { $0[r] } }
    }

    private func logits(vocab: Int, width: Int, dtype: DType, salt: Int) -> MLXArray {
        let values: [Float] = (0 ..< (ids.count * width * vocab)).map {
            Float((($0 + salt) * 73 + 19) % 509 - 254) / 97
        }
        return MLXArray(values, [ids.count, width, vocab]).asType(dtype)
    }

    func testPenalizedWindowEqualsSequentialSamplingOnTheAcceptedPrefix() throws {
        let vocab = 97
        let histories = histories(vocab: vocab)
        for dtype in [DType.float32, .bfloat16] {
            for seed in [UInt64(0), 7, 1234] {
                for width in [1, 2, 4] {
                    let p = params(seed: seed == 7 ? nil : seed)
                    let x = logits(vocab: vocab, width: width, dtype: dtype, salt: Int(seed))
                    let ordinary = ordinaryDraws(
                        logits: x, params: p, histories: histories, seed: 991)
                    let verifier = CBv2DefaultSampler(fallbackSeed: 991)
                    XCTAssertTrue(verifier.supportsMTPPenaltyVerification)
                    let outputs = histories.map { ArraySlice($0) }
                    let bases = histories.map(\.count)

                    // Oracle drafts (the ordinary draws): every position matches.
                    let oracle = MLXArray(
                        ordinary.flatMap { Array($0.prefix(width - 1)) }, [ids.count, width - 1])
                    let verified = try XCTUnwrap(
                        verifier.mtpVerifySample(
                            logits: x, params: p, requestIDs: ids, stepBases: bases,
                            penalties: .init(
                                outputTokens: outputs, drafts: width > 1 ? oracle : nil)))
                    eval(verified)
                    for r in 0 ..< ids.count {
                        XCTAssertEqual(
                            verified[r].asArray(Int32.self), ordinary[r],
                            "dtype=\(dtype) seed=\(seed) width=\(width) row=\(r)")
                    }

                    // Serial columns (columnOffset) give the same draws.
                    for column in 0 ..< width {
                        let one = try XCTUnwrap(
                            verifier.mtpVerifySample(
                                logits: x[0..., column ..< (column + 1), 0...], params: p,
                                requestIDs: ids, stepBases: bases.map { $0 + column },
                                penalties: .init(
                                    outputTokens: outputs, drafts: width > 1 ? oracle : nil,
                                    columnOffset: column)))
                        eval(one)
                        XCTAssertEqual(
                            one.asArray(Int32.self), ordinary.map { $0[column] },
                            "serial column \(column) dtype=\(dtype) seed=\(seed)")
                    }

                    // Batch order/composition does not change a row's draws.
                    for r in (0 ..< ids.count).reversed() {
                        let solo = try XCTUnwrap(
                            verifier.mtpVerifySample(
                                logits: x[r ..< (r + 1), 0..., 0...], params: [p[r]],
                                requestIDs: [ids[r]], stepBases: [bases[r]],
                                penalties: .init(
                                    outputTokens: [outputs[r]],
                                    drafts: width > 1 ? oracle[r ..< (r + 1), 0...] : nil)))
                        eval(solo)
                        XCTAssertEqual(solo[0].asArray(Int32.self), ordinary[r])
                    }

                    // A wrong draft at m: positions 0 ... m still match.
                    guard width > 1 else { continue }
                    for m in 0 ..< (width - 1) {
                        let wrong = ordinary.map { row -> [Int32] in
                            var d = Array(row.prefix(width - 1))
                            d[m] = (d[m] + 1) % Int32(vocab)
                            return d
                        }
                        let scored = try XCTUnwrap(
                            verifier.mtpVerifySample(
                                logits: x, params: p, requestIDs: ids, stepBases: bases,
                                penalties: .init(
                                    outputTokens: outputs,
                                    drafts: MLXArray(wrong.flatMap { $0 }, [ids.count, width - 1])))
                        )
                        eval(scored)
                        for r in 0 ..< ids.count {
                            XCTAssertEqual(
                                Array(scored[r].asArray(Int32.self).prefix(m + 1)),
                                Array(ordinary[r].prefix(m + 1)), "wrong draft at \(m) row \(r)")
                        }
                    }
                }
            }
        }
    }

    /// Without penalties the penalty-aware entry point is the historical one.
    func testNilPenaltiesMatchTheHistoricalVerifier() throws {
        let vocab = 61
        let x = logits(vocab: vocab, width: 3, dtype: .float32, salt: 5)
        let p = params(seed: 3).map { q -> CBv2SamplingParams in
            var q = q
            q.presencePenalty = 0
            q.frequencyPenalty = 0
            return q
        }
        let sampler = CBv2DefaultSampler(fallbackSeed: 5)
        let bases = [0, 4, 9, 2]
        let a = try XCTUnwrap(
            sampler.mtpVerifySample(logits: x, params: p, requestIDs: ids, stepBases: bases))
        let b = try XCTUnwrap(
            sampler.mtpVerifySample(
                logits: x, params: p, requestIDs: ids, stepBases: bases, penalties: nil))
        eval(a, b)
        XCTAssertEqual(a.asArray(Int32.self), b.asArray(Int32.self))
    }

    /// Penalty rows accept exactly even when the engine is configured for
    /// typical acceptance; rows without a penalty keep the typical rule.
    func testPenaltyRowsAcceptExactlyUnderTypical() throws {
        let vocab = 8
        // Flat logits: typical's floor is low, so a non-target draft would be
        // kept by the typical rule.
        let x = MLXArray.zeros([2, 3, vocab], type: Float.self)
        let rowIDs = [CBv2RequestID(1), CBv2RequestID(2)]
        let p: [CBv2SamplingParams] = [
            .init(temperature: 1, presencePenalty: 1.5, seed: 9),
            .init(temperature: 1, seed: 9),
        ]
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let tokens = try XCTUnwrap(
            sampler.mtpVerifySample(
                logits: x, params: p, requestIDs: rowIDs, stepBases: [0, 0],
                penalties: .init(
                    outputTokens: [[], []], drafts: MLXArray([Int32]([0, 0, 0, 0]), [2, 2]))))
        eval(tokens)
        let t = tokens.asArray(Int32.self)
        // Drafts that differ from the target draw at position 0.
        let drafts = MLXArray([(t[0] + 1) % 8, t[1], (t[3] + 1) % 8, t[4]], [2, 2])
        let scored = try XCTUnwrap(
            sampler.mtpVerifyTypical(
                logits: x, draftIDs: drafts, delta: 0.3, params: p, requestIDs: rowIDs,
                stepBases: [0, 0],
                penalties: .init(outputTokens: [[], []], drafts: drafts)))
        eval(scored.tokens, scored.accept)
        let accept = scored.accept.asArray(Bool.self)
        XCTAssertFalse(accept[0], "penalty row: a non-target draft is rejected")
        XCTAssertTrue(accept[2], "plain row: typical keeps a likely draft")
    }

    /// Incremental confirmed counts equal a fresh rebuild, and a reused id
    /// with a different history is rebuilt (also after `requestDidFinish`).
    func testConfirmedCountsFoldIncrementallyAndRebuildOnDivergence() throws {
        let vocab = 53
        let id = [CBv2RequestID(77)]
        let p: [CBv2SamplingParams] = [
            .init(temperature: 0, frequencyPenalty: 2, presencePenalty: 1.5)
        ]
        let x = MLXArray((0 ..< vocab).map { Float($0 % 7) / 3 }, [1, 1, vocab])
        func score(_ sampler: CBv2DefaultSampler, _ history: [Int]) -> Int32 {
            let t = sampler.mtpVerifySample(
                logits: x, params: p, requestIDs: id, stepBases: [history.count],
                penalties: .init(outputTokens: [ArraySlice(history)], drafts: nil))!
            eval(t)
            return t.asArray(Int32.self)[0]
        }
        let long = (0 ..< 30).map { ($0 * 5 + 6) % vocab }
        let warm = CBv2DefaultSampler(fallbackSeed: 0)
        for n in [0, 1, 5, 6, 18, 30] {
            let fresh = CBv2DefaultSampler(fallbackSeed: 0)
            XCTAssertEqual(score(warm, Array(long.prefix(n))), score(fresh, Array(long.prefix(n))))
        }
        // Same id, different (shorter / diverging) history.
        let other = [6, 6, 6]
        XCTAssertEqual(score(warm, other), score(CBv2DefaultSampler(fallbackSeed: 0), other))
        warm.requestDidFinish(id[0])
        XCTAssertEqual(score(warm, long), score(CBv2DefaultSampler(fallbackSeed: 0), long))
    }

    /// Distribution: many independent rows (distinct request ids) drawing at
    /// one penalized position follow softmax((x - α·c - β·[c > 0]) / T).
    func testPenalizedDrawFollowsThePenalizedDistribution() throws {
        let vocab = 5
        let n = 6000
        let base: [Float] = [1.2, 0.9, 0.4, 0.0, -0.3]
        let history = [0, 0, 2]
        let draft: Int32 = 1
        let temperature: Float = 0.9
        let presence: Float = 1.5
        let frequency: Float = 0.25
        let rowIDs = (0 ..< n).map { CBv2RequestID(UInt64(10_000 + $0)) }
        let p = [CBv2SamplingParams](
            repeating: .init(
                temperature: temperature, frequencyPenalty: frequency,
                presencePenalty: presence),
            count: n)
        let x = broadcast(MLXArray(base, [1, 1, vocab]), to: [n, 2, vocab])
        let sampler = CBv2DefaultSampler(fallbackSeed: 2024)
        let tokens = try XCTUnwrap(
            sampler.mtpVerifySample(
                logits: x, params: p, requestIDs: rowIDs,
                stepBases: [Int](repeating: history.count, count: n),
                penalties: .init(
                    outputTokens: [ArraySlice<Int>](repeating: ArraySlice(history), count: n),
                    drafts: MLXArray([Int32](repeating: draft, count: n), [n, 1]))))
        eval(tokens)
        let position1 = tokens[0..., 1].asArray(Int32.self)
        var counts = [Int](repeating: 0, count: vocab)
        for t in position1 { counts[Int(t)] += 1 }
        // Expected: counts = history + the draft.
        var c = [Float](repeating: 0, count: vocab)
        for t in history { c[t] += 1 }
        c[Int(draft)] += 1
        let z = (0 ..< vocab).map {
            (base[$0] - frequency * c[$0] - (c[$0] > 0 ? presence : 0)) / temperature
        }
        let m = z.max()!
        let e = z.map { Foundation.exp(Double($0 - m)) }
        let total = e.reduce(0, +)
        var chi2 = 0.0
        for v in 0 ..< vocab {
            let expected = Double(n) * e[v] / total
            chi2 += pow(Double(counts[v]) - expected, 2) / expected
        }
        // dof 4: P(chi2 > 18.47) = 0.001.
        XCTAssertLessThan(chi2, 18.47, "observed \(counts), chi2 \(chi2)")
    }
}
