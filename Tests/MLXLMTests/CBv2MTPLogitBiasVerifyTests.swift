import MLX
import XCTest

@testable import MLXLMCommon

/// Bias-aware target-prefix verification: every window position must be the
/// draw the ordinary pipeline (`LogitsPipelineV2.process` + `SamplerV2`)
/// makes from the same logits, with each row's `logitBias` added first.
final class CBv2MTPLogitBiasVerifyTests: XCTestCase {
    private let ids = (0 ..< 4).map { CBv2RequestID(UInt64(401 + $0 * 17)) }

    private func params(seed: UInt64?, ban: Int) -> [CBv2SamplingParams] {
        [
            // A ban on the greedy row's own argmax, and a boost.
            .init(temperature: 0, seed: seed, logitBias: [ban: -100, 5: 1.5]),
            .init(temperature: 1, topP: 0.95, topK: 20, seed: seed, logitBias: [ban: -100]),
            // A zero bias, and a bias on a token outside the vocabulary.
            .init(temperature: 0.7, topP: 0.9, seed: seed, logitBias: [3: 0, 1_000_000: 50]),
            // No bias beside biased rows.
            .init(temperature: 0.8, topK: 12, seed: seed),
        ]
    }

    private func histories(vocab: Int) -> [[Int]] {
        [[5, 9, 5, 17, 2, 2, 31], (0 ..< 40).map { ($0 * 7) % vocab }, [], [1, 2, 3]]
    }

    /// Ordinary sequential sampling: one `sample` per column.
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
        return (0 ..< ids.count).map { r in columns.map { $0[r] } }
    }

    private func logits(vocab: Int, width: Int, dtype: DType, salt: Int) -> MLXArray {
        let values: [Float] = (0 ..< (ids.count * width * vocab)).map {
            Float((($0 + salt) * 73 + 19) % 509 - 254) / 97
        }
        return MLXArray(values, [ids.count, width, vocab]).asType(dtype)
    }

    func testBiasedWindowEqualsSequentialSampling() throws {
        let vocab = 97
        let histories = histories(vocab: vocab)
        let verifier = CBv2DefaultSampler(fallbackSeed: 991)
        XCTAssertTrue(verifier.supportsMTPLogitBiasVerification)
        for dtype in [DType.float32, .bfloat16] {
            for seed in [UInt64(0), 7, 1234] {
                for width in [1, 2, 4] {
                    let x = logits(vocab: vocab, width: width, dtype: dtype, salt: Int(seed))
                    // Ban the greedy row's raw argmax at column 0, so the bias matters.
                    let raw = argMax(x[0 ..< 1, 0, 0...].asType(.float32), axis: -1)
                    eval(raw)
                    let ban = Int(raw.asArray(Int32.self)[0])
                    let p = params(seed: seed == 7 ? nil : seed, ban: ban)
                    let ordinary = ordinaryDraws(
                        logits: x, params: p, histories: histories, seed: 991)
                    XCTAssertNotEqual(ordinary[0][0], Int32(ban), "the ban must bite")
                    let bases = histories.map(\.count)

                    let verified = try XCTUnwrap(
                        verifier.mtpVerifySample(
                            logits: x, params: p, requestIDs: ids, stepBases: bases))
                    eval(verified)
                    for r in 0 ..< ids.count {
                        XCTAssertEqual(
                            verified[r].asArray(Int32.self), ordinary[r],
                            "dtype=\(dtype) seed=\(seed) width=\(width) row=\(r)")
                    }
                    // The penalty-aware entry point with no penalty rows is the same.
                    let viaPenalties = try XCTUnwrap(
                        verifier.mtpVerifySample(
                            logits: x, params: p, requestIDs: ids, stepBases: bases,
                            penalties: .init(
                                outputTokens: histories.map { ArraySlice($0) }, drafts: nil,
                                promptTokens: [])))
                    eval(viaPenalties)
                    XCTAssertEqual(viaPenalties.asArray(Int32.self), verified.asArray(Int32.self))

                    // Serial columns give the same draws.
                    for column in 0 ..< width {
                        let one = try XCTUnwrap(
                            verifier.mtpVerifySample(
                                logits: x[0..., column ..< (column + 1), 0...], params: p,
                                requestIDs: ids, stepBases: bases.map { $0 + column }))
                        eval(one)
                        XCTAssertEqual(
                            one.asArray(Int32.self), ordinary.map { $0[column] },
                            "serial column \(column) dtype=\(dtype) seed=\(seed)")
                    }

                    // A row's draws do not depend on its batchmates' biases.
                    for r in 0 ..< ids.count {
                        let solo = try XCTUnwrap(
                            verifier.mtpVerifySample(
                                logits: x[r ..< (r + 1), 0..., 0...], params: [p[r]],
                                requestIDs: [ids[r]], stepBases: [bases[r]]))
                        eval(solo)
                        XCTAssertEqual(solo[0].asArray(Int32.self), ordinary[r], "solo row \(r)")
                    }
                }
            }
        }
    }

    /// A zero bias changes no draw.
    func testZeroBiasIsTheIdentity() throws {
        let vocab = 61
        let x = logits(vocab: vocab, width: 3, dtype: .float32, salt: 5)
        let plain: [CBv2SamplingParams] = [
            .init(temperature: 0),
            .init(temperature: 1, seed: 3),
            .init(temperature: 0.7, topP: 0.9),
            .init(temperature: 0.5, topK: 8, seed: 9),
        ]
        let zero = plain.map { q -> CBv2SamplingParams in
            var q = q
            q.logitBias = [0: 0, 7: 0]
            return q
        }
        let sampler = CBv2DefaultSampler(fallbackSeed: 5)
        let bases = [0, 4, 9, 2]
        let a = try XCTUnwrap(
            sampler.mtpVerifySample(logits: x, params: plain, requestIDs: ids, stepBases: bases))
        let b = try XCTUnwrap(
            sampler.mtpVerifySample(logits: x, params: zero, requestIDs: ids, stepBases: bases))
        eval(a, b)
        XCTAssertEqual(a.asArray(Int32.self), b.asArray(Int32.self))
    }

    /// Bias rows accept exactly under typical acceptance; rows without a bias
    /// keep the typical rule.
    func testBiasRowsAcceptExactlyUnderTypical() throws {
        let vocab = 8
        let x = MLXArray.zeros([2, 3, vocab], type: Float.self)
        let rowIDs = [CBv2RequestID(1), CBv2RequestID(2)]
        let p: [CBv2SamplingParams] = [
            .init(temperature: 1, seed: 9, logitBias: [7: 0]),
            .init(temperature: 1, seed: 9),
        ]
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let tokens = try XCTUnwrap(
            sampler.mtpVerifySample(logits: x, params: p, requestIDs: rowIDs, stepBases: [0, 0]))
        eval(tokens)
        let t = tokens.asArray(Int32.self)
        let drafts = MLXArray([(t[0] + 1) % 8, t[1], (t[3] + 1) % 8, t[4]], [2, 2])
        let scored = try XCTUnwrap(
            sampler.mtpVerifyTypical(
                logits: x, draftIDs: drafts, delta: 0.3, params: p, requestIDs: rowIDs,
                stepBases: [0, 0]))
        eval(scored.tokens, scored.accept)
        let accept = scored.accept.asArray(Bool.self)
        XCTAssertEqual(scored.tokens.asArray(Int32.self), t, "typical keeps the same draws")
        XCTAssertFalse(accept[0], "the bias row rejects a draft that differs from its draw")
        XCTAssertTrue(accept[2], "the plain row keeps a likely draft under the typical rule")
    }

    func testOnlyTheDefaultSamplerVerifiesBias() {
        XCTAssertTrue(CBv2DefaultSampler().supportsMTPLogitBiasVerification)
        XCTAssertFalse(CBv2GreedySampler().supportsMTPLogitBiasVerification)
    }
}
