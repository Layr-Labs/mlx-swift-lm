import MLX
import XCTest

@testable import MLXLMCommon

/// The typical-acceptance mask must follow the CPU rule on the same filtered
/// rows the verify draw samples from, keep greedy rows exact, and stay
/// batch-invariant. This is a synthetic sampler oracle, not a claim about
/// whole-model logits.
final class CBv2MTPTypicalAcceptanceTests: XCTestCase {
    private let ids = (0 ..< 3).map { CBv2RequestID(UInt64(500 + $0)) }
    private let bases = [0, 11, 4097]
    private let delta: Float = 0.2

    /// CPU oracle: temperature divide, optional top-k keep, softmax, Shannon
    /// entropy in nats, `min(1, delta * exp(-H))`, strict `>`.
    private func oracle(
        logits: [Float], temperature: Float, topK: Int, draft: Int, delta: Float
    ) -> (p: Double, floor: Double, accept: Bool) {
        var scaled = logits.map { Double($0) / Double(temperature) }
        if topK > 0, topK < scaled.count {
            let cutoff = scaled.sorted(by: >)[topK - 1]
            scaled = scaled.map { $0 >= cutoff ? $0 : -Double.infinity }
        }
        let m = scaled.max()!
        let e = scaled.map { $0 == -Double.infinity ? 0 : exp($0 - m) }
        let z = e.reduce(0, +)
        let p = e.map { $0 / z }
        var h = 0.0
        for v in p where v > 0 { h -= v * log(v) }
        let floor = min(1.0, Double(delta) * exp(-h))
        return (p[draft], floor, p[draft] > floor)
    }

    /// Three rows, window width 3 (two drafts plus the bonus column), vocab 4.
    private var rowLogits: [[[Float]]] {
        [
            // Row 0 (stochastic, T=1): position 0 splits mass over two tokens
            // (H = ln 2, floor = delta / 2); the draft carries p = 0.5.
            // Position 1 is a point mass on token 0 (floor = delta); the
            // draft is token 1 with p ≈ 2e-9.
            [[10, 10, -30, -30], [20, 0, 0, 0], [1, 2, 3, 4]],
            // Row 1 (greedy): exact comparison regardless of mass.
            [[1, 5, 2, 0], [0, 0, 9, 0], [3, 3, 3, 3]],
            // Row 2 (stochastic, T=1): three-way split at position 0 with the
            // draft on one of the three (p = 1/3 > delta / 3); position 1 is
            // confident with the draft on the argmax.
            [[2, 2, 2, -30], [3, 0, 0, 0], [0, 1, 0, 0]],
        ]
    }
    private let drafts: [[Int32]] = [[0, 1], [2, 2], [0, 0]]
    private var params: [CBv2SamplingParams] {
        [
            .init(temperature: 1, topP: 1, topK: 0, seed: 9),
            .init(temperature: 0, seed: 9),
            .init(temperature: 1, topP: 1, topK: 0, seed: 9),
        ]
    }

    private func logitsArray(_ rows: [[[Float]]], dtype: DType = .float32) -> MLXArray {
        let b = rows.count
        let w = rows[0].count
        let v = rows[0][0].count
        return MLXArray(rows.flatMap { $0.flatMap { $0 } }, [b, w, v]).asType(dtype)
    }

    private func draftArray(_ rows: [[Int32]]) -> MLXArray {
        MLXArray(rows.flatMap { $0 }, [rows.count, rows[0].count])
    }

    func testMaskFollowsTheEntropyFloorAndGreedyRowsStayExact() throws {
        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
        let logits = logitsArray(rowLogits)
        let draftIDs = draftArray(drafts)
        let scored = try XCTUnwrap(sampler.mtpVerifyTypical(
            logits: logits, draftIDs: draftIDs, delta: delta,
            params: params, requestIDs: ids, stepBases: bases))
        let sampled = try XCTUnwrap(sampler.mtpVerifySample(
            logits: logits, params: params, requestIDs: ids, stepBases: bases))
        eval(scored.tokens, scored.accept, sampled)

        // The keyed draw is unchanged: typical mode commits the same target
        // token at the first rejection and bonus position as exact mode.
        XCTAssertEqual(scored.tokens.asArray(Int32.self), sampled.asArray(Int32.self))
        XCTAssertEqual(scored.accept.shape, [3, 2])
        let mask = scored.accept.asArray(Bool.self)
        let tokens = scored.tokens.asArray(Int32.self)

        for row in 0 ..< 3 {
            for position in 0 ..< 2 {
                let draft = Int(drafts[row][position])
                let expected: Bool
                if params[row].temperature < LogitsPipelineV2.greedyEpsilon {
                    expected = tokens[row * 3 + position] == Int32(draft)
                } else {
                    expected = oracle(
                        logits: rowLogits[row][position], temperature: params[row].temperature,
                        topK: params[row].topK, draft: draft, delta: delta
                    ).accept
                }
                XCTAssertEqual(mask[row * 2 + position], expected, "row=\(row) position=\(position)")
            }
        }
        // Pin the hand-computed expectations too, so a broken oracle cannot
        // agree with a broken mask.
        XCTAssertEqual(mask, [true, false, false, true, true, true])
    }

    func testFilteredRowsUseTheSamplerTransformBeforeTheFloor() throws {
        // Top-k 2 at T=0.5 keeps two tokens: the draft on the runner-up has
        // p ≈ 0.12 in the filtered row against a floor of ≈ 0.14 (reject),
        // while the same draft under T=1 without top-k passes (p ≈ 0.19 vs
        // floor ≈ 0.06). The mask must follow the filtered row.
        let logits: [[[Float]]] = [[[3, 2, 1, 0], [0, 0, 0, 9]]]
        let draftIDs = draftArray([[1]])
        let sampler = CBv2DefaultSampler(fallbackSeed: 5)
        for (temperature, topK) in [(Float(0.5), 2), (Float(1), 0)] {
            let params = [CBv2SamplingParams(temperature: temperature, topP: 1, topK: topK, seed: 3)]
            let scored = try XCTUnwrap(sampler.mtpVerifyTypical(
                logits: logitsArray(logits), draftIDs: draftIDs, delta: delta,
                params: params, requestIDs: [ids[0]], stepBases: [0]))
            eval(scored.accept)
            let expected = oracle(
                logits: logits[0][0], temperature: temperature, topK: topK, draft: 1, delta: delta)
            XCTAssertEqual(
                scored.accept.asArray(Bool.self), [expected.accept],
                "temperature=\(temperature) topK=\(topK) p=\(expected.p) floor=\(expected.floor)")
        }
    }

    func testAllGreedyBatchKeepsTheArgmaxComparison() throws {
        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
        let logits = logitsArray(rowLogits)
        let greedy = [CBv2SamplingParams](repeating: .init(temperature: 0), count: 3)
        let scored = try XCTUnwrap(sampler.mtpVerifyTypical(
            logits: logits, draftIDs: draftArray(drafts), delta: delta,
            params: greedy, requestIDs: ids, stepBases: bases))
        let argmax = argMax(logits, axis: -1).asType(.int32)
        eval(scored.tokens, scored.accept, argmax)
        XCTAssertEqual(scored.tokens.asArray(Int32.self), argmax.asArray(Int32.self))
        let tokens = scored.tokens.asArray(Int32.self)
        let expected = (0 ..< 3).flatMap { row in
            (0 ..< 2).map { tokens[row * 3 + $0] == drafts[row][$0] }
        }
        XCTAssertEqual(scored.accept.asArray(Bool.self), expected)
    }

    func testMaskIsBatchInvariantAcrossDTypes() throws {
        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
        let reference = try XCTUnwrap(sampler.mtpVerifyTypical(
            logits: logitsArray(rowLogits), draftIDs: draftArray(drafts), delta: delta,
            params: params, requestIDs: ids, stepBases: bases))
        eval(reference.accept)
        let expected = reference.accept.asArray(Bool.self)
        for dtype in [DType.float32, .float16, .bfloat16] {
            let logits = logitsArray(rowLogits, dtype: dtype)
            let batched = try XCTUnwrap(sampler.mtpVerifyTypical(
                logits: logits, draftIDs: draftArray(drafts), delta: delta,
                params: params, requestIDs: ids, stepBases: bases))
            eval(batched.accept)
            XCTAssertEqual(batched.accept.asArray(Bool.self), expected, "dtype=\(dtype)")
            for row in (0 ..< 3).reversed() {
                let solo = try XCTUnwrap(sampler.mtpVerifyTypical(
                    logits: logits[row ..< (row + 1), 0..., 0...],
                    draftIDs: draftArray([drafts[row]]), delta: delta,
                    params: [params[row]], requestIDs: [ids[row]], stepBases: [bases[row]]))
                eval(solo.accept)
                XCTAssertEqual(
                    solo.accept.asArray(Bool.self), Array(expected[row * 2 ..< row * 2 + 2]),
                    "dtype=\(dtype) row=\(row)")
            }
        }
    }

    func testSingleDraftColumnMatchesTheFirstWindowPosition() throws {
        // The serial verify path decides one column at a time: `[B, 1]`
        // drafts against a `[B, 1]` window must equal column 0 of the
        // rectangular decision.
        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
        let full = try XCTUnwrap(sampler.mtpVerifyTypical(
            logits: logitsArray(rowLogits), draftIDs: draftArray(drafts), delta: delta,
            params: params, requestIDs: ids, stepBases: bases))
        let column = try XCTUnwrap(sampler.mtpVerifyTypical(
            logits: logitsArray(rowLogits)[0..., 0 ..< 1, 0...],
            draftIDs: draftArray(drafts.map { [$0[0]] }), delta: delta,
            params: params, requestIDs: ids, stepBases: bases))
        eval(full.accept, column.accept, full.tokens, column.tokens)
        XCTAssertEqual(column.accept.shape, [3, 1])
        XCTAssertEqual(
            column.accept.asArray(Bool.self),
            full.accept[0..., 0 ..< 1].asArray(Bool.self))
        XCTAssertEqual(
            column.tokens.asArray(Int32.self),
            full.tokens[0..., 0 ..< 1].asArray(Int32.self))
    }

    func testLargerDeltaNeverAcceptsMore() throws {
        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
        var previous: [Bool]?
        for delta in [Float(0.05), 0.2, 0.5, 1.0, 2.0] {
            let scored = try XCTUnwrap(sampler.mtpVerifyTypical(
                logits: logitsArray(rowLogits), draftIDs: draftArray(drafts), delta: delta,
                params: params, requestIDs: ids, stepBases: bases))
            eval(scored.accept)
            let mask = scored.accept.asArray(Bool.self)
            if let previous {
                for (index, kept) in mask.enumerated() where kept {
                    XCTAssertTrue(previous[index], "delta=\(delta) accepted a draft a smaller delta rejected")
                }
            }
            previous = mask
        }
    }
}
