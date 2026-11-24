// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Repetition-penalty-aware target-prefix verification: every window
    /// position must be the draw the ordinary pipeline
    /// (`LogitsPipelineV2.process` + `SamplerV2`) makes from the same logits
    /// once the preceding drafts were sampled, with the repetition window over
    /// prompt + confirmed output + those drafts (and presence/frequency
    /// penalties on top where a row has them). Synthetic rows; a sampler
    /// oracle, not a claim about model logits.
    @Suite
    struct CBv2MTPRepetitionPenaltyVerifyTests {
        static let ids = (0 ..< 5).map { CBv2RequestID(UInt64(501 + $0 * 11)) }
        static let prompt = [4, 8, 15, 16, 23, 42]

        static func params(seed: UInt64?) -> [CBv2SamplingParams] {
            [
                // Greedy, window longer than the history.
                .init(temperature: 0, repetitionPenalty: 1.3, seed: seed),
                // Seeded T > 0 with filters and presence on top; a window of 3
                // so drafts inside one round evict earlier tokens.
                .init(
                    temperature: 1, topP: 0.95, topK: 20, repetitionPenalty: 1.5,
                    repetitionContextSize: 3, presencePenalty: 1.5, seed: seed),
                // Greedy with frequency and presence, window of 5.
                .init(
                    temperature: 0, repetitionPenalty: 2, repetitionContextSize: 5,
                    frequencyPenalty: 0.3, presencePenalty: 0.4, seed: seed),
                // A penalty below one (rewards repeats), window of 1.
                .init(
                    temperature: 0.7, minP: 0.05, repetitionPenalty: 0.8,
                    repetitionContextSize: 1, seed: seed),
                // A plain row riding the same call.
                .init(temperature: 0.8, topK: 12, seed: seed),
            ]
        }

        static func histories(vocab: Int) -> [[Int]] {
            [
                [5, 9, 5, 17, 2, 2, 31],
                (0 ..< 40).map { ($0 * 7) % vocab },
                [],
                [1, 2, 3],
                [6, 6],
            ]
        }

        /// Mixed-sign logits, so both the divide (positive) and the multiply
        /// (negative) branch of the penalty are exercised.
        static func logits(vocab: Int, width: Int, dtype: DType, salt: Int) -> MLXArray {
            let values: [Float] = (0 ..< (ids.count * width * vocab)).map {
                Float((($0 + salt) * 73 + 19) % 509 - 254) / 97
            }
            return MLXArray(values, [ids.count, width, vocab]).asType(dtype)
        }

        static func ints(_ a: MLXArray) -> [Int32] {
            a.reshaped([-1]).asType(.int32).asArray(Int32.self)
        }

        /// Ordinary sequential sampling: one `sample` per column, committing
        /// each draw (the pipeline's own ring buffer and counts). `[B][W]`.
        static func ordinaryDraws(
            logits: MLXArray, params: [CBv2SamplingParams], histories: [[Int]], seed: UInt64
        ) -> [[Int32]] {
            let sampler = CBv2DefaultSampler(fallbackSeed: seed)
            let rows = (0 ..< ids.count).map { r in
                CBv2SamplerRow(
                    id: ids[r], params: params[r], promptTokens: prompt,
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

        static func penalties(
            _ histories: [[Int]], drafts: MLXArray?, columnOffset: Int = 0
        ) -> CBv2MTPVerifyPenalties {
            .init(
                outputTokens: histories.map { ArraySlice($0) }, drafts: drafts,
                columnOffset: columnOffset,
                promptTokens: histories.map { _ in ArraySlice(prompt) })
        }

        /// Oracle drafts verify to the ordinary stream; serial columns,
        /// batch composition and a wrong draft at any position keep every
        /// position up to the first mismatch equal to it. T 0 and seeded
        /// T > 0, float32 and bfloat16, windows shorter and longer than
        /// the round.
        @Test(arguments: [DType.float32, .bfloat16])
        func repetitionWindowEqualsSequentialSamplingOnTheAcceptedPrefix(dtype: DType) throws {
            let vocab = 97
            let histories = Self.histories(vocab: vocab)
            for seed in [UInt64(0), 7, 1234] {
                for width in [1, 2, 4, 6] {
                    let p = Self.params(seed: seed == 7 ? nil : seed)
                    let x = Self.logits(vocab: vocab, width: width, dtype: dtype, salt: Int(seed))
                    let ordinary = Self.ordinaryDraws(
                        logits: x, params: p, histories: histories, seed: 991)
                    let verifier = CBv2DefaultSampler(fallbackSeed: 991)
                    #expect(verifier.supportsMTPRepetitionPenaltyVerification)
                    let bases = histories.map(\.count)
                    let oracle = MLXArray(
                        ordinary.flatMap { Array($0.prefix(width - 1)) },
                        [Self.ids.count, width - 1])
                    let drafts = width > 1 ? oracle : nil
                    let verified = try #require(
                        verifier.mtpVerifySample(
                            logits: x, params: p, requestIDs: Self.ids, stepBases: bases,
                            penalties: Self.penalties(histories, drafts: drafts)))
                    eval(verified)
                    for r in 0 ..< Self.ids.count {
                        #expect(
                            verified[r].asArray(Int32.self) == ordinary[r],
                            "dtype=\(dtype) seed=\(seed) width=\(width) row=\(r)")
                    }

                    for column in 0 ..< width {
                        let one = try #require(
                            verifier.mtpVerifySample(
                                logits: x[0..., column ..< (column + 1), 0...], params: p,
                                requestIDs: Self.ids, stepBases: bases.map { $0 + column },
                                penalties: Self.penalties(
                                    histories, drafts: drafts, columnOffset: column)))
                        eval(one)
                        #expect(
                            one.asArray(Int32.self) == ordinary.map { $0[column] },
                            "serial column \(column) dtype=\(dtype) seed=\(seed)")
                    }

                    for r in (0 ..< Self.ids.count).reversed() {
                        let solo = try #require(
                            verifier.mtpVerifySample(
                                logits: x[r ..< (r + 1), 0..., 0...], params: [p[r]],
                                requestIDs: [Self.ids[r]], stepBases: [bases[r]],
                                penalties: .init(
                                    outputTokens: [ArraySlice(histories[r])],
                                    drafts: drafts.map { $0[r ..< (r + 1), 0...] },
                                    promptTokens: [ArraySlice(Self.prompt)])))
                        eval(solo)
                        #expect(solo[0].asArray(Int32.self) == ordinary[r], "solo row \(r)")
                    }

                    guard width > 1 else { continue }
                    for m in 0 ..< (width - 1) {
                        let wrong = ordinary.map { row -> [Int32] in
                            var d = Array(row.prefix(width - 1))
                            d[m] = (d[m] + 1) % Int32(vocab)
                            return d
                        }
                        let scored = try #require(
                            verifier.mtpVerifySample(
                                logits: x, params: p, requestIDs: Self.ids, stepBases: bases,
                                penalties: Self.penalties(
                                    histories,
                                    drafts: MLXArray(
                                        wrong.flatMap { $0 }, [Self.ids.count, width - 1]))))
                        eval(scored)
                        for r in 0 ..< Self.ids.count {
                            #expect(
                                Array(scored[r].asArray(Int32.self).prefix(m + 1))
                                    == Array(ordinary[r].prefix(m + 1)),
                                "wrong draft at \(m) row \(r)")
                        }
                    }
                }
            }
        }

        /// After a rejected draft the next round is scored from the confirmed
        /// output only (the target's token at the rejection replaces the
        /// draft): a verifier that scored the rejected round first gives the
        /// draws of a fresh sampler, and of ordinary decode, on that history.
        @Test func rejectedDraftsDoNotCarryIntoTheNextRound() throws {
            let vocab = 61
            let histories = Self.histories(vocab: vocab)
            let p = Self.params(seed: 31)
            let first = Self.logits(vocab: vocab, width: 3, dtype: .float32, salt: 3)
            let second = Self.logits(vocab: vocab, width: 3, dtype: .float32, salt: 11)
            let ordinary = Self.ordinaryDraws(
                logits: first, params: p, histories: histories, seed: 5)
            // Reject at position 0: every row's first draft is wrong.
            let wrong = MLXArray(
                ordinary.flatMap { [($0[0] + 1) % Int32(vocab), $0[1]] }, [Self.ids.count, 2])
            let warm = CBv2DefaultSampler(fallbackSeed: 5)
            let round1 = try #require(
                warm.mtpVerifySample(
                    logits: first, params: p, requestIDs: Self.ids,
                    stepBases: histories.map(\.count),
                    penalties: Self.penalties(histories, drafts: wrong)))
            eval(round1)
            let committed = (0 ..< Self.ids.count).map { r in
                histories[r] + [Int(round1[r, 0].item(Int32.self))]
            }
            for r in 0 ..< Self.ids.count {
                #expect(committed[r].last == Int(ordinary[r][0]), "row \(r) commits the target")
            }
            let next = Self.ordinaryDraws(logits: second, params: p, histories: committed, seed: 5)
            let nextDrafts = MLXArray(
                next.flatMap { Array($0.prefix(2)) }, [Self.ids.count, 2])
            let bases = committed.map(\.count)
            let fromWarm = try #require(
                warm.mtpVerifySample(
                    logits: second, params: p, requestIDs: Self.ids, stepBases: bases,
                    penalties: Self.penalties(committed, drafts: nextDrafts)))
            let fromFresh = try #require(
                CBv2DefaultSampler(fallbackSeed: 5).mtpVerifySample(
                    logits: second, params: p, requestIDs: Self.ids, stepBases: bases,
                    penalties: Self.penalties(committed, drafts: nextDrafts)))
            eval(fromWarm, fromFresh)
            #expect(Self.ints(fromWarm) == Self.ints(fromFresh))
            #expect(Self.ints(fromWarm) == next.flatMap { $0 })
        }

        /// The penalty's sign handling, on one greedy position: a repeated
        /// token with a positive logit is divided, one with a negative logit
        /// multiplied, so in both cases the next token wins.
        @Test func positiveLogitsAreDividedAndNegativeLogitsMultiplied() throws {
            let id = [CBv2RequestID(9)]
            let p: [CBv2SamplingParams] = [.init(temperature: 0, repetitionPenalty: 1.5)]
            let sampler = CBv2DefaultSampler(fallbackSeed: 0)
            for row in [[Float(1.2), 1.0, 0.1, -2], [-0.5, -0.6, -2, -3]] {
                let x = MLXArray(row, [1, 1, 4])
                let scored = try #require(
                    sampler.mtpVerifySample(
                        logits: x, params: p, requestIDs: id, stepBases: [1],
                        penalties: .init(
                            outputTokens: [[0]], drafts: nil, promptTokens: [[2, 3]])))
                eval(scored)
                #expect(Self.ints(scored) == [1], "\(row)")
                let ordinary = Self.ordinaryOne(x: x, params: p[0], prompt: [2, 3], output: [0])
                #expect(ordinary == 1, "\(row)")
            }
        }

        static func ordinaryOne(
            x: MLXArray, params: CBv2SamplingParams, prompt: [Int], output: [Int]
        ) -> Int32 {
            let sampler = CBv2DefaultSampler(fallbackSeed: 0)
            let row = CBv2SamplerRow(
                id: CBv2RequestID(9), params: params, promptTokens: prompt, outputTokens: output)
            let t = sampler.sample(
                logits: x[0..., 0, 0...], params: [params], requestIDs: [row.id], stepIndex: 0,
                pendingSampledTokens: nil, rowContext: { [row] })
            eval(t)
            return t.asArray(Int32.self)[0]
        }

        /// Repetition-penalty rows accept exactly under typical acceptance;
        /// rows without a penalty keep the typical rule.
        @Test func repetitionRowsAcceptExactlyUnderTypical() throws {
            let vocab = 8
            let x = MLXArray.zeros([2, 3, vocab], type: Float.self)
            let rowIDs = [CBv2RequestID(1), CBv2RequestID(2)]
            let p: [CBv2SamplingParams] = [
                .init(temperature: 1, repetitionPenalty: 1.3, seed: 9),
                .init(temperature: 1, seed: 9),
            ]
            let sampler = CBv2DefaultSampler(fallbackSeed: 1)
            let empty = CBv2MTPVerifyPenalties(
                outputTokens: [[], []], drafts: MLXArray([Int32]([0, 0, 0, 0]), [2, 2]),
                promptTokens: [[5], [5]])
            let tokens = try #require(
                sampler.mtpVerifySample(
                    logits: x, params: p, requestIDs: rowIDs, stepBases: [0, 0],
                    penalties: empty))
            eval(tokens)
            let t = Self.ints(tokens)
            let drafts = MLXArray([(t[0] + 1) % 8, t[1], (t[3] + 1) % 8, t[4]], [2, 2])
            let scored = try #require(
                sampler.mtpVerifyTypical(
                    logits: x, draftIDs: drafts, delta: 0.3, params: p, requestIDs: rowIDs,
                    stepBases: [0, 0],
                    penalties: .init(
                        outputTokens: [[], []], drafts: drafts, promptTokens: [[5], [5]])))
            eval(scored.tokens, scored.accept)
            let accept = scored.accept.asArray(Bool.self)
            #expect(!accept[0], "repetition row: a non-target draft is rejected")
            #expect(accept[2], "plain row: typical keeps a likely draft")
        }

        /// Which rows the verifier treats as repetition rows: exactly those the
        /// ordinary pipeline penalizes.
        @Test func repetitionRowsAreThePipelinesRepetitionRows() {
            #expect(!CBv2MTPVerifyPenalties.appliesRepetition(.init()))
            #expect(CBv2MTPVerifyPenalties.appliesRepetition(.init(repetitionPenalty: 1.1)))
            #expect(
                !CBv2MTPVerifyPenalties.appliesRepetition(
                    .init(repetitionPenalty: 1.1, repetitionContextSize: 0)))
            #expect(CBv2MTPVerifyPenalties.appliesAny(.init(presencePenalty: 1)))
            #expect(!CBv2MTPVerifyPenalties.applies(.init(repetitionPenalty: 1.1)))
            #expect(!CBv2GreedySampler().supportsMTPRepetitionPenaltyVerification)
        }
    }
}
