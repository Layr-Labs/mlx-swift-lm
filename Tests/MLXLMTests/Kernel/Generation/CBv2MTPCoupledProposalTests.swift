// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// `CBv2DefaultSampler.mtpCoupledProposal`: a draft row raced with the
    /// target's temperature and the keyed noise of the output index it
    /// proposes for, against the draw `mtpVerifySample` makes there.
    /// Synthetic rows; a sampler oracle, not a claim about model logits.
    @Suite
    struct CBv2MTPCoupledProposalTests {
        static let vocabulary = 257
        static let bases = [0, 1, 37, 4097]
        static let ids = (0 ..< 4).map { CBv2RequestID(UInt64(41 + $0 * 17)) }

        /// Four rows `[4, 1, V]`, flat enough that the draws vary with the key.
        static func rows(_ salt: Int) -> MLXArray {
            let values: [Float] = (0 ..< (4 * vocabulary)).map {
                Float(($0 * 73 + 19 + salt * 31) % 509 - 254) / 47
            }
            return MLXArray(values, [4, 1, vocabulary])
        }

        static func ints(_ a: MLXArray) -> [Int32] {
            a.reshaped([-1]).asType(.int32).asArray(Int32.self)
        }

        /// Unfiltered rows: the proposal from the target's own row is the
        /// target's draw at that index, in any dtype, batched or alone.
        @Test(arguments: [DType.float32, .float16, .bfloat16])
        func unfilteredProposalsAreTheTargetsDraws(dtype: DType) throws {
            try Device.withDefaultDevice(.cpu) {
                for seed in [UInt64(0), 7, UInt64.max - 1] {
                    let params: [CBv2SamplingParams] = [
                        .init(temperature: 0, seed: seed),
                        .init(temperature: 0.7, seed: seed),
                        .init(temperature: 1, seed: nil),
                        .init(temperature: 1.3, seed: seed),
                    ]
                    for salt in 0 ..< 4 {
                        let logits = Self.rows(salt).asType(dtype)
                        let sampler = CBv2DefaultSampler(fallbackSeed: 991)
                        let proposed = sampler.mtpCoupledProposal(
                            logits: logits, params: params, requestIDs: Self.ids,
                            stepBases: Self.bases)
                        let drawn = try #require(
                            sampler.mtpVerifySample(
                                logits: logits, params: params, requestIDs: Self.ids,
                                stepBases: Self.bases))
                        eval(proposed, drawn)
                        #expect(proposed.shape == [4, 1] && proposed.dtype == .int32)
                        #expect(
                            Self.ints(proposed) == Self.ints(drawn),
                            "dtype \(dtype) seed \(seed) salt \(salt)")
                        for row in 0 ..< 4 {
                            let alone = sampler.mtpCoupledProposal(
                                logits: logits[row ..< (row + 1)], params: [params[row]],
                                requestIDs: [Self.ids[row]], stepBases: [Self.bases[row]])
                            #expect(Self.ints(alone) == [Self.ints(proposed)[row]])
                        }
                    }
                }
            }
        }

        /// Filtered rows: the draw, or the row's mode where the race's winner
        /// is outside the top-k / top-p / min-p set; top-k 1 is the mode.
        @Test func filteredProposalsAreTheDrawOrTheMode() throws {
            try Device.withDefaultDevice(.cpu) {
                let params: [CBv2SamplingParams] = [
                    .init(temperature: 0.7, topP: 0.9, topK: 12, minP: 0.05, seed: 5),
                    .init(temperature: 1, topP: 0.6, seed: 5),
                    .init(temperature: 1.3, topP: 0.6, topK: 31, minP: 0.1, seed: 5),
                    .init(temperature: 1, topK: 1, seed: 5),
                ]
                var agreed = 0
                var total = 0
                for salt in 0 ..< 16 {
                    let logits = Self.rows(salt)
                    let sampler = CBv2DefaultSampler(fallbackSeed: 991)
                    let proposed = Self.ints(
                        sampler.mtpCoupledProposal(
                            logits: logits, params: params, requestIDs: Self.ids,
                            stepBases: Self.bases))
                    let drawn = Self.ints(
                        try #require(
                            sampler.mtpVerifySample(
                                logits: logits, params: params, requestIDs: Self.ids,
                                stepBases: Self.bases)))
                    let modes = Self.ints(argMax(logits.reshaped([4, Self.vocabulary]), axis: -1))
                    for row in 0 ..< 4 {
                        #expect(
                            proposed[row] == drawn[row] || proposed[row] == modes[row],
                            "salt \(salt) row \(row)")
                        agreed += proposed[row] == drawn[row] ? 1 : 0
                        total += 1
                    }
                    #expect(proposed[3] == modes[3] && drawn[3] == modes[3])
                }
                // The fallback is the exception even with filters this tight
                // (top_p 0.6 drops ~40 % of a flat row's mass).
                #expect(agreed * 2 > total, "\(agreed) of \(total)")
            }
        }

        /// The noise is the index's: other step bases give other proposals.
        @Test func proposalsAreKeyedOnTheirOutputIndex() {
            Device.withDefaultDevice(.cpu) {
                let params = Array(repeating: CBv2SamplingParams(temperature: 1, seed: 3), count: 4)
                let sampler = CBv2DefaultSampler(fallbackSeed: 991)
                let logits = Self.rows(0)
                let here = Self.ints(
                    sampler.mtpCoupledProposal(
                        logits: logits, params: params, requestIDs: Self.ids, stepBases: Self.bases)
                )
                let next = Self.ints(
                    sampler.mtpCoupledProposal(
                        logits: logits, params: params, requestIDs: Self.ids,
                        stepBases: Self.bases.map { $0 + 1 }))
                let again = Self.ints(
                    sampler.mtpCoupledProposal(
                        logits: logits, params: params, requestIDs: Self.ids, stepBases: Self.bases)
                )
                #expect(here == again && here != next)
            }
        }
    }
}
