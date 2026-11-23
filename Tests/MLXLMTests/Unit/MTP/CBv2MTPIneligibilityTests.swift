import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// The whole-request MTP gates a row fails (`CBv2MTPIneligibility`): one
    /// bit per gate, every failing gate recorded, and the timing field that
    /// carries them.
    struct CBv2MTPIneligibilityTests {
        typealias Gate = CBv2MTPIneligibility

        static let gates: [Gate] = [
            .media, .reasoningBudget, .grammar, .sampling, .logprobs, .logitBias,
            .repetitionPenalty, .penalties, .stopStrings, .multimodalHistory,
        ]

        /// Every combination of the ten gates: each failing gate sets its own bit,
        /// and all of them are recorded. The timing field keeps "not evaluated"
        /// (nil) apart from "evaluated, none failed" (empty).
        @Test func eachFailingGateSetsItsOwnBit() {
            #expect(Set(Self.gates.map(\.rawValue)).count == Self.gates.count)
            for gate in Self.gates {
                #expect(gate.rawValue.nonzeroBitCount == 1)
                #expect(gate.rawValue & Gate.evaluatedBit == 0)
            }
            for mask in 0 ..< (1 << Self.gates.count) {
                let fails = Self.gates.indices.map { mask & (1 << $0) != 0 }
                let reasons = EngineLoopV2.mtpIneligibility(
                    targetOnlyMedia: fails[0], reasoningBudget: fails[1],
                    constraintEligible: !fails[2], samplingEligible: !fails[3], logprobs: fails[4],
                    logitBias: fails[5], repetitionPenalty: fails[6], penaltiesEligible: !fails[7],
                    stopStringsEligible: !fails[8], multimodalHistoryEligible: !fails[9])
                let expected = Gate(Self.gates.indices.filter { fails[$0] }.map { Self.gates[$0] })
                #expect(reasons == expected, "\(mask)")
                var timing = CBv2RequestTiming()
                timing.mtpIneligibility = reasons
                #expect(timing.mtpIneligibilityBits != 0)
                #expect(timing.mtpIneligibility == reasons, "\(mask)")
            }
            var timing = CBv2RequestTiming()
            #expect(timing.mtpIneligibility == nil)
            timing.mtpIneligibility = []
            #expect(timing.mtpIneligibility == [])
            timing.mtpIneligibility = nil
            #expect(timing.mtpIneligibilityBits == 0)
        }
    }
}
