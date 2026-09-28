import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXLLM

/// Regression equivalent of oMLX #3978's abandoned-cycle history fold.
/// Tests the native request-state seam, not a full scheduler/HTTP park event.
final class MiMoV26MTPParkResumeTests: XCTestCase {
    private func ids(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init), [1, values.count])
    }
    private func fence(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState) throws {
        eval(assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
    }
    private func drafts(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
                        seed: Int, hidden: MLXArray, depth: Int) throws -> [Int32] {
        var token = ids([seed]), feature = hidden
        var outputs: [Int32] = []
        for _ in 0..<depth {
            let result = assistant.draftStep(tokens: token, hidden: feature,
                shortlist: nil, requestState: state)
            eval([result.tokens, result.hidden] + assistant.evaluationTargets(for: state))
            try assistant.requestStateDidFinishEvaluation(state)
            outputs.append(result.tokens.item(Int32.self))
            token = result.tokens.reshaped([1, 1]); feature = result.hidden
        }
        return outputs
    }
    func testAbandonedRoundsDoNotTurnParkedHistoryIntoDrafts() throws {
        let (target, predictor) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: predictor)
        let tokens = (0..<40).map { 1 + ($0 * 7) % 31 }
        let hidden = try target.forward(inputIDs: ids(tokens)).normalizedHiddenStates
        eval(hidden)
        for depth in 1...3 {
            let candidate = assistant.makeRequestState() as! MiMoV26MTPState
            let control = assistant.makeRequestState() as! MiMoV26MTPState
            try assistant.configureRequestState(candidate, maximumSequenceLength: 64)
            try assistant.configureRequestState(control, maximumSequenceLength: 64)
            defer {
                assistant.releaseRequestState(candidate)
                assistant.releaseRequestState(control)
            }
            var offset = 0
            // Two abandoned rounds, each followed by a multi-token parked
            // history and a separate one-token re-entry seam, then continue.
            for length in [4, 6, 1, 1, 9, 1, 1] {
                if offset == 4 || offset == 12 {
                    let before = candidate.headInputCounts
                    _ = try drafts(assistant, candidate, seed: 7,
                        hidden: hidden[0..., (offset - 1)..<offset, 0...], depth: depth)
                    XCTAssertGreaterThan(candidate.stagedInputCount, 0)
                    assistant.discardRound(requestState: candidate)
                    try fence(assistant, candidate)
                    XCTAssertNil(candidate.round)
                    XCTAssertEqual(candidate.stagedInputCount, 0)
                    XCTAssertEqual(candidate.headInputCounts, before)
                }
                let end = offset + length
                for state in [candidate, control] {
                    assistant.observeCommittedTarget(.init(tokens: ids(Array(tokens[offset..<end])),
                        hidden: hidden[0..., offset..<end, 0...]), requestState: state)
                    try fence(assistant, state)
                    XCTAssertEqual(state.headInputCounts, (0..<3).map { max(0, end - $0 - 1) })
                }
                XCTAssertEqual(candidate.observedCount, control.observedCount)
                XCTAssertEqual(candidate.ownedArrays.count, control.ownedArrays.count)
                for (actual, expected) in zip(candidate.ownedArrays, control.ownedArrays) {
                    XCTAssertEqual(actual.shape, expected.shape)
                    XCTAssertEqual(actual.dtype, expected.dtype)
                    XCTAssertEqual(actual.asData(access: .copy).data,
                                   expected.asData(access: .copy).data)
                }
                offset = end
            }
            let feature = hidden[0..., (offset - 1)..<offset, 0...]
            XCTAssertEqual(try drafts(assistant, candidate, seed: 9, hidden: feature, depth: depth),
                           try drafts(assistant, control, seed: 9, hidden: feature, depth: depth))
            assistant.discardRound(requestState: candidate)
            assistant.discardRound(requestState: control)
            try fence(assistant, candidate); try fence(assistant, control)
        }
    }
}
