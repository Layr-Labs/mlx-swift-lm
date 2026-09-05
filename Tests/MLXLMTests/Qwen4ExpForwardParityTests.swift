import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpForwardParityTests: XCTestCase {
    // MARK: - (b) CBv2 forward is token-exact against the legacy path

    /// Prompt short enough that the visible context stays inside the indexer
    /// budget: the keep mask never fires and attention is plain causal.
    func testCBv2MatchesLegacyBelowTheIndexerBudget() throws {
        try assertCBv2MatchesLegacy(prompt: [3, 9, 14, 2], decodeSteps: 3)
    }

    /// Prompt long enough that the tape passes the budget during prefill, so
    /// every full-attention layer runs with a keep mask.
    func testCBv2MatchesLegacyAboveTheIndexerBudget() throws {
        try assertCBv2MatchesLegacy(
            prompt: [3, 9, 14, 2, 21, 6, 11, 30, 4, 17, 25, 8], decodeSteps: 4)
    }

    private func assertCBv2MatchesLegacy(prompt: [Int], decodeSteps: Int) throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        let expected = qwen4ExpLegacyGreedy(
            model: model, prompt: prompt, count: decodeSteps + 1)

        let session = try Qwen4ExpCBv2Session(model: model)
        var produced: [Int] = []
        var next = session.greedy(try session.forward(prompt).logits)
        produced.append(next)
        for _ in 0 ..< decodeSteps {
            next = session.greedy(try session.forward([next]).logits)
            produced.append(next)
        }

        XCTAssertEqual(produced, expected)
        // The mask must really have fired on the long prompt, or this test
        // would be asserting the dense path twice.
        let tapeLength = session.caches[0].indexerTapeLength
        XCTAssertEqual(tapeLength, prompt.count + decodeSteps)
        if prompt.count > Qwen4ExpFixture.indexerBudget {
            XCTAssertGreaterThan(tapeLength, Qwen4ExpFixture.indexerBudget)
        }
    }

    // MARK: - (c) MTP draft and verify are lossless at depths 1...3

    func testMTPRoundsAreLosslessAtEveryPermittedDepth() throws {
        let prompt = [3, 9, 14, 2, 21, 6, 11, 30, 4]
        let steps = 6

        let baselineModel = try Qwen4ExpFixture.model()
        let baselineSession = try Qwen4ExpCBv2Session(model: baselineModel)
        var baseline: [Int] = []
        var token = baselineSession.greedy(try baselineSession.forward(prompt).logits)
        baseline.append(token)
        for _ in 1 ..< steps {
            token = baselineSession.greedy(try baselineSession.forward([token]).logits)
            baseline.append(token)
        }

        for depth in 1 ... Qwen4ExpInlineMTPAssistant.maximumDepth {
            let produced = try speculativeGreedy(prompt: prompt, steps: steps, depth: depth)
            XCTAssertEqual(
                produced, baseline,
                "MTP output diverged from serial output at depth \(depth)")
        }
    }

    /// Greedy generation driven by the embedded head, with the accept walk the
    /// engine's serial-target verification performs: chain `depth` drafts,
    /// then feed the window one column at a time and stop at the first
    /// divergence. A rejected draft is never fed, so the committed history is
    /// exactly the history serial decode would have written.
    private func speculativeGreedy(prompt: [Int], steps: Int, depth: Int) throws -> [Int] {
        let model = try Qwen4ExpFixture.model()
        let session = try Qwen4ExpCBv2Session(model: model)
        let drafter = try XCTUnwrap(Qwen4ExpInlineMTPAssistant(target: model))
        XCTAssertEqual(drafter.maximumDraftTokens, 3)
        XCTAssertEqual(drafter.maximumSpeculativeBatch, 1)
        XCTAssertEqual(drafter.requiredVerificationMode, .serialTarget)
        XCTAssertEqual(drafter.mtpTargetIdentity, ObjectIdentifier(model))

        let requestState = drafter.makeRequestState()
        defer { drafter.releaseRequestState(requestState) }

        // Prefill, then hand the head the prompt's trusted transitions.
        let prefill = try session.forward(prompt)
        drafter.observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(
                tokens: MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]),
                hidden: prefill.multi),
            requestState: requestState)

        var produced: [Int] = []
        var carryToken = session.greedy(prefill.logits)
        var carryMulti = prefill.multi[0..., -1 ..< prefill.multi.dim(1), 0...]
        produced.append(carryToken)

        while produced.count < steps {
            // Chain the draft.
            var proposals: [Int] = []
            var chainTokens = MLXArray([Int32(carryToken)]).reshaped([1, 1])
            var chainMulti = carryMulti
            for _ in 0 ..< depth {
                let draft = drafter.draftStep(
                    tokens: chainTokens, hidden: chainMulti, shortlist: nil,
                    requestState: requestState)
                eval(draft.tokens, draft.hidden)
                proposals.append(draft.tokens.item(Int.self))
                chainTokens = draft.tokens.reshaped([1, 1])
                chainMulti = draft.hidden
            }

            // Verify column by column against the target itself.
            var accepted = 0
            var conditioningMulti: [MLXArray] = []
            var feed = carryToken
            var emitted = 0
            while true {
                let out = try session.forward([feed])
                conditioningMulti.append(out.multi[0..., -1 ..< out.multi.dim(1), 0...])
                let argmax = session.greedy(out.logits)
                carryMulti = conditioningMulti[conditioningMulti.count - 1]
                produced.append(argmax)
                emitted += 1
                carryToken = argmax
                guard accepted < proposals.count, argmax == proposals[accepted],
                    produced.count < steps
                else { break }
                accepted += 1
                feed = argmax
            }
            XCTAssertGreaterThan(emitted, 0)

            // Close the round on the head: the accepted proposals became
            // canonical target history, each paired with the multi stream
            // that conditioned it.
            let committedTokens =
                accepted > 0
                ? MLXArray(proposals[0 ..< accepted].map { Int32($0) }).reshaped([1, accepted])
                : MLXArray([Int32]()).reshaped([1, 0])
            let committedHidden =
                accepted > 0
                ? concatenated(Array(conditioningMulti[0 ..< accepted]), axis: 1)
                : MLXArray.zeros([1, 0, carryMulti.dim(2)], dtype: carryMulti.dtype)
            drafter.finalizeRound(
                requestState: requestState,
                confirmedInputTokens: accepted + 1,
                committedDraftTokens: committedTokens,
                committedTargetHidden: committedHidden)
        }
        return Array(produced.prefix(steps))
    }
}
