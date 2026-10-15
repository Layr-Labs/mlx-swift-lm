import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// One-request serial speculation driven by hand over the real Super target
/// transactions and the real assistant: draft, one `[1, 1]` target column per
/// input, accept walk with forced early rejection, recurrent and attention
/// rollback, assistant finalize. After every round the target state must equal
/// plain greedy decode bit for bit. Phase timings are synthetic fixture data.
@Suite(.tags(.integration))
struct NemotronSuperMTPRoundTests {
    private final class Request {
        let rows: [CBv2SequenceKV?]
        let state: CBv2RecurrentRequestState
        init(_ model: NemotronSuperModel, backend: CBv2ContiguousKVBackend, prompt: Int) throws {
            rows = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: prompt, maxLength: 96)
            state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        }
    }

    private struct Snapshot {
        let offsets: [Int]
        let tensors: [MLXArray]
    }

    private static func snapshot(_ request: Request) throws -> Snapshot {
        let confirmed = try #require(request.state.confirmedStateSnapshot())
        var tensors: [MLXArray] = []
        for spec in request.state.spec.layers {
            let layer = try #require(confirmed[spec.modelLayerIndex])
            tensors += [try #require(layer.conv) + 0, try #require(layer.ssm) + 0]
        }
        var offsets: [Int] = []
        for row in request.rows.compactMap({ $0 }) {
            let kv = row.snapshot()
            offsets.append(kv.offset)
            tensors += [kv.keys + 0, kv.values + 0]
        }
        eval(tensors)
        return Snapshot(offsets: offsets, tensors: tensors)
    }

    private static func exact(_ lhs: Snapshot, _ rhs: Snapshot, _ context: String) {
        #expect(lhs.offsets == rhs.offsets, "\(context)")
        #expect(lhs.tensors.count == rhs.tensors.count)
        for (a, b) in zip(lhs.tensors, rhs.tensors) {
            #expect(a.shape == b.shape && arrayEqual(a, b).item(Bool.self), "\(context)")
        }
    }

    /// One target forward with hidden capture inside its own recurrent transaction.
    private static func forward(
        _ target: NemotronSuperMTPTextTarget, bank: CBv2LayerCacheBank, request: Request,
        tokens: [Int], prefill: Bool
    ) throws -> (logits: MLXArray, hidden: MLXArray, transaction: CBv2RecurrentStateEvaluation) {
        let caches = try bank.layerCaches(rowStates: [request.rows]).map {
            try #require($0 as? KVCache)
        }
        let transaction = try request.state.bind()
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let output =
            prefill
            ? target.cbv2ForwardWithHiddenForPrefill(
                input, caches: caches, recurrentState: [transaction], positionIds: nil,
                requirement: .lastPositionLogits)
            : target.cbv2ForwardWithHidden(
                input, caches: caches, recurrentState: [transaction], positionIds: nil)
        eval(
            [output.logits, output.lastHidden] + (try transaction.evaluate())
                + caches.flatMap { $0.innerState() })
        return (output.logits, output.lastHidden, transaction)
    }

    private static func argmax(_ logits: MLXArray) -> Int {
        Int(argMax(logits[0..., -1, 0...], axis: -1).item(Int32.self))
    }

    @Test(.timeLimit(.minutes(3)), arguments: [1, 3])
    func forcedRejectionsRestoreExactTargetStateEveryRound(depth: Int) throws {
        let stack = try NemotronSuperMTPFixture.stack(
            kind: .predictive, depth: depth, mixerScale: 0.25)
        let target = stack.target
        let assistant = stack.assistant
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 16 << 20, kvDType: .float32))
        let bank = CBv2LayerCacheBank(
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let prompt = [5, 9, 1, 4, 22, 7, 3, 8, 30, 12, 2]
        let generated = 30

        // Plain greedy oracle: snapshot after every fed generated token.
        let oracle = try Request(stack.model, backend: backend, prompt: prompt.count)
        var first = try Self.forward(
            target, bank: bank, request: oracle, tokens: prompt, prefill: true)
        try first.transaction.commit()
        var reference = [Self.argmax(first.logits)]
        var oracleStates = [try Self.snapshot(oracle)]
        for index in 0 ..< generated {
            let step = try Self.forward(
                target, bank: bank, request: oracle, tokens: [reference[index]], prefill: false)
            try step.transaction.commit()
            reference.append(Self.argmax(step.logits))
            oracleStates.append(try Self.snapshot(oracle))
        }

        let actual = try Request(stack.model, backend: backend, prompt: prompt.count)
        defer {
            bank.releaseBoundRows()
            backend.release(oracle.rows)
            backend.release(actual.rows)
        }
        let requestState = assistant.makeRequestState()
        try assistant.configureRequestState(requestState, maximumSequenceLength: 96)
        first = try Self.forward(target, bank: bank, request: actual, tokens: prompt, prefill: true)
        try first.transaction.commit()
        assistant.observeCommittedTarget(
            .init(
                tokens: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
                hidden: first.hidden),
            requestState: requestState)
        var seed = Self.argmax(first.logits)
        var carry = first.hidden[0..., (prompt.count - 1)..., 0...]
        var emitted = [seed]
        var fed = 0
        var round = 0
        var naturalAccepted = 0
        var forcedRejections: Set<Int> = []
        var rejectionPositions: Set<Int> = []
        var draftNanos: UInt64 = 0
        var verifyNanos: UInt64 = 0
        var rollbackNanos: UInt64 = 0
        while fed < generated - depth {
            let limit = [depth, 0, min(1, depth), min(2, depth)][round % 4]
            let draftStart = DispatchTime.now().uptimeNanoseconds
            var proposals: [MLXArray] = []
            var token = MLXArray([Int32(seed)]).reshaped(1, 1)
            var hidden = carry
            for _ in 0 ..< depth {
                let step = assistant.draftStep(
                    tokens: token, hidden: hidden, shortlist: nil, requestState: requestState)
                proposals.append(step.tokens)
                token = step.tokens.reshaped(1, 1)
                hidden = step.hidden
            }
            eval(assistant.evaluationTargets(for: requestState))
            try assistant.requestStateDidFinishEvaluation(requestState)
            let drafts = proposals.map { Int($0.item(Int32.self)) }
            let verifyStart = DispatchTime.now().uptimeNanoseconds
            draftNanos += verifyStart - draftStart
            for row in actual.rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            var transactions: [CBv2RecurrentStateEvaluation] = []
            var targets: [Int] = []
            var targetHidden: [MLXArray] = []
            for input in [seed] + drafts {
                let column = try Self.forward(
                    target, bank: bank, request: actual, tokens: [input], prefill: false)
                transactions.append(column.transaction)
                targets.append(Self.argmax(column.logits))
                targetHidden.append(column.hidden)
            }
            let rollbackStart = DispatchTime.now().uptimeNanoseconds
            verifyNanos += rollbackStart - verifyStart
            var accepted = 0
            while accepted < depth, drafts[accepted] == targets[accepted] { accepted += 1 }
            naturalAccepted += accepted
            if limit < accepted { forcedRejections.insert(limit) }
            accepted = min(accepted, limit)
            if accepted < depth { rejectionPositions.insert(accepted) }
            let rejected = depth - accepted
            for transaction in transactions.suffix(rejected).reversed() {
                try transaction.rollback()
            }
            for transaction in transactions.prefix(accepted + 1) { try transaction.commit() }
            for row in actual.rows.compactMap({ $0 }) {
                if rejected > 0 { row.rollback(rejected) }
                row.commitSpeculativeWrite()
            }
            bank.invalidateBoundComposition()
            let committed = Array(drafts.prefix(accepted))
            assistant.finalizeRound(
                requestState: requestState, confirmedInputTokens: 1 + accepted,
                committedDraftTokens: MLXArray(committed.map(Int32.init)).reshaped(1, accepted),
                committedTargetHidden: accepted == 0
                    ? MLXArray.zeros([1, 0, NemotronSuperMTPFixture.hidden])
                    : concatenated(Array(targetHidden.prefix(accepted)), axis: 1))
            rollbackNanos += DispatchTime.now().uptimeNanoseconds - rollbackStart
            fed += 1 + accepted
            emitted += committed + [targets[accepted]]
            Self.exact(
                try Self.snapshot(actual), oracleStates[fed],
                "round \(round) accepted \(accepted) of \(depth)")
            seed = targets[accepted]
            carry = targetHidden[accepted]
            round += 1
        }
        #expect(Array(emitted.prefix(fed + 1)) == Array(reference.prefix(fed + 1)))
        if depth > 1 { #expect(rejectionPositions.count >= 2 && !forcedRejections.isEmpty) }
        assistant.releaseRequestState(requestState)
        try oracle.state.release()
        try actual.state.release()
        print(
            "SYNTHETIC super-mtp manual serial rounds depth=\(depth) rounds=\(round) "
                + "naturalAccepted=\(naturalAccepted) rejectionPositions=\(rejectionPositions.sorted()) "
                + "forced=\(forcedRejections.sorted()) "
                + "meanDraftNs=\(draftNanos / UInt64(max(round, 1))) "
                + "meanVerifyNs=\(verifyNanos / UInt64(max(round, 1))) "
                + "meanRollbackNs=\(rollbackNanos / UInt64(max(round, 1)))")
    }
}
