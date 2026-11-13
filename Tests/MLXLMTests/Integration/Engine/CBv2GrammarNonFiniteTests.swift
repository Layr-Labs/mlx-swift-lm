// Copyright © 2026 Eigen Labs.

import MLX
import Testing

@testable import MLXLMCommon

@Suite("Grammar MTP failure and lifecycle", .serialized, .tags(.integration))
struct CBv2GrammarNonFiniteTests {
    /// Reuses the ordinary non-finite engine fixture: real cache writes,
    /// token+1 logits, and one poisoned logit only when consuming token 6.
    private final class Target: CBv2MTPSteppableModel {
        let model = CBv2NonFiniteLogitsTests.TriggerModel(poison: .nan)
        let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
        var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            model.forward(tokens: tokens, caches: caches)
        }

        func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
            -> (logits: MLXArray, lastHidden: MLXArray)
        {
            (forward(tokens: tokens, caches: caches), tokens.asType(.float32)[0..., 0..., .newAxis])
        }
    }

    private final class Drafter: CBv2MTPDrafter {
        private final class Capture: CBv2MTPPreparedCapture {}
        let mtpTargetIdentity: ObjectIdentifier?
        let commitsAcceptedPrefixPerRow: Bool
        var onFirstPrepare: (() -> Void)?

        init(target: Target, independentWidths: Bool) {
            mtpTargetIdentity = ObjectIdentifier(target)
            commitsAcceptedPrefixPerRow = independentWidths
        }

        func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
            let callback = onFirstPrepare
            onFirstPrepare = nil
            callback?()
            return Capture()
        }

        func draftStep(
            tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
        ) -> (tokens: MLXArray, hidden: MLXArray) {
            (((tokens + 1) % 32).reshaped([-1]), hidden)
        }
    }

    private func engine(
        speculating: Bool, mode: CBv2MTPVerificationMode, independentWidths: Bool,
        pauseAfterFirstDraft: Bool = false, nativeTracking: Bool = false,
        constrainedSpeculation: Bool = true, maxSpeculativeBatch: Int = 2
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let target = Target()
        let kind = CBv2NonFiniteLogitsTests.kind
        let storage = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 8 << 20, kvDType: .float32))
        let drafter =
            speculating ? Drafter(target: target, independentWidths: independentWidths) : nil
        let candidate = EngineV2(
            model: target, layerKinds: [kind], backend: storage,
            cacheProvider: CBv2LayerCacheBank(layerKinds: [kind]),
            sampler: CBv2DefaultSampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 4),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: speculating, maxDraftTokens: 2,
                maxSpeculativeBatch: maxSpeculativeBatch, fixedDraftTokens: 2,
                verificationMode: mode, constrainedSpeculation: constrainedSpeculation),
            nativeCompletionTracking: nativeTracking)
        if pauseAfterFirstDraft {
            drafter?.onFirstPrepare = { [weak candidate] in
                guard let loop = candidate?.loopForTesting else { return }
                loop.suspendStepExecutionAtCountForTesting = loop.stepCount + 1
            }
        }
        return (candidate, storage)
    }

    /// Admit the entire mixed batch before any row can seed or finish alone.
    private func submitTogether(_ requests: [CBv2Request], to engine: EngineV2) throws
        -> [AsyncStream<CBv2Event>]
    {
        let loop = engine.loopForTesting
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = loop.stepCount }
        defer { loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = nil } }
        return try requests.map { try engine.submit($0) }
    }

    @Test(.timeLimit(.minutes(2)), arguments: [false, true])
    func nonFiniteFailureKeepsConfirmedPrefixAndNeighbor(independentWidths: Bool) async throws {
        // Late verify, first verify column, ordinary seed, and an unconsumed
        // poisoned suffix. The last case must never report NaN.
        for path in [
            Array(4 ... 11), Array(5 ... 12), Array(6 ... 13), [4, 5, 8, 9, 10, 11, 12, 13],
        ] {
            for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
                let paths = [path, Array(21 ... 28)]
                var expected: [CBv2SchedCollected] = []
                for row in paths.indices {
                    let (oracle, storage) = engine(
                        speculating: false, mode: mode, independentWidths: independentWidths)
                    let grammar = CBv2MTPInterningConstraint(paths[row])
                    var request = CBv2Request(
                        id: CBv2RequestID(UInt64(row + 1)),
                        promptTokens: [1, 2, paths[row][0] - 1],
                        sampling: .init(temperature: 0), maxTokens: paths[row].count,
                        stopTokens: [0])
                    request.tokenConstraint = grammar
                    let result = await cbv2SchedCollect(try oracle.submit(request))
                    // Known ordinary-path fault: it confirms masked token 7
                    // before rejecting its raw non-finite logits. Preserve
                    // that baseline explicitly; it is NOT commit-only state.
                    let discardedBaselineToken = row == 0 && path.contains(6) ? [7] : []
                    if !discardedBaselineToken.isEmpty {
                        withKnownIssue(
                            "Ordinary finalization confirms a non-finite token before discarding it"
                        ) {
                            #expect(grammar.confirmedTransitions == result.tokens)
                        }
                    }
                    #expect(grammar.confirmedTransitions == result.tokens + discardedBaselineToken)
                    expected.append(result)
                    await oracle.shutdown()
                    #expect(storage.bytesReserved == 0)
                }
                let shouldFail = path.contains(6)
                #expect(expected[0].tokens == (shouldFail ? Array(path.prefix { $0 <= 6 }) : path))
                #expect(
                    expected[0].finishReason
                        == (shouldFail ? CBv2NonFiniteLogits.finishReason : .length))

                let (candidate, storage) = engine(
                    speculating: true, mode: mode, independentWidths: independentWidths)
                let grammars = paths.map(CBv2MTPInterningConstraint.init)
                let requests = paths.indices.map { row in
                    var request = CBv2Request(
                        id: CBv2RequestID(UInt64(row + 1)),
                        promptTokens: [1, 2, paths[row][0] - 1],
                        sampling: .init(temperature: 0), maxTokens: paths[row].count,
                        stopTokens: [0])
                    request.tokenConstraint = grammars[row]
                    return request
                }
                let streams = try submitTogether(requests, to: candidate)
                async let first = cbv2SchedCollect(streams[0])
                async let second = cbv2SchedCollect(streams[1])
                let actual = await [first, second]
                for row in paths.indices {
                    #expect(actual[row].tokens == expected[row].tokens)
                    #expect(actual[row].finishReason == expected[row].finishReason)
                    #expect(
                        actual[row].usage?.completionTokens == expected[row].usage?.completionTokens
                    )
                    #expect(grammars[row].confirmedTransitions == actual[row].tokens)
                }
                #expect((candidate.mtpMetricsSnapshot()?.rounds ?? 0) > 0)
                await candidate.shutdown()
                #expect(storage.bytesReserved == 0)
                #expect(candidate.capacity().activeRequests == 0)
            }
        }
    }

    private func request(
        _ id: UInt64, grammar: CBv2MTPInterningConstraint, stop: Int? = nil
    ) -> CBv2Request {
        var request = CBv2Request(
            id: CBv2RequestID(id), promptTokens: [1, 2, grammar.path[0] - 1],
            sampling: .init(temperature: 0), maxTokens: grammar.maxTokens,
            stopTokens: Set([0] + (stop.map { [$0] } ?? [])))
        request.tokenConstraint = grammar
        return request
    }

    @Test(.timeLimit(.minutes(2)), arguments: [false, true])
    func earlyStopConfirmsOnlyTheVisiblePrefix(independentWidths: Bool) async throws {
        for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
            for stop in [23, 24, 25] {
                let paths = [Array(21 ... 28), Array(8 ... 15)]
                var expected: [CBv2SchedCollected] = []
                for row in paths.indices {
                    let (oracle, storage) = engine(
                        speculating: false, mode: mode, independentWidths: independentWidths)
                    let grammar = CBv2MTPInterningConstraint(paths[row])
                    let result = await cbv2SchedCollect(
                        try oracle.submit(
                            request(
                                UInt64(row + 1), grammar: grammar,
                                stop: row == 0 ? stop : nil)))
                    #expect(grammar.confirmedTransitions == result.tokens)
                    expected.append(result)
                    await oracle.shutdown()
                    #expect(storage.bytesReserved == 0)
                }
                #expect(expected[0].tokens == Array(21 ... stop))
                #expect(expected[0].finishReason == .stop)

                let (candidate, storage) = engine(
                    speculating: true, mode: mode, independentWidths: independentWidths)
                let grammars = paths.map(CBv2MTPInterningConstraint.init)
                let requests = paths.indices.map { row in
                    request(UInt64(row + 1), grammar: grammars[row], stop: row == 0 ? stop : nil)
                }
                let streams = try submitTogether(requests, to: candidate)
                async let first = cbv2SchedCollect(streams[0])
                async let second = cbv2SchedCollect(streams[1])
                let actual = await [first, second]
                for row in paths.indices {
                    #expect(actual[row].tokens == expected[row].tokens)
                    #expect(actual[row].finishReason == expected[row].finishReason)
                    #expect(
                        actual[row].usage?.completionTokens == expected[row].usage?.completionTokens
                    )
                    #expect(grammars[row].confirmedTransitions == actual[row].tokens)
                    #expect((actual[row].usage?.timing.mtpRounds ?? 0) > 0)
                }
                await candidate.shutdown()
                #expect(storage.bytesReserved == 0)
            }
        }
    }

    @Test(.timeLimit(.minutes(2)), arguments: [false, true])
    func ordinaryFallbackConfirmsOnlyFiniteTokensWhenGrammarMTPIsOptedIn(optedIn: Bool)
        async throws
    {
        let (oracle, oracleStorage) = engine(
            speculating: false, mode: .serialTarget, independentWidths: false)
        let oracleGrammar = CBv2MTPInterningConstraint(Array(5 ... 12))
        let expected = await cbv2SchedCollect(try oracle.submit(request(1, grammar: oracleGrammar)))
        #expect(expected.tokens == [5, 6])
        #expect(expected.finishReason == CBv2NonFiniteLogits.finishReason)
        #expect(oracleGrammar.confirmedTransitions == [5, 6, 7])
        await oracle.shutdown()
        #expect(oracleStorage.bytesReserved == 0)

        // Two live rows exceed the configured speculative batch cap. The
        // poison must therefore be consumed by ordinary finalization.
        let (candidate, storage) = engine(
            speculating: true, mode: .serialTarget, independentWidths: false,
            constrainedSpeculation: optedIn, maxSpeculativeBatch: 1)
        let grammar = CBv2MTPInterningConstraint(Array(5 ... 12))
        let neighbor = CBv2MTPInterningConstraint(Array(21 ... 28))
        let streams = try submitTogether(
            [request(1, grammar: grammar), request(2, grammar: neighbor)], to: candidate)
        async let first = cbv2SchedCollect(streams[0])
        async let second = cbv2SchedCollect(streams[1])
        let (actual, healthy) = await (first, second)
        #expect(actual.tokens == expected.tokens)
        #expect(actual.finishReason == expected.finishReason)
        #expect(actual.usage?.completionTokens == expected.usage?.completionTokens)
        if optedIn {
            #expect(grammar.confirmedTransitions == actual.tokens)
        } else {
            // Flag-off retains the documented ordinary-path defect.
            #expect(grammar.confirmedTransitions == [5, 6, 7])
        }
        #expect(healthy.tokens == neighbor.path)
        #expect(healthy.finishReason == .length)
        #expect(neighbor.confirmedTransitions == healthy.tokens)
        let failures = candidate.loopForTesting.onEngineQueueSync {
            (
                candidate.loopForTesting.nonFiniteSampledFailures,
                candidate.loopForTesting.nonFiniteVerifyFailures
            )
        }
        #expect(failures.0 == 1)
        #expect(failures.1 == 0)
        #expect((actual.usage?.timing.mtpRounds ?? 0) == 0)
        await candidate.shutdown()
        #expect(storage.bytesReserved == 0)
    }

    @Test(.timeLimit(.minutes(2)), arguments: [false, true])
    func cancellationAfterVerifyLaunchDoesNotConfirmDiscardedDrafts(independentWidths: Bool)
        async throws
    {
        for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
            let (candidate, storage) = engine(
                speculating: true, mode: mode, independentWidths: independentWidths,
                pauseAfterFirstDraft: true)
            let loop = candidate.loopForTesting
            let grammar = CBv2MTPInterningConstraint(Array(21 ... 28))
            let victim = request(1, grammar: grammar)
            let stream = try candidate.submit(victim)
            let launched = await cbv2SchedWait {
                loop.onEngineQueueSync {
                    guard let boundary = loop.suspendStepExecutionAtCountForTesting else {
                        return false
                    }
                    return loop.stepCount == boundary && loop.hasInFlightStepForTesting
                }
            }
            #expect(launched, "held the first actual verify launch before its finalization")
            let confirmedBeforeCancel = grammar.confirmedTransitions
            #expect(!confirmedBeforeCancel.isEmpty)
            #expect(confirmedBeforeCancel.count < grammar.maxTokens)
            candidate.cancel(victim.id)
            loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = nil }
            let cancelled = await cbv2SchedCollect(stream)
            #expect(cancelled.finishReason == .cancelled)
            #expect(cancelled.tokens == Array(grammar.path.prefix(confirmedBeforeCancel.count)))
            #expect(grammar.confirmedTransitions == confirmedBeforeCancel)
            #expect(grammar.confirmedTransitions == cancelled.tokens)
            #expect(cancelled.usage?.completionTokens == cancelled.tokens.count)

            // Reusing the ID exercises sampler retirement and admission after
            // the launched window was discarded, with a different grammar.
            let followerGrammar = CBv2MTPInterningConstraint(Array(8 ... 15))
            let follower = await cbv2SchedCollect(
                try candidate.submit(request(1, grammar: followerGrammar)))
            #expect(follower.tokens == followerGrammar.path)
            #expect(follower.finishReason == .length)
            #expect(followerGrammar.confirmedTransitions == follower.tokens)
            #expect((follower.usage?.timing.mtpRounds ?? 0) > 0)
            await candidate.shutdown()
            #expect(storage.bytesReserved == 0)
            #expect(candidate.capacity().activeRequests == 0)
        }
    }

    @Test func grammarOptInCannotBypassNativeContractRefusal() async throws {
        let (candidate, storage) = engine(
            speculating: true, mode: .serialTarget, independentWidths: false,
            nativeTracking: true)
        let grammar = CBv2MTPInterningConstraint(Array(21 ... 28))
        let item = request(1, grammar: grammar)
        do {
            _ = try candidate.submitWithNativeRetirement(item)
            Issue.record("grammar opt-in must not grant an unsupported native contract")
        } catch CBv2NativeShutdownError.unsupportedConsumer {
            // The synthetic target has no issued native ownership contract.
        } catch {
            Issue.record("unexpected native refusal: \(error)")
        }
        #expect(grammar.confirmedTransitions.isEmpty)
        #expect((candidate.mtpMetricsSnapshot()?.rounds ?? 0) == 0)
        #expect(storage.bytesReserved == 0)
        await candidate.shutdown()
    }
}
