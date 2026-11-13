// Copyright © 2026 Eigen Labs.

import MLX
import Testing

@testable import MLXLMCommon

@Suite("Grammar MTP non-finite prefix", .serialized, .tags(.integration))
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

        init(target: Target, independentWidths: Bool) {
            mtpTargetIdentity = ObjectIdentifier(target)
            commitsAcceptedPrefixPerRow = independentWidths
        }

        func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Capture() }

        func draftStep(
            tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
        ) -> (tokens: MLXArray, hidden: MLXArray) {
            (((tokens + 1) % 32).reshaped([-1]), hidden)
        }
    }

    private func engine(
        speculating: Bool, mode: CBv2MTPVerificationMode, independentWidths: Bool
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let target = Target()
        let kind = CBv2NonFiniteLogitsTests.kind
        let storage = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 8 << 20, kvDType: .float32))
        return (
            EngineV2(
                model: target, layerKinds: [kind], backend: storage,
                cacheProvider: CBv2LayerCacheBank(layerKinds: [kind]),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 64,
                    prefillChunkSize: 16, maxWaiting: 4),
                admissionConfig: .init(watermarkFraction: 0),
                mtpDrafter: speculating
                    ? Drafter(target: target, independentWidths: independentWidths) : nil,
                mtpConfig: .init(
                    enabled: speculating, maxDraftTokens: 2, fixedDraftTokens: 2,
                    verificationMode: mode, constrainedSpeculation: true)),
            storage
        )
    }

    @Test(.timeLimit(.minutes(2)), arguments: [false, true])
    func nonFiniteFailureKeepsConfirmedPrefixAndNeighbor(independentWidths: Bool) async throws {
        // Late fault, first-column fault, and a poisoned suffix discarded at
        // the first grammar mismatch. The last case must never report NaN.
        for path in [Array(4 ... 11), Array(5 ... 12), [4, 5, 8, 9, 10, 11, 12, 13]] {
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
                    #expect(grammar.confirmedTransitions == result.tokens)
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
                let streams = try paths.indices.map { row in
                    var request = CBv2Request(
                        id: CBv2RequestID(UInt64(row + 1)),
                        promptTokens: [1, 2, paths[row][0] - 1],
                        sampling: .init(temperature: 0), maxTokens: paths[row].count,
                        stopTokens: [0])
                    request.tokenConstraint = grammars[row]
                    return try candidate.submit(request)
                }
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
}
