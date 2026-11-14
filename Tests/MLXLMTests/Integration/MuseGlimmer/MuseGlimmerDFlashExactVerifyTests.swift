// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The DFlash drafter on a tower with exact verify windows: the engine scores
/// each round in one rectangular forward, and the output is the target's
/// alone, drafter on or off, greedy or sampled, cold or restored.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashExactVerifyTests {
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter
    static let chunk = max(32, CBv2AttentionV1.queryBlockSize)
    static let maxTokens = 32
    static let taps = [1, 3, 5]
    static let greedy = CBv2SamplingParams(temperature: 0)
    static let sampled = CBv2SamplingParams(temperature: 1, topP: 0.95, topK: 64, seed: 99)

    private func target(_ dtype: DType, exact: Bool = true) throws -> MuseGlimmerTextModel {
        let model = try MuseGlimmerExactWindowTests.mixed(layers: 6, dtype: dtype)
        model.mtpHiddenLayerIDs = Self.taps
        try model.setRowExactWindows(exact)
        return model
    }

    private func drafter(
        _ target: MuseGlimmerTextModel, settings: MuseGlimmerDFlashDrafter.Settings? = nil,
        columnsAsRows: Bool = true
    ) throws -> MuseGlimmerDFlashDrafter {
        let configuration = try MuseGlimmerDFlashConfiguration(
            hiddenSize: MuseGlimmerExactWindowTests.hidden, intermediateSize: 256,
            hiddenLayers: 2, attentionHeads: 4, kvHeads: 2, headDimensions: 16, blockSize: 16,
            maskTokenID: 95, targetLayerIDs: Self.taps, slidingWindow: 32, rmsNormEps: 1e-5,
            ropeTheta: 500_000)
        return try MuseGlimmerDFlashDrafter(
            target: target,
            model: try MuseGlimmerDFlashFixture.model(
                configuration, seed: 5, dtype: target.activationDType),
            settings: settings
                ?? .verifying(
                    target, maximumSpeculativeRows: 2,
                    batchesVerifyColumnsAsDecodeRows: columnsAsRows))
    }

    private func engine(
        _ m: MuseGlimmerTextModel, drafter: (any CBv2MTPDrafter)?,
        store: CompleteCheckpointFixtureStore? = nil
    ) throws -> (EngineV2, PagedKVBackend, [PagedLayerCache]) {
        let backend = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 256 << 20, dtype: m.activationDType, maxPrefillChunk: Self.chunk,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
        let caches = backend.makeLayerCaches()
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(m), layerKinds: m.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: Self.chunk,
                prefillChunkSize: Self.chunk, soloPrefillStripeTokens: Self.chunk, maxWaiting: 4,
                enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 7, maxSpeculativeBatch: 2,
                fixedDraftTokens: 7, verificationMode: .rectangularExact,
                maxAutomaticRectangularTokens: 16))
        if store != nil {
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.historicalCheckpointStrideTokens = Self.chunk
            }
        }
        return (engine, backend, caches)
    }

    /// Layer calls that attended a round as decode rows.
    private func decodeRowCalls(_ caches: [PagedLayerCache]) -> Int {
        caches.map(\.mtpColumnsAsRowsCalls).reduce(0, +)
    }

    private func request(
        _ prompt: [Int], _ sampling: CBv2SamplingParams, id: UInt64 = 1, receipt: UInt64? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: sampling, maxTokens: Self.maxTokens,
            cacheSalt: receipt == nil ? nil : "tenant",
            prefixCacheReceiptID: receipt.map { .init($0) })
    }

    private func prompt(_ length: Int, salt: Int) -> [Int] {
        MuseGlimmerFixture.tokens(length, vocabulary: 96, salt: salt).map(Int.init)
    }

    private func plain(_ m: MuseGlimmerTextModel, _ request: CBv2Request) async throws -> [Int] {
        let (engine, _, _) = try engine(m, drafter: nil)
        let result = await cbv2SchedCollect(try engine.submit(request))
        await engine.shutdown()
        #expect(result.tokens.count == Self.maxTokens)
        return result.tokens
    }

    /// Every verification was one rectangular forward. `rounds` counts
    /// request rounds; one forward verifies the rounds of every row it holds.
    @discardableResult
    private func expectRectangular(_ engine: EngineV2) throws -> CBv2MTPMetrics {
        let metrics = try #require(engine.mtpMetricsSnapshot())
        #expect(metrics.verificationMode == .rectangularExact)
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(metrics.rectangularVerificationRounds <= metrics.rounds)
        #expect(metrics.serialVerificationRounds == 0)
        return metrics
    }

    @Test func settingsFollowTheTargetAndExactVerificationNeedsAnExactTarget() throws {
        let plainTower = try target(.bfloat16, exact: false)
        let serial = MuseGlimmerDFlashDrafter.Settings.verifying(
            plainTower, maximumSpeculativeRows: 2)
        #expect(serial.verification == .serialTarget && !serial.commitsAcceptedPrefixPerRow)
        #expect(serial.maximumSpeculativeRows == 1 && serial.maximumDraftTokens == 7)
        #expect(throws: MuseGlimmerDFlashError.target("exact verify windows")) {
            try drafter(plainTower, settings: .init(verification: .rectangularExact))
        }
        let exactTower = try target(.bfloat16)
        let exact = MuseGlimmerDFlashDrafter.Settings.verifying(
            exactTower, maximumSpeculativeRows: 2)
        #expect(exact.verification == .rectangularExact && exact.commitsAcceptedPrefixPerRow)
        #expect(exact.maximumSpeculativeRows == 2)
        let d = try drafter(exactTower)
        #expect(d.requiredVerificationMode == .rectangularExact && d.commitsAcceptedPrefixPerRow)
        // The tower taps other layers of the same count: refused.
        exactTower.mtpHiddenLayerIDs = [0, 2, 4]
        #expect(throws: MuseGlimmerDFlashError.target("tapped layers")) { try drafter(exactTower) }
        exactTower.mtpHiddenLayerIDs = Self.taps
        // Decode-row attention: on for exact verification unless declined,
        // never for serial verification, off in a bare `Settings`.
        #expect(exact.batchesVerifyColumnsAsDecodeRows && d.batchesVerifyColumnsAsDecodeRows)
        #expect(!serial.batchesVerifyColumnsAsDecodeRows)
        #expect(!MuseGlimmerDFlashDrafter.Settings().batchesVerifyColumnsAsDecodeRows)
        #expect(!(try drafter(exactTower, columnsAsRows: false)).batchesVerifyColumnsAsDecodeRows)
    }

    /// With attention as decode rows and as the column loop.
    @Test(.timeLimit(.minutes(6)), arguments: [DType.bfloat16, .float32], [true, false])
    func outputEqualsTheTargetAloneGreedyAndSampled(dtype: DType, columnsAsRows: Bool)
        async throws
    {
        let m = try target(dtype)
        let prompt = prompt(53, salt: 2)
        for sampling in [Self.greedy, Self.sampled] {
            let expected = try await plain(m, request(prompt, sampling))
            for mode in [Scripted.Mode.native, .right, .wrong(at: 0), .wrong(at: 3)] {
                let scripted = Scripted(
                    try drafter(m, columnsAsRows: columnsAsRows), mode: mode,
                    scripts: [prompt + expected], vocabulary: 96)
                let (engine, backend, caches) = try engine(m, drafter: scripted)
                let result = await cbv2SchedCollect(try engine.submit(request(prompt, sampling)))
                #expect(result.tokens == expected, "\(mode) \(sampling.temperature)")
                try expectRectangular(engine)
                #expect((decodeRowCalls(caches) > 0) == columnsAsRows)
                let metrics = try #require(engine.mtpMetricsSnapshot())
                if mode == .right { #expect(metrics.acceptedTokens == metrics.draftedTokens) }
                await engine.shutdown()
                #expect(backend.bytesReserved == 0)
            }
        }
    }

    /// Two requests share each verify forward; each row commits its own prefix.
    @Test(.timeLimit(.minutes(6)), arguments: [DType.bfloat16, .float32])
    func twoRequestsVerifiedTogetherEqualTheirSoloOutputs(dtype: DType) async throws {
        let m = try target(dtype)
        let prompts = [prompt(53, salt: 2), prompt(29, salt: 7)]
        for sampling in [Self.greedy, Self.sampled] {
            var expected = [[Int]]()
            for (index, prompt) in prompts.enumerated() {
                expected.append(
                    try await plain(m, request(prompt, sampling, id: UInt64(index + 1))))
            }
            for mode in [Scripted.Mode.right, .wrong(at: 2)] {
                let scripted = Scripted(
                    try drafter(m), mode: mode,
                    scripts: zip(prompts, expected).map { $0 + $1 }, vocabulary: 96)
                let (engine, backend, caches) = try engine(m, drafter: scripted)
                let first = try engine.submit(request(prompts[0], sampling, id: 1))
                let second = try engine.submit(request(prompts[1], sampling, id: 2))
                async let a = cbv2SchedCollect(first)
                async let b = cbv2SchedCollect(second)
                let together = await [a, b]
                #expect(together.map(\.tokens) == expected, "\(mode) \(sampling.temperature)")
                // Some forward verified both requests.
                let metrics = try expectRectangular(engine)
                #expect(metrics.rectangularVerificationRounds < metrics.rounds)
                #expect(decodeRowCalls(caches) > 0)
                await engine.shutdown()
                #expect(backend.bytesReserved == 0)
            }
        }
    }

    @Test(.timeLimit(.minutes(6)))
    func restoredRequestEqualsColdRequestWithTheDrafterOn() async throws {
        let m = try target(.bfloat16)
        let prompt = prompt(3 * Self.chunk + 5, salt: 4)
        for sampling in [Self.greedy, Self.sampled] {
            let coldStore = CompleteCheckpointFixtureStore(segmentBytes: 4096)
            let (cold, _, _) = try engine(m, drafter: try drafter(m), store: coldStore)
            let expected = await cbv2SchedCollect(
                try cold.submit(request(prompt, sampling, receipt: 1001)))
            try expectRectangular(cold)
            await cold.shutdown()
            #expect(!coldStore.saved.isEmpty)
            #expect(expected.tokens == (try await plain(m, request(prompt, sampling))))

            let reopened = CompleteCheckpointFixtureStore(archives: coldStore.saved)
            let scripted = Scripted(
                try drafter(m), mode: .right, scripts: [prompt + expected.tokens], vocabulary: 96)
            let (warm, _, _) = try engine(m, drafter: scripted, store: reopened)
            let warmRequest = request(prompt, sampling, receipt: 2002)
            #expect(try reopened.stage(engine: warm, request: warmRequest))
            let restored = await cbv2SchedCollect(try warm.submit(warmRequest))
            #expect(try #require(restored.usage?.prefixCachePrefillTokensSaved) > 0)
            #expect(restored.tokens == expected.tokens)
            try expectRectangular(warm)
            await warm.shutdown()
        }
    }
}
