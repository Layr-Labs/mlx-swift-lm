// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The DFlash drafter bound to the native Muse Glimmer text tower through
/// `MuseGlimmerCBv2Adapter`, on the paged backend. Tiny models, production
/// attention shape and dtype.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashNativeTargetTests {
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter
    static let chunk = max(32, CBv2AttentionV1.queryBlockSize)
    static let maxTokens = 32
    static let taps = [1, 3, 5]

    private func target() throws -> MuseGlimmerTextModel {
        let model = try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: 16), dtype: .bfloat16)
        model.mtpHiddenLayerIDs = Self.taps
        return model
    }

    private func drafter(_ target: MuseGlimmerTextModel) throws -> MuseGlimmerDFlashDrafter {
        let configuration = try MuseGlimmerDFlashConfiguration(
            hiddenSize: 64, intermediateSize: 128, hiddenLayers: 2, attentionHeads: 4,
            kvHeads: 2, headDimensions: 16, blockSize: 16, maskTokenID: 95,
            targetLayerIDs: Self.taps, slidingWindow: 32, rmsNormEps: 1e-5, ropeTheta: 500_000)
        return try MuseGlimmerDFlashDrafter(
            target: target,
            model: try MuseGlimmerDFlashFixture.model(configuration, seed: 5, dtype: .bfloat16),
            settings: .init(maximumSpeculativeRows: 2))
    }

    private func engine(
        _ m: MuseGlimmerTextModel, drafter: (any CBv2MTPDrafter)?,
        verification: CBv2MTPVerificationMode = .serialTarget,
        store: CompleteCheckpointFixtureStore? = nil
    ) throws -> (EngineV2, PagedKVBackend) {
        let backend = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 192 << 20, dtype: m.activationDType, maxPrefillChunk: Self.chunk,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(m), layerKinds: m.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: Self.chunk,
                prefillChunkSize: Self.chunk, soloPrefillStripeTokens: Self.chunk, maxWaiting: 4,
                enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 7, maxSpeculativeBatch: 2,
                fixedDraftTokens: 7, verificationMode: verification,
                maxAutomaticRectangularTokens: 16))
        if store != nil {
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.historicalCheckpointStrideTokens = Self.chunk
            }
        }
        return (engine, backend)
    }

    private func request(
        _ prompt: [Int], _ sampling: CBv2SamplingParams, receipt: UInt64? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: .init(1), promptTokens: prompt, sampling: sampling, maxTokens: Self.maxTokens,
            cacheSalt: receipt == nil ? nil : "tenant",
            prefixCacheReceiptID: receipt.map { .init($0) })
    }

    static let greedy = CBv2SamplingParams(temperature: 0)
    static let sampled = CBv2SamplingParams(temperature: 1, topP: 0.95, topK: 64, seed: 99)

    @Test func theNativeTowerIsADrafterTarget() throws {
        let m = try target()
        let d = try drafter(m)
        #expect(d.mtpTargetIdentity == MuseGlimmerCBv2Adapter(m).mtpTargetIdentity)
        #expect(m.dflashContextWidth == 192 && m.dflashHiddenSize == 64)
        #expect(m.dflashVocabularySize == 96)
        let ids = MLXArray([Int32(3), 95]).reshaped(1, 2)
        #expect(
            MuseGlimmerFixture.sameBits(m.dflashTokenEmbeddings(ids), m.rawTokenEmbeddings(ids)))
        let (engine, _) = try engine(m, drafter: d)
        #expect(engine.mtpInactiveReason == nil)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.usesRequestStatefulDrafter && driver.historyIsAdvisory)
        #expect(driver.config.maxDraftTokens == 7)
    }

    @Test(.timeLimit(.minutes(4)))
    func outputEqualsTheTargetAloneGreedyAndSampled() async throws {
        let m = try target()
        let prompt = MuseGlimmerFixture.tokens(53, vocabulary: 96, salt: 2).map(Int.init)
        for sampling in [Self.greedy, Self.sampled] {
            let (plain, _) = try engine(m, drafter: nil)
            let expected = await cbv2SchedCollect(try plain.submit(request(prompt, sampling)))
            await plain.shutdown()
            #expect(expected.tokens.count == Self.maxTokens)
            for mode in [Scripted.Mode.native, .right, .wrong(at: 0), .wrong(at: 3)] {
                let scripted = Scripted(
                    try drafter(m), mode: mode, scripts: [prompt + expected.tokens],
                    vocabulary: 96)
                let (engine, backend) = try engine(m, drafter: scripted)
                let result = await cbv2SchedCollect(try engine.submit(request(prompt, sampling)))
                #expect(result.tokens == expected.tokens, "\(mode)")
                let metrics = try #require(engine.mtpMetricsSnapshot())
                #expect(metrics.rounds > 0)
                if mode == .right { #expect(metrics.acceptedTokens == metrics.draftedTokens) }
                #expect(
                    zip(scripted.blockAnchors, scripted.blockContextRows).allSatisfy {
                        $1 == min($0, 32)
                    })
                await engine.shutdown()
                #expect(backend.bytesReserved == 0)
            }
        }
    }

    @Test(.timeLimit(.minutes(4)))
    func restoredRequestEqualsColdRequestWithTheDrafterOn() async throws {
        let m = try target()
        let prompt = MuseGlimmerFixture.tokens(3 * Self.chunk + 5, vocabulary: 96, salt: 4).map(
            Int.init)
        for sampling in [Self.greedy, Self.sampled] {
            let coldStore = CompleteCheckpointFixtureStore(segmentBytes: 4096)
            let (cold, _) = try engine(m, drafter: try drafter(m), store: coldStore)
            let expected = await cbv2SchedCollect(
                try cold.submit(request(prompt, sampling, receipt: 1001)))
            #expect(try #require(cold.mtpMetricsSnapshot()).rounds > 0)
            await cold.shutdown()
            #expect(!coldStore.saved.isEmpty)
            #expect(coldStore.saved.allSatisfy { $0.manifest.assistantCodecID == nil })

            let (plain, _) = try engine(m, drafter: nil)
            let reference = await cbv2SchedCollect(try plain.submit(request(prompt, sampling)))
            await plain.shutdown()
            #expect(expected.tokens == reference.tokens)

            let reopened = CompleteCheckpointFixtureStore(archives: coldStore.saved)
            let scripted = Scripted(
                try drafter(m), mode: .right, scripts: [prompt + expected.tokens], vocabulary: 96)
            let (warm, _) = try engine(m, drafter: scripted, store: reopened)
            let warmRequest = request(prompt, sampling, receipt: 2002)
            #expect(try reopened.stage(engine: warm, request: warmRequest))
            let restored = await cbv2SchedCollect(try warm.submit(warmRequest))
            let boundary = try #require(restored.usage?.prefixCachePrefillTokensSaved)
            #expect(boundary > 0)
            #expect(restored.tokens == expected.tokens)
            #expect(scripted.observedPositions.first == boundary)
            let metrics = try #require(warm.mtpMetricsSnapshot())
            #expect(metrics.rounds > 0 && metrics.acceptedTokens == metrics.draftedTokens)
            await warm.shutdown()
        }
    }
}
