import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Native miniature GDN + QSA target. No real model load or production route.
final class Qwen4NativeCheckpointEngineTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    private func model() -> Qwen4ExpTextModel {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 64; c.hiddenLayers = 2; c.attentionHeads = 2; c.kvHeads = 1; c.headDim = 64
        c.linearNumValueHeads = 2; c.linearNumKeyHeads = 1
        c.linearKeyHeadDim = 64; c.linearValueHeadDim = 64
        c.vocabularySize = 64; c.maxPositionEmbeddings = 512
        c.fullAttentionInterval = 2; c.layerTypes = ["linear_attention", "qwen_sparse_attention"]
        c.hcCount = 2; c.hcLowrank = 8; c.pleLayerIds = []; c.pleEmbedDim = 64
        c.indexerNHeads = 2; c.indexerKVHeads = 1; c.indexerHeadDim = 32
        c.indexerBudget = 16; c.indexerCompressRatio = 4
        c.numExperts = 1; c.numExpertsPerTok = 1
        c.sharedExpertIntermediateSize = 32; c.moeIntermediateSize = 32
        c.mropeSection = [2, 1, 1]; c.partialRotaryFactor = 0.25; c.mtpNumHiddenLayers = 1
        MLXRandom.seed(8301)
        let model = Qwen4ExpTextModel(c)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(model)
        return model
    }

    private func engine(store: CompleteCheckpointFixtureStore?, mtp: Bool,
                        verification: CBv2MTPVerificationMode,
                        skipColdPromptReplay: Bool = false) throws -> (EngineV2, PagedKVBackend) {
        let target = model()
        let adapter = CBv2SteppableLanguageModelAdapter(target)
        let kinds = target.cbv2LayerKinds
        let observed = try CBv2NativeKVTypeProbe.run(model: adapter, layerKinds: kinds,
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        XCTAssertEqual(target.cbv2CompleteCheckpointKVDTypes, observed.layerDTypes)
        let geometry = try XCTUnwrap(adapter.cbv2Qwen4CheckpointGeometries)
        XCTAssertEqual(geometry.count, 1)
        XCTAssertEqual(geometry[0].layer, 1)
        XCTAssertEqual(geometry[0].headDim, 32)
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 256 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
            segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let caches = backend.makeLayerCaches()
        let assistant = try mtp ? Qwen4ExpInlineMTPAssistant(configuration: target.configuration,
            blockSize: 3, target: target, verificationMode: verification,
            skipColdPromptReplay: skipColdPromptReplay) : nil
        if let assistant { eval(assistant) }
        let engine = EngineV2(model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: assistant, mtpConfig: .init(enabled: mtp, maxDraftTokens: 2,
                fixedDraftTokens: 2, verificationMode: verification))
        XCTAssertNil(engine.hybridPrefixCache)
        if store != nil {
            XCTAssertNotNil(engine.completeCheckpointCodec)
            XCTAssertEqual(engine.completeCheckpointCodec?.qwen4Geometries, geometry)
        }
        return (engine, backend)
    }

    private func released(_ engine: EngineV2, _ backend: PagedKVBackend) {
        let bytes = engine.loopForTesting.onEngineQueueSync {
            (engine.admissionForTesting.bytesReserved, backend.bytesReserved, backend.bytesWired,
             engine.loopForTesting.recurrentStates.isEmpty)
        }
        XCTAssertEqual(bytes.0, 0); XCTAssertEqual(bytes.1, 0); XCTAssertEqual(bytes.2, 0)
        XCTAssertTrue(bytes.3)
    }

    func testMetadataObservesNativeProjectionRatherThanWeightStorageDtype() throws {
        let target = model()
        let projection = try XCTUnwrap(target.model.layers[1].selfAttn).indexer.indexQKProj
        projection.update(parameters: ModuleParameters.unflattened([
            ("weight", projection.weight.asType(.float32))
        ]))
        eval(target)
        let geometries = try XCTUnwrap(target.cbv2Qwen4CheckpointGeometries)
        XCTAssertEqual(projection.weight.dtype, .float32)
        let projected = Qwen4ExpAffineQMM.apply(projection,
            MLXArray.ones([1, 4, 64], dtype: .bfloat16))
        // The established Qwen4 numerical profile preserves BF16 activations
        // even when a projection's parameter storage is FP32. Observe output,
        // not weight dtype, and never widen or narrow a checkpoint to fit it.
        XCTAssertEqual(projected.dtype, .bfloat16)
        XCTAssertEqual(geometries[0].keyDType, CBv2CheckpointDType(projected.dtype))
        XCTAssertEqual(geometries[0].pooledDType, .bfloat16)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: CBv2SteppableLanguageModelAdapter(target), layerKinds: target.cbv2LayerKinds,
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        XCTAssertEqual(observed.layerDTypes, [.bfloat16])
        XCTAssertEqual(target.cbv2CompleteCheckpointKVDTypes, observed.layerDTypes)
    }

    func testColdNoReplayRemainsTargetAuthoritative() async throws {
        let (baseline, baselineBackend) = try engine(
            store: nil, mtp: false, verification: .rectangular)
        let (candidate, candidateBackend) = try engine(
            store: nil, mtp: true, verification: .rectangular,
            skipColdPromptReplay: true)
        let prompt = (0..<chunk + 17).map { 1 + ($0 * 7) % 61 }
        let expected = await cbv2SchedCollect(try baseline.submit(.init(
            id: .init(801), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: 12)))
        let actual = await cbv2SchedCollect(try candidate.submit(.init(
            id: .init(802), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: 12)))
        XCTAssertEqual(actual.finishReason, .length)
        XCTAssertEqual(actual.tokens, expected.tokens)
        let metrics = try XCTUnwrap(candidate.mtpMetricsSnapshot())
        XCTAssertGreaterThan(metrics.draftedTokens, 0)
        XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
        released(baseline, baselineBackend)
        released(candidate, candidateBackend)
        await baseline.shutdown()
        await candidate.shutdown()
    }

    func testNativePagedRestoreMatchesColdAtEarlierAndLatestBoundaries() async throws {
        let modes: [(Bool, CBv2MTPVerificationMode)] = [
            (false, .serialTarget), (true, .serialTarget), (true, .rectangularExact)
        ]
        for (mtp, verification) in modes {
            let store = CompleteCheckpointFixtureStore(segmentBytes: 1024)
            let (donor, donorBackend) = try engine(store: store, mtp: mtp, verification: verification)
            let prompt = (0..<2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
            let request = CBv2Request(id: .init(1), promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 8, cacheSalt: "tenant",
                prefixCacheReceiptID: .init(101))
            let donated = await cbv2SchedCollect(try donor.submit(request))
            XCTAssertEqual(donated.finishReason, .length)
            XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
            XCTAssertTrue(store.saved.allSatisfy { $0.manifest.tensors.contains { $0.role == .indexKeys } })
            released(donor, donorBackend)
            await donor.shutdown()
            let reopened = CompleteCheckpointFixtureStore(archives: store.saved, segmentBytes: 1024)
            let (warm, warmBackend) = try engine(store: reopened, mtp: mtp, verification: verification)
            let (cold, coldBackend) = try engine(store: nil, mtp: mtp, verification: verification)
            var branch = prompt
            branch[chunk + 3] = 2 + branch[chunk + 3] % 59
            for (index, item) in [(prompt, 2 * chunk), (branch, chunk)].enumerated() {
                let next = CBv2Request(id: .init(UInt64(10 + index)), promptTokens: item.0,
                    sampling: .init(temperature: 0), maxTokens: 8, cacheSalt: "tenant",
                    prefixCacheReceiptID: .init(UInt64(110 + index)))
                let expected = await cbv2SchedCollect(try cold.submit(next))
                XCTAssertTrue(try reopened.stage(engine: warm, request: next))
                let actual = await cbv2SchedCollect(try warm.submit(next))
                XCTAssertEqual(actual.finishReason, .length)
                XCTAssertEqual(actual.tokens, expected.tokens, "mtp=\(mtp), boundary=\(item.1)")
                XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, item.1)
                XCTAssertEqual(actual.usage?.prefixCacheReplayTokens, 0)
                released(warm, warmBackend); released(cold, coldBackend)
            }
            if mtp { XCTAssertGreaterThan(try XCTUnwrap(warm.mtpMetricsSnapshot()).draftedTokens, 0) }
            await warm.shutdown(); await cold.shutdown()
        }
    }
}
