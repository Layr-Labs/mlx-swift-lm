import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Production Qwen3.5 attention/recurrent execution with small seeded weights.
/// Checkpoint compression is lossy; only cold runs and identical restores are
/// compared exactly, not a restored run against the original native donor.
final class Qwen35NativeQuantizedCheckpointTests: XCTestCase {
    private struct Fixture {
        let engine: EngineV2
        let backend: PagedKVBackend
        let recurrentSpec: CBv2RecurrentStateSpec
    }

    private var chunk: Int { max(256, CBv2AttentionV1.queryBlockSize) }

    private func model() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data("""
                {
                  "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                  "num_hidden_layers": 4, "intermediate_size": 64,
                  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                  "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                  "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                  "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                  "vocab_size": 64, "num_experts": 0, "num_experts_per_tok": 0,
                  "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                  "norm_topk_prob": true, "mtp_num_hidden_layers": 0
                }
                """.utf8))
        MLXRandom.seed(7029)
        let model = Qwen35TextModel(configuration)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        eval(model)
        return model
    }

    private func fixture(
        store: CompleteCheckpointFixtureStore,
        checkpointQuantization: PagedKVQuantizationConfig?
    ) throws -> Fixture {
        let model = try model()
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 2 * chunk + 16,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
            ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
        })
        let caches = model.newCacheV2 { index, _ in storage[indices[index]!] }
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            checkpointQuantization: checkpointQuantization)
        XCTAssertNotNil(engine.completeCheckpointCodec)
        XCTAssertEqual(engine.completeCheckpointCodec?.checkpointQuantization, checkpointQuantization)
        XCTAssertNil(engine.completeCheckpointCodec?.assistant)
        XCTAssertNil(engine.mtpMetricsSnapshot())
        let fixture = Fixture(
            engine: engine, backend: backend, recurrentSpec: model.cbv2RecurrentStateSpec)
        assertNative(fixture)
        return fixture
    }

    private func assertNative(_ fixture: Fixture) {
        let backend = fixture.backend
        XCTAssertNil(backend.pool.config.quantization)
        XCTAssertFalse(backend.usesQuantizedStorage)
        XCTAssertTrue(backend.supportsOrdinaryDecodeChaining)
        var nativeRate = 0
        for (index, kind) in backend.layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            XCTAssertNil(backend.pool.groupKey(forLayer: index).quantization)
            if case .full = kind.attention {
                nativeRate += kind.kvHeads * (kind.headDim + kind.valueHeadDim)
                    * backend.pool.layerDTypes[index].size
            }
        }
        XCTAssertGreaterThan(nativeRate, 0)
        XCTAssertEqual(fixture.engine.admissionForTesting.fullKVBytesPerToken, nativeRate)
    }

    private func collect(
        _ fixture: Fixture, store: CompleteCheckpointFixtureStore,
        request: CBv2Request, restore: Bool = false
    ) async throws -> CBv2SchedCollected {
        do {
            if restore { XCTAssertTrue(try store.stage(engine: fixture.engine, request: request)) }
            let result = await cbv2SchedCollect(try fixture.engine.submit(request))
            store.finishPublicationCallbacks(engine: fixture.engine)
            assertNative(fixture)
            let state = fixture.engine.loopForTesting.onEngineQueueSync {
                (fixture.engine.admissionForTesting.bytesReserved, fixture.backend.bytesReserved,
                 fixture.backend.bytesWired, fixture.engine.loopForTesting.recurrentStates.isEmpty)
            }
            XCTAssertEqual(state.0, 0)
            XCTAssertEqual(state.1, 0)
            XCTAssertEqual(state.2, 0)
            XCTAssertTrue(state.3)
            await fixture.engine.shutdown()
            return result
        } catch {
            await fixture.engine.shutdown()
            throw error
        }
    }

    func testNativeColdOutputAndRepeatedLossyCheckpointSuffix() async throws {
        let profile = PagedKVQuantizationConfig()
        XCTAssertEqual(profile.recentTokenCount, 128)
        let prompt = (0 ..< 2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
        let request = CBv2Request(
            id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        let nativeStore = CompleteCheckpointFixtureStore(segmentBytes: 257)
        let native = try fixture(store: nativeStore, checkpointQuantization: nil)
        let cold = try await collect(native, store: nativeStore, request: request)
        XCTAssertEqual(cold.finishReason, .length)
        XCTAssertEqual(cold.tokens.count, 4)
        XCTAssertEqual(cold.usage?.prefixCachePrefillTokensSaved, 0)

        let store = CompleteCheckpointFixtureStore(segmentBytes: 257)
        let donor = try fixture(store: store, checkpointQuantization: profile)
        XCTAssertEqual(
            donor.engine.admissionForTesting.fullKVBytesPerToken,
            native.engine.admissionForTesting.fullKVBytesPerToken)
        XCTAssertEqual(
            donor.engine.admissionForTesting.fixedBytesPerRequest,
            native.engine.admissionForTesting.fixedBytesPerRequest)
        let donated = try await collect(donor, store: store, request: request)
        XCTAssertEqual(donated.finishReason, .length)
        XCTAssertEqual(donated.tokens, cold.tokens,
            "Storage-only quantization must not change a cold native forward")
        XCTAssertEqual(donated.usage?.prefixCachePrefillTokensSaved, 0)
        let archives = store.saved
        XCTAssertEqual(archives.map(\.manifest.position), [2 * chunk, chunk])
        let boundary = try XCTUnwrap(archives.map(\.manifest.position).max())
        XCTAssertGreaterThanOrEqual(boundary, 256)
        XCTAssertGreaterThan(boundary, profile.recentTokenCount)
        let nativeDTypes = donor.backend.pool.layerDTypes.compactMap(CBv2CheckpointDType.init)
        for archive in archives {
            let manifest = archive.manifest
            XCTAssertEqual(manifest.backendLayout, CBv2CompleteCheckpointManifest.nativeQuantizedPagedLayout)
            XCTAssertEqual(manifest.checkpointQuantization, profile)
            XCTAssertEqual(manifest.checkpointNativeDTypes, nativeDTypes)
            XCTAssertNil(manifest.assistantCodecID)
            let target = manifest.tensors.filter { $0.role == .keys || $0.role == .values }
            XCTAssertEqual(target.count, donor.backend.layerKinds.count * 2)
            XCTAssertTrue(target.allSatisfy { $0.dtype == .uint8 })
            XCTAssertLessThan(
                target.reduce(0) { $0 + $1.byteCount },
                manifest.position * donor.engine.admissionForTesting.fullKVBytesPerToken)
            XCTAssertEqual(
                manifest.tensors.filter { $0.role == .convolution }.count,
                donor.recurrentSpec.layers.count)
            XCTAssertEqual(
                manifest.tensors.filter { $0.role == .recurrent }.count,
                donor.recurrentSpec.layers.count)
            for layer in donor.recurrentSpec.layers {
                let convolution = try XCTUnwrap(manifest.tensors.first {
                    $0.role == .convolution && $0.layer == layer.modelLayerIndex
                })
                let recurrent = try XCTUnwrap(manifest.tensors.first {
                    $0.role == .recurrent && $0.layer == layer.modelLayerIndex
                })
                XCTAssertEqual(convolution.dtype, try XCTUnwrap(CBv2CheckpointDType(layer.convDType)))
                XCTAssertEqual(recurrent.dtype, try XCTUnwrap(CBv2CheckpointDType(layer.ssmDType)))
            }
        }

        // Both fresh engines receive the same encoded checkpoint and a real
        // suffix that differs from the donor after its deepest saved boundary.
        var suffixPrompt = prompt
        let suffixIndex = 2 * chunk + 3
        suffixPrompt[suffixIndex] = suffixPrompt[suffixIndex] % 61 + 1
        var restoredTokens: [Int]?
        for index in 0 ..< 2 {
            let reopened = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 257)
            let restored = try fixture(store: reopened, checkpointQuantization: profile)
            XCTAssertEqual(
                restored.engine.admissionForTesting.fullKVBytesPerToken,
                donor.engine.admissionForTesting.fullKVBytesPerToken)
            XCTAssertEqual(
                restored.engine.admissionForTesting.fixedBytesPerRequest,
                donor.engine.admissionForTesting.fixedBytesPerRequest)
            let suffixRequest = CBv2Request(
                id: .init(UInt64(10 + index)), promptTokens: suffixPrompt,
                sampling: .init(temperature: 0), maxTokens: 4, cacheSalt: "tenant",
                prefixCacheReceiptID: .init(UInt64(2000 + index)))
            let actual = try await collect(
                restored, store: reopened, request: suffixRequest, restore: true)
            XCTAssertEqual(actual.finishReason, .length)
            XCTAssertEqual(actual.tokens.count, 4)
            XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, boundary)
            XCTAssertEqual(actual.usage?.prefixCacheReplayTokens, 0)
            XCTAssertEqual(actual.usage?.prefixCacheTier, .snapshot)
            XCTAssertEqual(reopened.releaseCount, 1)
            if let restoredTokens {
                XCTAssertEqual(actual.tokens, restoredTokens,
                    "Identical encoded state and seeded target must execute the same suffix")
            } else {
                restoredTokens = actual.tokens
            }
        }
    }
}
