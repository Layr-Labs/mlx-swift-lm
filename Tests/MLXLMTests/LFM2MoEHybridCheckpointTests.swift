// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Hybrid prefix reuse through the real EngineV2: an exact checkpoint of
/// full-attention KV plus every short-conv history at a prefill-chunk
/// boundary. A restored request must commit the same tokens as a cold run
/// and save exactly the checkpointed prefix. Covers the resident hybrid bank
/// (contiguous KV) and the complete-checkpoint codec (contiguous and native
/// paged KV), whose encoding omits the absent SSM tensor of conv-only layers.
final class LFM2MoEHybridCheckpointTests: XCTestCase {
    private var chunk: Int { LFM2MoEFixture.chunk }

    private final class Positions: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [[Int]] = []
        func append(_ positions: [Int]) { lock.withLock { values.append(positions) } }
        var snapshot: [[Int]] { lock.withLock { values } }
    }

    private func engine(
        _ model: LFM2MoEModel, storage: LFM2MoECBv2Storage, resident: Bool,
        store: CompleteCheckpointFixtureStore? = nil
    ) -> EngineV2 {
        EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: storage.backend, cacheProvider: storage.bank, sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 2, enablePrefixCache: resident || store != nil),
            admissionConfig: .init(watermarkFraction: 0),
            hybridPrefixCache: resident
                ? .init(
                    maximumBytes: 64 << 20, modelID: "lfm2moe-tiny",
                    promptContractID: "fixture-template", buildID: "fixture-build")
                : nil,
            completePrefixCache: store)
    }

    private func cold(_ model: LFM2MoEModel, prompt: [Int], paged: Bool) async throws -> [Int] {
        let storage = try LFM2MoECBv2Storage(model: model, paged: paged, maximumLength: prompt.count + 8)
        let engine = engine(model, storage: storage, resident: false)
        let result = await cbv2SchedCollect(
            try engine.submit(.init(id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 6)),
            timeoutSeconds: 120)
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
        await engine.shutdown()
        XCTAssertEqual(storage.backend.bytesReserved, 0)
        return result.tokens
    }

    func testResidentHybridCheckpointRestoresConvAndKVAtChunkBoundary() async throws {
        for precision in [LFM2MoEFixture.Precision.float32, .bfloat16, .quantizedMixed] {
            let model = try LFM2MoEFixture.model(precision)
            let donorPrompt = LFM2MoEFixture.prompt(2 * chunk + 7)
            let suffix = donorPrompt + LFM2MoEFixture.prompt(chunk / 2, salt: 3)
            let branch: [Int] = {
                var p = suffix
                p[chunk + 3] = (p[chunk + 3] % 90) + 2  // diverges inside the second chunk
                return p
            }()
            let storage = try LFM2MoECBv2Storage(model: model, paged: false, maximumLength: suffix.count + 8)
            let engine = engine(model, storage: storage, resident: true)
            let published = Positions()
            engine.setResidentPrefixPublicationHandler { _, positions in published.append(positions) }

            let donor = await cbv2SchedCollect(
                try engine.submit(.init(
                    id: .init(10), promptTokens: donorPrompt, sampling: .init(temperature: 0),
                    maxTokens: 6, cacheSalt: "tenant", prefixCacheReceiptID: .init(110))),
                timeoutSeconds: 120)
            XCTAssertEqual(donor.finishReason, .length)
            let coldDonor = try await cold(model, prompt: donorPrompt, paged: false)
            XCTAssertEqual(donor.tokens, coldDonor)
            let publishedDonor = await cbv2SchedWait { !published.snapshot.isEmpty }
            XCTAssertTrue(publishedDonor, "\(precision): donor checkpoints were not published")
            XCTAssertEqual(published.snapshot.first, [chunk, 2 * chunk])

            for (label, prompt, boundary) in [("extension", suffix, 2 * chunk), ("branch", branch, chunk)] {
                let warm = await cbv2SchedCollect(
                    try engine.submit(.init(
                        id: .init(11), promptTokens: prompt, sampling: .init(temperature: 0),
                        maxTokens: 6, cacheSalt: "tenant")),
                    timeoutSeconds: 120)
                XCTAssertEqual(warm.finishReason, .length)
                XCTAssertEqual(warm.usage?.prefixCacheTier, .resident, "\(precision) \(label)")
                XCTAssertEqual(
                    warm.usage?.prefixCachePrefillTokensSaved, boundary, "\(precision) \(label)")
                let coldTokens = try await cold(model, prompt: prompt, paged: false)
                XCTAssertEqual(
                    warm.tokens, coldTokens,
                    "\(precision) \(label): restored conv+KV state must reproduce the cold tokens")
            }
            // Another tenant never sees the donor's state.
            let other = await cbv2SchedCollect(
                try engine.submit(.init(
                    id: .init(12), promptTokens: suffix, sampling: .init(temperature: 0),
                    maxTokens: 2, cacheSalt: "other-tenant")),
                timeoutSeconds: 120)
            XCTAssertEqual(other.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
            await engine.shutdown()
            XCTAssertEqual(storage.backend.bytesReserved, 0)
            XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
            XCTAssertEqual(engine.hybridPrefixCache?.stats.retainedBytes, 0)
        }
    }

    private struct Run {
        let tokens: [Int]
        let archives: [CompleteCheckpointFixtureStore.Archive]
    }

    private func completeRun(
        _ model: LFM2MoEModel, paged: Bool, prompt: [Int],
        archives: [CompleteCheckpointFixtureStore.Archive] = [], expectedSaved: Int = 0
    ) async throws -> Run {
        let store = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 64 << 10)
        let storage = try LFM2MoECBv2Storage(model: model, paged: paged, maximumLength: prompt.count + 8)
        let engine = engine(model, storage: storage, resident: false, store: store)
        XCTAssertNotNil(engine.completeCheckpointCodec, "paged=\(paged): complete checkpoint codec unavailable")
        let published = Positions()
        engine.setCompletePrefixPublicationHandler { _, positions in published.append(positions) }
        let request = CBv2Request(
            id: .init(25), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 6,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(125))
        if expectedSaved > 0 {
            XCTAssertTrue(try store.stage(engine: engine, request: request), "checkpoint was not staged")
        }
        let result = await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: 180)
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved ?? 0, expectedSaved)
        XCTAssertEqual(result.usage?.prefixCacheReplayTokens ?? 0, 0)
        let eligible = stride(from: chunk, to: prompt.count, by: chunk).filter { $0 > expectedSaved }
        if !eligible.isEmpty {
            let settled = await cbv2SchedWait(timeoutSeconds: 10) {
                Set(published.snapshot.flatMap { $0 }).contains(eligible.last!)
            }
            XCTAssertTrue(settled, "paged=\(paged): publication did not settle")
        }
        let saved = Array(store.saved.dropFirst(archives.count))
        await engine.shutdown()
        XCTAssertEqual(storage.backend.bytesReserved, 0)
        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        if let pages = storage.paged { XCTAssertEqual(pages.bytesWired, 0) }
        return Run(tokens: result.tokens, archives: saved)
    }

    func testCompleteCheckpointExportOmitsSSMAndRestoresOnBothBackends() async throws {
        let model = try LFM2MoEFixture.model(.bfloat16)
        let spec = model.cbv2RecurrentStateSpec
        let prompt = LFM2MoEFixture.prompt(2 * chunk + 7)
        for paged in [false, true] {
            let donor = try await completeRun(model, paged: paged, prompt: prompt)
            XCTAssertEqual(donor.archives.map(\.manifest.position).sorted(), [chunk, 2 * chunk])
            for archive in donor.archives {
                let tensors = archive.manifest.tensors
                XCTAssertEqual(
                    tensors.filter { $0.role == .convolution }.count, spec.layers.count,
                    "one conv history per short-conv layer")
                XCTAssertTrue(
                    tensors.filter { $0.role == .recurrent }.isEmpty,
                    "conv-only layers encode no SSM tensor")
                XCTAssertEqual(
                    tensors.filter { $0.role == .keys || $0.role == .values }.count,
                    2 * model.cbv2LayerKinds.count, "one K and one V per attention layer")
                for descriptor in tensors where descriptor.role == .convolution {
                    XCTAssertEqual(descriptor.shape, [1, 2, 256])
                }
            }
            var suffix = prompt + LFM2MoEFixture.prompt(chunk, salt: 5)
            let extended = try await completeRun(
                model, paged: paged, prompt: suffix, archives: donor.archives,
                expectedSaved: 2 * chunk)
            let coldExtended = try await cold(model, prompt: suffix, paged: paged)
            XCTAssertEqual(extended.tokens, coldExtended)
            suffix[chunk + 3] = (suffix[chunk + 3] % 90) + 2
            let branched = try await completeRun(
                model, paged: paged, prompt: suffix, archives: donor.archives, expectedSaved: chunk)
            let coldBranched = try await cold(model, prompt: suffix, paged: paged)
            XCTAssertEqual(branched.tokens, coldBranched)
        }
    }
}
