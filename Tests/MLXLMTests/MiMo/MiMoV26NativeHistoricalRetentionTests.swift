import Foundation
import XCTest

@testable import MLXLMCommon

/// Actual default native-contiguous capture/import and retirement. The tiny
/// artifact and encoded transport do not establish full-model TPS or SSD proof.
final class MiMoV26NativeHistoricalRetentionTests: XCTestCase {
    func testDemandedMiddleFrontierSurvivesNativePublicationAndReopenedFork() async throws {
        for mtp in [false, true] {
            var fixtures: [MiMoNativeHistoricalRetentionFixture] = []
            do {
                let donor = try await MiMoNativeHistoricalRetentionFixture.load(mtp: mtp)
                fixtures.append(donor)
                donor.engine.loopForTesting.onEngineQueueSync { [weak engine = donor.engine] in
                    engine?.loopForTesting.nativeRetirementBoundaryForTesting = {
                        [weak engine] _, _ in
                        guard let engine, let capture = engine.completeCheckpointCapture else {
                            return
                        }
                        XCTAssertLessThanOrEqual(capture.staged.values.map(\.count).max() ?? 0, 3)
                        XCTAssertLessThanOrEqual(
                            capture.stagedHistoricalBytes
                                + max(0, capture.inFlightHistoricalBytes),
                            capture.historicalSlotStagedByteCap)
                        XCTAssertLessThanOrEqual(
                            engine.admissionForTesting.bytesReserved,
                            engine.admissionForTesting.bytesCapacity)
                    }
                }
                var seed = request(95_100, count: 897)
                seed.prefixCheckpointTargetTokens = 512
                _ = try await donor.collect(seed)
                try assertExecutedMTPMode(donor, enabled: mtp)
                donor.engine.loopForTesting.onEngineQueueSync {
                    donor.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                }
                XCTAssertEqual(donor.store.base.saved.map(\.manifest.position), [896, 512, 128])
                let target = try XCTUnwrap(
                    donor.store.base.saved.first { $0.manifest.position == 512 })
                XCTAssertEqual(target.manifest.assistantCodecID != nil, mtp)
                XCTAssertEqual(target.manifest.cacheSalt, seed.cacheSalt)
                XCTAssertEqual(target.manifest.prefixTokens, Array(seed.promptTokens.prefix(512)))
                for (index, tensor) in target.manifest.tensors.enumerated() {
                    let chunks = target.chunks.filter { $0.tensor == index }.sorted {
                        $0.offset < $1.offset
                    }
                    var offset = 0
                    for chunk in chunks {
                        XCTAssertEqual(chunk.offset, offset)
                        offset += chunk.bytes.count
                    }
                    XCTAssertEqual(
                        offset, tensor.byteCount,
                        "reopening uses complete encoded bytes, never donor native aliases")
                }
                try await donor.stop()
                let warm = try await MiMoNativeHistoricalRetentionFixture.load(
                    mtp: mtp, archives: [target])
                fixtures.append(warm)
                let cold = try await MiMoNativeHistoricalRetentionFixture.load(mtp: mtp)
                fixtures.append(cold)
                var fork = request(95_101, count: 513)
                fork.promptTokens[512] = fork.promptTokens[512] % 29 + 1
                XCTAssertTrue(try warm.store.base.stage(engine: warm.engine, request: fork))
                let restored = try await warm.collect(fork)
                try assertExecutedMTPMode(warm, enabled: mtp)
                fork.prefixCacheEnabled = false
                let reference = try await cold.collect(fork)
                try assertExecutedMTPMode(cold, enabled: mtp)
                XCTAssertEqual(restored.tokens, reference.tokens)
                XCTAssertEqual(restored.usage?.prefixCachePrefillTokensSaved, 512)
                XCTAssertEqual(restored.usage?.prefixCacheReplayTokens ?? 0, 0)
                XCTAssertEqual(reference.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
                XCTAssertEqual(restored.usage?.prefixCacheHitTokens, 512)
                try await warm.stop()
                try await cold.stop()
            } catch {
                for fixture in fixtures { try? await fixture.stop() }
                throw error
            }
        }
    }

    /// Sampled rows never speculate on this drafter (no target-prefix
    /// acceptance). They still publish frontiers that carry the assistant:
    /// the same bytes a greedy donor publishes, restorable by either kind.
    func testSampledDonorPublishesFrontiersThatSampledAndGreedyForksRestore() async throws {
        var fixtures: [MiMoNativeHistoricalRetentionFixture] = []
        do {
            let donor = try await MiMoNativeHistoricalRetentionFixture.load(mtp: true)
            fixtures.append(donor)
            donor.engine.loopForTesting.onEngineQueueSync { [weak engine = donor.engine] in
                engine?.loopForTesting.nativeRetirementBoundaryForTesting = {
                    [weak engine] _, _ in
                    guard let engine, let capture = engine.completeCheckpointCapture else {
                        return
                    }
                    XCTAssertLessThanOrEqual(capture.staged.values.map(\.count).max() ?? 0, 3)
                    XCTAssertLessThanOrEqual(
                        capture.stagedHistoricalBytes + max(0, capture.inFlightHistoricalBytes),
                        capture.historicalSlotStagedByteCap)
                    XCTAssertLessThanOrEqual(
                        engine.admissionForTesting.bytesReserved,
                        engine.admissionForTesting.bytesCapacity)
                }
            }
            // 900 = 7 chunks of 128 + a 4-token final chunk that takes the
            // recording prefill, not a 1-token decode row.
            var seed = request(95_200, count: 900, sampling: Self.sampled)
            seed.prefixCheckpointTargetTokens = 512
            let donated = try await donor.collect(seed)
            assertNoDrafts(donor)
            donor.engine.loopForTesting.onEngineQueueSync {
                donor.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            }
            XCTAssertEqual(donor.store.base.saved.map(\.manifest.position), [896, 512, 128])
            let target = try XCTUnwrap(
                donor.store.base.saved.first { $0.manifest.position == 512 })
            XCTAssertNotNil(target.manifest.assistantCodecID)
            XCTAssertEqual(target.manifest.prefixTokens, Array(seed.promptTokens.prefix(512)))
            let sampledArchives = donor.store.base.saved
            try await donor.stop()

            let greedy = try await MiMoNativeHistoricalRetentionFixture.load(mtp: true)
            fixtures.append(greedy)
            var greedySeed = request(95_202, count: 900)
            greedySeed.prefixCheckpointTargetTokens = 512
            _ = try await greedy.collect(greedySeed)
            try assertExecutedMTPMode(greedy, enabled: true)
            XCTAssertEqual(greedy.store.base.saved.map(\.manifest.position), [896, 512, 128])
            for (sampled, speculating) in zip(sampledArchives, greedy.store.base.saved) {
                let position = sampled.manifest.position
                XCTAssertEqual(sampled.manifest, speculating.manifest, "position \(position)")
                XCTAssertNotNil(sampled.manifest.assistantCodecID, "position \(position)")
                XCTAssertFalse(sampled.chunks.isEmpty, "position \(position)")
                XCTAssertTrue(Self.sameTensors(sampled, speculating), "position \(position)")
            }
            try await greedy.stop()

            let plain = try await MiMoNativeHistoricalRetentionFixture.load(mtp: true)
            fixtures.append(plain)
            seed.prefixCacheEnabled = false
            let plainTokens = try await plain.collect(seed).tokens
            XCTAssertEqual(plainTokens, donated.tokens)
            XCTAssertTrue(plain.store.base.saved.isEmpty)
            try await plain.stop()

            for sampling in [Self.sampled, CBv2SamplingParams(temperature: 0)] {
                let warm = try await MiMoNativeHistoricalRetentionFixture.load(
                    mtp: true, archives: [target])
                fixtures.append(warm)
                let cold = try await MiMoNativeHistoricalRetentionFixture.load(mtp: true)
                fixtures.append(cold)
                var fork = request(95_201, count: 513, sampling: sampling)
                fork.promptTokens[512] = fork.promptTokens[512] % 29 + 1
                XCTAssertTrue(try warm.store.base.stage(engine: warm.engine, request: fork))
                let restored = try await warm.collect(fork)
                fork.prefixCacheEnabled = false
                let reference = try await cold.collect(fork)
                if sampling.temperature == 0 {
                    try assertExecutedMTPMode(warm, enabled: true)
                    try assertExecutedMTPMode(cold, enabled: true)
                } else {
                    assertNoDrafts(warm)
                    assertNoDrafts(cold)
                }
                XCTAssertEqual(restored.tokens, reference.tokens)
                XCTAssertEqual(restored.usage?.prefixCachePrefillTokensSaved, 512)
                XCTAssertEqual(restored.usage?.prefixCacheReplayTokens ?? 0, 0)
                XCTAssertEqual(restored.usage?.prefixCacheHitTokens, 512)
                XCTAssertEqual(reference.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
                try await warm.stop()
                try await cold.stop()
            }
        } catch {
            for fixture in fixtures { try? await fixture.stop() }
            throw error
        }
    }

    private func assertNoDrafts(_ fixture: MiMoNativeHistoricalRetentionFixture) {
        let metrics = fixture.engine.mtpMetricsSnapshot()
        XCTAssertEqual(metrics?.active, true)
        XCTAssertEqual(metrics?.serialVerificationRounds ?? -1, 0)
        XCTAssertEqual(metrics?.draftedTokens ?? -1, 0)
    }

    private static func sameTensors(
        _ a: CompleteCheckpointFixtureStore.Archive, _ b: CompleteCheckpointFixtureStore.Archive
    ) -> Bool {
        func parts(_ archive: CompleteCheckpointFixtureStore.Archive) -> [(Int, Int, Data)] {
            archive.chunks.map { ($0.tensor, $0.offset, $0.bytes) }
                .sorted { ($0.0, $0.1) < ($1.0, $1.1) }
        }
        let (left, right) = (parts(a), parts(b))
        return a.manifest.position == b.manifest.position && left.count == right.count
            && zip(left, right).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 && $0.2 == $1.2 }
    }

    private static let sampled = CBv2SamplingParams(temperature: 1, topP: 0.95, seed: 7)

    private func assertExecutedMTPMode(
        _ fixture: MiMoNativeHistoricalRetentionFixture, enabled: Bool
    ) throws {
        let metrics = fixture.engine.mtpMetricsSnapshot()
        XCTAssertEqual(metrics?.active == true, enabled)
        if enabled {
            let active = try XCTUnwrap(metrics)
            XCTAssertEqual(active.verificationMode, .serialTarget)
            XCTAssertGreaterThan(active.serialVerificationRounds, 0)
            XCTAssertGreaterThan(active.draftedTokens, 0)
        } else {
            XCTAssertNil(metrics)
        }
    }

    private func request(
        _ id: UInt64, count: Int, sampling: CBv2SamplingParams = .init(temperature: 0)
    ) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< count).map { 1 + ($0 * 7) % 29 },
            sampling: sampling, maxTokens: 24,
            cacheSalt: "isolated-contiguous-tenant", prefixCacheEnabled: true,
            prefixCacheReceiptID: .init(id + 10_000))
    }
}
