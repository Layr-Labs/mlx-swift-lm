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

    private func request(_ id: UInt64, count: Int) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< count).map { 1 + ($0 * 7) % 29 },
            sampling: .init(temperature: 0), maxTokens: 24,
            cacheSalt: "isolated-contiguous-tenant", prefixCacheEnabled: true,
            prefixCacheReceiptID: .init(id + 10_000))
    }
}
