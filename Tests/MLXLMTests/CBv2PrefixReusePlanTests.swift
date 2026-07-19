import XCTest

@testable import MLXLMCommon

final class CBv2FrozenReplayPlanTests: XCTestCase {
    private func full(sharedWith source: Int? = nil) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            sharesKVWithLayer: source,
            headDim: 8,
            kvHeads: 2,
            queryHeads: 4)
    }

    private func sliding(_ window: Int) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window),
            headDim: 8,
            kvHeads: 2,
            queryHeads: 4)
    }

    func testInterleavedHybridProducesExplicitFrozenPlan() throws {
        let kinds = [full(), sliding(16), sliding(16), full()]
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .contiguousUnquantized)
        XCTAssertTrue(capability.isSupported)
        XCTAssertEqual(capability.strategy, .frozenFullReplay)
        XCTAssertEqual(capability.conservativeReplayBoundTokens, 32)

        let plan = try XCTUnwrap(capability.plan(matchedBoundary: 72))
        XCTAssertEqual(plan.matchedBoundary, 72)
        XCTAssertEqual(plan.replayStart, 40)
        XCTAssertEqual(plan.replayTokens, 32)
        XCTAssertEqual(plan.prefillTokensSaved, 40)
        XCTAssertEqual(plan.restoredFullTokens, 72)
        XCTAssertEqual(plan.capacityReservationTokens, 72)
        XCTAssertEqual(plan.fullKVBytesPerToken, 128)
        XCTAssertEqual(plan.stagedFullKVBytes, 72 * 128)
        XCTAssertEqual(plan.residentFullKVBytes, 72 * 128)
        XCTAssertEqual(plan.clampedChunk(start: 32, proposed: 16), 8)
        XCTAssertEqual(plan.clampedChunk(start: 40, proposed: 40), 32)
        XCTAssertEqual(plan.clampedChunk(start: 72, proposed: 16), 16)

        let nativeFP32 = try XCTUnwrap(
            capability.plan(
                matchedBoundary: 72,
                exactStagedFullKVBytes: 72 * 256,
                maximumSequenceLength: 200))
        XCTAssertEqual(nativeFP32.fullKVBytesPerToken, 256)
        XCTAssertEqual(nativeFP32.nominalFullKVBytesPerToken, 128)
        XCTAssertEqual(nativeFP32.additionalFullKVBytesPerToken, 128)
        XCTAssertEqual(nativeFP32.fullCapacityTokensReserved, 200)
        XCTAssertEqual(
            nativeFP32.initialAdditionalCapacityBytes,
            200 * 256 - 72 * 128)
        XCTAssertEqual(nativeFP32.stagedFullKVBytes, 72 * 256)
        XCTAssertEqual(nativeFP32.residentFullKVBytes, 72 * 256)
    }

    func testBackendSupportFailsColdForPagedQuantizedAndUnknownHybrid() {
        let kinds = [full(), sliding(16), full()]
        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .pagedFP16)
        XCTAssertFalse(paged.isSupported)
        XCTAssertEqual(paged.unsupportedReason, .pagedHybridRequiresDualCursor)

        let quantized = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .contiguousQuantized)
        XCTAssertFalse(quantized.isSupported)
        XCTAssertEqual(quantized.unsupportedReason, .quantizedRowsUnsupported)

        let unknown = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .unknown)
        XCTAssertFalse(unknown.isSupported)
        XCTAssertEqual(unknown.unsupportedReason, .unknownBackend)

        let forwardShared = [
            full(sharedWith: 1),
            full(),
        ]
        XCTAssertEqual(
            CBv2PrefixReuseCapability.derive(
                layerKinds: forwardShared,
                backend: .contiguousUnquantized
            ).unsupportedReason,
            .invalidLayout)
        let mismatchedShared = [
            sliding(16),
            full(sharedWith: 0),
        ]
        XCTAssertEqual(
            CBv2PrefixReuseCapability.derive(
                layerKinds: mismatchedShared,
                backend: .contiguousUnquantized
            ).unsupportedReason,
            .invalidLayout)
    }

    func testMatchesBelowAtAndAboveReplayBoundFailOrAdoptExactly() {
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(16), sliding(16), full()],
            backend: .contiguousUnquantized)
        XCTAssertEqual(capability.conservativeReplayBoundTokens, 32)
        XCTAssertNil(capability.plan(matchedBoundary: 31))
        XCTAssertNil(capability.plan(matchedBoundary: 32))
        let above = capability.plan(matchedBoundary: 33)
        XCTAssertEqual(above?.replayStart, 1)
        XCTAssertEqual(above?.replayTokens, 32)
        XCTAssertEqual(above?.clampedChunk(start: 0, proposed: 2), 1)
        XCTAssertEqual(above?.clampedChunk(start: 1, proposed: 31), 31)
        XCTAssertEqual(above?.clampedChunk(start: 1, proposed: 33), 32)
        XCTAssertEqual(above?.clampedChunk(start: 33, proposed: 1), 1)
    }

    func testSafeLayoutsPreserveDirectAndTailReplay() throws {
        let direct = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), full()],
            backend: .pagedFP16)
        XCTAssertEqual(direct.strategy, .direct)
        let directPlan = try XCTUnwrap(direct.plan(matchedBoundary: 512))
        XCTAssertEqual(directPlan.replayStart, 512)
        XCTAssertEqual(directPlan.restoredFullTokens, 512)

        let tail = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(64), sliding(128)],
            backend: .pagedFP16)
        XCTAssertEqual(tail.strategy, .tailReplay)
        let tailPlan = try XCTUnwrap(tail.plan(matchedBoundary: 512))
        XCTAssertEqual(tailPlan.replayTokens, 256)
        XCTAssertEqual(tailPlan.replayStart, 256)
        XCTAssertEqual(tailPlan.restoredFullTokens, 256)
        XCTAssertEqual(tailPlan.capacityReservationTokens, 256)

        let shared = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(64), full(sharedWith: 0)],
            backend: .contiguousUnquantized)
        XCTAssertEqual(shared.strategy, .tailReplay)
    }

    func testFrozenReplayReservesMOnceAndRollbackBalancesExactly() throws {
        let kinds = [full(), sliding(16), sliding(16), full()]
        let plan = try XCTUnwrap(
            CBv2PrefixReuseCapability.derive(
                layerKinds: kinds,
                backend: .contiguousUnquantized
            ).plan(
                matchedBoundary: 72,
                exactStagedFullKVBytes: 72 * 256,
                maximumSequenceLength: 74))
        let admission = AdmissionV2(
            layerKinds: kinds,
            bytesCapacity: 1 << 28,
            config: .init(watermarkFraction: 0))
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 64,
                prefillChunkSize: 32),
            capacity: admission)
        let request = CBv2Request(
            id: CBv2RequestID(991),
            promptTokens: makePromptTokens(length: 73, seed: 991),
            sampling: .init(temperature: 0),
            maxTokens: 1)
        let rec = try scheduler.enqueue(request)
        rec.prefixReusePlan = plan
        rec.numComputedTokens = plan.replayStart
        try admission.reserve(
            id: rec.id,
            additionalTokens: plan.capacityReservationTokens,
            additionalBytes: plan.initialAdditionalCapacityBytes)
        let reservedAtM =
            admission.estimatedBytes(forTokens: plan.matchedBoundary)
            + plan.initialAdditionalCapacityBytes
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        let replay = scheduler.plan()
        XCTAssertEqual(replay.assignments.map(\.numTokens), [32])
        XCTAssertEqual(
            admission.bytesReserved,
            reservedAtM,
            "replay below M must consume the atomic M reservation, not add R")
        scheduler.rollback(replay)
        XCTAssertEqual(rec.numComputedTokens, plan.replayStart)
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        rec.numComputedTokens = plan.matchedBoundary
        let tail = scheduler.plan()
        XCTAssertEqual(tail.assignments.map(\.numTokens), [1])
        XCTAssertEqual(
            admission.bytesReserved,
            reservedAtM,
            "full native span and fixed sliding rings were prepaid at adoption")
        scheduler.rollback(tail)
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        XCTAssertTrue(scheduler.requeueOnCapacity(rec.id))
        XCTAssertNil(rec.prefixReusePlan)
        XCTAssertEqual(rec.numComputedTokens, 0)
        XCTAssertEqual(admission.bytesReserved, 0)
    }
}
