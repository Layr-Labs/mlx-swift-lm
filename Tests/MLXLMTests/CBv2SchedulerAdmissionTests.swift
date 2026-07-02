// Copyright © 2026 Eigen Labs.
//
// WS-B: AdmissionV2 tests — truthful worst-case estimates (window-capped,
// KV-sharing aware), soft reservations with a watermark, rollback symmetry.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerAdmissionTests: XCTestCase {

    private func id(_ raw: UInt64) -> CBv2RequestID { CBv2RequestID(raw) }

    /// full + sliding-window(128) + KV-shared layer, kvHeads 2, headDim 8.
    private var hybridKinds: [CBv2LayerKind] {
        [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 0, headDim: 8, kvHeads: 2, queryHeads: 4),
        ]
    }

    func testEstimateRespectsWindowsAndKVSharing() {
        let admission = AdmissionV2(
            layerKinds: hybridKinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        // Per token per storage-owning layer: 2(K+V) × 2 heads × 8 dim × 2 B = 64 B.
        // 1000 tokens: full layer = 64_000; windowed = min(1000,128)×64 = 8192;
        // KV-shared layer owns NOTHING.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 1000), 64_000 + 8192)
        // Below the window, both storage layers charge fully.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 100), 100 * 64 * 2)
        XCTAssertEqual(admission.estimatedBytes(forTokens: 0), 0)
    }

    func testCanEverFitIsWorstCase() {
        // Capacity fits exactly 100 tokens of the single full layer
        // (32 B/token: 2 × 1 head × 8 dim × 2 B).
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0))
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
        XCTAssertFalse(admission.canEverFit(promptTokens: 50, maxTokens: 51))
        // The check ignores current usage — feasibility, not availability.
        try? admission.reserve(id: id(1), additionalTokens: 100)
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
    }

    /// Regression (admission/reserve livelock): `canEverFit` judged against
    /// FULL capacity while `reserve` enforces `capacity - watermark`. A
    /// request whose worst case landed in `(capacity - watermark, capacity]`
    /// was admitted, could never reserve its final tokens, self-preempted,
    /// restarted, and livelocked until its deadline. Feasibility must use
    /// the same watermark-adjusted ceiling `reserve` enforces.
    func testCanEverFitRespectsWatermark() throws {
        // 1 full layer, kv1 hd8 eb2 → 32 B/token. Capacity 3200, watermark
        // 5% (160 B) → 3040 usable → 95 tokens is the true ceiling.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0.05))
        XCTAssertEqual(admission.admissibleBytesCapacity, 3040)

        // 95 tokens: admissible AND reservable end to end.
        XCTAssertTrue(admission.canEverFit(promptTokens: 45, maxTokens: 50))
        try admission.reserve(id: id(1), additionalTokens: 95)
        admission.releaseAll(id: id(1))

        // 96..100 tokens: previously admitted (<= full capacity) but the
        // solo reservation can never complete — must be rejected up front.
        for total in 96 ... 100 {
            XCTAssertFalse(
                admission.canEverFit(promptTokens: total - 50, maxTokens: 50),
                "\(total)-token worst case exceeds capacity - watermark")
            XCTAssertThrowsError(
                try admission.reserve(id: id(2), additionalTokens: total),
                "reserve agrees: \(total) tokens can never be granted")
            admission.releaseAll(id: id(2))
        }
    }

    func testSoftReserveThrowsAtWatermark() throws {
        // 1 full layer, kv1 hd1 eb2 → 4 B/token. Capacity 1000, watermark
        // 10% → 900 usable → 225 tokens.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0.1))
        try admission.reserve(id: id(1), additionalTokens: 225)
        XCTAssertThrowsError(try admission.reserve(id: id(2), additionalTokens: 1)) { error in
            guard case CBv2KVError.capacityExhausted(let needed, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(needed, 4)
            XCTAssertEqual(available, 0)
        }
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 1))
    }

    func testUnreserveAndReleaseRestoreHeadroom() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 400, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 60)  // 240 B
        try admission.reserve(id: id(2), additionalTokens: 40)  // 400 B
        XCTAssertEqual(admission.bytesReserved, 400)
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 1))

        // Optimistic-advance rollback path.
        admission.unreserve(id: id(2), tokens: 10)
        XCTAssertEqual(admission.bytesReserved, 360)
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 10))

        // Finish/cancel/preempt path.
        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 120)
        try admission.reserve(id: id(3), additionalTokens: 70)
        XCTAssertEqual(admission.bytesReserved, 400)
    }

    func testWindowedReservationPlateaus() throws {
        // Sliding window 4 → per-request bytes stop growing past 4 tokens,
        // so long decodes on windowed layers do not exhaust the ledger.
        let kinds = [CBv2LayerKind(attention: .slidingWindow(4), headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 64, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 4)  // 16 B (at the plateau)
        XCTAssertEqual(admission.bytesReserved, 16)
        for _ in 0 ..< 1000 {
            try admission.reserve(id: id(1), additionalTokens: 1)  // +0 B each
        }
        XCTAssertEqual(admission.bytesReserved, 16)
        // A second sequence still pays its own window.
        try admission.reserve(id: id(2), additionalTokens: 100)
        XCTAssertEqual(admission.bytesReserved, 32)
    }

    func testSnapshotReportsLedgerOrBackendTruth() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 25)  // 100 B
        let soft = admission.snapshot(activeRequests: 1, waitingRequests: 2, activeTokens: 25)
        XCTAssertEqual(soft.kvBytesInUse, 100)
        XCTAssertEqual(soft.kvBytesCapacity, 1000)
        XCTAssertEqual(soft.activeRequests, 1)
        XCTAssertEqual(soft.waitingRequests, 2)
        XCTAssertEqual(soft.activeTokens, 25)

        let hard = admission.snapshot(
            activeRequests: 1, waitingRequests: 2, activeTokens: 25, backendBytesInUse: 88)
        XCTAssertEqual(hard.kvBytesInUse, 88, "backend truth wins when provided")
    }
}
