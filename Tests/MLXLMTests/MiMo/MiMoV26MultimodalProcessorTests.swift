import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class MiMoV26MultimodalProcessorTests: XCTestCase {
    private func prepared(_ f: MiMoMediaFixture.Models, _ content: [MiMoV26MultimodalContent])
        throws -> MiMoV26PreparedMultimodal
    {
        let plan = try f.processor.plan(MiMoMediaFixture.request(content))
        return try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
    }
    func testActualVisionFeaturesArePairedInPromptOrderIncludingOddVideo() throws {
        let f = try MiMoMediaFixture.models()
        let a = MiMoMediaFixture.image(40)
        let b = MiMoMediaFixture.image(90)
        let video = MiMoV26SilentVideo(frames: [a, b, a], timestamps: [0, 1, 2])
        let value = try prepared(f, [.image(b), .text("between"), .silentVideo(video), .image(a)])
        let request = try value.makeRequest(binding: f.binding, id: .init(1))
        XCTAssertFalse(request.prefixCacheEnabled)
        XCTAssertNil(request.positionState)
        XCTAssertNil(request.hybridPrefixIdentity)
        XCTAssertEqual(request.multimodal?.attention, .causal)
        let arrays = try XCTUnwrap(request.multimodal).embeddings()
        XCTAssertEqual(arrays.count, 4)
        var expected: [MLXArray] = []
        for content in [MiMoV26MultimodalContent.image(b), .silentVideo(video), .image(a)] {
            let pixels: MiMoV26Pixels.Prepared
            switch content {
            case .image(let frame):
                pixels = try MiMoV26Pixels.image(
                    frame, settings: f.processor.profile.settings,
                    limits: MiMoMediaFixture.limits.pixels)
            case .silentVideo(let clip):
                pixels = try MiMoV26Pixels.video(
                    frames: clip.frames, sampledFrameCount: clip.frames.count,
                    settings: f.processor.profile.settings, limits: MiMoMediaFixture.limits.pixels)
            default: throw MiMoV26MultimodalError.incompatiblePlan
            }
            let g = pixels.geometry
            let feature = try f.vision.forward(
                patches: MLXArray(pixels.patchValues, [g.patchCount, g.patchVectorSize]),
                grids: [.init(temporal: g.gridT, height: g.gridH, width: g.gridW)],
                limits: MiMoMediaFixture.limits.vision)
            let per = g.gridH * g.gridW / 4
            for t in 0 ..< g.gridT { expected.append(feature[(t * per) ..< ((t + 1) * per), 0...]) }
        }
        for (a, b) in zip(arrays, expected) {
            eval(a, b)
            XCTAssertEqual(a.shape, b.shape)
            XCTAssertLessThanOrEqual(abs(a - b).max().item(Float.self), 1e-6)
        }
        XCTAssertThrowsError(try XCTUnwrap(request.multimodal).embeddings())
    }
    func testForeignPlanAndBindingRefuseBeforeAuthorizationOrConsumption() throws {
        let a = try MiMoMediaFixture.models()
        let b = try MiMoMediaFixture.models()
        let plan = try a.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        var authorizations = 0
        XCTAssertThrowsError(
            try b.processor.prepare(
                plan,
                authorize: {
                    authorizations += 1
                    return MiMoMediaFixture.Reservation($0)
                }))
        XCTAssertEqual(authorizations, 0)
        let value = try a.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        XCTAssertThrowsError(try value.makeRequest(binding: b.binding, id: .init(2)))
        let request = try value.makeRequest(binding: a.binding, id: .init(3))
        XCTAssertNotNil(request.multimodal)
        XCTAssertThrowsError(try value.makeRequest(binding: a.binding, id: .init(4)))
    }
    func testGenerationInvalidationRejectsPlanPreparedAndQueuedHandoffs() throws {
        let f = try MiMoMediaFixture.models()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        let first = try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let second = try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let queued = try second.makeRequest(binding: f.binding, id: .init(5))
        f.generation.invalidate()
        var calls = 0
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: {
                    calls += 1
                    return MiMoMediaFixture.Reservation($0)
                }))
        XCTAssertEqual(calls, 0)
        XCTAssertThrowsError(try first.makeRequest(binding: f.binding, id: .init(6)))
        XCTAssertThrowsError(try XCTUnwrap(queued.multimodal).embeddings())
    }
    func testMandatoryReservationRejectionAndCancellationRetainThenReleaseOwner() throws {
        let f = try MiMoMediaFixture.models()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        var calls = 0
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: {
                    calls += 1
                    return MiMoMediaFixture.Reservation($0)
                }, isCancelled: { true }))
        XCTAssertEqual(calls, 0)
        weak var observed: MiMoMediaFixture.Reservation?
        var cancelled = false
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: { value in
                    let reservation = MiMoMediaFixture.Reservation(value)
                    observed = reservation
                    cancelled = true
                    return reservation
                }, isCancelled: { cancelled }))
        XCTAssertNil(observed, "preparation failure drained and released the only work owner")
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: { value in
                    let reservation = MiMoMediaFixture.Reservation(value)
                    reservation.revoked = true
                    return reservation
                }))
    }
    func testPreparedAndRequestKeepWorkOwnerWithoutArraySnapshots() throws {
        let f = try MiMoMediaFixture.models()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        weak var observed: MiMoMediaFixture.Reservation?
        var prepared: MiMoV26PreparedMultimodal? = try f.processor.prepare(
            plan,
            authorize: { value in
                let reservation = MiMoMediaFixture.Reservation(value)
                observed = reservation
                return reservation
            })
        var request: CBv2Request? = try prepared!.makeRequest(binding: f.binding, id: .init(7))
        prepared = nil
        XCTAssertNotNil(observed)
        _ = try request!.multimodal!.embeddings()
        XCTAssertNotNil(observed)
        request = nil
        XCTAssertNil(observed)
    }
    func testUnloadedOrWrongPrecisionTowerIsNotSilentlyAccepted() throws {
        let f = try MiMoMediaFixture.models("bfloat16")
        try f.vision.update(parameters: f.vision.parameters(), verify: .all)
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        XCTAssertThrowsError(
            try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) }))
        let source = try MiMoV26VisionTower.expectedTensorShapes(
            configuration: f.vision.configuration
        )
        .mapValues { MiMoMediaFixture.values("wrong-type", $0, .float32) }
        try f.vision.loadNativeWeights(source, expectedDType: .float32)
        XCTAssertThrowsError(
            try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        ) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .invalidFeatures)
        }
    }
    func testFailedMediaDrainKeepsActualRootSourceAndLeaseUntilRecovery() throws {
        let f = try MiMoMediaFixture.models()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        enum Failure: Error { case injected }
        var quarantined: MiMoV26FailedMediaWork?
        var lease: MiMoMediaFixture.Reservation? = .init(plan)
        weak var weakLease = lease
        lease!.quarantine = { quarantined = $0 }
        var source: NSObject? = NSObject()
        weak var weakSource = source
        var array: MLXArray? = MLXArray([Float(1), 2])
        weak var weakArray = array
        var work: MiMoV26FailedMediaWork? = .init(
            plan: plan, owner: source!, codec: nil, reservation: lease!)
        work!.track([array!])
        array = nil
        source = nil
        lease = nil
        XCTAssertThrowsError(
            try work!.drain(generation: f.generation, synchronize: { throw Failure.injected })
        ) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .drainFailed)
        }
        work = nil
        XCTAssertNotNil(weakArray)
        XCTAssertNotNil(weakSource)
        XCTAssertNotNil(weakLease)
        XCTAssertEqual(quarantined?.retainedRootCount, 1)
        XCTAssertThrowsError(try f.generation.validate())
        XCTAssertThrowsError(
            try quarantined!.verifyRecoveryDrain(synchronize: { throw Failure.injected }))
        XCTAssertNotNil(weakArray)
        XCTAssertNotNil(weakSource)
        XCTAssertNotNil(weakLease)
        try quarantined!.recoverAndReleaseAfterDrain()
        XCTAssertNil(weakArray)
        XCTAssertNil(weakSource)
        XCTAssertNil(weakLease)
        XCTAssertThrowsError(try quarantined!.recoverAndReleaseAfterDrain())
        quarantined = nil
    }
}
