import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXVLM

private final class MiMoAudioWorkLog {
    var acquired = 0
    var released = 0
    var failed: MiMoV26FailedAudioWork?
}
private final class MiMoAudioTestWorkOwner: MiMoV26AudioWorkReservation {
    let log: MiMoAudioWorkLog
    let plan: MiMoV26AudioInputPlan
    let sourceIdentity: String
    let generation: UUID
    init(_ log: MiMoAudioWorkLog, plan: MiMoV26AudioInputPlan, weights: MiMoV26AudioInputWeights) {
        self.log = log; self.plan = plan; sourceIdentity = weights.sourceIdentity; generation = weights.generation
        log.acquired += 1
    }
    func validate(plan: MiMoV26AudioInputPlan, sourceIdentity: String, generation: UUID) throws {
        guard plan == self.plan, sourceIdentity == self.sourceIdentity, generation == self.generation else {
            throw MiMoV26AudioInputError.input("test reservation binding")
        }
    }
    func retainAfterFailedDrain(_ work: MiMoV26FailedAudioWork) { log.failed = work }
    deinit { log.released += 1 }
}

final class MiMoV26AudioInputTests: XCTestCase {
    private func bundle() throws -> MiMoV26AudioInputWeights {
        let oracle = try mimoAudioCodecOracle(), c = try MiMoV26AudioInputConfiguration.fixture()
        return try MiMoV26AudioTokenizerWeights.fixtureBundle(configuration:c,weights:oracle.weights.mapValues { $0.array() })
    }
    private func clip(_ name: String, frames: Int, amplitude: Float = 0.25) throws -> MiMoV26DecodedPCM {
        let samples = (0..<frames).map { amplitude*Float(sin(Double($0)*0.1)) }
        return try .init(samples:samples,descriptor:.init(sourceIdentity:name,channels:1,frameCount:frames,sampleRate:24000))
    }

    func testPCMCompositionPreservesItemOrderAndRetiresWorkOwner() throws {
        try mimoAudioInputRequireNative()
        let weights = try bundle(), input = MiMoV26AudioInput(weights:weights), log = MiMoAudioWorkLog()
        let clips = [try clip("first",frames:481),try clip("second",frames:1200)]
        var admitted: MiMoV26AudioInputPlan?
        let codes = try input.encode(clips:clips,limits:mimoAudioInputTestLimits(),retaining:input,authorize:{ plan in
            admitted = plan;return MiMoAudioTestWorkOwner(log,plan:plan,weights:weights)
        })
        let plan = try XCTUnwrap(admitted)
        XCTAssertEqual(plan.sourceIdentities,["first","second"])
        XCTAssertEqual(codes.map(\.frameCount),plan.codeFrameCounts)
        XCTAssertEqual(codes.count,2)
        for code in codes {
            XCTAssertEqual(code.codes.count,code.frameCount*weights.configuration.quantizers)
            XCTAssertTrue(code.codes.allSatisfy { $0 >= 0 && $0 < 4 })
        }
        XCTAssertEqual(log.acquired,1);XCTAssertEqual(log.released,1)
    }

    func testAuthorizationRefusalHappensBeforeFrontendNativeArrays() throws {
        try mimoAudioInputRequireNative()
        let input = MiMoV26AudioInput(weights:try bundle()), clips = [try clip("denied",frames:481)]
        StreamOrDevice.default.stream.synchronize()
        let resources = Memory.numResources
        XCTAssertThrowsError(try input.encode(clips:clips,limits:mimoAudioInputTestLimits(),retaining:input,authorize:{ _ in
            throw MiMoV26AudioInputError.limit("intentional real-owner refusal")
        }))
        XCTAssertEqual(Memory.numResources,resources)
    }

    func testCancellationAfterFrontendDrainReturnsNoPartialCodesAndCanRetry() throws {
        try mimoAudioInputRequireNative()
        let input = MiMoV26AudioInput(weights:try bundle()), clips = [try clip("cancel",frames:1200)]
        let log = MiMoAudioWorkLog();var polls = 0
        XCTAssertThrowsError(try input.encode(clips:clips,limits:mimoAudioInputTestLimits(),retaining:input,authorize:{ plan in
            MiMoAudioTestWorkOwner(log,plan:plan,weights:input.weights)
        },isCancelled:{ polls += 1;return polls > 8 })) {
            XCTAssertEqual($0 as? MiMoV26AudioInputError,.cancelled)
        }
        XCTAssertEqual(log.acquired,1);XCTAssertEqual(log.released,1)
        let retry = try input.encode(clips:clips,limits:mimoAudioInputTestLimits(),retaining:input,
            authorize:{ MiMoAudioTestWorkOwner(log,plan:$0,weights:input.weights) })
        XCTAssertEqual(retry.count,1);XCTAssertEqual(log.acquired,2);XCTAssertEqual(log.released,2)
    }

    func testSupportedNetworkReloadInvalidatesPreviouslyComposedWeights() throws {
        try mimoAudioInputRequireNative()
        let weights = try bundle(), input = MiMoV26AudioInput(weights:weights), oracle = try mimoAudioCodecOracle()
        let old = weights.generation
        try weights.encoder.loadNativeNetworkWeights(oracle.weights.filter { !$0.key.hasPrefix("encoder.quantizer.") }.mapValues { $0.array() })
        XCTAssertNotEqual(weights.encoder.loadedGeneration,old)
        var authorizations = 0
        XCTAssertThrowsError(try input.encode(clips:[try clip("stale",frames:481)],limits:mimoAudioInputTestLimits(),retaining:input,authorize:{ plan in
            authorizations += 1;return MiMoAudioTestWorkOwner(MiMoAudioWorkLog(),plan:plan,weights:weights)
        })) { XCTAssertEqual($0 as? MiMoV26AudioInputError,.weightsNotLoaded) }
        XCTAssertEqual(authorizations,0)
    }

    func testFailedDrainRetainsOriginalRootsLoadedOwnerAndLeaseUntilRecovery() throws {
        try mimoAudioInputRequireNative()
        enum Failure: Error { case injected }
        let weights = try bundle(), clips = [try clip("quarantine",frames:481)]
        let plan = try MiMoV26AudioInputPlan.make(clips:clips.map(\.descriptor),configuration:weights.configuration,limits:mimoAudioInputTestLimits())
        let log = MiMoAudioWorkLog()
        var source: NSObject? = NSObject(); weak var weakSource = source
        var root: MLXArray? = MLXArray([Float(1),2]); weak var weakRoot = root
        var lease: MiMoAudioTestWorkOwner? = .init(log,plan:plan,weights:weights)
        var work: MiMoV26FailedAudioWork? = try .init(weights:weights,clips:clips,plan:plan,sourceOwner:source!,reservation:lease!)
        work!.track([root!]); source = nil; root = nil; lease = nil
        var invalidated = false
        XCTAssertThrowsError(try work!.drain(invalidate:{ invalidated = true },synchronize:{ throw Failure.injected })) {
            XCTAssertEqual($0 as? MiMoV26AudioWorkFailure,.drainFailed)
        }
        work = nil
        XCTAssertTrue(invalidated); XCTAssertEqual(log.failed?.retainedRootCount,1)
        XCTAssertNotNil(weakSource); XCTAssertNotNil(weakRoot); XCTAssertEqual(log.released,0)
        XCTAssertThrowsError(try log.failed!.verifyRecoveryDrain(synchronize:{ throw Failure.injected }))
        XCTAssertNotNil(weakSource); XCTAssertNotNil(weakRoot); XCTAssertEqual(log.released,0)
        try log.failed!.recoverAndReleaseAfterDrain()
        XCTAssertNil(weakSource); XCTAssertNil(weakRoot); XCTAssertEqual(log.released,1)
        XCTAssertThrowsError(try log.failed!.recoverAndReleaseAfterDrain())
        XCTAssertEqual(log.released,1); log.failed = nil
    }

    func testAudioWorkReservationBindsFullPlanSourceAndGeneration() throws {
        try mimoAudioInputRequireNative()
        let weights = try bundle(), clips = [try clip("bound",frames:481)]
        let plan = try MiMoV26AudioInputPlan.make(clips:clips.map(\.descriptor),configuration:weights.configuration,limits:mimoAudioInputTestLimits())
        let lease = MiMoAudioTestWorkOwner(MiMoAudioWorkLog(),plan:plan,weights:weights)
        XCTAssertThrowsError(try lease.validate(plan:plan,sourceIdentity:"foreign",generation:weights.generation))
        XCTAssertThrowsError(try lease.validate(plan:plan,sourceIdentity:weights.sourceIdentity,generation:UUID()))
        try lease.validate(plan:plan,sourceIdentity:weights.sourceIdentity,generation:weights.generation)
    }
}
