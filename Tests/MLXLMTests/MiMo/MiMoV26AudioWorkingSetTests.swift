import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class MiMoV26AudioWorkingSetTests: XCTestCase {
    func testCompletedScratchRetiresArrayRegistrationsWithoutDroppingOwnerOrLoan() throws {
        let state = CBv2NativeShutdownState(engineID: UUID(), contractID: UUID(), supported: true)
        let owner = NSObject()
        let loan = try state.beginLoan(owner: owner)
        let work = CBv2NativeMediaPreparation(tracking: state, loan: loan)
        work.rootIDs.append(state.retain(owners: [owner]))
        for _ in 0 ..< 32 {
            let values = MLXArray.ones([16, 16]) + 1
            try work.beforeNativeWork([values])
            XCTAssertEqual(state.debugRetainedRootCount, 2)
            try work.evaluateScratchCheckpoint([values])
            XCTAssertEqual(state.debugRetainedRootCount, 1)
            XCTAssertEqual(work.rootIDs.count, 1)
            XCTAssertTrue(state.hasLoans)
        }
        try work.beforeNativeWork([MLXArray.ones([16])])
        _ = state.fail(.nativeWorkFailed)
        XCTAssertThrowsError(try work.evaluateScratchCheckpoint([MLXArray.ones([16])]))
        XCTAssertEqual(state.debugRetainedRootCount, 2)
        XCTAssertEqual(work.rootIDs.count, 2)
        XCTAssertTrue(state.hasLoans)
        XCTAssertNil(work.detachAfterCompletion())
        XCTAssertEqual(state.debugRetainedRootCount, 2)
    }

    func testBoundedEncoderAndQuantizerMatchLazyAcrossMixedClips() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let weights = try MiMoV26AudioTokenizerWeights.fixtureBundle(
            configuration: c,
            weights: MiMoV26AudioTokenizerWeights.inputSourceShapes(configuration: c).mapValues {
                shape in
                MiMoMediaFixture.values("audio", shape, .bfloat16)
            }.reduce(into: [String: MLXArray]()) { result, item in
                result[item.key] =
                    item.key.hasPrefix("encoder.quantizer.")
                    ? item.value.asType(.float32) : item.value
            })
        let clips = try [481, 1201, 2201].map { frames in
            try MiMoV26DecodedPCM(
                samples: (0 ..< frames).map { Float(sin(Double($0) * 0.02)) },
                descriptor: .init(
                    sourceIdentity: "\(frames)", channels: 1,
                    frameCount: frames, sampleRate: 24000))
        }
        let plan = try MiMoV26AudioInputPlan.make(
            clips: clips.map(\.descriptor),
            configuration: c, limits: mimoAudioInputTestLimits())
        let frontend = MiMoV26AudioFrontend(configuration: c)
        let mels = try clips.enumerated().map {
            try frontend.prepare(pcm: $0.element, plan: plan, clipIndex: $0.offset)
        }
        try withError { eval(mels.map(\.values)) }
        let lazy = try weights.encoder.encodeFeatures(mels: mels, plan: plan)
        try withError { eval(lazy.features) }
        var checkpoints = 0
        let bounded = try weights.encoder.encodeFeaturesBounded(
            mels: mels, plan: plan,
            isCancelled: { false }
        ) { roots in
            checkpoints += 1
            try withError { eval(roots) }
        }
        XCTAssertGreaterThan(checkpoints, c.layers)
        XCTAssertEqual(lazy.features.asArray(Float.self), bounded.features.asArray(Float.self))
        let expected = try weights.quantizer.quantize(
            features: lazy.features,
            frameCounts: plan.codeFrameCounts, tileFrames: 2)
        try withError { eval(expected.codes, expected.allFinite) }
        enum Stop: Error { case now }
        XCTAssertThrowsError(
            try weights.quantizer.quantizeBounded(
                features: bounded.features,
                frameCounts: plan.codeFrameCounts, tileFrames: 2,
                isCancelled: { false }, checkpoint: { _ in throw Stop.now }))
        let actual = try weights.quantizer.quantizeBounded(
            features: bounded.features,
            frameCounts: plan.codeFrameCounts, tileFrames: 2, isCancelled: { false }
        ) { roots in
            try withError { eval(roots) }
        }
        XCTAssertTrue(actual.allFinite.item(Bool.self))
        XCTAssertEqual(expected.codes.asArray(Int32.self), actual.codes.asArray(Int32.self))
    }

    func testShortAudioChargesActualTileAndOneEncoderBlock() throws {
        let root = try XCTUnwrap(
            Bundle.module.url(
                forResource: "config.json", withExtension: nil, subdirectory: "MiMoOpenRouter"))
        let main = try JSONDecoder().decode(MiMoV26Configuration.self, from: Data(contentsOf: root))
        let sidecar = root.deletingLastPathComponent().appendingPathComponent("audio-config.json")
        let c = try MiMoV26AudioInputConfiguration(
            sidecarJSON: Data(contentsOf: sidecar), mainConfiguration: main)
        func plan(_ seconds: Int, tile: Int) throws -> MiMoV26AudioInputPlan {
            try .make(
                clips: [
                    .init(
                        sourceIdentity: "wave", channels: 2,
                        frameCount: seconds * 32000, sampleRate: 32000)
                ], configuration: c,
                limits: mimoAudioInputTestLimits(block: 6000, tile: tile))
        }
        let small = try plan(5, tile: 1500)
        let same = try plan(5, tile: 300)
        let larger = try plan(30, tile: 1500)
        XCTAssertEqual(
            try MiMoV26AudioWorkingSet.bytes(small), try MiMoV26AudioWorkingSet.bytes(same))
        XCTAssertLessThan(try MiMoV26AudioWorkingSet.bytes(small), 256 << 20)
        XCTAssertGreaterThan(
            try MiMoV26AudioWorkingSet.bytes(larger), try MiMoV26AudioWorkingSet.bytes(small))
        XCTAssertLessThan(
            try MiMoV26AudioWorkingSet.bytes(small), small.workingElementUpperBound * 4)
    }
}
