import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Audio patch plan and encoder with the tiny audio geometry of
/// MiMoV26TinyCheckpoint: hidden 64, two heads of 32, one layer, MLP 128,
/// 20 channels, speech vocabulary 7, group 4. Expected plan values are
/// computed by hand from the formulas in MiMoV26AudioPatch.swift.
final class MiMoV26TinyAudioPatchTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    private let limits = MiMoV26AudioPatchLimits(
        maximumClips: 8, maximumFrames: 64, maximumPatches: 32,
        maximumWorkingElements: 1_000_000)

    private func configuration(_ change: ((inout [String: Any]) -> Void)? = nil) throws
        -> MiMoV26AudioConfiguration
    {
        var fields = Fixture.baseFields(dtype: "float32")
        if let change {
            var audio = try XCTUnwrap(fields["audio_config"] as? [String: Any])
            change(&audio)
            fields["audio_config"] = audio
        }
        return try XCTUnwrap(Fixture.configuration(fields).audio)
    }
    /// One frame is 20 channel codes; frame `f` uses code `(f + channel) % 7`.
    private func clip(frames: Int, shift: Int32 = 0) -> MiMoV26AudioCodeClip {
        .init(
            codes: (0 ..< frames).flatMap { frame in
                (0 ..< 20).map { Int32((frame + $0) % 7 + Int(shift)) % 7 }
            }, frameCount: frames)
    }
    private func encoder() throws -> MiMoV26AudioPatchEncoder {
        let c = try configuration()
        let encoder = try MiMoV26AudioPatchEncoder(configuration: c)
        try encoder.loadNativeWeights(
            MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: c).mapValues {
                MiMoMediaFixture.values("audio", $0, .float32)
            }, expectedDType: .float32)
        return encoder
    }

    func testPlanPadsEachClipByRepeatingItsLastFrame() throws {
        let c = try configuration()
        let a = clip(frames: 1)
        let b = clip(frames: 5)
        let plan = try MiMoV26AudioPatchPlan.make(clips: [a, b], configuration: c, limits: limits)
        // Clip a: one frame pads to one patch of 4; clip b: five frames, two patches.
        XCTAssertEqual(plan.patchCount, 3)
        XCTAssertEqual(plan.clipPatchRanges, [0 ..< 1, 1 ..< 3])
        XCTAssertEqual(plan.originalFrameCount, 6)
        var expected: [Int32] = []
        for _ in 0 ..< 4 { expected += a.codes }
        for frame in 0 ..< 8 {
            let source = min(frame, 4) * 20
            expected += b.codes[source ..< (source + 20)]
        }
        XCTAssertEqual(plan.groupedCodes, expected)
        // codes 12*20 + hidden 12*(2*64 + (5*64 + 3*128)) + scores 3*2*4*4*1
        // + projection 3*(1024 + 64) = 240 + 9984 + 96 + 3264.
        XCTAssertEqual(plan.workingElements, 13_584)
        XCTAssertNoThrow(
            try MiMoV26AudioPatchPlan.make(
                clips: [a, b], configuration: c,
                limits: .init(
                    maximumClips: 2, maximumFrames: 6, maximumPatches: 3,
                    maximumWorkingElements: 13_584)))
        let empty = try MiMoV26AudioPatchPlan.make(
            clips: [], configuration: c,
            limits: .init(
                maximumClips: 0, maximumFrames: 0, maximumPatches: 0, maximumWorkingElements: 0))
        XCTAssertEqual(empty.groupedCodes, [])
        XCTAssertEqual(empty.patchCount, 0)
        XCTAssertEqual(empty.workingElements, 0)
    }

    func testPlanRejectsMalformedClipsAndEachLimit() throws {
        let c = try configuration()
        let valid = clip(frames: 1)
        for (bad, reason) in [
            (MiMoV26AudioCodeClip(codes: [], frameCount: 0), "zero-frame item"),
            (.init(codes: [], frameCount: -1), "zero-frame item"),
            (.init(codes: [0], frameCount: 1), "exact frame/channel geometry"),
            (
                .init(codes: [Int32](repeating: 7, count: 20), frameCount: 1),
                "speech code outside its embedding vocabulary"
            ),
            (
                .init(codes: [Int32](repeating: -1, count: 20), frameCount: 1),
                "speech code outside its embedding vocabulary"
            ),
        ] {
            XCTAssertThrowsError(
                try MiMoV26AudioPatchPlan.make(
                    clips: [valid, bad], configuration: c, limits: limits), reason
            ) { XCTAssertEqual($0 as? MiMoV26AudioPatchError, .invalidInput(reason)) }
        }
        for (restricted, reason) in [
            (
                MiMoV26AudioPatchLimits(
                    maximumClips: 0, maximumFrames: 64, maximumPatches: 32,
                    maximumWorkingElements: 1_000_000), "clip limits"
            ),
            (
                .init(
                    maximumClips: -1, maximumFrames: 64, maximumPatches: 32,
                    maximumWorkingElements: 1_000_000), "clip limits"
            ),
            (
                .init(
                    maximumClips: 8, maximumFrames: 0, maximumPatches: 32,
                    maximumWorkingElements: 1_000_000), "frame/patch limit"
            ),
            (
                .init(
                    maximumClips: 8, maximumFrames: 64, maximumPatches: 0,
                    maximumWorkingElements: 1_000_000), "frame/patch limit"
            ),
            (
                .init(
                    maximumClips: 8, maximumFrames: 64, maximumPatches: 32,
                    maximumWorkingElements: 1), "working element limit"
            ),
        ] {
            XCTAssertThrowsError(
                try MiMoV26AudioPatchPlan.make(
                    clips: [valid], configuration: c, limits: restricted), reason
            ) { XCTAssertEqual($0 as? MiMoV26AudioPatchError, .executionLimit(reason)) }
        }
    }

    func testUnsupportedReferencePoliciesAndLargeGeometryAreRefused() throws {
        XCTAssertEqual(
            try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: configuration()).count,
            35)
        for change: (inout [String: Any]) -> Void in [
            { $0["input_full_attention"] = false }, { $0["add_post_norm"] = false },
            { $0["projection_layers"] = 1 }, { $0["partial_rotary_factor"] = 0.5 },
            { $0["input_local_hidden_dropout"] = 0.1 },
        ] {
            let c = try configuration(change)
            XCTAssertThrowsError(
                try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: c)
            ) {
                XCTAssertEqual(
                    $0 as? MiMoV26AudioPatchError,
                    .invalidConfiguration(
                        "requires full attention/full RoPE/post-norm/group4/two projections"))
            }
            XCTAssertThrowsError(try MiMoV26AudioPatchEncoder(configuration: c))
        }
        let deep = try configuration { $0["input_local_layers"] = 7 }
        XCTAssertThrowsError(try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: deep))
        {
            XCTAssertEqual(
                $0 as? MiMoV26AudioPatchError, .invalidConfiguration("dimensions or RoPE theta"))
        }
        let wide = try configuration { $0["input_local_intermediate_size"] = 4097 }
        XCTAssertThrowsError(try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: wide))
        {
            XCTAssertEqual(
                $0 as? MiMoV26AudioPatchError, .invalidConfiguration("dimensions or RoPE theta"))
        }
    }

    func testStrictLoadRejectsTypeClosureAndShapeAndResetsOnUpdate() throws {
        let c = try configuration()
        let encoder = try MiMoV26AudioPatchEncoder(configuration: c)
        XCTAssertThrowsError(try encoder.forward(clips: [clip(frames: 1)], limits: limits)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .weightsNotLoaded)
        }
        let shapes = try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: c)
        let weights = shapes.mapValues { MiMoMediaFixture.values("audio", $0, .float32) }
        XCTAssertThrowsError(try encoder.loadNativeWeights(weights, expectedDType: .int32)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .invalidWeights("unsupported dtype"))
        }
        var missing = weights
        missing.removeValue(forKey: "audio_encoder.projection.mlp.2.weight")
        XCTAssertThrowsError(try encoder.loadNativeWeights(missing, expectedDType: .float32)) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioPatchError, .invalidWeights("missing/unmapped audio tensors"))
        }
        var wrongShape = weights
        wrongShape["speech_embeddings.3.weight"] = MLXArray.zeros([8, 64])
        XCTAssertThrowsError(try encoder.loadNativeWeights(wrongShape, expectedDType: .float32)) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioPatchError, .invalidWeights("speech_embeddings.3.weight"))
        }
        XCTAssertThrowsError(try encoder.loadNativeWeights(weights, expectedDType: .bfloat16))
        XCTAssertThrowsError(try encoder.loadPackedWeights(weights, expectedDType: .float32)) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioPatchError, .invalidWeights("packed audio activation dtype"))
        }
        try encoder.loadNativeWeights(weights, expectedDType: .float32)
        let empty = try encoder.forward(clips: [], limits: limits)
        XCTAssertEqual(empty.features.shape, [0, 64])
        XCTAssertEqual(empty.features.dtype, .float32)
        XCTAssertEqual(empty.clipPatchRanges, [])
        try encoder.update(parameters: encoder.parameters(), verify: .all)
        XCTAssertThrowsError(try encoder.forward(clips: [clip(frames: 1)], limits: limits)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .weightsNotLoaded)
        }
    }

    func testForwardKeepsPatchesIndependentAndPadsWithTheLastFrame() throws {
        let encoder = try self.encoder()
        let a = clip(frames: 1)
        let b = clip(frames: 5)
        let joint = try encoder.forward(clips: [a, b], limits: limits)
        let alone = try encoder.forward(clips: [b], limits: limits)
        // Four copies of a's only frame give the same grouped codes as a.
        let repeated = MiMoV26AudioCodeClip(
            codes: Array([a.codes, a.codes, a.codes, a.codes].joined()), frameCount: 4)
        let padded = try encoder.forward(clips: [repeated], limits: limits)
        let other = try encoder.forward(clips: [clip(frames: 1, shift: 1)], limits: limits)
        eval(joint.features, alone.features, padded.features, other.features)
        XCTAssertEqual(joint.features.shape, [3, 64])
        XCTAssertEqual(joint.clipPatchRanges, [0 ..< 1, 1 ..< 3])
        XCTAssertTrue(joint.features.asArray(Float.self).allSatisfy(\.isFinite))
        // Float32. Each patch is an independent length-4 sequence; only the
        // batch size of the matrix products changes the order of sums: 1e-4.
        XCTAssertLessThanOrEqual(
            abs(joint.features[1 ..< 3, 0...] - alone.features).max().item(Float.self), 1e-4)
        XCTAssertLessThanOrEqual(
            abs(joint.features[0 ..< 1, 0...] - padded.features).max().item(Float.self), 1e-4)
        // Other codes must reach the output.
        XCTAssertGreaterThan(
            abs(padded.features - other.features).max().item(Float.self), 1e-4)
        XCTAssertThrowsError(
            try encoder.forward(
                clips: [a, b],
                limits: .init(
                    maximumClips: 1, maximumFrames: 64, maximumPatches: 32,
                    maximumWorkingElements: 1_000_000)))
    }
}
