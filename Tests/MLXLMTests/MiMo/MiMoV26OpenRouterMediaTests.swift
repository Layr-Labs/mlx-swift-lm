import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Real encoded OpenRouter fixtures and production geometry; no language-model
/// weights or downloads. Native tower equivalence uses a separate tiny model.
final class MiMoV26OpenRouterMediaTests: XCTestCase {
    private func asset(_ name: String) throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "MiMoOpenRouter")
        )
    }
    private func configuration() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(
            MiMoV26Configuration.self, from: Data(contentsOf: asset("config.json")))
    }
    private func settings(_ c: MiMoV26Configuration) throws -> MiMoV26MediaGeometry.Settings {
        let p = try XCTUnwrap(c.processorFields)
        func n(_ key: String) throws -> Int { try MiMoV26MultimodalProfile.integer(p, key) }
        return try .init(
            patchSize: n("patch_size"), mergeSize: n("merge_size"),
            temporalPatchSize: n("temporal_patch_size"),
            temporalCompressionRatio: n("temporal_compression_ratio"),
            imageMinPixels: n("image_min_pixels"), imageMaxPixels: n("image_max_pixels"),
            videoMinPixels: n("video_min_pixels"), videoMaxPixels: n("video_max_pixels"),
            videoTotalMaxPixels: n("video_total_max_pixels"))
    }
    private let limits = MiMoV26EncodedVisualDecoder.Limits(
        maximumPixels: 8_388_608, maximumWorkingBytes: 1 << 30,
        maximumSourceFrames: 360_000, maximumSampledFrames: 3600,
        maximumEncodedBytes: 32 << 20)

    func testCapturedImagesDecodeAndFitBoundedVisionWork() throws {
        let c = try configuration()
        let vision = try XCTUnwrap(c.vision)
        let fixtures = [
            (
                "input-image-url.jpg", 1280, 851,
                "757af6887ac1fabc5968818cbedf4adcf13018c727abe4572acc2a4df465f825"
            ),
            (
                "input-image-base64.jpg", 640, 461,
                "d6a32382d2d500fdec576ebc070c416e783acd896a2cd94b9571052826201d73"
            ),
        ]
        for (name, width, height, digest) in fixtures {
            let data = try Data(contentsOf: asset(name))
            XCTAssertEqual(
                SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), digest)
            let rgb = try MiMoV26EncodedVisualDecoder.image(data, limits: limits)
            XCTAssertEqual(rgb.width, width)
            XCTAssertEqual(rgb.height, height)
            let geometry = try MiMoV26MediaGeometry.image(
                height: rgb.height, width: rgb.width, settings: settings(c))
            let working = try MiMoV26VisionWorkingSet.frameBytes(geometry, configuration: vision)
            XCTAssertLessThan(working, 12 << 30, name)
            XCTAssertGreaterThan(working, 0)
            if width == 1280 {
                XCTAssertEqual(geometry.patchCount, 4320)
                // The shipped whole-graph quote exceeded 340 GiB before
                // retained pixels/features. The new path completes each block.
                let oldScores = geometry.patchCount * geometry.patchCount * vision.queryHeads * 16
                let oldActivations =
                    geometry.patchCount * (vision.hiddenSize * 64 + vision.intermediateSize * 16)
                    * 4
                XCTAssertGreaterThan((oldScores + oldActivations) * (vision.depth + 2), 340 << 30)
            }
        }
    }

    func testCapturedVideosDecodeAudioAndKeepVisionPeakIndependentOfFrameCount() async throws {
        let c = try configuration()
        let vision = try XCTUnwrap(c.vision)
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(configuration: c)
        for (name, width, height, count, hasAudio) in [
            ("input-video-url.mp4", 1280, 720, 8, true),
            ("input-video-base64.mov", 1490, 534, 10, false),
        ] {
            let owner = try MemoryBackedVideoAsset(videoData: Data(contentsOf: asset(name)))
            let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
                owner, sampling: sampling, limits: limits)
            XCTAssertEqual(plan.hasAudioTrack, hasAudio)
            XCTAssertEqual(plan.sampledIndices.count, count)
            let frames: [MiMoV26Pixels.DecodedRGB]
            var audioBytes = 0
            if hasAudio {
                let audioLimits = MiMoV26EncodedAudiovisualDecoder.Limits(
                    maximumFrames: 1_000_000, maximumWorkingBytes: 1 << 30)
                let audiovisual = try await MiMoV26EncodedAudiovisualDecoder.inspect(
                    plan, limits: audioLimits)
                let decoded = try await MiMoV26EncodedAudiovisualDecoder.decode(
                    audiovisual, videoLimits: limits, audioLimits: audioLimits)
                frames = decoded.frames
                // Encoded ingress preserves the original PCM format. The
                // native audio frontend owns resampling/downmix semantics.
                XCTAssertEqual(decoded.wholeAudio.descriptor.sampleRate, 32000)
                XCTAssertEqual(decoded.wholeAudio.descriptor.channels, 2)
                XCTAssertFalse(decoded.wholeAudio.samples.isEmpty)
                let codec = try MiMoV26AudioInputConfiguration(
                    sidecarJSON: Data(contentsOf: asset("audio-config.json")), mainConfiguration: c)
                let inputLimits = MiMoV26AudioInputLimits(
                    maximumClips: 24, maximumChannels: 8, maximumSampleRate: 192000,
                    maximumInputSamples: Int(Int32.max), maximumResampledSamples: Int(Int32.max),
                    maximumResampleCoefficients: Int(Int32.max), maximumMelFrames: Int(Int32.max),
                    maximumSegments: 1_000_000, maximumPaddedMelFrames: Int(Int32.max),
                    maximumWorkingElements: (230 << 30) / 4,
                    frontendFrameBlockSize: 6000, rvqTileFrames: 1500)
                let input = try MiMoV26AudioInputPlan.make(
                    clips: [decoded.wholeAudio.descriptor], configuration: codec,
                    limits: inputLimits)
                XCTAssertEqual(input.configuration.sampleRate, 24000)
                audioBytes = try MiMoV26ManagedAudioCommitment.additionalBytes(
                    input: input, patchConfiguration: XCTUnwrap(c.audio),
                    limits: .init(
                        maximumClips: 24, maximumFrames: Int(Int32.max),
                        maximumPatches: c.maxPositionEmbeddings,
                        maximumWorkingElements: (230 << 30) / 4))
            } else {
                frames = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: limits)
                    .frames
            }
            XCTAssertEqual(frames.count, count)
            XCTAssertEqual(frames.first?.width, width)
            XCTAssertEqual(frames.first?.height, height)
            let geometry = try MiMoV26MediaGeometry.video(
                height: height, width: width, sampledFrames: count, settings: settings(c))
            let longer = try MiMoV26MediaGeometry.video(
                height: height, width: width, sampledFrames: count * 2, settings: settings(c))
            let working = try MiMoV26VisionWorkingSet.frameBytes(geometry, configuration: vision)
            XCTAssertLessThan(working, 12 << 30, name)
            XCTAssertLessThan(
                working + audioBytes, 48 << 30, "vision and actual AAC audio reservation: \(name)")
            XCTAssertEqual(
                working, try MiMoV26VisionWorkingSet.frameBytes(longer, configuration: vision))
            // Retained decoded frames still grow with the clip; only the
            // completed frame/block graph is reused by the bounded path.
            XCTAssertGreaterThan(longer.patchElementCount, geometry.patchElementCount)
        }
    }

    func testBoundedTowerMatchesLazyTowerAndRecoversAfterCheckpointRefusal() throws {
        var fields = try JSONDecoder().decode(
            [String: MiMoV26JSONValue].self, from: Data(contentsOf: asset("config.json")))
        guard case .object(var v) = fields["vision_config"] else {
            return XCTFail("missing vision config")
        }
        for (key, value) in [
            "hidden_size": 8, "intermediate_size": 16, "num_heads": 2,
            "num_key_value_heads": 1, "num_query_groups": 2, "depth": 4,
            "patch_size": 2, "spatial_patch_size": 2, "qk_channels": 8, "kv_channels": 8,
        ] {
            v[key] = .number(Decimal(value))
        }
        v["fullatt_block_indexes"] = .array([.number(0)])
        v["vit_window_attn_types"] = .array([.number(-1), .number(0), .number(1), .number(0)])
        fields["vision_config"] = .object(v)
        let c = try MiMoV26Configuration(rawFields: fields)
        let vision = try XCTUnwrap(c.vision)
        let tower = try MiMoV26VisionTower(configuration: vision)
        let shapes = try MiMoV26VisionTower.expectedTensorShapes(configuration: vision)
        try tower.loadNativeWeights(
            shapes.mapValues { MiMoMediaFixture.values("vision", $0, .float32) },
            expectedDType: .float32)
        let grids = [
            MiMoV26VisionGrid(temporal: 2, height: 12, width: 12),
            .init(temporal: 1, height: 4, width: 6),
        ]
        let patches = MLXArray((0 ..< (312 * 24)).map { Float($0 % 17) / 17 }, [312, 24])
        let towerLimits = MiMoV26VisionLimits(
            maximumPatches: 1024, maximumAttentionScoreElements: 1_000_000)
        let expected = try tower.forward(patches: patches, grids: grids, limits: towerLimits)
        try withError { eval(expected) }
        enum Refusal: Error { case stop }
        var calls = 0
        XCTAssertThrowsError(
            try tower.forwardBounded(patches: patches, grids: grids, limits: towerLimits) { roots in
                calls += 1
                if calls == 2 { throw Refusal.stop }
                try withError { eval(roots) }
            })
        calls = 0
        let actual = try tower.forwardBounded(patches: patches, grids: grids, limits: towerLimits) {
            roots in
            calls += 1
            try withError { eval(roots) }
        }
        XCTAssertEqual(calls, 3 * (vision.depth + 2) + 1)
        XCTAssertEqual(actual.shape, expected.shape)
        let difference = abs(actual - expected).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-4)
    }
}
