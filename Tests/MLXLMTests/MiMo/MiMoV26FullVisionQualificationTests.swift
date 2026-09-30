import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Explicit local qualification with the artifact's actual vision weights.
/// No language-model weights, generation, network or automatic downloads.
final class MiMoV26FullVisionQualificationTests: XCTestCase {
    func testCapturedMediaWithActualVisionWeights() async throws {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_FULL_VISION_WEIGHTS"] else {
            throw XCTSkip(
                "Set MIMO_V26_FULL_VISION_WEIGHTS to the verified 364-tensor vision artifact")
        }
        func asset(_ name: String) throws -> URL {
            try XCTUnwrap(
                Bundle.module.url(
                    forResource: name, withExtension: nil, subdirectory: "MiMoOpenRouter"))
        }
        let config = try JSONDecoder().decode(
            MiMoV26Configuration.self, from: Data(contentsOf: asset("config.json")))
        let vision = try XCTUnwrap(config.vision)
        let p = try XCTUnwrap(config.processorFields)
        func n(_ key: String) throws -> Int { try MiMoV26MultimodalProfile.integer(p, key) }
        let settings = try MiMoV26MediaGeometry.Settings(
            patchSize: n("patch_size"), mergeSize: n("merge_size"),
            temporalPatchSize: n("temporal_patch_size"),
            temporalCompressionRatio: n("temporal_compression_ratio"),
            imageMinPixels: n("image_min_pixels"),
            imageMaxPixels: n("image_max_pixels"), videoMinPixels: n("video_min_pixels"),
            videoMaxPixels: n("video_max_pixels"), videoTotalMaxPixels: n("video_total_max_pixels"))
        let tower = try MiMoV26VisionTower(configuration: vision)
        let weights = try loadArrays(url: URL(fileURLWithPath: path))
        XCTAssertEqual(weights.count, 364)
        try tower.loadNativeWeights(weights, expectedDType: .bfloat16, flattenedPatchStorage: true)
        try withError { eval(Array(weights.values)) }
        let limits = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 8_388_608, maximumWorkingBytes: 1 << 30,
            maximumSourceFrames: 360_000, maximumSampledFrames: 3600, maximumEncodedBytes: 32 << 20)
        let pixels = MiMoV26Pixels.Limits(
            maximumInputElements: 100_000_000,
            maximumOutputElements: 100_000_000, maximumWorkingBytes: 1 << 30)
        let towerLimits = MiMoV26VisionLimits(
            maximumPatches: 100_000, maximumAttentionScoreElements: 40_000_000_000)
        func run(_ prepared: MiMoV26Pixels.Prepared, name: String) throws {
            Memory.clearCache()
            let baseline = Memory.snapshot().activeMemory
            Memory.peakMemory = 0
            let geometry = prepared.geometry
            let patches = MLXArray(
                prepared.patchValues, [geometry.patchCount, geometry.patchVectorSize])
            var checkpoints = 0
            let output = try tower.forwardBounded(
                patches: patches,
                grids: [
                    .init(temporal: geometry.gridT, height: geometry.gridH, width: geometry.gridW)
                ],
                limits: towerLimits
            ) { roots in
                checkpoints += 1
                try withError { eval(roots) }
            }
            XCTAssertEqual(output.shape, [geometry.mediaTokens, vision.outputHiddenSize])
            let finite = all(isFinite(output))
            try withError { eval(finite) }
            XCTAssertTrue(finite.item(Bool.self))
            XCTAssertEqual(checkpoints, geometry.gridT * (vision.depth + 2) + 1)
            let snapshot = Memory.snapshot()
            let quote = try MiMoV26VisionWorkingSet.frameBytes(geometry, configuration: vision)
            let retained =
                geometry.patchElementCount * 4 + geometry.mediaTokens * vision.outputHiddenSize * 8
            let incremental = max(0, snapshot.peakMemory - baseline)
            XCTAssertLessThanOrEqual(incremental, quote + retained, name)
            let report: [String: Any] = [
                "fixture": name, "media_tokens": geometry.mediaTokens,
                "frame_working_quote_bytes": quote,
                "baseline_bytes": baseline, "incremental_peak_bytes": incremental,
                "retained_patch_feature_bytes": retained,
                "active_bytes": snapshot.activeMemory, "process_peak_bytes": snapshot.peakMemory,
                "checkpoints": checkpoints,
            ]
            print(
                String(
                    decoding: try JSONSerialization.data(
                        withJSONObject: report, options: [.sortedKeys]), as: UTF8.self))
        }
        for name in ["input-image-url.jpg", "input-image-base64.jpg"] {
            let rgb = try MiMoV26EncodedVisualDecoder.image(
                Data(contentsOf: asset(name)), limits: limits)
            try run(MiMoV26Pixels.image(rgb, settings: settings, limits: pixels), name: name)
        }
        for name in ["input-video-url.mp4", "input-video-base64.mov"] {
            let owner = try MemoryBackedVideoAsset(videoData: Data(contentsOf: asset(name)))
            let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
                owner, sampling: .init(configuration: config), limits: limits)
            // This test qualifies only the vision tower. The mandatory fixture
            // test separately checks the MP4's actual AAC decode and audio quote.
            let decoded =
                plan.hasAudioTrack
                ? try await MiMoV26EncodedVisualDecoder.audiovisualFrames(plan, limits: limits)
                : try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: limits)
            try run(
                MiMoV26Pixels.video(
                    frames: decoded.frames, sampledFrameCount: decoded.frames.count,
                    settings: settings, limits: pixels), name: name)
        }
        if ProcessInfo.processInfo.environment["MIMO_V26_VISION_MEMORY_MATRIX"] == "1" {
            // Same published weights, broader processor envelope, one at a time.
            // The maximum-area cases include portrait and panoramic geometry.
            for (height, width) in [(1080, 1920), (2160, 3840), (4096, 2048), (1024, 8192)] {
                let rgb = MiMoV26Pixels.DecodedRGB(
                    height: height, width: width,
                    planarRGB: [Float](repeating: 127, count: height * width * 3))
                try run(
                    MiMoV26Pixels.image(rgb, settings: settings, limits: pixels),
                    name: "matrix-\(width)x\(height)")
            }
            let frame = MiMoV26Pixels.DecodedRGB(
                height: 480, width: 640,
                planarRGB: [Float](repeating: 127, count: 480 * 640 * 3))
            try run(
                MiMoV26Pixels.video(
                    frames: Array(repeating: frame, count: 32),
                    sampledFrameCount: 32, settings: settings, limits: pixels),
                name: "matrix-32-frames")
        }
    }
}
