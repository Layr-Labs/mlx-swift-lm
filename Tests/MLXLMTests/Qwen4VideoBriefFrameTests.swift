import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXVLM

/// Opt-in probe against the immutable synthetic 20-frame, 2-fps qualification clip.
/// Separates frame extraction from the model's ability to recognize a brief event.
final class Qwen4VideoBriefFrameTests: XCTestCase {
    func testNativeSamplerRetainsBriefBlueFrameNine() async throws {
        guard let path = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN4_VIDEO_FIXTURE"] else {
            throw XCTSkip("Set the synthetic brief-blue qualification clip path")
        }
        let config = try JSONDecoder().decode(Qwen4ExpVideoConfiguration.self, from: Data(#"""
        {"size":{"shortest_edge":4096,"longest_edge":25165824},"patch_size":16,
         "temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],
         "image_std":[0.5,0.5,0.5],"video_processor_type":"Qwen3VLVideoProcessor"}
        """#.utf8))
        let sample = try await Qwen4ExpVideoSampler.sample(.url(URL(fileURLWithPath: path)),
            config: config, process: { frame, _ in frame.toSRGB() })
        XCTAssertEqual(sample.indices, Array(0..<20))
        XCTAssertEqual(sample.sourceFPS, 2, accuracy: 0.001)
        XCTAssertEqual(sample.frames.count, 20)
        for (index, frame) in sample.frames.enumerated() {
            let channels = mean(frame, axes: [0, 2, 3])
            eval(channels)
            let rgb = channels.asArray(Float.self)
            XCTAssertEqual(rgb.count, 3)
            if index == 9 {
                XCTAssertGreaterThan(rgb[2], 0.8, "Blue frame must survive native decoding")
                XCTAssertLessThan(rgb[0], 0.1)
            } else {
                XCTAssertGreaterThan(rgb[0], 0.8)
                XCTAssertLessThan(rgb[2], 0.1)
            }
        }
    }
}
