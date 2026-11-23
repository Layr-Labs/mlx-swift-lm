// Copyright © 2026 Eigen Labs.

import CoreImage
import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {

    /// Geometry, configuration, placeholder and message rules of the Inkling 2
    /// image path. The expected values are exact.
    @Suite
    struct Inkling2ImageLayoutTests {
        /// `vision_config` of `thinkingmachines/Inkling-2-Mini` @ 6e7c8248.
        static let checkpointJSON = """
            {"model_type": "inkling2_vision", "text_hidden_size": 2304,
             "temporal_patch_size": 2, "native_patch_size": 20, "adapter_merge_size": 2,
             "projection_hidden_dim": 4096, "projector_bias": false, "hidden_size": 1024,
             "num_hidden_layers": 23, "num_attention_heads": 16, "mlp_ratio": 4.0,
             "posemb_grid_size": 32, "layer_norm_eps": 1e-05}
            """

        @Test func decodesTheCheckpointVisionConfig() throws {
            let config = try JSONDecoder().decode(
                Inkling2VisionConfiguration.self, from: Data(Self.checkpointJSON.utf8))
            try config.validate()
            #expect(config == Inkling2VisionConfiguration())
            #expect(config.headDim == 64)
            #expect(config.intermediateSize == 4096)
            #expect(config.tokenSide == 40)
            #expect(config.patchFeatures == 1200)
            #expect(config.ropeTheta == 10_000)
        }

        @Test(arguments: [
            (40, 40, 1, 1), (41, 40, 1, 2), (17, 23, 1, 1), (130, 75, 2, 4),
            (4096, 4096, 103, 103), (517, 333, 9, 13),
        ])
        func tokenGrid(width: Int, height: Int, rows: Int, columns: Int) {
            let layout = Inkling2ImageLayout(width: width, height: height)
            #expect(layout.rows == rows && layout.columns == columns)
            #expect(layout.tokenCount == rows * columns)
            #expect(layout.grid.product == rows * columns)
        }

        @Test func sidesUpTo4096KeepTheirPixels() {
            #expect(Inkling2ImageLayout.processedSize(width: 4096, height: 4096) == (4096, 4096))
            #expect(Inkling2ImageLayout.processedSize(width: 1, height: 4096) == (1, 4096))
            #expect(Inkling2ImageLayout.processedSize(width: 39, height: 3) == (39, 3))
        }

        @Test func longerSidesClampTo4096() {
            #expect(Inkling2ImageLayout.processedSize(width: 5000, height: 300) == (4096, 246))
            #expect(Inkling2ImageLayout.processedSize(width: 300, height: 8192) == (150, 4096))
            #expect(Inkling2ImageLayout.processedSize(width: 9000, height: 9000) == (4096, 4096))
            #expect(Inkling2ImageLayout.processedSize(width: 100_000, height: 2) == (4096, 1))
            let clamped = Inkling2ImageLayout(width: 4096, height: 246)
            #expect(clamped.tokenCount == 103 * 7)
        }

        @Test func callerBoundScalesDown() {
            let bound = (width: 512, height: 512)
            #expect(
                Inkling2ImageLayout.processedSize(width: 1024, height: 768, bound: bound) == (
                    512, 384
                ))
            #expect(
                Inkling2ImageLayout.processedSize(width: 200, height: 100, bound: bound) == (
                    200, 100
                ))
        }

        @Test func placeholdersExpandPerImage() throws {
            let tokens = [1, 200_005, 200_054, 200_010, 7, 200_005, 200_054, 200_010]
            let expanded = try Inkling2Processor.expandPlaceholders(
                tokens, imageTokenId: 200_054, tokenCounts: [3, 1])
            #expect(
                expanded == [
                    1, 200_005, 200_054, 200_054, 200_054, 200_010, 7, 200_005, 200_054, 200_010,
                ])
            #expect(throws: VLMError.self) {
                try Inkling2Processor.expandPlaceholders(
                    tokens, imageTokenId: 200_054, tokenCounts: [3])
            }
            #expect(throws: VLMError.self) {
                try Inkling2Processor.expandPlaceholders(
                    tokens, imageTokenId: 200_054, tokenCounts: [3, 1, 2])
            }
        }

        @Test func imageOnlyTurnsRenderNoEmptyText() {
            var message = Chat.Message.user("", images: [.ciImage(CIImage.empty())])
            var rendered = Inkling2MessageGenerator().generate(message: message)
            #expect((rendered["content"] as? [[String: String]]) == [["type": "image"]])
            message.content = "What is this?"
            rendered = Inkling2MessageGenerator().generate(message: message)
            #expect(
                (rendered["content"] as? [[String: String]])
                    == [["type": "image"], ["type": "text", "text": "What is this?"]])
            rendered = Inkling2MessageGenerator().generate(message: .user("hi"))
            #expect(rendered["content"] as? String == "hi")
        }

        @Test func decodesTheCheckpointConfig() throws {
            let configuration = try SyntheticModel.configuration(
                Inkling2VLMConfiguration.self, Inkling2TinyVLM.configuration)
            try configuration.validateModelConfiguration()
            #expect(configuration.model.imageTokenId == Inkling2TinyVLM.imageTokenId)
            #expect(configuration.vision?.textHiddenSize == 128)
            var mismatched = configuration
            mismatched.vision?.textHiddenSize = 64
            #expect(throws: VLMError.self) { try mismatched.validateModelConfiguration() }
        }
    }
}
