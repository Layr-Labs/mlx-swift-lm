// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Video input for the base Gemma 4 (`gemma4`) VLM
//
// Gemma 4 has no separate video encoder: each frame runs through the same vision
// tower as images and is trimmed to the smaller per-frame video budget
// (`vision_soft_tokens_per_video_frame`, 70 for E4B) before scattering onto the
// `<video>` soft-token positions (`video_token_id`, 258884). These tests cover the
// wiring on the non-unified `Gemma4` class and its `Gemma4Processor`.

struct Gemma4VideoInputTests {

    /// The vMLX Gemma4 wrapper defaults the video placeholder token when a
    /// checkpoint omits it, while the processor owns the per-frame budget.
    @Test("Gemma4Configuration and processor decode video defaults")
    func decodesVideoConfiguration() throws {
        let withVideo = try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 258884,")
        #expect(withVideo.videoTokenId == 258884)

        let withoutVideo = try Self.decodeConfig(videoTokenLine: "")
        #expect(withoutVideo.videoTokenId == 258884)

        let processor = try Self.makeProcessorConfig()
        #expect(processor.videoMaxSoftTokens == 70)
    }

    /// With the per-frame video budget equal to the vision tower's output length,
    /// the same pixels produce identical visual features for image and video
    /// inputs. Their final logits are intentionally not compared: Gemma4's
    /// per-layer input embeddings retain the distinct image/video token ids.
    @Test("Image and video pixels share vision features without truncation")
    func imageAndVideoAgree() throws {
        let model = Gemma4(try Self.makeTinyConfig())
        eval(model)

        let imageTokenId = 100
        let videoTokenId = 101
        func prompt(_ mm: Int) -> [Int] { [5, 6, mm, mm, mm, mm, 7, 8] }
        // 8x8 with patch_size 4 → a 2x2 patch grid → 4 soft tokens per image
        // under the aspect-preserving tower (which emits patch-count tokens).
        let pixels = (MLXArray(0 ..< 192).reshaped([1, 3, 8, 8]).asType(.float32)) / 192.0

        let imageFeatures = try #require(model.perImageVisionFeatures(pixels: pixels, frames: nil).first)
        let videoFeatures = try #require(
            model.perVideoFrameVisionFeatures(pixels: pixels, frames: nil).first)
        eval(imageFeatures, videoFeatures)
        #expect(
            allClose(imageFeatures, videoFeatures, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "An equal image/video feature budget must preserve vision-tower output.")

        func lastLogits(mm: Int, asVideo: Bool) throws -> MLXArray {
            let tokens = MLXArray(prompt(mm).map { Int32($0) }).expandedDimensions(axis: 0)
            let text = LMInput.Text(tokens: tokens)
            let input =
                asVideo
                ? LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))
                : LMInput(text: text, image: LMInput.ProcessedImage(pixels: pixels))
            let result = try model.prepare(
                input, cache: try model.newCache(parameters: nil), windowSize: 1024)
            guard case .logits(let out) = result else {
                Issue.record("Expected .logits from Gemma4.prepare (multimodal branch)")
                return MLXArray(0)
            }
            let logits = out.logits[0..., -1, 0...]
            eval(logits)
            return logits
        }

        let imageLogits = try lastLogits(mm: imageTokenId, asVideo: false)
        let videoLogits = try lastLogits(mm: videoTokenId, asVideo: true)
        #expect(imageLogits.shape == [1, 200])
        #expect(videoLogits.shape == [1, 200])
    }

    /// A multi-frame video uses the same per-frame feature geometry as the
    /// processor's placeholder expansion.
    @Test("Multi-frame video scatters every frame's soft tokens")
    func videoScattersPerFrameFeatures() throws {
        let numFrames = 2
        let videoTokenId = 101
        let model = Gemma4(try Self.makeTinyConfig())
        eval(model)

        var tokens: [Int32] = [5, 6]
        tokens += Array(repeating: Int32(videoTokenId), count: numFrames * 2)
        tokens += [7]
        let text = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        // 4x8 frames with patch size 4 produce two soft tokens per frame.
        let pixels =
            (MLXArray(0 ..< (numFrames * 3 * 4 * 8)).reshaped([numFrames, 3, 4, 8])
                .asType(.float32)) / 400.0
        let input = LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))

        let result = try model.prepare(
            input, cache: try model.newCache(parameters: nil), windowSize: 1024)
        guard case .logits(let out) = result else {
            Issue.record("Expected .logits from Gemma4.prepare")
            return
        }
        eval(out.logits)
        #expect(out.logits.shape == [1, tokens.count, 200])
    }

    // MARK: - Helpers

    private static func makeTinyConfig() throws -> Gemma4Configuration {
        try decodeConfig(videoTokenLine: "\"video_token_id\": 101,")
    }

    /// Tiny `gemma4` config. `pooling_kernel_size 1` + `default_output_length 4` make the
    /// vision tower emit 4 soft tokens per frame regardless of patch count.
    private static func decodeConfig(videoTokenLine: String, extraLines: String = "") throws
        -> Gemma4Configuration
    {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              \(videoTokenLine)
              \(extraLines)
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
                "sliding_window": 16, "sliding_window_pattern": 2, "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_attention_heads": 2, "num_key_value_heads": 2,
                "head_dim": 8, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    private static func makeProcessorConfig() throws -> Gemma4ProcessorConfiguration {
        let json = """
            {
              "processor_class": "Gemma4Processor",
              "do_normalize": true,
              "image_seq_length": 280,
              "image_token_id": 258880,
              "video_token_id": 258884,
              "video_processor": { "max_soft_tokens": 70 }
            }
            """
        return try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
    }
}
