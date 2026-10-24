// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Video input for the base Gemma 4 (`gemma4`) VLM
//
// Gemma 4 has no separate video encoder: each frame runs through the same vision
// tower as images and scatters onto the `<video>` soft-token positions
// (`video_token_id`, 258884). These tests cover the wiring on the non-unified
// `Gemma4` class and its `Gemma4Processor`.
//
// Fork: the fork's MLXVLM `Gemma4` keeps its own video path (fork.yaml section
// "Vision-language models"; Gemma 4 video seam for CBv2, fork PR #68). The
// processor resizes each frame to the `video_processor.max_soft_tokens` budget
// inside `prepare(input:)`, and the model gives each frame the soft-token count
// of its resized patch grid, `(h / patch) * (w / patch) / pool^2`, where upstream
// truncates to `vision_soft_tokens_per_video_frame`. `video_token_id` defaults
// to 258884 when config.json omits it. The cases below check upstream's
// properties against these fork entry points. `CBv2VideoSeamTests` covers the
// CBv2 seam.

struct Gemma4VideoInputTests {

    /// `Gemma4Configuration` decodes `video_token_id` and defaults it to 258884, and
    /// `Gemma4ProcessorConfiguration` decodes the per-frame video budget
    /// (`video_processor.max_soft_tokens`) and defaults it to 70.
    @Test("Gemma4Configuration decodes video_token_id")
    func decodesVideoTokenId() throws {
        let withVideo = try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 258884,")
        #expect(withVideo.videoTokenId == 258884)

        let withoutVideo = try Self.decodeConfig(videoTokenLine: "")
        #expect(withoutVideo.videoTokenId == 258884)

        #expect(try Self.makeProcessorConfig(videoBudgetLine: "").videoMaxSoftTokens == 70)
        #expect(
            try Self.makeProcessorConfig(
                videoBudgetLine: "\"video_processor\": { \"max_soft_tokens\": 4 },"
            ).videoMaxSoftTokens == 4)
    }

    /// When a frame's patch grid gives the same soft-token count as the image budget,
    /// the same pixels fed as an image and as a video produce identical logits:
    /// video reuses the vision tower and scatters at `video_token_id`.
    @Test("Image and video pixels scatter identically (no truncation)")
    func imageAndVideoAgree() throws {
        let model = Gemma4(try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 101,"))
        eval(model)

        let imageTokenId = 100
        let videoTokenId = 101
        func prompt(_ mm: Int) -> [Int] { [5, 6, mm, mm, mm, mm, 7, 8] }
        // 32x32 with patch_size 8 and pooling 2 -> a 4x4 patch grid -> 4 soft tokens,
        // equal to the image budget (`default_output_length` 4).
        let pixels = (MLXArray(0 ..< 3072).reshaped([1, 3, 32, 32]).asType(.float32)) / 3072.0

        func lastLogits(mm: Int, asVideo: Bool) throws -> MLXArray {
            let tokens = MLXArray(prompt(mm).map { Int32($0) }).expandedDimensions(axis: 0)
            let text = LMInput.Text(tokens: tokens)
            let input =
                asVideo
                ? LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))
                : LMInput(text: text, image: LMInput.ProcessedImage(pixels: pixels))
            let result = try model.prepare(
                input, cache: model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 1024))
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
        #expect(imageLogits.shape == [1, Self.vocabularySize])
        #expect(
            allClose(imageLogits, videoLogits, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "Video pixels must scatter through the vision tower identically to image pixels.")
    }

    /// A multi-frame video scatters the soft-token count of each frame's patch grid.
    /// 48x48 frames give a 6x6 grid, so 36 / 2^2 = 9 tokens per frame: a 2-frame
    /// clip fills exactly 18 placeholders, and a prompt sized to the image budget
    /// (4 per frame) is rejected.
    @Test("Multi-frame video scatters the per-frame patch-grid count")
    func videoTruncatesPerFrame() throws {
        let softPerFrame = 9
        let numFrames = 2
        let videoTokenId = 101
        let model = Gemma4(try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 101,"))
        eval(model)

        let pixels =
            (MLXArray(0 ..< (numFrames * 3 * 48 * 48)).reshaped([numFrames, 3, 48, 48])
                .asType(.float32)) / Float(numFrames * 3 * 48 * 48)

        func input(placeholdersPerFrame: Int) -> (LMInput, Int) {
            var tokens: [Int32] = [5, 6]
            tokens += Array(repeating: Int32(videoTokenId), count: numFrames * placeholdersPerFrame)
            tokens += [7]
            let text = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
            return (
                LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels)), tokens.count
            )
        }

        let (matching, tokenCount) = input(placeholdersPerFrame: softPerFrame)
        let result = try model.prepare(
            matching, cache: model.newCache(parameters: nil), state: nil,
            prefill: .init(stepSize: 1024))
        guard case .logits(let out) = result else {
            Issue.record("Expected .logits from Gemma4.prepare")
            return
        }
        eval(out.logits)
        #expect(out.logits.shape == [1, tokenCount, Self.vocabularySize])

        let (imageBudgetSized, _) = input(placeholdersPerFrame: 4)
        #expect(throws: (any Error).self) {
            _ = try model.prepare(
                imageBudgetSized, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 1024))
        }
    }

    /// `Gemma4Processor.prepare(input:)` samples and preprocesses frames into a
    /// `[totalFrames, C, H, W]` tensor at the video frame size. With patch 8, pooling 2
    /// and a video budget of 4 tokens, a 64x64 frame resizes to 32x32.
    @Test("Processor extracts video frames to [F, C, 32, 32]")
    func processorExtractsFrames() async throws {
        let processor = Gemma4Processor(
            try Self.makeProcessorConfig(
                videoBudgetLine: "\"video_processor\": { \"max_soft_tokens\": 4 },"),
            tokenizer: VideoTestTokenizer())

        // Four synthetic 64x64 frames at 1s spacing.
        let frames = (0 ..< 4).map { i in
            UserInput.VideoFrame(
                image: .ciImage(
                    CIImage(color: CIColor(red: 0.3, green: 0.5, blue: 0.7))
                        .cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))),
                timeStamp: CMTime(value: Int64(i), timescale: 1))
        }
        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("describe the clip", videos: [.frames(frames)])]))
        let video = try #require(prepared.video)
        let pixels = video.pixels
        eval(pixels)

        let frameGrid = try #require(video.frames)
        #expect(pixels.dim(0) == frameGrid.count)
        #expect(pixels.dim(0) >= 1)
        #expect(Array(pixels.shape.dropFirst()) == [3, 32, 32])
        #expect(frameGrid.allSatisfy { $0.h == 32 && $0.w == 32 })

        // One placeholder run of 4 tokens per sampled frame.
        let placeholders = prepared.text.tokens.asArray(Int32.self).filter {
            Int($0) == VideoTestTokenizer.videoMarkerId
        }
        #expect(placeholders.count == 4 * pixels.dim(0))
    }

    // MARK: - Helpers

    private static let vocabularySize = 512

    /// Tiny `gemma4` config in the fork's schema (the geometry of
    /// `CBv2VideoSeamTests`): patch 8, pooling 2, image budget 4.
    private static func decodeConfig(videoTokenLine: String) throws -> Gemma4Configuration {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              \(videoTokenLine)
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8,
                "intermediate_size": 64, "vocab_size": \(vocabularySize), "rms_norm_eps": 1e-6,
                "sliding_window": 16, "layer_types": ["sliding_attention", "full_attention"],
                "tie_word_embeddings": true, "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 0, "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false, "enable_moe_block": false,
                "use_bidirectional_attention": "vision"
              },
              "vision_config": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 1e-6, "patch_size": 8, "position_embedding_size": 64,
                "default_output_length": 4, "pooling_kernel_size": 2
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    private static func makeProcessorConfig(videoBudgetLine: String) throws
        -> Gemma4ProcessorConfiguration
    {
        let json = """
            {
              \(videoBudgetLine)
              "processor_class": "Gemma4Processor",
              "patch_size": 8,
              "max_soft_tokens": 16,
              "pooling_kernel_size": 2,
              "image_seq_length": 16,
              "boi_token_id": 10,
              "eoi_token_id": 11
            }
            """
        return try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
    }
}

/// Minimal tokenizer so `Gemma4Processor` can run its chat template: `<|image|>` and
/// `<|video|>` map to fixed ids, text maps per UTF-8 byte into a band away from them,
/// and the template writes one marker token per media content part.
private struct VideoTestTokenizer: Tokenizer {
    static let imageMarkerId = 6
    static let videoMarkerId = 7

    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { 20 + Int($0 % 60) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|image|>": return Self.imageMarkerId
        case "<|video|>": return Self.videoMarkerId
        default: return nil
        }
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var out = [1]
        for message in messages {
            if let parts = message["content"] as? [[String: String]] {
                for part in parts {
                    switch part["type"] {
                    case "image": out.append(Self.imageMarkerId)
                    case "video": out.append(Self.videoMarkerId)
                    default:
                        out.append(
                            contentsOf: encode(text: part["text"] ?? "", addSpecialTokens: false))
                    }
                }
            } else if let text = message["content"] as? String {
                out.append(contentsOf: encode(text: text, addSpecialTokens: false))
            }
        }
        return out
    }
}
