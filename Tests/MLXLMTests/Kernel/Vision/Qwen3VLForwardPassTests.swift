import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {

    /// Forward-pass tests of `Qwen3VL` with a tiny random model and random
    /// pixels (no media files).
    ///
    /// The text model has hidden size 32, 2 layers, 4 query heads and 2 KV
    /// heads with head size 8, and an untied LM head. The vision tower has 2
    /// blocks with hidden size 32, patch size 4, temporal patch size 2,
    /// spatial merge 2 and one DeepStack merger after block 0.
    ///
    /// An image has the grid `THW(1, 4, 4)`: 16 patch rows of width
    /// `3 * 2 * 4 * 4 = 96` and 16 / 2^2 = 4 image tokens. A video has the
    /// grid `THW(2, 4, 4)`: 32 patch rows and 8 video tokens.
    @Suite
    struct Qwen3VLForwardPassTests {

        static let vocabularySize = 64
        static let imageToken = 60
        static let videoToken = 61
        static let visionStart = 57
        static let visionEnd = 58

        /// Width of one patch row: channels * temporal patch * patch * patch.
        static let patchWidth = 3 * 2 * 4 * 4

        // Tolerance of the float32 comparisons: the compared paths differ
        // only in the order of the attention sums (cache or no cache, one
        // or two images in one vision call). Differences are near 1e-6 for
        // values of size 1 to 5.
        static let tolerance: Float = 1e-4

        static func configurationDictionary(tied: Bool = false) -> [String: Any] {
            [
                "model_type": "qwen3_vl",
                "image_token_id": imageToken,
                "video_token_id": videoToken,
                "vision_start_token_id": visionStart,
                "vision_end_token_id": visionEnd,
                "vision_token_id": 59,
                "text_config": [
                    "model_type": "qwen3_vl_text", "hidden_size": 32, "intermediate_size": 48,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 256,
                    "vocab_size": vocabularySize, "rope_theta": 10000, "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": tied,
                    "rope_scaling": [
                        "type": "mrope", "mrope_interleaved": true, "mrope_section": [1, 1, 2],
                    ],
                ] as [String: Any],
                "vision_config": [
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 48, "out_hidden_size": 32, "num_heads": 4,
                    "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
                ] as [String: Any],
            ]
        }

        static func makeModel(seed: UInt64, tied: Bool = false) throws -> Qwen3VL {
            let configuration = try SyntheticModel.configuration(
                Qwen3VLConfiguration.self, configurationDictionary(tied: tied))
            let model = Qwen3VL(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// Random normal patch rows `[rows, 96]`.
        static func pixels(_ rows: Int, seed: UInt64) -> MLXArray {
            let x = MLXRandom.normal([rows, patchWidth], key: MLXRandom.key(seed))
            eval(x)
            return x
        }

        static func tokens(_ ids: [Int]) -> MLXArray {
            MLXArray(ids.map { Int32($0) })
        }

        static func ints(_ array: MLXArray) -> [Int] {
            array.asType(.int32).asArray(Int32.self).map(Int.init)
        }

        /// One media layout of the prompt: an image, a video, or both.
        struct Media: Sendable, CustomTestStringConvertible {
            let name: String
            let image: Bool
            let video: Bool

            var testDescription: String { name }

            /// Text, the image block, the video block, then more text.
            var prompt: [Int] {
                typealias Outer = Qwen3VLForwardPassTests
                var ids = [5]
                if image {
                    ids += [Outer.visionStart] + Array(repeating: Outer.imageToken, count: 4)
                    ids += [Outer.visionEnd]
                }
                if video {
                    ids += [Outer.visionStart] + Array(repeating: Outer.videoToken, count: 8)
                    ids += [Outer.visionEnd]
                }
                return ids + [9, 11]
            }

            func input(_ prompt: [Int], seed: UInt64) -> LMInput {
                let tokens = SyntheticModel.batch([prompt])
                let processedImage =
                    image
                    ? LMInput.ProcessedImage(
                        pixels: Qwen3VLForwardPassTests.pixels(16, seed: seed),
                        frames: [THW(1, 4, 4)]) : nil
                let processedVideo =
                    video
                    ? LMInput.ProcessedVideo(
                        pixels: Qwen3VLForwardPassTests.pixels(32, seed: seed &+ 100),
                        frames: [THW(2, 4, 4)]) : nil
                return LMInput(
                    text: .init(tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .int32)),
                    image: processedImage, video: processedVideo)
            }
        }

        static let media: [Media] = [
            Media(name: "image", image: true, video: false),
            Media(name: "video", image: false, video: true),
            Media(name: "image and video", image: true, video: true),
        ]

        /// Runs `prepare` and returns the logits of the prompt.
        static func prefill(_ model: Qwen3VL, _ input: LMInput, cache: [KVCache]) throws
            -> MLXArray
        {
            guard case .logits(let output) = try model.prepare(input, cache: cache, windowSize: nil)
            else {
                Issue.record("prepare must return logits")
                return MLXArray.zeros([1])
            }
            eval(output.logits)
            return output.logits
        }

        // MARK: - Prefill with media

        @Test(arguments: media) func prefillLogitsHaveTheExpectedShapeAndAreFinite(_ m: Media)
            throws
        {
            let model = try Self.makeModel(seed: 1)
            let logits = try Self.prefill(
                model, m.input(m.prompt, seed: 1), cache: model.newCache(parameters: nil))
            #expect(
                logits.shape == [1, m.prompt.count, Self.vocabularySize],
                "\(m.name): logits shape [1, prompt, vocabulary]")
            #expect(isFinite(logits).all().item(Bool.self), "\(m.name): a logit is not finite")
        }

        /// The same pixels give the same logits; other pixels give other
        /// logits at the last position, which comes after the media tokens.
        @Test(arguments: media) func theMediaChangesTheLogitsAfterIt(_ m: Media) throws {
            let model = try Self.makeModel(seed: 1)
            func run(_ seed: UInt64) throws -> MLXArray {
                try Self.prefill(
                    model, m.input(m.prompt, seed: seed), cache: model.newCache(parameters: nil))
            }
            let a = try run(1)
            let again = try run(1)
            let b = try run(2)
            // Exact: the same model runs the same input twice.
            #expect(
                SyntheticModel.maxAbsDifference(a, again) == 0,
                "\(m.name): same pixels, same logits")
            #expect(
                SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                "\(m.name): other pixels must change the logits")
        }

        /// After the prompt, a decode step with the cache gives the logits
        /// of the last position of the same prompt with that token appended.
        /// The decode step uses the M-RoPE delta that the prompt stored.
        @Test(arguments: media) func decodeAfterThePromptMatchesALongerPrompt(_ m: Media) throws {
            let model = try Self.makeModel(seed: 1)
            let cache = model.newCache(parameters: nil)
            _ = try Self.prefill(model, m.input(m.prompt, seed: 1), cache: cache)
            let step = model(SyntheticModel.batch([[17]]), cache: cache)
            eval(step)
            let longer = try Self.prefill(
                model, m.input(m.prompt + [17], seed: 1), cache: model.newCache(parameters: nil))
            let difference = SyntheticModel.maxAbsDifference(step[0..., -1], longer[0..., -1])
            #expect(
                difference <= Self.tolerance,
                "\(m.name): decode step differs from the longer prompt by \(difference)")
        }

        /// The vision features must fill exactly the media tokens. One
        /// image token too few makes `prepare` throw.
        @Test func prepareRejectsATokenCountThatDoesNotMatchTheImage() throws {
            let model = try Self.makeModel(seed: 1)
            let prompt =
                [5, Self.visionStart] + Array(repeating: Self.imageToken, count: 3)
                + [Self.visionEnd, 9]
            let input = Self.media[0].input(prompt, seed: 1)
            #expect(throws: (any Error).self, "3 image tokens for 4 image features") {
                try model.prepare(input, cache: model.newCache(parameters: nil), windowSize: nil)
            }
        }

        // MARK: - Text only

        @Test func textOnlyPrepareMatchesTheDirectCall() throws {
            let model = try Self.makeModel(seed: 1)
            let prompt = [5, 7, 9, 11, 13, 15]
            let logits = try Self.prefill(
                model, LMInput(text: .init(tokens: SyntheticModel.batch([prompt]))),
                cache: model.newCache(parameters: nil))
            let direct = ForwardPassChecks.logits(model, [prompt])
            let difference = SyntheticModel.maxAbsDifference(logits, direct)
            #expect(logits.shape == direct.shape, "text prepare logits shape")
            #expect(
                difference <= Self.tolerance,
                "text prepare differs from the direct call by \(difference)")
        }

        @Test func textOnlyForwardPassChecks() throws {
            let model = try Self.makeModel(seed: 1)
            ForwardPassChecks.checkShapeDTypeAndFinite(
                model, vocabularySize: Self.vocabularySize, length: 7)
            let rowA = SyntheticModel.tokens(count: 11, vocabularySize: 64, seed: 1)
            let rowB = SyntheticModel.tokens(count: 11, vocabularySize: 64, seed: 2)
            // Batch size 1 only. With batch size 2, a cached step adds the
            // per-row delta `[2]` on the position axis of `[2, L]`
            // (Qwen3VL.swift:1611), and the broadcast fails.
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [rowA], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
            ForwardPassChecks.checkBatchInvariance(
                model, rowA: rowA, rowB: rowB, tolerance: Self.tolerance)
            ForwardPassChecks.checkCausality(
                model, row: rowA, position: 6, vocabularySize: Self.vocabularySize,
                tolerance: Self.tolerance)
        }

        /// A tied model has no LM head and projects with the embeddings.
        @Test func tiedModelUsesTheEmbeddingsAsTheHead() throws {
            let model = try Self.makeModel(seed: 1, tied: true)
            let keys = Set(SyntheticModel.flatParameters(model).keys)
            #expect(!keys.contains("language_model.lm_head.weight"), "tied: no lm_head")
            let logits = try Self.prefill(
                model, Self.media[0].input(Self.media[0].prompt, seed: 1),
                cache: model.newCache(parameters: nil))
            #expect(
                logits.shape == [1, Self.media[0].prompt.count, Self.vocabularySize],
                "tied: logits shape")
            #expect(isFinite(logits).all().item(Bool.self), "tied: a logit is not finite")
        }

        // MARK: - Checkpoint loading

        /// The parameters in the MLX layout load without change.
        @Test func loaderAcceptsAnMLXLayoutCheckpoint() throws {
            let reference = try Self.makeModel(seed: 5)
            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(SyntheticModel.flatParameters(reference), into: loaded)
            let input = Self.media[2].input(Self.media[2].prompt, seed: 1)
            let a = try Self.prefill(reference, input, cache: reference.newCache(parameters: nil))
            let b = try Self.prefill(loaded, input, cache: loaded.newCache(parameters: nil))
            // Exact: the loaded weights are copies.
            #expect(SyntheticModel.maxAbsDifference(a, b) == 0, "MLX layout: loaded logits")
        }

        /// The Hugging Face layout: `model.visual.*`, `model.language_model.*`,
        /// `lm_head.*`, a PyTorch convolution layout and a `position_ids`
        /// buffer. `sanitize(weights:)` must convert all of it.
        @Test func loaderAcceptsAHuggingFaceCheckpoint() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in SyntheticModel.flatParameters(reference) {
                var name = key
                var tensor = value
                if name.hasPrefix("vision_tower.") {
                    name = "model.visual." + name.dropFirst("vision_tower.".count)
                } else if name.hasPrefix("language_model.model.") {
                    name = "model.language_model." + name.dropFirst("language_model.model.".count)
                } else if name.hasPrefix("language_model.lm_head.") {
                    name = "lm_head." + name.dropFirst("language_model.lm_head.".count)
                }
                if name.hasSuffix("patch_embed.proj.weight") {
                    // MLX [O, T, p, p, C] to PyTorch [O, C, T, p, p].
                    tensor = value.movedAxis(source: -1, destination: 1)
                }
                checkpoint[name] = tensor
            }
            checkpoint["model.visual.position_ids"] = MLXArray.zeros([1, 16])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let input = Self.media[0].input(Self.media[0].prompt, seed: 1)
            let a = try Self.prefill(reference, input, cache: reference.newCache(parameters: nil))
            let b = try Self.prefill(loaded, input, cache: loaded.newCache(parameters: nil))
            // Exact: renames and a transpose do not change values.
            #expect(SyntheticModel.maxAbsDifference(a, b) == 0, "HF layout: loaded logits")
        }

        /// A tied model drops every LM head key, and the vision sanitizer
        /// drops `position_ids`.
        @Test func sanitizeDropsTheHeadOfATiedModelAndPositionIds() throws {
            let model = try Self.makeModel(seed: 1, tied: true)
            let sanitized = model.sanitize(weights: [
                "lm_head.weight": MLXArray.zeros([64, 32]),
                "language_model.lm_head.weight": MLXArray.zeros([64, 32]),
                "model.visual.position_ids": MLXArray.zeros([4]),
            ])
            #expect(sanitized.isEmpty, "tied head and position_ids are dropped: \(sanitized.keys)")

            let kept = model.sanitize(weights: [
                "model.visual.patch_embed.proj.weight": MLXArray.zeros([32, 2, 4, 4, 3])
            ])
            // MLX layout (last axis = 3 input channels) stays as it is.
            #expect(
                kept["vision_tower.patch_embed.proj.weight"]?.shape == [32, 2, 4, 4, 3],
                "MLX convolution layout is kept")
        }

        // MARK: - Model properties

        @Test func modelPropertiesDescribeTheTinyConfiguration() throws {
            let model = try Self.makeModel(seed: 1)
            #expect(model.vocabularySize == 64, "vocabularySize")
            #expect(model.kvHeads == [2, 2], "kvHeads per layer")
            #expect(model.loraLayers.count == 2, "one LoRA layer per decoder layer")
            #expect(model.imagePlaceholderTokenId == 60, "image placeholder")
            #expect(model.videoPlaceholderTokenId == 61, "video placeholder")
            #expect(model.deepstackLayerCount == 1, "one DeepStack layer")
            #expect(model.cbv2PositionAxisCount == 3, "three M-RoPE axes")
            #expect(!model.supportsVisionSpanPrefill, "no bidirectional vision spans")
            #expect(model.supportsCausalVisionPrefill, "causal vision prefill")

            let kinds = model.cbv2LayerKinds
            #expect(kinds.count == 2, "one CBv2 layer kind per layer")
            for (index, kind) in kinds.enumerated() {
                #expect(kind.attention == .full, "layer \(index): full attention")
                #expect(kind.headDim == 8, "layer \(index): head size")
                #expect(kind.kvHeads == 2, "layer \(index): KV heads")
                #expect(kind.queryHeads == 4, "layer \(index): query heads")
                #expect(kind.modelLayerIndex == index, "layer \(index): model layer index")
            }

            let capabilities = model.cbv2Capabilities
            #expect(!capabilities.supportsPrefixReuse, "no prefix reuse")
            #expect(!capabilities.supportsPagedKV, "no paged KV")
            #expect(!capabilities.supportsCompiledDecode, "no compiled decode")
            #expect(!capabilities.supportsPackedPrefill, "no packed prefill")
            #expect(!capabilities.supportsMTP, "no MTP")

            let embeddings = model.scaledInputEmbeddings(SyntheticModel.batch([[1, 2, 3]]))
            #expect(embeddings.shape == [1, 3, 32], "input embeddings shape")
        }

        /// The fused gate-up projection of a sparse layer looks up the
        /// quantization of the split checkpoint projections too.
        @Test func quantizationPathAliasesCoverFusedAndSplitExperts() throws {
            let model = try Self.makeModel(seed: 1)
            let suffix = "layers.0.mlp.switch_mlp"

            let down = model.quantizationPathAliases(
                for: "language_model.model.\(suffix).down_proj")
            #expect(
                down == [
                    "model.language_model.\(suffix).down_proj",
                    "model.\(suffix).down_proj",
                    "language_model.\(suffix).down_proj",
                    "\(suffix).down_proj",
                ], "down_proj aliases")

            let fused = model.quantizationPathAliases(
                for: "language_model.model.\(suffix).gate_up_proj")
            // 4 aliases of the fused path, then 5 for each split path.
            #expect(fused.count == 14, "gate_up_proj alias count")
            #expect(
                fused.contains("model.language_model.\(suffix).gate_proj"), "gate_proj alias")
            #expect(fused.contains("\(suffix).up_proj"), "up_proj alias")

            #expect(
                model.quantizationPathAliases(for: "language_model.model.layers.0.mlp.gate_proj")
                    .isEmpty, "a dense projection has no aliases")
        }

        // MARK: - Request-owned positions

        @Test func positionResultOfTextIsSequential() throws {
            let model = try Self.makeModel(seed: 1)
            let result = try model.positionResult(tokens: Self.tokens([1, 2, 3, 4]))
            #expect(result.promptPositionIds.shape == [3, 1, 4], "text positions shape")
            #expect(
                Self.ints(result.promptPositionIds) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "text positions")
            #expect(result.decodeState.deltas == [0], "text delta")
            #expect(result.promptLength == 4, "text prompt length")
        }

        /// Image grid 4x4 with merge 2 gives a 2x2 block of positions. The
        /// two text tokens after it continue from the largest position.
        @Test func positionResultOfAnImageUsesThreeAxes() throws {
            let model = try Self.makeModel(seed: 1)
            let ids = [5, 57, 60, 60, 60, 60, 58, 9]
            let expectedT = [0, 1, 2, 2, 2, 2, 4, 5]
            let expectedH = [0, 1, 2, 2, 3, 3, 4, 5]
            let expectedW = [0, 1, 2, 3, 2, 3, 4, 5]

            let result = try model.positionResult(
                tokens: Self.tokens(ids), imageGrids: [THW(1, 4, 4)])
            #expect(result.promptPositionIds.shape == [3, 1, 8], "image positions shape")
            #expect(
                Self.ints(result.promptPositionIds) == expectedT + expectedH + expectedW,
                "image positions")
            // Largest position 5, plus 1, minus 8 tokens.
            #expect(result.decodeState.deltas == [-2], "image delta")

            // A 1-D mask of ones gives the same positions.
            let masked = try model.positionResult(
                tokens: Self.tokens(ids), imageGrids: [THW(1, 4, 4)],
                attentionMask: MLXArray(Array(repeating: Int32(1), count: 8)))
            #expect(
                Self.ints(masked.promptPositionIds) == expectedT + expectedH + expectedW,
                "image positions with a mask of ones")
        }

        /// Video grid THW(2, 4, 4): 2 frames of a 2x2 merged block.
        @Test func positionResultOfAVideoUsesTheTimeAxis() throws {
            let model = try Self.makeModel(seed: 1)
            let ids = [5, 57] + Array(repeating: 61, count: 8) + [58]
            let result = try model.positionResult(
                tokens: Self.tokens(ids), videoGrids: [THW(2, 4, 4)])
            let expectedT = [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4]
            let expectedH = [0, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4]
            let expectedW = [0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4]
            #expect(
                Self.ints(result.promptPositionIds) == expectedT + expectedH + expectedW,
                "video positions")
            // Largest position 4, plus 1, minus 11 tokens.
            #expect(result.decodeState.deltas == [-6], "video delta")
        }

        @Test func positionResultRejectsBadInput() throws {
            let model = try Self.makeModel(seed: 1)
            let image = Self.tokens([5, 57, 60, 60, 60, 60, 58, 9])

            #expect(throws: Qwen35PositionSeamError.invalidInputRank(3), "rank 3 tokens") {
                try model.positionResult(tokens: MLXArray.zeros([1, 1, 3], dtype: .int32))
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidAttentionMaskRank(3), "rank 3 mask"
            ) {
                try model.positionResult(
                    tokens: image, attentionMask: MLXArray.ones([1, 1, 8], dtype: .int32))
            }
            #expect(
                throws: Qwen35PositionSeamError.attentionMaskShapeMismatch, "mask too short"
            ) {
                try model.positionResult(
                    tokens: image, attentionMask: MLXArray.ones([1, 7], dtype: .int32))
            }
            #expect(
                throws: Qwen35PositionSeamError.multimodalBatchUnsupported(2), "batch 2 media"
            ) {
                try model.positionResult(
                    tokens: SyntheticModel.batch([[5, 60, 60, 60, 60], [5, 60, 60, 60, 60]]),
                    imageGrids: [THW(1, 4, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "image", index: 0),
                "height 3 is not a multiple of merge 2"
            ) {
                try model.positionResult(tokens: image, imageGrids: [THW(1, 3, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "video", index: 0),
                "video with 0 frames"
            ) {
                try model.positionResult(tokens: image, videoGrids: [THW(0, 4, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 3),
                "3 image tokens for a 4-token grid"
            ) {
                try model.positionResult(
                    tokens: Self.tokens([5, 57, 60, 60, 60, 58]), imageGrids: [THW(1, 4, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 0, actual: 4),
                "image tokens without a grid"
            ) {
                try model.positionResult(tokens: image)
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 0),
                "image grid without tokens"
            ) {
                try model.positionResult(
                    tokens: Self.tokens([5, 9, 11]), imageGrids: [THW(1, 4, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "video", gridIndex: 0, expected: 8, actual: 0),
                "video grid without tokens"
            ) {
                try model.positionResult(
                    tokens: Self.tokens([5, 9, 11]), videoGrids: [THW(2, 4, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 0),
                "masked image tokens do not count"
            ) {
                try model.positionResult(
                    tokens: image, imageGrids: [THW(1, 4, 4)],
                    attentionMask: MLXArray([Int32(1), 1, 0, 0, 0, 0, 1, 1]).reshaped(1, 8))
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "image", index: 0),
                "token count of the grid overflows"
            ) {
                try model.positionResult(
                    tokens: Self.tokens([57, 60, 58]), imageGrids: [THW(Int.max, 4, 2)])
            }
        }

        // MARK: - CBv2 vision seam

        /// Two images in one call give one feature block per image. The
        /// vision attention is per image, so the first block equals the
        /// features of the first image alone.
        @Test func cbv2VisionFeaturesSplitPerImage() throws {
            let model = try Self.makeModel(seed: 1)
            let first = Self.pixels(16, seed: 1)
            let second = Self.pixels(32, seed: 2)
            let both = try model.cbv2VisionFeatures(
                imagePixels: concatenated([first, second]),
                imageGrids: [THW(1, 4, 4), THW(1, 4, 8)])
            #expect(both.features.map(\.shape) == [[1, 4, 32], [1, 8, 32]], "feature shapes")
            #expect(both.deepstack.count == 1, "one DeepStack layer")
            #expect(
                both.deepstack.first?.map(\.shape) == [[1, 4, 32], [1, 8, 32]],
                "DeepStack shapes")

            let alone = try model.cbv2VisionFeatures(
                imagePixels: first, imageGrids: [THW(1, 4, 4)])
            #expect(alone.features.map(\.shape) == [[1, 4, 32]], "single image feature shape")
            let difference = SyntheticModel.maxAbsDifference(alone.features[0], both.features[0])
            #expect(
                difference <= Self.tolerance,
                "first image features depend on the second image: \(difference)")
        }

        @Test func cbv2VisionFeaturesRejectBadInput() throws {
            let model = try Self.makeModel(seed: 1)
            let pixels = Self.pixels(16, seed: 1)

            #expect(throws: (any Error).self, "no image grid") {
                try model.cbv2VisionFeatures(imagePixels: pixels, imageGrids: [])
            }
            #expect(
                throws: Qwen35VisionSeamError.invalidGrid(kind: "image", index: 0),
                "height 3 is not a multiple of merge 2"
            ) {
                try model.cbv2VisionFeatures(imagePixels: pixels, imageGrids: [THW(1, 3, 4)])
            }
            #expect(
                throws: Qwen35VisionSeamError.invalidGrid(kind: "image", index: 0),
                "row count of the grid overflows"
            ) {
                try model.cbv2VisionFeatures(imagePixels: pixels, imageGrids: [THW(Int.max, 2, 2)])
            }
            #expect(throws: (any Error).self, "patch rows of width 95, not 96") {
                try model.cbv2VisionFeatures(
                    imagePixels: MLXArray.zeros([16, 95]), imageGrids: [THW(1, 4, 4)])
            }
            #expect(
                throws: Qwen35VisionSeamError.pixelCountMismatch(
                    kind: "image", expected: 16, actual: 15),
                "15 patch rows for a 16-row grid"
            ) {
                try model.cbv2VisionFeatures(
                    imagePixels: pixels[0 ..< 15], imageGrids: [THW(1, 4, 4)])
            }
        }

        // MARK: - Configuration

        /// The optional keys of the configuration take their defaults.
        @Test func configurationDefaults() throws {
            let minimal: [String: Any] = [
                "model_type": "qwen3_vl",
                "text_config": [
                    "model_type": "qwen3_vl_text", "hidden_size": 32, "intermediate_size": 48,
                    "num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 8,
                    "max_position_embeddings": 256, "vocab_size": 64,
                ] as [String: Any],
                "vision_config": [
                    "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32,
                    "intermediate_size": 48, "out_hidden_size": 32, "num_heads": 4,
                    "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 16,
                ] as [String: Any],
            ]
            let configuration = try SyntheticModel.configuration(
                Qwen3VLConfiguration.self, minimal)
            let text = configuration.textConfiguration
            #expect(text.numKeyValueHeads == 4, "KV heads default to the query heads")
            #expect(text.ropeTheta == 1_000_000, "rope_theta default")
            #expect(text.rmsNormEps == 1e-6, "rms_norm_eps default")
            #expect(text.ropeScaling?.mropeSection == nil, "no rope_scaling")
            #expect(text.normTopKProb, "norm_topk_prob default")
            #expect(text.numExperts == 0, "dense by default")
            #expect(text.numExpertsPerTok == 0, "no experts per token")
            #expect(text.decoderSparseStep == 1, "decoder_sparse_step default")
            #expect(text.mlpOnlyLayers.isEmpty, "mlp_only_layers default")
            #expect(text.moeIntermediateSize == 48, "moe size defaults to the MLP size")
            #expect(text.tieWordEmbeddings, "tied by default")
            #expect(!text.attentionBias, "no attention bias by default")
            #expect(text.hiddenAct == "silu", "text activation default")
            #expect(!text.usesSparseMoE(layerIndex: 0), "dense layer 0")

            let vision = configuration.visionConfiguration
            #expect(vision.inChannels == 3, "in_channels default")
            #expect(vision.hiddenAct == "gelu", "vision activation default")
            #expect(vision.deepstackVisualIndexes.isEmpty, "no DeepStack by default")

            #expect(configuration.ignoreIndex == -100, "ignore_index default")
            #expect(configuration.imageTokenId == 151_655, "image_token_id default")
            #expect(configuration.videoTokenId == 151_656, "video_token_id default")
            #expect(configuration.imageTokenIndex == 151_655, "image index follows the id")
            #expect(configuration.videoTokenIndex == 151_656, "video index follows the id")
            #expect(configuration.visionStartTokenId == 151_652, "vision start default")
            #expect(configuration.visionEndTokenId == 151_653, "vision end default")
            #expect(configuration.visionTokenId == 151_654, "vision token default")
            #expect(configuration.vocabSize == 64, "vocab_size from text_config")
            #expect(configuration.eosTokenId == nil, "no eos_token_id")

            let overridden = try SyntheticModel.configuration(
                Qwen3VLConfiguration.self, minimal,
                overrides: [
                    "image_token_index": 70, "video_token_index": 71, "vocab_size": 80,
                    "eos_token_id": [1, 2],
                ])
            #expect(overridden.imageTokenIndex == 70, "image_token_index overrides")
            #expect(overridden.videoTokenIndex == 71, "video_token_index overrides")
            #expect(overridden.vocabSize == 80, "top-level vocab_size overrides")
            #expect(overridden.eosTokenId == [1, 2], "eos_token_id list")

            let built = Qwen3VLConfiguration(
                textConfiguration: text, visionConfiguration: vision, imageTokenId: 60,
                videoTokenId: 61, vocabSize: 64)
            #expect(built.modelType == "qwen3_vl", "memberwise model_type default")
            #expect(built.imageTokenIndex == 60, "memberwise image index follows the id")
            #expect(built.videoTokenIndex == 61, "memberwise video index follows the id")
            #expect(built.vocabSize == 64, "memberwise vocab size")

            let scaling = Qwen3VLConfiguration.RoPEScaling(
                type: "mrope", mropeInterleaved: true, mropeSection: [1, 1, 2])
            #expect(scaling.type == "mrope", "RoPEScaling type")
            #expect(scaling.mropeInterleaved == true, "RoPEScaling interleaved")
            #expect(scaling.mropeSection == [1, 1, 2], "RoPEScaling sections")
        }

        @Test func messageGeneratorPutsMediaBeforeText() {
            let image = UserInput.Image.ciImage(
                CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5))
                    .cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8)))
            let video = UserInput.Video.frames([])
            let message = Qwen3VLMessageGenerator().generate(
                message: .user("hello", images: [image], videos: [video]))
            #expect(message["role"] as? String == "user", "role")
            let content = message["content"] as? [[String: String]]
            #expect(
                content == [
                    ["type": "image"], ["type": "video"], ["type": "text", "text": "hello"],
                ],
                "image, video, then text")
        }
    }
}
