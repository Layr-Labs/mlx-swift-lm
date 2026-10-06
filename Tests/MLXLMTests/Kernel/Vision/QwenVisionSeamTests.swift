import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// The tower attention geometry of a seam model as `[hiddenSize, numHeads]`.
private func seamTowerGeometry(_ model: any QwenVisionSeamModel) -> [Int] {
    let value = model.visionTowerAttentionGeometry
    return [value.hiddenSize, value.numHeads]
}

extension KernelTests {

    /// `QwenVisionSeamSupport.positionResult`: M-RoPE prompt positions and
    /// the checks on token runs, grids, ranks and masks.
    ///
    /// Token IDs: 9 is vision start, 10 is the image position token, 11 is
    /// the video position token, 14 is vision end. Other IDs are text.
    @Suite
    struct QwenVisionSeamPositionKernelTests {

        static func seam(merge: Int) -> Qwen35VisionSeamConfiguration {
            Qwen35VisionSeamConfiguration(
                imagePlaceholderTokenId: 12, videoPlaceholderTokenId: 13,
                imagePositionTokenId: 10, videoPositionTokenId: 11,
                visionStartTokenId: 9, visionEndTokenId: 14,
                spatialMergeSize: merge, temporalPatchSize: 1, attention: .causal)
        }

        private func positions(
            _ tokens: [Int32], image: [THW]? = nil, video: [THW]? = nil,
            mask: MLXArray? = nil, merge: Int = 2
        ) throws -> Qwen35PositionResult {
            try QwenVisionSeamSupport.positionResult(
                tokens: MLXArray(tokens), imageGrids: image, videoGrids: video,
                attentionMask: mask, seam: Self.seam(merge: merge))
        }

        /// The temporal, height and width planes of batch row 0.
        static func planes(_ result: Qwen35PositionResult) -> [[Int]] {
            let ids = result.promptPositionIds
            let values = ids.asType(.int32).asArray(Int32.self).map(Int.init)
            let row = ids.dim(1) * ids.dim(2)
            return (0 ..< 3).map { Array(values[($0 * row) ..< ($0 * row + ids.dim(2))]) }
        }

        @Test
        func textOnlyPromptGetsSequentialPositions() throws {
            let result = try positions([1, 2, 3, 4])
            #expect(result.promptPositionIds.shape == [3, 1, 4], "rank-1 tokens get a batch axis")
            #expect(result.promptLength == 4, "prompt length")
            #expect(
                Self.planes(result) == [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                "text positions are 0 ..< 4 on every plane")
            #expect(result.decodeState.deltas == [0], "no media, so no delta")
        }

        @Test
        func textOnlyBatchIsAllowedAndEmptyGridListsCountAsNone() throws {
            let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 8]).reshaped(2, 4)
            let result = try QwenVisionSeamSupport.positionResult(
                tokens: tokens, imageGrids: [], videoGrids: [], attentionMask: nil,
                seam: Self.seam(merge: 2))
            #expect(result.promptPositionIds.shape == [3, 2, 4], "two rows")
            #expect(result.decodeState.deltas == [0, 0], "one zero delta for each row")
        }

        @Test
        func imageRunGetsThreeDimensionalPositions() throws {
            // Grid 1x4x4 with merge 2 is a 2 x 2 grid of 4 tokens. The text
            // token 9 is at 0. The image tokens start at 0 + 1 = 1 and add
            // their (t, h, w) index. The text after the image starts at the
            // largest image position (2) + 1 = 3.
            let result = try positions([9, 10, 10, 10, 10, 14, 5], image: [THW(1, 4, 4)])
            #expect(
                Self.planes(result) == [
                    [0, 1, 1, 1, 1, 3, 4],
                    [0, 1, 1, 2, 2, 3, 4],
                    [0, 1, 2, 1, 2, 3, 4],
                ], "t, h and w planes")
            // The largest position is 4, so the delta is 4 + 1 - 7 = -2.
            #expect(result.decodeState.deltas == [-2], "decode delta")
            #expect(result.promptLength == 7, "prompt length")
        }

        @Test
        func videoRunGetsTemporalPositions() throws {
            // Grid 2x2x2 with merge 2 is 2 frames of 1 token each.
            let result = try positions([9, 11, 11, 14], video: [THW(2, 2, 2)])
            #expect(
                Self.planes(result) == [[0, 1, 2, 3], [0, 1, 1, 3], [0, 1, 1, 3]],
                "the temporal plane counts frames")
            #expect(result.decodeState.deltas == [0], "decode delta 3 + 1 - 4")
        }

        @Test
        func maskOfOnesGivesTheSamePositionsAsNoMask() throws {
            let tokens: [Int32] = [9, 10, 10, 14, 5]
            let plain = try positions(tokens, image: [THW(1, 2, 4)])
            let masked = try positions(
                tokens, image: [THW(1, 2, 4)], mask: MLXArray([Int32](repeating: 1, count: 5)))
            #expect(Self.planes(masked) == Self.planes(plain), "a full mask changes nothing")
            #expect(masked.decodeState == plain.decodeState, "same decode state")
        }

        @Test
        func maskedImageRunIsMissing() throws {
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 2, actual: 0)
            ) {
                try positions(
                    [9, 10, 10, 14], image: [THW(1, 2, 4)], mask: MLXArray([Int32(1), 0, 0, 1]))
            }
        }

        @Test
        func ranksAndShapesAreChecked() throws {
            #expect(throws: Qwen35PositionSeamError.invalidInputRank(3), "rank-3 tokens") {
                try QwenVisionSeamSupport.positionResult(
                    tokens: MLXArray.zeros([1, 1, 2], dtype: .int32), imageGrids: nil,
                    videoGrids: nil, attentionMask: nil, seam: Self.seam(merge: 2))
            }
            #expect(throws: Qwen35PositionSeamError.invalidInputRank(0), "scalar token") {
                try QwenVisionSeamSupport.positionResult(
                    tokens: MLXArray(Int32(5)), imageGrids: nil, videoGrids: nil,
                    attentionMask: nil, seam: Self.seam(merge: 2))
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidAttentionMaskRank(3), "rank-3 mask"
            ) {
                try positions([1, 2], mask: MLXArray.ones([1, 1, 2], dtype: .int32))
            }
            #expect(
                throws: Qwen35PositionSeamError.attentionMaskShapeMismatch, "mask too short"
            ) {
                try positions([1, 2, 3], mask: MLXArray([Int32(1), 1]))
            }
            #expect(
                throws: Qwen35PositionSeamError.multimodalBatchUnsupported(2), "media batch of 2"
            ) {
                try QwenVisionSeamSupport.positionResult(
                    tokens: MLXArray.zeros([2, 4], dtype: .int32), imageGrids: [THW(1, 2, 2)],
                    videoGrids: nil, attentionMask: nil, seam: Self.seam(merge: 2))
            }
        }

        @Test
        func gridsThatDoNotFitTheMergeAreRejected() throws {
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "image", index: 0),
                "height 3 is not a multiple of 2"
            ) {
                try positions([1], image: [THW(1, 3, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "video", index: 1),
                "second video has t = 0"
            ) {
                try positions([1], video: [THW(1, 2, 2), THW(0, 2, 2)])
            }
            // (2^62 / 2) * (2^62 / 2) overflows Int.
            #expect(
                throws: Qwen35PositionSeamError.invalidGrid(kind: "image", index: 0),
                "token count overflows"
            ) {
                try positions([1, 2, 3], image: [THW(1, 1 << 62, 1 << 62)])
            }
        }

        @Test
        func tokenRunsMustMatchTheGrids() throws {
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 2, actual: 3),
                "run of 3 for a grid of 2"
            ) {
                try positions([9, 10, 10, 10, 14], image: [THW(1, 2, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "video", gridIndex: 0, expected: 0, actual: 1),
                "video run without a video grid"
            ) {
                try positions([9, 10, 10, 14, 9, 11, 14], image: [THW(1, 2, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "image", gridIndex: 1, expected: 2, actual: 0),
                "second image grid has no run"
            ) {
                try positions([9, 10, 10, 14], image: [THW(1, 2, 4), THW(1, 2, 4)])
            }
            #expect(
                throws: Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: "video", gridIndex: 0, expected: 1, actual: 0),
                "video grid has no run"
            ) {
                try positions([1, 2, 3], video: [THW(1, 2, 2)])
            }
        }
    }

    /// `QwenVisionSeamSupport.visionFeatures` with a tiny Qwen3-VL tower:
    /// hidden 8, one block, patch 1, one input channel, merge 1.
    @Suite
    struct QwenVisionSeamFeatureKernelTests {

        static var visionConfig: [String: Any] {
            [
                "model_type": "qwen3_5", "depth": 1, "hidden_size": 8,
                "intermediate_size": 16, "out_hidden_size": 8, "num_heads": 1,
                "patch_size": 1, "spatial_merge_size": 1, "temporal_patch_size": 1,
                "num_position_embeddings": 16, "in_channels": 1,
            ]
        }

        private let tower: Qwen3VLVision.VisionModel

        init() throws {
            MLXRandom.seed(11)
            tower = Qwen3VLVision.VisionModel(
                try SyntheticModel.configuration(
                    Qwen3VLConfiguration.VisionConfiguration.self, Self.visionConfig))
        }

        private func features(
            image: MLXArray? = nil, imageGrids: [THW]? = nil, video: MLXArray? = nil,
            videoGrids: [THW]? = nil, merge: Int = 1, dtype: DType = .float16
        ) throws -> Qwen35VisionFeatures {
            try QwenVisionSeamSupport.visionFeatures(
                imagePixels: image, imageGrids: imageGrids, videoPixels: video,
                videoGrids: videoGrids, tower: tower, spatialMergeSize: merge,
                textHiddenSize: 8, textDType: dtype)
        }

        private static func column(_ values: [Float]) -> MLXArray {
            MLXArray(values).reshaped(values.count, 1)
        }

        @Test
        func noMediaGivesAnEmptyResult() throws {
            let result = try features()
            #expect(result.ordered.isEmpty, "no slices")
            #expect(result.flattenedFeatures.shape == [0, 8], "zero rows of text width")
            #expect(result.flattenedFeatures.dtype == .float16, "text dtype")
        }

        @Test
        func imagesComeFirstThenEachVideoFrame() throws {
            let result = try features(
                image: Self.column([1, 2, 3]), imageGrids: [THW(1, 1, 1), THW(1, 1, 2)],
                video: Self.column([4, 5, 6, 7]), videoGrids: [THW(2, 1, 2)], dtype: .bfloat16)
            eval(result.flattenedFeatures, result.ordered.map(\.features))
            #expect(result.flattenedFeatures.shape == [7, 8], "3 image + 4 video tokens")
            #expect(result.flattenedFeatures.dtype == .bfloat16, "cast to the text dtype")
            #expect(
                result.ordered.map(\.kind) == [
                    .image(index: 0), .image(index: 1),
                    .videoFrame(videoIndex: 0, frameIndex: 0),
                    .videoFrame(videoIndex: 0, frameIndex: 1),
                ], "slice order")
            #expect(
                result.ordered.map(\.features.shape) == [
                    [1, 1, 8], [1, 2, 8], [1, 2, 8], [1, 2, 8],
                ],
                "slice shapes")
            let joined = concatenated(result.ordered.map(\.features), axis: 1).squeezed(axis: 0)
            // Exact: the slices are copies of the flattened rows.
            #expect(
                SyntheticModel.maxAbsDifference(joined, result.flattenedFeatures) == 0,
                "the slices rebuild the flattened features")
        }

        @Test
        func videoOnlyInputUsesTheVideoPixels() throws {
            let result = try features(video: Self.column([1, 2, 3, 4]), videoGrids: [THW(2, 1, 2)])
            #expect(
                result.ordered.map(\.kind) == [
                    .videoFrame(videoIndex: 0, frameIndex: 0),
                    .videoFrame(videoIndex: 0, frameIndex: 1),
                ], "two frame slices")
            #expect(result.flattenedFeatures.shape == [4, 8], "4 video tokens")
            #expect(
                isFinite(result.flattenedFeatures).all().item(Bool.self), "features are finite")
        }

        @Test
        func pixelsAndGridsMustComeTogether() throws {
            let one = Self.column([1])
            #expect(throws: Qwen35VisionSeamError.missingImageGrids, "image pixels, no grids") {
                try features(image: one)
            }
            #expect(
                throws: Qwen35VisionSeamError.missingImageGrids, "image pixels, empty grids"
            ) {
                try features(image: one, imageGrids: [])
            }
            #expect(throws: Qwen35VisionSeamError.missingImagePixels, "image grids, no pixels") {
                try features(imageGrids: [THW(1, 1, 1)])
            }
            #expect(throws: Qwen35VisionSeamError.missingVideoGrids, "video pixels, no grids") {
                try features(video: one)
            }
            #expect(throws: Qwen35VisionSeamError.missingVideoPixels, "video grids, no pixels") {
                try features(videoGrids: [THW(1, 1, 1)])
            }
        }

        @Test
        func gridsRanksAndCountsAreChecked() throws {
            #expect(
                throws: Qwen35VisionSeamError.invalidGrid(kind: "image", index: 0), "t = 0"
            ) {
                try features(image: Self.column([1]), imageGrids: [THW(0, 1, 1)])
            }
            #expect(
                throws: Qwen35VisionSeamError.invalidGrid(kind: "video", index: 0),
                "width 1 is not a multiple of merge 2"
            ) {
                try features(video: Self.column([1, 2]), videoGrids: [THW(1, 2, 1)], merge: 2)
            }
            #expect(
                throws: Qwen35VisionSeamError.invalidPixelRank(kind: "image", actual: 1),
                "rank-1 image pixels"
            ) {
                try features(image: MLXArray([Float(1), 2]), imageGrids: [THW(1, 1, 2)])
            }
            #expect(
                throws: Qwen35VisionSeamError.invalidPixelRank(kind: "video", actual: 3),
                "rank-3 video pixels"
            ) {
                try features(
                    video: MLXArray.zeros([1, 2, 1]), videoGrids: [THW(1, 1, 2)])
            }
            #expect(
                throws: Qwen35VisionSeamError.pixelCountMismatch(
                    kind: "image", expected: 2, actual: 3),
                "3 image rows for a grid of 2"
            ) {
                try features(image: Self.column([1, 2, 3]), imageGrids: [THW(1, 1, 2)])
            }
            #expect(
                throws: Qwen35VisionSeamError.pixelCountMismatch(
                    kind: "video", expected: 4, actual: 3),
                "3 video rows for a grid of 4"
            ) {
                try features(
                    image: Self.column([1, 2]), imageGrids: [THW(1, 1, 2)],
                    video: Self.column([1, 2, 3]), videoGrids: [THW(1, 1, 4)])
            }
            // t * h * w = Int.max * 4 overflows Int.
            #expect(
                throws: Qwen35VisionSeamError.invalidGrid(kind: "image", index: 0),
                "pixel count overflows"
            ) {
                try features(image: Self.column([1, 2]), imageGrids: [THW(Int.max, 2, 2)])
            }
        }
    }

    /// The `MLXVLM.Qwen4Exp` wrapper: vision seam, media rejection,
    /// sanitize, configuration coding and the forwarded properties.
    ///
    /// The text model has no layers (hidden 8, vocabulary 16). The tower is
    /// hidden 8, one head, patch 1, one input channel, merge 2.
    @Suite
    struct Qwen4ExpVisionWrapperKernelTests {

        static func text() -> Qwen4ExpConfiguration {
            var text = Qwen4ExpTextConfiguration()
            text.hiddenLayers = 0
            text.hiddenSize = 8
            text.vocabularySize = 16
            text.pleLayerIds = []
            text.numExperts = 0
            text.tieWordEmbeddings = false
            return Qwen4ExpConfiguration(textConfig: text)
        }

        static var visionConfig: [String: Any] {
            [
                "model_type": "qwen4_exp", "depth": 1, "hidden_size": 8,
                "intermediate_size": 16, "out_hidden_size": 8, "num_heads": 1,
                "patch_size": 1, "spatial_merge_size": 2, "temporal_patch_size": 1,
                "num_position_embeddings": 16, "in_channels": 1,
                "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [Int](),
            ]
        }

        static func vision() throws -> Qwen4ExpVLMConfiguration.VisionConfiguration {
            try SyntheticModel.configuration(
                Qwen4ExpVLMConfiguration.VisionConfiguration.self, visionConfig)
        }

        static func withTower(languageModelOnly: Bool = false) throws -> MLXVLM.Qwen4Exp {
            MLXRandom.seed(5)
            return MLXVLM.Qwen4Exp(
                Qwen4ExpVLMConfiguration(
                    text: text(), visionConfiguration: try vision(),
                    languageModelOnly: languageModelOnly, imageTokenId: 10, videoTokenId: 11,
                    visionStartTokenId: 9, visionEndTokenId: 14))
        }

        static func textOnly() -> MLXVLM.Qwen4Exp {
            // The convenience initializer for a bare text configuration.
            MLXVLM.Qwen4Exp(text())
        }

        private static func imageInput() -> LMInput {
            LMInput(
                text: .init(tokens: MLXArray([Int32(1)]).reshaped(1, 1)),
                image: .init(pixels: MLXArray.zeros([4, 1]), frames: [THW(1, 2, 2)]))
        }

        @Test
        func textOnlyWrapperUsesFlashNextDefaultsAndRejectsMedia() throws {
            let model = Self.textOnly()
            #expect(!model.servesVision, "no tower")
            #expect(seamTowerGeometry(model) == [0, 0], "no tower geometry")
            let seam = model.visionSeamConfiguration
            #expect(
                seam
                    == Qwen35VisionSeamConfiguration(
                        imagePlaceholderTokenId: 248_056, videoPlaceholderTokenId: 248_057,
                        imagePositionTokenId: 248_056, videoPositionTokenId: 248_057,
                        visionStartTokenId: 248_053, visionEndTokenId: 248_054,
                        spatialMergeSize: 2, temporalPatchSize: 2, attention: .causal),
                "Flash-Next token IDs and merge 2 without a tower")
            #expect(model.imagePlaceholderTokenId == 248_056, "image placeholder")
            #expect(model.videoPlaceholderTokenId == 248_057, "video placeholder")

            let rejected = VLMError.processing(MLXVLM.Qwen4Exp.mediaRejectedMessage)
            #expect(throws: rejected, "vision features without a tower") {
                try model.visionFeatures(
                    imagePixels: MLXArray.zeros([4, 1]), imageGrids: [THW(1, 2, 2)])
            }
            #expect(throws: rejected, "legacy prepare with an image") {
                try model.prepare(Self.imageInput(), cache: [], windowSize: nil)
            }
            #expect(throws: rejected, "static media check") {
                try MLXVLM.Qwen4Exp.rejectMediaIfPresent(Self.imageInput())
            }
            try MLXVLM.Qwen4Exp.rejectMediaIfPresent(
                LMInput(tokens: MLXArray([Int32(1)]).reshaped(1, 1)))
        }

        @Test
        func textPrepareReturnsTheTokens() throws {
            let model = Self.textOnly()
            let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            let result = try model.prepare(LMInput(tokens: tokens), cache: [], windowSize: nil)
            guard case .tokens(let text) = result else {
                Issue.record("a short text prompt must come back as tokens")
                return
            }
            #expect(
                text.tokens.asType(.int32).asArray(Int32.self) == [1, 2, 3],
                "the prompt tokens are kept")
        }

        @Test
        func towerWrapperUsesConfiguredTokenIDs() throws {
            let model = try Self.withTower()
            #expect(model.servesVision, "tower present")
            #expect(seamTowerGeometry(model) == [8, 1], "hidden 8, one head")
            #expect(
                model.visionSeamConfiguration
                    == Qwen35VisionSeamConfiguration(
                        imagePlaceholderTokenId: 10, videoPlaceholderTokenId: 11,
                        imagePositionTokenId: 10, videoPositionTokenId: 11,
                        visionStartTokenId: 9, visionEndTokenId: 14,
                        spatialMergeSize: 2, temporalPatchSize: 1, attention: .causal),
                "configured token IDs and tower merge")
            #expect(
                throws: VLMError.processing(
                    "Qwen4-Exp media is served through the CBv2 vision prefill path, not legacy prepare"
                ), "legacy prepare rejects media with a tower too"
            ) {
                try model.prepare(Self.imageInput(), cache: [], windowSize: nil)
            }
            let languageOnly = try Self.withTower(languageModelOnly: true)
            #expect(!languageOnly.servesVision, "language_model_only removes the tower")
        }

        @Test
        func videoGridsAreSplitIntoOneGridPerTemporalGroup() throws {
            // Grid 2x2x2 with merge 2 is two groups of one token. Each group
            // has its own start, pad and end tokens, as the video prompt
            // writes them. Without the split, the first run (1 token) would
            // not match a grid of 2 tokens.
            let model = try Self.withTower()
            let result = try model.positionResult(
                tokens: MLXArray([Int32(9), 11, 14, 9, 11, 14]), videoGrids: [THW(2, 2, 2)])
            #expect(
                QwenVisionSeamPositionKernelTests.planes(result)
                    == Array(repeating: [0, 1, 2, 3, 4, 5], count: 3),
                "one position for each token")
            #expect(result.decodeState.deltas == [0], "decode delta")

            let image = try model.positionResult(
                tokens: MLXArray([Int32(9), 10, 14]), imageGrids: [THW(1, 2, 2)])
            #expect(
                QwenVisionSeamPositionKernelTests.planes(image)
                    == Array(repeating: [0, 1, 2], count: 3),
                "image grid passes through")

            #expect(throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry, "video t = 0") {
                try model.positionResult(
                    tokens: MLXArray([Int32(1)]), videoGrids: [THW(0, 2, 2)])
            }
        }

        @Test
        func towerFeaturesUseTheTextEmbeddingDType() throws {
            let model = try Self.withTower()
            let result = try model.visionFeatures(
                imagePixels: MLXArray((0 ..< 4).map { Float($0) / 4 }).reshaped(4, 1),
                imageGrids: [THW(1, 2, 2)],
                videoPixels: MLXArray((0 ..< 8).map { Float($0) / 8 }).reshaped(8, 1),
                videoGrids: [THW(2, 2, 2)])
            eval(result.flattenedFeatures)
            let textDType = model.scaledInputEmbeddings(MLXArray([Int32(0)]).reshaped(1, 1))
                .dtype
            #expect(result.flattenedFeatures.shape == [3, 8], "4 / 4 + 8 / 4 merged tokens")
            #expect(result.flattenedFeatures.dtype == textDType, "text embedding dtype")
            #expect(
                result.ordered.map(\.kind) == [
                    .image(index: 0),
                    .videoFrame(videoIndex: 0, frameIndex: 0),
                    .videoFrame(videoIndex: 0, frameIndex: 1),
                ], "image first, then the video frames")
            #expect(
                isFinite(result.flattenedFeatures).all().item(Bool.self), "features are finite")
        }

        @Test
        func sanitizeKeepsTheTowerOnlyWhenItIsServed() throws {
            let weights: [String: MLXArray] = [
                "model.visual.patch_embed.proj.weight": MLXArray.zeros([8, 2, 1, 1, 3]),
                "model.visual.blocks.0.attn.qkv.weight": MLXArray.zeros([24, 8]),
                "model.visual.pos_embed.position_ids": MLXArray.zeros([16]),
                "model.language_model.norm.weight": MLXArray.zeros([8]),
                "lm_head.weight": MLXArray.zeros([16, 8]),
                "mtp.fc.weight": MLXArray.zeros([8, 16]),
            ]
            let served = try Self.withTower().sanitize(weights: weights)
            #expect(
                Set(served.keys) == [
                    "vision_tower.patch_embed.proj.weight",
                    "vision_tower.blocks.0.attn.qkv.weight",
                    "language_model.model.norm.weight",
                    "language_model.lm_head.weight",
                ], "tower renamed, text moved under language_model, mtp and ids dropped")
            #expect(
                served["vision_tower.patch_embed.proj.weight"]?.shape == [8, 1, 1, 3, 2],
                "HF patch weight moves channels last")

            let textOnly = Self.textOnly().sanitize(weights: weights)
            #expect(
                Set(textOnly.keys) == [
                    "language_model.model.norm.weight", "language_model.lm_head.weight",
                ], "no tower: every vision key is dropped")
        }

        @Test
        func loadFilterFollowsTheTower() throws {
            let served = try Self.withTower().checkpointWeightLoadFilter
            let textOnly = Self.textOnly().checkpointWeightLoadFilter
            #expect(served("model.visual.blocks.0.attn.qkv.weight"), "tower keys load")
            #expect(!textOnly("model.visual.blocks.0.attn.qkv.weight"), "tower keys skip")
            #expect(!served("mtp.fc.weight"), "mtp keys never load")
        }

        @Test
        func configurationRoundTripsWithAllTokenFields() throws {
            let config = Qwen4ExpVLMConfiguration(
                text: Self.text(), visionConfiguration: try Self.vision(), imageTokenId: 10,
                videoTokenId: 11, visionStartTokenId: 9, visionEndTokenId: 14)
            var object = try #require(
                try JSONSerialization.jsonObject(with: JSONEncoder().encode(config))
                    as? [String: Any])
            #expect(object["image_token_id"] as? Int == 10, "image token encoded")
            #expect(object["language_model_only"] as? Bool == false, "flag encoded")
            object["image_token_index"] = 12
            object["video_token_index"] = 13
            object["language_model_only"] = true
            let decoded = try JSONDecoder().decode(
                Qwen4ExpVLMConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
            #expect(decoded.imageTokenId == 10, "image position token")
            #expect(decoded.videoTokenId == 11, "video position token")
            #expect(decoded.imageTokenIndex == 12, "image placeholder from image_token_index")
            #expect(decoded.videoTokenIndex == 13, "video placeholder from video_token_index")
            #expect(decoded.visionStartTokenId == 9, "vision start")
            #expect(decoded.visionEndTokenId == 14, "vision end")
            #expect(decoded.languageModelOnly, "language_model_only decoded")
            #expect(!decoded.servesVision, "language_model_only turns the tower off")
            #expect(decoded.visionConfiguration?.spatialMergeSize == 2, "vision config decoded")
            #expect(decoded.text.textConfig.hiddenSize == 8, "text config decoded")
            #expect(decoded.modelType == "qwen4_exp", "model type from the text config")

            let defaults = try JSONDecoder().decode(
                Qwen4ExpVLMConfiguration.self,
                from: JSONEncoder().encode(Qwen4ExpVLMConfiguration(text: Self.text())))
            #expect(defaults.imageTokenId == 248_056, "default image token")
            #expect(defaults.videoTokenIndex == 248_057, "default video placeholder")
            #expect(defaults.visionStartTokenId == 248_053, "default vision start")
            #expect(defaults.visionEndTokenId == 248_054, "default vision end")
            #expect(defaults.visionConfiguration == nil, "no vision config")
            #expect(!defaults.languageModelOnly, "language_model_only defaults to false")
        }

        @Test
        func forwardedPropertiesMatchTheTextModel() throws {
            let model = Self.textOnly()
            let inner = model.qwen4ExpTextTarget
            #expect(model.vocabularySize == 16, "vocabulary")
            #expect(model.kvHeads.isEmpty, "no layers, no KV heads")
            #expect(model.loraLayers.isEmpty, "no LoRA layers")
            #expect(model.newCache(parameters: nil).isEmpty, "no caches")
            #expect(model.cbv2LayerKinds.isEmpty, "no CBv2 layers")
            #expect(model.cbv2RecurrentStateSpec == inner.cbv2RecurrentStateSpec, "state spec")
            #expect(model.cbv2Capabilities == inner.cbv2Capabilities, "capabilities")
            #expect(model.cbv2Capabilities.supportsMTP, "Qwen4 serves MTP")
            #expect(
                (model.cbv2TargetAuxiliaryAllocationSpecs == nil)
                    == (inner.cbv2TargetAuxiliaryAllocationSpecs == nil),
                "auxiliary allocation specs")
            #expect(model.cbv2PositionAxisCount == 3, "three M-RoPE axes")
            #expect(model.cbv2SupportsPackedPrefill == inner.cbv2SupportsPackedPrefill, "packed")
            #expect(!model.supportsVisionSpanPrefill, "no vision span prefill")
            #expect(model.supportsCausalVisionPrefill, "causal vision prefill")
            #expect(
                model.cbv2MTPTargetIdentity == ObjectIdentifier(inner), "MTP target identity")
            #expect(model.skipWholeShardPrefetch == inner.mmapPLE, "prefetch follows mmap PLE")
            #expect(!model.needsIncrementalCheckpointMaterialization, "no fused experts")
            try model.materializeCheckpointWeightsIncrementally()
            try model.validateExternalPLEResources()
            #expect(model.cbv2CompleteCheckpointKVDTypes == [], "no QSA layers, no KV types")
            #expect(model.cbv2Qwen4CheckpointGeometries?.count == 0, "no QSA geometries")
            #expect(model.checkpointPerLayerQuantization == nil, "no policy staged")
            model.checkpointPerLayerQuantization = nil
            #expect(inner.checkpointPerLayerQuantization == nil, "setter reaches the text model")
            let path = "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj"
            #expect(
                model.quantizationPathAliases(for: path)
                    == qwen35GateUpQuantizationAliases(for: path), "gate/up aliases")
            #expect(
                throws: GenericGenerationError.nativeCBv2Required(modelType: "qwen4_exp_text")
            ) {
                try model.validateGenericGeneration()
            }
            let caches = try model.newCacheV2 { _, _ in throw CancellationError() }
            #expect(caches.isEmpty, "no layers, so the factory is not called")

            let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            let embedded = model.scaledInputEmbeddings(tokens)
            #expect(embedded.shape == [1, 3, 8], "embedding shape")
            // Exact: the same embedding table and the same tokens.
            #expect(
                SyntheticModel.maxAbsDifference(embedded, inner.scaledInputEmbeddings(tokens))
                    == 0, "the wrapper embeds like the text model")
        }
    }

    /// `PrismHadamardQwen35` built from a tiny configuration: one
    /// full-attention text layer (hidden 8, vocabulary 64) and the merge-1
    /// tower. The packed Prism weights are not loaded; these tests use only
    /// construction and the vision seam.
    @Suite
    struct PrismHadamardQwen35VisionKernelTests {

        static func configuration(
            vision: Bool = true, outHiddenSize: Int = 8, deepstack: [Int] = []
        ) -> [String: Any] {
            var visionConfig = QwenVisionSeamFeatureKernelTests.visionConfig
            visionConfig["out_hidden_size"] = outHiddenSize
            visionConfig["deepstack_visual_indexes"] = deepstack
            return [
                "schema_version": 2, "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5", "gdn_activation_layout": "grouped",
                "tensor_namespace": "mlx-vlm-qwen3_5", "hadamard_config": "hadamard.json",
                "components": ["text": true, "vision": vision, "mtp": false],
                "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
                "modules": [
                    [
                        "path": "model.embed_tokens", "block": 1024, "embedding": true,
                        "dtype": "float16",
                    ],
                    ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
                ],
                "image_token_id": 10, "video_token_id": 11,
                "image_token_index": 12, "video_token_index": 13,
                "vision_start_token_id": 9, "vision_end_token_id": 14,
                "text_config": [
                    "model_type": "qwen3_5_text", "hidden_size": 8, "num_hidden_layers": 1,
                    "intermediate_size": 16, "num_attention_heads": 1,
                    "num_key_value_heads": 1, "head_dim": 8, "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1, "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8, "linear_conv_kernel_dim": 2, "vocab_size": 64,
                    "full_attention_interval": 1, "num_experts": 0, "num_experts_per_tok": 0,
                    "mtp_num_hidden_layers": 0,
                ],
                "vision_config": visionConfig,
            ]
        }

        static func model() throws -> MLXVLM.PrismHadamardQwen35 {
            MLXRandom.seed(3)
            return try MLXVLM.PrismHadamardQwen35(
                configurationData: JSONSerialization.data(withJSONObject: configuration()))
        }

        @Test
        func unsupportedVisionDeclarationsAreRejected() throws {
            for (name, object) in [
                ("no vision component", Self.configuration(vision: false)),
                ("tower width differs from text width", Self.configuration(outHiddenSize: 16)),
                ("deepstack levels", Self.configuration(deepstack: [0])),
            ] {
                #expect(throws: PrismCheckpointError.self, "\(name)") {
                    try MLXVLM.PrismHadamardQwen35(
                        configurationData: JSONSerialization.data(withJSONObject: object))
                }
            }
        }

        @Test
        func seamFactsComeFromTheConfiguration() throws {
            let model = try Self.model()
            #expect(model.prismCheckpoint.hasVision, "vision declared")
            #expect(
                model.visionSeamConfiguration
                    == Qwen35VisionSeamConfiguration(
                        imagePlaceholderTokenId: 12, videoPlaceholderTokenId: 13,
                        imagePositionTokenId: 10, videoPositionTokenId: 11,
                        visionStartTokenId: 9, visionEndTokenId: 14,
                        spatialMergeSize: 1, temporalPatchSize: 1, attention: .causal),
                "placeholder and position token IDs")
            #expect(model.imagePlaceholderTokenId == 12, "image placeholder")
            #expect(model.videoPlaceholderTokenId == 13, "video placeholder")
            #expect(seamTowerGeometry(model) == [8, 1], "tower geometry")
        }

        @Test
        func positionsAndFeaturesUseTheSharedSeam() throws {
            let model = try Self.model()
            // Grid 1x1x2 with merge 1 is 2 tokens. Image positions start at 1;
            // the last text token starts after the largest position (2).
            let positions = try model.positionResult(
                tokens: MLXArray([Int32(9), 10, 10, 14]), imageGrids: [THW(1, 1, 2)])
            #expect(
                QwenVisionSeamPositionKernelTests.planes(positions)
                    == [[0, 1, 1, 3], [0, 1, 1, 3], [0, 1, 2, 3]], "t, h and w planes")
            #expect(positions.decodeState.deltas == [0], "decode delta 3 + 1 - 4")

            let features = try model.visionFeatures(
                imagePixels: MLXArray([Float(0.25), 0.75]).reshaped(2, 1),
                imageGrids: [THW(1, 1, 2)])
            #expect(features.flattenedFeatures.shape == [2, 8], "2 tokens of text width")
            #expect(features.flattenedFeatures.dtype == .float16, "Prism text dtype is float16")
            #expect(features.ordered.map(\.kind) == [.image(index: 0)], "one image slice")
        }

        @Test
        func legacyPrepareRejectsMediaAndKeepsText() throws {
            let model = try Self.model()
            let image = LMInput(
                text: .init(tokens: MLXArray([Int32(1)]).reshaped(1, 1)),
                image: .init(pixels: MLXArray.zeros([2, 1]), frames: [THW(1, 1, 2)]))
            #expect(
                throws: VLMError.processing("Prism media requires native CBv2 vision prefill")
            ) {
                try model.prepare(image, cache: [], windowSize: nil)
            }
            let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            let result = try model.prepare(LMInput(tokens: tokens), cache: [], windowSize: nil)
            guard case .tokens(let text) = result else {
                Issue.record("a short text prompt must come back as tokens")
                return
            }
            #expect(text.tokens.shape == [1, 3], "the prompt tokens are kept")
        }

        @Test
        func forwardedPropertiesMatchTheTextModel() throws {
            let model = try Self.model()
            let inner = model.languageModel
            #expect(model.vocabularySize == 64, "vocabulary")
            #expect(model.kvHeads == [1], "one attention layer with one KV head")
            #expect(model.loraLayers.count == inner.loraLayers.count, "LoRA layers")
            #expect(model.newCache(parameters: nil).count == 1, "one cache")
            #expect(model.cbv2LayerKinds.count == inner.cbv2LayerKinds.count, "CBv2 layers")
            #expect(model.cbv2RecurrentStateSpec == inner.cbv2RecurrentStateSpec, "state spec")
            var capabilities = inner.cbv2Capabilities
            capabilities.supportsMTP = false
            #expect(model.cbv2Capabilities == capabilities, "same capabilities without MTP")
            #expect(model.cbv2SupportsPackedPrefill == inner.cbv2SupportsPackedPrefill, "packed")
            #expect(!model.supportsVisionSpanPrefill, "no vision span prefill")
            #expect(model.supportsCausalVisionPrefill, "causal vision prefill")
            #expect(
                model.cbv2CompleteCheckpointKVDTypes == inner.cbv2CompleteCheckpointKVDTypes,
                "complete checkpoint KV types")
            let weights = ["a.weight": MLXArray.zeros([2])]
            #expect(
                Set(model.sanitize(weights: weights).keys) == ["a.weight"], "sanitize keeps keys")
            #expect(throws: CancellationError.self, "the cache factory error goes to the caller") {
                try model.newCacheV2 { _, _ in throw CancellationError() }
            }
            let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
            // Exact: the same embedding table and the same tokens.
            #expect(
                SyntheticModel.maxAbsDifference(
                    model.scaledInputEmbeddings(tokens), inner.scaledInputEmbeddings(tokens))
                    == 0, "the wrapper embeds like the text model")
        }
    }

    /// `MLXVLM.Qwen35MoE` sanitize overrides and the `Qwen35` extensions in
    /// `Qwen35MoE.swift` and `QwenVisionSeam.swift`. The tiny model has no
    /// experts, so the tests check only the weight dictionary.
    @Suite
    struct Qwen35MoEVLMSanitizeKernelTests {

        static func model() throws -> MLXVLM.Qwen35MoE {
            let object: [String: Any] = [
                "model_type": "qwen3_5_moe_vl",
                "image_token_id": 10, "video_token_id": 11,
                "image_token_index": 12, "video_token_index": 13,
                "vision_start_token_id": 9, "vision_end_token_id": 14,
                "text_config": [
                    "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                    "num_attention_heads": 1, "num_key_value_heads": 1,
                    "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 2, "vocab_size": 64,
                    "full_attention_interval": 1, "num_experts": 0, "num_experts_per_tok": 0,
                ],
                "vision_config": QwenVisionSeamFeatureKernelTests.visionConfig,
            ]
            return MLXVLM.Qwen35MoE(
                try SyntheticModel.configuration(MLXVLM.Qwen35Configuration.self, object))
        }

        /// Raw HF stacked experts in layer 0, MLX split experts in layer 1,
        /// a norm, the head and an MTP tensor.
        static func weights() -> [String: MLXArray] {
            [
                "model.language_model.layers.0.mlp.experts.gate_up_proj":
                    MLXArray(0 ..< 48).asType(.float32).reshaped(2, 6, 4),
                "model.language_model.layers.0.mlp.experts.down_proj":
                    MLXArray.zeros([2, 4, 3]),
                "model.language_model.layers.1.mlp.switch_mlp.gate_proj.weight":
                    MLXArray.ones([2, 3, 4]),
                "model.language_model.layers.1.mlp.switch_mlp.up_proj.weight":
                    MLXArray.ones([2, 3, 4]) * 2,
                "model.language_model.norm.weight": MLXArray.zeros([8]),
                "lm_head.weight": MLXArray.zeros([64, 8]),
                "mtp.fc.weight": MLXArray.zeros([8, 16]),
            ]
        }

        static let sanitizedKeys: Set<String> = [
            "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
            "language_model.model.layers.1.mlp.switch_mlp.gate_up_proj.weight",
            "language_model.model.norm.weight",
            "language_model.lm_head.weight",
        ]

        @Test
        func sourceLayoutIsFusedRenamedAndNormShifted() throws {
            let result = try Self.model().sanitize(weights: Self.weights())
            #expect(Set(result.keys) == Self.sanitizedKeys, "fused, renamed, mtp dropped")
            let stacked = try #require(
                result["language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight"])
            // Exact: the stacked tensor is moved, not changed.
            #expect(
                stacked.asArray(Float.self) == (0 ..< 48).map(Float.init),
                "raw stacked gate/up keeps its values")
            let fused = try #require(
                result["language_model.model.layers.1.mlp.switch_mlp.gate_up_proj.weight"])
            #expect(fused.shape == [2, 6, 4], "gate and up join on the output axis")
            // Exact: concatenation copies the values.
            #expect(
                fused[0..., ..<3, 0...].min().item(Float.self) == 1
                    && fused[0..., 3..., 0...].min().item(Float.self) == 2,
                "gate rows first, then up rows")
            // Exact: 0 + 1 in float32.
            #expect(
                result["language_model.model.norm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1, count: 8), "norm weight gets + 1")
        }

        @Test
        func metadataPathsFuseWithoutRenamingMLXCheckpoints() throws {
            let model = try Self.model()
            let mlx = model.sanitize(weights: Self.weights(), metadata: ["format": "mlx"])
            #expect(
                Set(mlx.keys) == [
                    "model.language_model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                    "model.language_model.layers.0.mlp.switch_mlp.down_proj.weight",
                    "model.language_model.layers.1.mlp.switch_mlp.gate_up_proj.weight",
                    "model.language_model.norm.weight",
                    "lm_head.weight",
                ], "MLX checkpoints keep their names and lose only mtp")
            // Exact: an MLX checkpoint norm is already converted.
            #expect(
                mlx["model.language_model.norm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 0, count: 8), "MLX norm is not shifted")

            let source = model.sanitize(weights: Self.weights(), metadata: [:])
            #expect(Set(source.keys) == Self.sanitizedKeys, "other formats use sanitize(weights:)")
            // Exact: the second fuse pass must not add 1 again.
            #expect(
                source["language_model.model.norm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1, count: 8), "norm shifted once")
        }

        @Test
        func baseClassExtensionsAnswerForTheMoEModel() throws {
            let model = try Self.model()
            let path = "model.language_model.layers.0.mlp.switch_mlp.gate_up_proj"
            #expect(
                model.quantizationPathAliases(for: path)
                    == qwen35GateUpQuantizationAliases(for: path), "gate/up aliases")
            #expect(model.quantizationPathAliases(for: "mtp.layers.0.mlp").isEmpty, "mtp has none")
            let seamModel: any QwenVisionSeamModel = model
            #expect(seamTowerGeometry(seamModel) == [8, 1], "hidden 8, one head")
        }
    }
}
