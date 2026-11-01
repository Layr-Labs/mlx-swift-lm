// Copyright © 2025 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_vl

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen3VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
    case missingImageGrid
    case invalidImagePixels(shape: [Int], expectedWidth: Int)
}

private let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35vl.ropeDeltas")

// MARK: - Processor

public struct Qwen3VLProcessor: UserInputProcessor {

    /// Uniform linspace sample cap per clip. Qwen's tower attends over the full
    /// T×H×W grid, so tower and prefill work scale with every sampled frame.
    /// Eight frames preserve whole-clip coverage while fitting the serving
    /// first-content budget; the previous 16-frame cap still missed it.
    public static let maxVideoFrames = 8

    /// Per-sampled-frame pixel ceiling for video. Artifact image processors can
    /// advertise multi-megapixel image limits (16 MP in the Qwen 3.5 serving
    /// artifact), but the video tower attends over every frame in one sequence:
    /// keeping a 720p raster for eight frames creates a multi-GiB score tensor
    /// on Qwen's unfused 72-dim attention heads. A 512² budget preserves useful
    /// spatial detail while bounding tower memory and first-content latency.
    public static let maxVideoFramePixels = 512 * 512

    private let config: Qwen3VLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let checkpointImageBounds: (min: Int, max: Int)?

    public init(_ config: Qwen3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.checkpointImageBounds = nil
    }

    init(
        _ config: Qwen3VLProcessorConfiguration, tokenizer: any Tokenizer,
        checkpointImageBounds: (min: Int, max: Int)
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.checkpointImageBounds = checkpointImageBounds
    }

    private var mediaProcessor: Qwen3VLMediaProcessor {
        Qwen3VLMediaProcessor(config, checkpointImageBounds: checkpointImageBounds)
    }

    static func defaultVisionTokenBudgetPixels(factor: Int, ceiling: Int) -> Int {
        Qwen3VLMediaProcessor.defaultVisionTokenBudgetPixels(factor: factor, ceiling: ceiling)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        try mediaProcessor.preprocess(images: images, processing: processing)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        if input.images.isEmpty, input.videos.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(tokens: promptArray)
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let concatenated = concatenated(imageFrames.map { $0.0 })
            processedImage = .init(pixels: concatenated, frames: imageFrames.map { $0.1 })

            if let frames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var accumulatedFrames: [[MLXArray]] = []

            for video in input.videos {
                accumulatedFrames.append(
                    try await mediaProcessor.videoFrames(video, processing: input.processing))
            }

            let videoFrames = try accumulatedFrames.map {
                try QwenVL.patchify(
                    images: $0,
                    mergeSize: config.mergeSize,
                    patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }

            let concatenated = concatenated(videoFrames.map { $0.0 })
            processedVideo = .init(pixels: concatenated, frames: videoFrames.map { $0.1 })

            if let frames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
}

public struct Qwen3VLProcessorConfiguration: Codable, Sendable {

    /// The synthesized min/max-pixel budget the processor consumes. Always
    /// resolved (non-optional) — see `var size`.
    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    /// The raw JSON `"size"` block. Newer Qwen3-VL configs (e.g. Qwen3.6-27B
    /// PARO) ship *only* `{longest_edge, shortest_edge}` — pixel-area budgets
    /// under the HF convention `shortest_edge → min_pixels`,
    /// `longest_edge → max_pixels` (same direct mapping GlmOcr uses); older
    /// configs ship `{min_pixels, max_pixels}`. Every field is optional so a
    /// config that omits any one still decodes: the processor loader swallows
    /// decode throws into a silent text-only fallback, so a hard requirement
    /// would make vision disappear instead of erroring.
    public struct SizeBudget: Codable, Sendable {
        public let maxPixels: Int?
        public let minPixels: Int?
        public let longestEdge: Int?
        public let shortestEdge: Int?

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
            case longestEdge = "longest_edge"
            case shortestEdge = "shortest_edge"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _minPixels: Int?
    private let _maxPixels: Int?
    private let _size: SizeBudget?
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String

    // Budget resolution order: explicit top-level legacy keys, then the
    // `size` block's legacy keys, then its new-style edges, then defaults.
    // The defaults are only reached when the config carries no budget at all.
    public var minPixels: Int {
        _minPixels ?? _size?.minPixels ?? _size?.shortestEdge ?? 4 * 28 * 28
    }  // default 3,136
    public var maxPixels: Int {
        _maxPixels ?? _size?.maxPixels ?? _size?.longestEdge ?? 16384 * 28 * 28
    }  // default 12,845,056

    public var size: Size { .init(maxPixels: maxPixels, minPixels: minPixels) }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case _size = "size"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
    }
}

extension Qwen3VLProcessorConfiguration {
    init(qwen35VisionConfiguration config: Qwen3VLConfiguration.VisionConfiguration) {
        self.imageMean = [0.5, 0.5, 0.5]
        self.imageStd = [0.5, 0.5, 0.5]
        self._minPixels = 65_536
        self._maxPixels = 16_777_216
        self._size = nil
        self.mergeSize = config.spatialMergeSize
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.imageProcessorType = "Qwen2VLImageProcessorFast"
    }
}

// MARK: - Model Configuration

public struct Qwen3VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numKeyValueHeads: Int?
        public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
        public let headDim: Int
        private let _ropeTheta: Double?
        public var ropeTheta: Double { _ropeTheta ?? 1_000_000 }
        public let maxPositionEmbeddings: Int
        private let _rmsNormEps: Double?
        public var rmsNormEps: Double { _rmsNormEps ?? 1e-6 }
        private let _ropeScaling: RoPEScaling?
        public var ropeScaling: RoPEScaling? { _ropeScaling }
        private let _normTopKProb: Bool?
        public var normTopKProb: Bool { _normTopKProb ?? true }
        private let _numExperts: Int?
        public var numExperts: Int { _numExperts ?? 0 }
        private let _numExpertsPerTok: Int?
        public var numExpertsPerTok: Int { _numExpertsPerTok ?? 0 }
        private let _decoderSparseStep: Int?
        public var decoderSparseStep: Int { _decoderSparseStep ?? 1 }
        private let _mlpOnlyLayers: [Int]?
        public var mlpOnlyLayers: [Int] { _mlpOnlyLayers ?? [] }
        private let _moeIntermediateSize: Int?
        public var moeIntermediateSize: Int { _moeIntermediateSize ?? intermediateSize }
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? false }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "silu" }
        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case _ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeScaling = "rope_scaling"
            case _normTopKProb = "norm_topk_prob"
            case _numExperts = "num_experts"
            case _numExpertsPerTok = "num_experts_per_tok"
            case _decoderSparseStep = "decoder_sparse_step"
            case _mlpOnlyLayers = "mlp_only_layers"
            case _moeIntermediateSize = "moe_intermediate_size"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
            case _hiddenAct = "hidden_act"
            case vocabSize = "vocab_size"
        }

        func validateForConstruction() {
            guard numExperts > 0 else { return }

            precondition(numExpertsPerTok > 0, "Qwen3-VL MoE top-k must be positive")
            precondition(
                numExpertsPerTok <= numExperts,
                "Qwen3-VL MoE top-k cannot exceed the expert count")
            precondition(
                moeIntermediateSize > 0,
                "Qwen3-VL MoE intermediate size must be positive")
            precondition(
                decoderSparseStep > 0,
                "Qwen3-VL MoE decoder sparse step must be positive")
            precondition(
                mlpOnlyLayers.allSatisfy { $0 >= 0 && $0 < numHiddenLayers },
                "Qwen3-VL MLP-only layer index is outside the decoder")
        }

        func usesSparseMoE(layerIndex: Int) -> Bool {
            numExperts > 0
                && !mlpOnlyLayers.contains(layerIndex)
                && (layerIndex + 1) % decoderSparseStep == 0
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let numPositionEmbeddings: Int
        private let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        private let _hiddenAct: String?
        // The fork keeps "gelu" (fe01df9f): MoE and Qwen4 vision configs must
        // declare gelu_pytorch_tanh, see visionMLPApproximationForConstruction.
        public var hiddenAct: String { _hiddenAct ?? "gelu" }
        private let _deepstackVisualIndexes: [Int]?
        public var deepstackVisualIndexes: [Int] { _deepstackVisualIndexes ?? [] }

        func visionMLPApproximationForConstruction() -> GELU.Approximation {
            // Converted dense checkpoints and the official dense/MoE vision configs
            // declare their activation explicitly. Preserve legacy behavior when the
            // field is missing or unsupported rather than changing old defaults.
            if modelType == "qwen3_5" || modelType == "qwen3_5_vision"
                || modelType == "qwen3_5_moe_vision"
            {
                switch _hiddenAct {
                case "gelu_pytorch_tanh": return .tanh
                case "gelu": return .none
                default: return .fast
                }
            }
            guard modelType == "qwen3_vl_moe" || modelType == "qwen4_exp" else { return .fast }
            precondition(
                hiddenAct == "gelu_pytorch_tanh",
                "Qwen3-VL MoE and Qwen4 vision blocks require gelu_pytorch_tanh")
            return .tanh
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case numPositionEmbeddings = "num_position_embeddings"
            case _inChannels = "in_channels"
            case _hiddenAct = "hidden_act"
            case _deepstackVisualIndexes = "deepstack_visual_indexes"
        }
    }

    public struct RoPEScaling: Codable, Sendable {
        public let type: String?
        public let mropeInterleaved: Bool?
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case type
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
        }

        public init(type: String? = nil, mropeInterleaved: Bool? = nil, mropeSection: [Int]? = nil)
        {
            self.type = type
            self.mropeInterleaved = mropeInterleaved
            self.mropeSection = mropeSection
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 151_655 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 151_656 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 151_652 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 151_653 }
    private let _visionTokenId: Int?
    public var visionTokenId: Int { _visionTokenId ?? 151_654 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabSize }
    private let _eosTokenId: [Int]?
    public var eosTokenId: [Int]? { _eosTokenId }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _visionTokenId = "vision_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }

    public init(
        textConfiguration: TextConfiguration, visionConfiguration: VisionConfiguration,
        modelType: String = "qwen3_vl", ignoreIndex: Int = -100, imageTokenId: Int = 151_655,
        videoTokenId: Int = 151_656, imageTokenIndex: Int? = nil, videoTokenIndex: Int? = nil,
        visionStartTokenId: Int = 151_652, visionEndTokenId: Int = 151_653,
        visionTokenId: Int = 151_654, vocabSize: Int? = nil, eosTokenId: [Int]? = nil
    ) {
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.modelType = modelType
        self._ignoreIndex = ignoreIndex
        self._imageTokenId = imageTokenId
        self._videoTokenId = videoTokenId
        self._imageTokenIndex = imageTokenIndex
        self._videoTokenIndex = videoTokenIndex
        self._visionStartTokenId = visionStartTokenId
        self._visionEndTokenId = visionEndTokenId
        self._visionTokenId = visionTokenId
        self._vocabSize = vocabSize
        self._eosTokenId = eosTokenId
    }
}

// MARK: - Vision

enum Qwen3VLVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    /// The fused SDPA Metal kernel supports head dims 64/80/128; the vision tower's is 72
    /// (1152/16), which otherwise falls back to composed ops that materialize the
    /// [L, L] score matrix per head. Zero-padding Q/K/V to 80 preserves the math exactly
    /// (padded dims contribute nothing to the dot products; `scale` is passed explicitly)
    /// and dispatches the O(L)-memory fused kernel. Same trick as `gemma4EnsureFusedSDPA`
    /// and mlx-vlm's Python `ensure_fused_sdpa`.
    static func ensureFusedSDPA(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray {
        let fusedDims = [64, 80, 128]
        let headDim = queries.dim(queries.ndim - 1)
        let target = fusedDims.first(where: { headDim <= $0 }) ?? headDim

        if target == headDim {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        }

        let widths: [IntOrPair] = [0, 0, 0, .init((0, target - headDim))]
        return MLXFast.scaledDotProductAttention(
            queries: MLX.padded(queries, widths: widths),
            keys: MLX.padded(keys, widths: widths),
            values: MLX.padded(values, widths: widths),
            scale: scale, mask: .none
        )[.ellipsis, ..<headDim]
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            // Attention is block-diagonal over the images in `cuSeqlens`, so each image is
            // attended independently instead of materializing a dense [L, L] additive mask
            // over the joint sequence (memory O((sum L_i)^2) for math that never crosses an
            // image boundary). Mirrors mlx-vlm's Python implementation.
            let seqlens = cuSeqlens.asArray(Int.self)
            var attendedParts: [MLXArray] = []
            for idx in 1 ..< seqlens.count {
                let start = seqlens[idx - 1]
                let end = seqlens[idx]
                guard end > start else { continue }
                attendedParts.append(
                    Qwen3VLVision.ensureFusedSDPA(
                        queries: queries[0..., 0..., start ..< end, 0...],
                        keys: keys[0..., 0..., start ..< end, 0...],
                        values: values[0..., 0..., start ..< end, 0...],
                        scale: scale))
            }
            let attended = concatenated(attendedParts, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int, approximation: GELU.Approximation) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(approximation: approximation)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(
                dim: config.hiddenSize,
                hiddenDim: config.intermediateSize,
                approximation: config.visionMLPApproximationForConstruction())
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            var states = hiddenStates
            states =
                states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {

        let config: Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else { continue }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0 ..< mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0 ..< merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0)  // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0 ..< h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0 ..< w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(posEmbed.weight.dtype)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: posEmbed.weight.dtype)

            for cornerIdx in 0 ..< 4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset ..< (offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        static func sanitize(
            weights: [String: MLXArray], config: Qwen3VLConfiguration.VisionConfiguration
        ) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            Self.sanitize(weights: weights, config: config)
        }
    }
}

// MARK: - Language

enum Qwen3VLLanguage {

    final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeIndices: MLXArray

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)

            // Precompute which of the three position planes (t/h/w) owns each
            // frequency so the interleave is a single takeAlong instead of a
            // per-frequency slice loop building ~dims lazy ops per forward.
            let sections = ropeScaling?.mropeSection ?? [24, 20, 20]
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, indices.count)
                for index in stride(from: offset, to: end, by: 3) {
                    indices[index] = Int32(dimension)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            if let attending = cache as? any CBv2AttendingLayerCache {
                let resolvedPositions: MLXArray
                if let positionIds {
                    precondition(
                        positionIds.shape == [3, batch, length],
                        "Qwen3-VL CBv2 positions must be [3, batch, length]")
                    resolvedPositions = positionIds
                } else {
                    // Capture every row's pre-update absolute offset before
                    // updateAndAttend advances the cache. Never consult the
                    // scalar KVCache.offset on a rectangular CBv2 batch.
                    let offsets = attending.positionOffsets + MLXArray(0, dtype: .int32)
                    precondition(
                        offsets.ndim == 1 && offsets.dim(0) == batch,
                        "Qwen3-VL CBv2 offsets must contain one value per row")
                    let steps = MLXArray(0 ..< length).asType(.int32)
                    let positions =
                        offsets.asType(.int32)[0..., .newAxis]
                        + steps[.newAxis, 0...]
                    resolvedPositions = broadcast(
                        positions[.newAxis, 0..., 0...],
                        to: [3, batch, length])
                }

                let (cosValues, sinValues) = rotaryEmbedding(
                    positionIds: resolvedPositions, dtype: x.dtype)
                (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                    q: queries, k: keys, cos: cosValues, sin: sinValues)
                let output = attending.updateAndAttend(
                    queries: queries, keys: keys, values: values,
                    scale: scale, sinks: nil
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)
                return wo(output)
            }

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)

            return result
        }
    }

    class FeedForward: Module, UnaryLayer {
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            preconditionFailure("Qwen3-VL feed-forward base class cannot execute")
        }
    }

    final class MLP: FeedForward {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class SparseMoeBlock: FeedForward {
        let numExperts: Int
        let topK: Int
        let normTopKProb: Bool

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            self.numExperts = config.numExperts
            self.topK = config.numExpertsPerTok
            self.normTopKProb = config.normTopKProb

            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                fuseGateUp: true)
        }

        func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
            let probabilities = softmax(gate(x), axis: -1, precise: true)
            let kth = numExperts - topK
            let indices = argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = takeAlong(probabilities, indices, axis: -1)
            if normTopKProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
            return (indices, scores)
        }

        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, scores) = route(x)
            let canFlatten: Bool
            if x.ndim == 2 {
                canFlatten =
                    indices.ndim == 2 && scores.ndim == 2
                    && indices.dim(0) == x.dim(0) && scores.dim(0) == x.dim(0)
                    && indices.dim(1) == topK && scores.dim(1) == topK
            } else if x.ndim == 3 {
                canFlatten =
                    indices.ndim == 3 && scores.ndim == 3
                    && indices.dim(0) == x.dim(0) && scores.dim(0) == x.dim(0)
                    && indices.dim(1) == x.dim(1) && scores.dim(1) == x.dim(1)
                    && indices.dim(2) == topK && scores.dim(2) == topK
            } else {
                canFlatten = false
            }
            guard canFlatten else {
                return weightedExpertSum(switchMLP(x, indices), scores)
            }

            // Gather kernels operate on one contiguous token axis. Reshaping
            // does not reorder tokens, selected experts, or scores.
            let flatX = x.ndim == 2 ? x : x.reshaped(-1, x.dim(-1))
            let flatIndices = indices.ndim == 2 ? indices : indices.reshaped(-1, topK)
            let flatScores = scores.ndim == 2 ? scores : scores.reshaped(-1, topK)
            let reduced = switchMLP.callAndWeightedReduce(
                flatX, flatIndices, weights: flatScores,
                fuseSortedReduction: false)
            return x.ndim == 2
                ? reduced
                : reduced.reshaped(x.dim(0), x.dim(1), x.dim(2))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: FeedForward

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if config.usesSparseMoE(layerIndex: layerIndex) {
                _mlp.wrappedValue = SparseMoeBlock(config)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            }
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }

        func cbv2Forward(
            _ x: MLXArray,
            cache: any CBv2AttendingLayerCache,
            positionIds: MLXArray?
        ) -> MLXArray {
            guard let cache = cache as? KVCache else {
                preconditionFailure("Qwen3-VL CBv2 layer cache must conform to KVCache")
            }
            return self(x, mask: nil, cache: cache, positionIds: positionIds)
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            config.validateForConstruction()
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualIndices: [Int]?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            // Loop-invariant: the scatter targets the same token positions in
            // every deepstack layer, so build the index array once rather than
            // re-uploading it per layer.
            var deepstackIndices: MLXArray?
            if deepstackEmbeds != nil, let visualIndices, !visualIndices.isEmpty {
                deepstackIndices = MLXArray(visualIndices.map { UInt32($0) })
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let deepstackIndices
                {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualIndices: deepstackIndices,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        func cbv2Forward(
            _ inputIds: MLXArray?,
            inputEmbeddings: MLXArray?,
            cache: [any CBv2AttendingLayerCache],
            positionIds: MLXArray?,
            deepstackEmbeds: [MLXArray]
        ) -> MLXArray {
            precondition(
                cache.count == layers.count,
                "Qwen3-VL CBv2 requires one cache per language layer")
            precondition(
                deepstackEmbeds.count <= layers.count,
                "Qwen3-VL DeepStack has more injections than language layers")

            let initial: MLXArray
            if let inputEmbeddings {
                initial = inputEmbeddings
            } else if let inputIds {
                initial = embedTokens(inputIds)
            } else {
                preconditionFailure("Either input ids or embeddings must be provided")
            }

            var hidden = initial
            for (index, layer) in layers.enumerated() {
                hidden = layer.cbv2Forward(
                    hidden, cache: cache[index], positionIds: positionIds)
                if index < deepstackEmbeds.count {
                    let injection = deepstackEmbeds[index]
                    precondition(
                        injection.shape == hidden.shape,
                        "Qwen3-VL DeepStack layer \(index) shape \(injection.shape) != hidden shape \(hidden.shape)"
                    )
                    hidden = hidden + injection.asType(hidden.dtype)
                }
            }
            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualIndices: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let result = hiddenStates
            result[0..., visualIndices, 0...] =
                result[0..., visualIndices, 0...] + visualEmbeds

            return result
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            state: LMOutput.State?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualIndices: [Int]?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            var state = state ?? .init()

            if pixelValues != nil {
                state[ropeDeltasKey] = nil
            }

            var positionIds = providedPositionIds

            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                // A stateless text continuation (no deltas in `state`, a warm
                // cache, no media) uses zero deltas in the cached branch below.
                // Recomputing here would restart positions at zero.
                let statelessTextContinuation =
                    state[ropeDeltasKey] == nil && (cache?.first?.offset ?? 0) > 0
                    && imageGridTHW == nil && videoGridTHW == nil
                if statelessTextContinuation {
                    state[ropeDeltasKey] = MLXArray.zeros([1], dtype: .int32)
                }
                if (cache?.first?.offset ?? 0) == 0 || state[ropeDeltasKey] == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)

                        positionIds = computed
                        state[ropeDeltasKey] = deltas
                    } else if let cache, state[ropeDeltasKey] == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue

                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas = state[ropeDeltasKey] {
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)

                    positionIds = Qwen3VLLanguage.cachedPositionIds(
                        batch: batch, seqLength: seqLength,
                        offset: cache.last?.offset ?? 0, ropeDeltas: ropeDeltas)
                }
            }

            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualIndices: visualIndices,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output, state: state)
        }

        func cbv2Forward(
            _ inputIds: MLXArray?,
            inputEmbeddings: MLXArray? = nil,
            cache: [any CBv2AttendingLayerCache],
            positionIds: MLXArray?,
            deepstackEmbeds: [MLXArray] = []
        ) -> LMOutput {
            var output = model.cbv2Forward(
                inputIds,
                inputEmbeddings: inputEmbeddings,
                cache: cache,
                positionIds: positionIds,
                deepstackEmbeds: deepstackEmbeds)
            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }
            return LMOutput(logits: output)
        }

    }
}

extension Qwen3VLLanguage {

    static func cachedPositionIds(
        batch: Int, seqLength: Int, offset: Int, ropeDeltas: MLXArray
    ) -> MLXArray {
        var delta = MLXArray(offset).asType(.int32) + ropeDeltas.asType(.int32)
        if delta.dim(0) == 1 && batch > 1 {
            delta = repeated(delta, count: batch, axis: 0)
        }
        let base = broadcast(
            MLXArray(0 ..< seqLength).asType(.int32)[.newAxis, 0...],
            to: [batch, seqLength])
        // One delta per row, not per position, even when seqLength == batch.
        let positions = base + delta.reshaped(-1, 1)
        return broadcast(positions[.newAxis, 0..., 0...], to: [3, batch, seqLength])
    }

    /// - Parameter positionOffset: the absolute M-RoPE position the first
    ///   token should occupy. Zero (the default) starts the index at absolute
    ///   zero — correct for a cold full prefill. A warm continuation that
    ///   resumes a cached prefix passes the **Position Anchor** (cache offset +
    ///   the rope delta accumulated by the cached images), so a *new* image in
    ///   the remainder gets its diverging temporal/height/width positions
    ///   computed *from* the anchor instead of resetting to zero. The returned
    ///   delta is shifted with the positions, so it remains
    ///   `maxPositionId + 1 − tokenCount` in the offset frame.
    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil,
        positionOffset: Int = 0
    ) -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            var positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            if positionOffset != 0 {
                positionIds3D = positionIds3D + MLXArray(Int32(positionOffset))
            }
            // Text-only positions are `arange + positionOffset`, so the delta
            // (`maxPos + 1 − tokenCount`) collapses to `positionOffset`.
            let deltas = MLXArray(Array(repeating: Int32(positionOffset), count: batchSize))
            return (positionIds3D, deltas)
        }

        positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIds .== MLXArray(visionStartTokenId))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIds))
            let visionStartIdx = argMax(visionStartWeighted).item(Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()).item(
                Int.self)
            let videoNums = ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()).item(
                Int.self)

            let inputTokens = batchInputIds.asArray(Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIdsList.isEmpty {
                var llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]
                // Shift the whole batch item into the anchor frame. Applied to
                // `llmPositions` (not just the final array) so `maxPosId` below
                // — and therefore the returned delta — is computed in the same
                // offset frame the positions live in.
                if positionOffset != 0 {
                    llmPositions = llmPositions + MLXArray(Int32(positionOffset))
                }

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIds = newPositions

                let maxPosId = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }
}

private func qwen3VLPolicyPathCandidates(for path: String) -> [String] {
    let prefixes = [
        "language_model.model.", "model.language_model.", "language_model.", "model.",
    ]
    var suffix = path
    for prefix in prefixes where path.hasPrefix(prefix) {
        suffix = String(path.dropFirst(prefix.count))
        break
    }

    var candidates = [path]
    for candidate in [
        "language_model.model." + suffix,
        "model.language_model." + suffix,
        "model." + suffix,
        "language_model." + suffix,
        suffix,
    ] where !candidates.contains(candidate) {
        candidates.append(candidate)
    }
    return candidates
}

private func qwen3VLGateUpQuantizationAliases(for path: String) -> [String] {
    var aliases: [String] = []
    func appendCandidates(for name: String) {
        for candidate in qwen3VLPolicyPathCandidates(for: name)
        where candidate != path && !aliases.contains(candidate) {
            aliases.append(candidate)
        }
    }

    let fusedSuffix = ".gate_up_proj"
    if path.hasSuffix(fusedSuffix) {
        let base = String(path.dropLast(fusedSuffix.count))
        appendCandidates(for: path)
        appendCandidates(for: "\(base).gate_proj")
        appendCandidates(for: "\(base).up_proj")
    } else if path.hasSuffix(".switch_mlp.gate_proj")
        || path.hasSuffix(".switch_mlp.up_proj")
        || path.hasSuffix(".switch_mlp.down_proj")
    {
        appendCandidates(for: path)
    }
    return aliases
}

// MARK: - Model

public final class Qwen3VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel

    public let config: Qwen3VLConfiguration
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ config: Qwen3VLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config)
    }

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    /// Qwen3-VL is a full-attention decoder. Dense and MoE checkpoints share
    /// the same KV geometry.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        let text = config.textConfiguration
        return (0 ..< text.numHiddenLayers).map { index in
            CBv2LayerKind(
                attention: .full,
                headDim: text.headDim,
                kvHeads: text.numKeyValueHeads,
                queryHeads: text.numAttentionHeads,
                modelLayerIndex: index)
        }
    }

    /// Paged projections and three-axis positioned/causal embedding forwards
    /// share the attending-cache contract. Prefix, compiled decode, packed
    /// prefill and MTP retain their separate qualification gates.
    public var cbv2Capabilities: CBv2ModelCapabilities {
        CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: true,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: false)
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { index, kind in
            try makeLayerCache(index, kind)
        }
    }

    public var imagePlaceholderTokenId: Int { config.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { config.videoTokenIndex }

    /// Compute immutable request-owned prompt positions and the decode delta.
    public func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]? = nil,
        videoGrids: [THW]? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> Qwen35PositionResult {
        guard tokens.ndim == 1 || tokens.ndim == 2 else {
            throw Qwen35PositionSeamError.invalidInputRank(tokens.ndim)
        }
        let batchedTokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens

        var mask = attentionMask
        if let attentionMask {
            guard attentionMask.ndim == 1 || attentionMask.ndim == 2 else {
                throw Qwen35PositionSeamError.invalidAttentionMaskRank(attentionMask.ndim)
            }
            mask =
                attentionMask.ndim == 1
                ? attentionMask.expandedDimensions(axis: 0) : attentionMask
            guard mask?.shape == batchedTokens.shape else {
                throw Qwen35PositionSeamError.attentionMaskShapeMismatch
            }
        }

        let imageGrids = imageGrids?.nilIfEmpty
        let videoGrids = videoGrids?.nilIfEmpty
        if imageGrids != nil || videoGrids != nil, batchedTokens.dim(0) != 1 {
            throw Qwen35PositionSeamError.multimodalBatchUnsupported(batchedTokens.dim(0))
        }
        let merge = config.visionConfiguration.spatialMergeSize
        for (kind, grids) in [("image", imageGrids ?? []), ("video", videoGrids ?? [])] {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validateVisualTokenRuns(
            tokens: batchedTokens, attentionMask: mask,
            imageGrids: imageGrids ?? [], videoGrids: videoGrids ?? [], merge: merge)

        let (positions, deltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: batchedTokens,
            imageGridTHW: imageGrids,
            videoGridTHW: videoGrids,
            spatialMergeSize: merge,
            imageTokenId: config.imageTokenIndex,
            videoTokenId: config.videoTokenIndex,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: mask)
        return Qwen35PositionResult(
            promptPositionIds: positions,
            decodeState: Qwen35PositionState(deltas: deltas.asArray(Int32.self)),
            promptLength: batchedTokens.dim(1))
    }

    private func validateVisualTokenRuns(
        tokens: MLXArray,
        attentionMask: MLXArray?,
        imageGrids: [THW],
        videoGrids: [THW],
        merge: Int
    ) throws {
        guard tokens.dim(0) == 1 else { return }
        let values = tokens[0, 0...].asArray(Int32.self).map(Int.init)
        let maskValues = attentionMask?.asType(.int32)[0, 0...].asArray(Int32.self)
        var imageIndex = 0
        var videoIndex = 0
        var cursor = 0
        while cursor < values.count {
            let visible = maskValues.map { $0[cursor] == 1 } ?? true
            let token = values[cursor]
            guard visible,
                token == config.imageTokenIndex || token == config.videoTokenIndex
            else {
                cursor += 1
                continue
            }

            let isImage = token == config.imageTokenIndex
            let kind = isImage ? "image" : "video"
            let gridIndex = isImage ? imageIndex : videoIndex
            let grids = isImage ? imageGrids : videoGrids
            var end = cursor + 1
            while end < values.count,
                maskValues.map { $0[end] == 1 } ?? true, values[end] == token
            {
                end += 1
            }
            let actual = end - cursor
            guard gridIndex < grids.count else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: 0, actual: actual)
            }
            let expected = try mergedTokenCount(
                grids[gridIndex], merge: merge, kind: kind, index: gridIndex)
            guard actual == expected else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: expected, actual: actual)
            }
            if isImage { imageIndex += 1 } else { videoIndex += 1 }
            cursor = end
        }

        if imageIndex < imageGrids.count {
            let expected = try mergedTokenCount(
                imageGrids[imageIndex], merge: merge, kind: "image", index: imageIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "image", gridIndex: imageIndex, expected: expected, actual: 0)
        }
        if videoIndex < videoGrids.count {
            let expected = try mergedTokenCount(
                videoGrids[videoIndex], merge: merge, kind: "video", index: videoIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "video", gridIndex: videoIndex, expected: expected, actual: 0)
        }
    }

    private func mergedTokenCount(
        _ grid: THW, merge: Int, kind: String, index: Int
    ) throws -> Int {
        let (spatial, spatialOverflow) = (grid.h / merge).multipliedReportingOverflow(
            by: grid.w / merge)
        let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
        guard !spatialOverflow, !temporalOverflow else {
            throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
        }
        return count
    }

    private func visionFeatureBundle(
        pixelParts: [MLXArray], grids: [THW], textDType: DType
    ) -> (features: MLXArray, deepstack: [MLXArray], counts: [Int]) {
        let pixels = concatenated(
            pixelParts.map {
                $0.asType(visionModel.patchEmbed.proj.weight.dtype)
            })
        let (hidden, deepstack) = visionModel(pixels, gridTHW: grids)
        let merge = config.visionConfiguration.spatialMergeSize
        let counts = grids.map { $0.product / (merge * merge) }
        let splitIndices = cumulativeSplitIndices(from: counts)
        let features = concatenated(
            hidden.split(indices: splitIndices)
        ).asType(textDType)
        let flattenedDeepstack = deepstack.map {
            concatenated($0.split(indices: splitIndices)).asType(textDType)
        }
        return (features, flattenedDeepstack, counts)
    }

    /// Image-only vision seam for EngineV2. Each final feature is
    /// `[1, spanLength, hidden]`; DeepStack is ordered language layer first,
    /// then image span.
    public func cbv2VisionFeatures(
        imagePixels: MLXArray, imageGrids: [THW]
    ) throws -> (features: [MLXArray], deepstack: [[MLXArray]]) {
        guard !imageGrids.isEmpty else { throw Qwen3VLError.missingImageGrid }
        let vision = config.visionConfiguration
        let merge = vision.spatialMergeSize
        var expectedRows = 0
        for (index, grid) in imageGrids.enumerated() {
            guard grid.t > 0, grid.h > 0, grid.w > 0,
                grid.h % merge == 0, grid.w % merge == 0
            else {
                throw Qwen35VisionSeamError.invalidGrid(kind: "image", index: index)
            }
            let (spatial, spatialOverflow) = grid.h.multipliedReportingOverflow(by: grid.w)
            let (rows, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
            let (total, totalOverflow) = expectedRows.addingReportingOverflow(rows)
            guard !spatialOverflow, !temporalOverflow, !totalOverflow else {
                throw Qwen35VisionSeamError.invalidGrid(kind: "image", index: index)
            }
            expectedRows = total
        }

        let expectedWidth =
            vision.inChannels * vision.temporalPatchSize * vision.patchSize * vision.patchSize
        guard imagePixels.ndim == 2, imagePixels.dim(1) == expectedWidth else {
            throw Qwen3VLError.invalidImagePixels(
                shape: imagePixels.shape, expectedWidth: expectedWidth)
        }
        guard imagePixels.dim(0) == expectedRows else {
            throw Qwen35VisionSeamError.pixelCountMismatch(
                kind: "image", expected: expectedRows, actual: imagePixels.dim(0))
        }

        // Quantized embeddings store packed weights in an integer dtype;
        // vision features must match the embedding OUTPUT activation dtype.
        let textDType = languageModel.model.embedTokens(
            MLXArray([Int32(0)]).reshaped([1, 1])
        ).dtype
        let bundle = visionFeatureBundle(
            pixelParts: [imagePixels], grids: imageGrids, textDType: textDType)
        let expectedFeatures = bundle.counts.reduce(0, +)
        guard bundle.features.ndim == 2, bundle.features.dim(0) == expectedFeatures,
            bundle.features.dim(1) == config.textConfiguration.hiddenSize
        else {
            throw Qwen35VisionSeamError.featureCountMismatch(
                expected: expectedFeatures, actual: bundle.features.dim(0))
        }
        let splitIndices = cumulativeSplitIndices(from: bundle.counts)
        func perImage(_ output: MLXArray) -> [MLXArray] {
            if bundle.counts.count == 1 {
                return [output.expandedDimensions(axis: 0)]
            }
            return output.split(indices: splitIndices).map {
                $0.expandedDimensions(axis: 0)
            }
        }
        let features = perImage(bundle.features)
        let deepstack = try bundle.deepstack.enumerated().map { layer, output in
            guard output.ndim == 2, output.shape == bundle.features.shape else {
                throw CBv2MultimodalError.embeddingMismatch(
                    "Qwen3-VL DeepStack layer \(layer) shape \(output.shape) != final vision shape \(bundle.features.shape)"
                )
            }
            return perImage(output)
        }
        guard deepstack.count == vision.deepstackVisualIndexes.count else {
            throw CBv2MultimodalError.embeddingMismatch(
                "Qwen3-VL vision returned \(deepstack.count) DeepStack layers; expected \(vision.deepstackVisualIndexes.count)"
            )
        }
        return (features: features, deepstack: deepstack)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, [Int]) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        let specialMask = (imageMask .|| videoMask)

        // Flat positions of the visual tokens, the same indexing `applyDeepstack` scatters
        // by. Everything below derives from these: broadcasting the mask across the hidden
        // dimension only to read it back would download `batch * seq * hidden` bools.
        let visualIndices = nonZero(specialMask)

        let hiddenSize = inputEmbeds.dim(-1)
        guard visualIndices.count * hiddenSize == imageFeatures.size else {
            throw Qwen3VLError.featureTokenMismatch(
                expected: visualIndices.count, actual: imageFeatures.dim(0))
        }
        guard !visualIndices.isEmpty else { return (inputEmbeds, visualIndices) }

        // Scatter over a [batch * seq, hidden] view so one row index places a whole embedding.
        let result = inputEmbeds.reshaped([-1, hiddenSize])
        result[MLXArray(visualIndices.map { Int32($0) }), 0...] =
            imageFeatures.reshaped([visualIndices.count, hiddenSize])

        return (result.reshaped(inputEmbeds.shape), visualIndices)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    /// The vision tower, the image→token merge, and the per-layer deepstack
    /// features for one call's inputs. All of it is relative to the tokens in
    /// this call, which is what lets a windowed forward slice it per chunk.
    private struct VisionInputs {
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?
        var inputEmbeddings: MLXArray?
        var visualIndices: [Int]?
        var deepstackEmbeds: [MLXArray]?
    }

    private func visionInputs(_ input: LMInput) throws -> VisionInputs {
        var result = VisionInputs()

        let dtype = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            result.imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            result.videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            result.pixelValues = concatenated(pixelParts)
        }

        guard let pixelValues = result.pixelValues,
            let framesList = combinedFrames(
                imageFrames: result.imageFrames, videoFrames: result.videoFrames
            ).nilIfEmpty
        else {
            return result
        }

        let inputIds = input.text.tokens
        let textEmbeds = languageModel.model.embedTokens(inputIds)
        // The same tower, split, and dtype cast the CBv2 vision seam uses.
        let bundle = visionFeatureBundle(
            pixelParts: [pixelValues], grids: framesList, textDType: textEmbeds.dtype)

        let (mergedEmbeds, visualIndices) = try mergeInputIdsWithImageFeatures(
            imageFeatures: bundle.features,
            inputEmbeds: textEmbeds,
            inputIds: inputIds,
            imageTokenIndex: config.imageTokenIndex,
            videoTokenIndex: config.videoTokenIndex)

        result.inputEmbeddings = mergedEmbeds
        result.visualIndices = visualIndices
        result.deepstackEmbeds = bundle.deepstack.isEmpty ? nil : bundle.deepstack

        return result
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        // Text-only follow-up processors may return rank-1 tokens. The language
        // model and continuation slicers require [batch, sequence], so normalize
        // before choosing a warm path rather than falling through and trapping.
        let inputIds2D = inputIds.ndim == 1 ? inputIds[.newAxis, 0...] : inputIds
        let cacheOffset = cache.first?.offset ?? 0
        // Resolved before the routing decision so a warm cache fails closed on either path.
        let positionOffset = try QwenVL.continuationAnchor(
            model: "Qwen3VL", key: ropeDeltasKey, cacheOffset: cacheOffset,
            batchSize: inputIds2D.dim(0), state: state)

        let window = prefill.resolvedStepSize()
        if inputIds2D.dim(0) == 1, inputIds2D.dim(-1) > 0,
            cacheOffset > 0 || inputIds2D.dim(-1) > window
        {
            return try prepareContinuation(
                input, inputIds: inputIds2D, cache: cache, cacheOffset: cacheOffset,
                positionOffset: positionOffset, prefill: prefill)
        }

        let vision = try visionInputs(input)
        let typedCache = castCache(cache)

        // `state` is forwarded so a warm text continuation that did not take the
        // windowed path above still anchors at the cache offset rather than zero.
        let languageOutput = languageModel(
            inputIds,
            cache: typedCache,
            state: state,
            inputEmbeddings: vision.inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualIndices: vision.visualIndices,
            deepstackEmbeds: vision.deepstackEmbeds,
            pixelValues: vision.pixelValues,
            imageGridTHW: vision.imageFrames,
            videoGridTHW: vision.videoFrames)

        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(languageOutput)
    }

    /// Warm, windowed continuation — the path `prepare` also uses for long cold prompts.
    ///
    /// The vision tower and image→token merge run once over the remainder, whose M-RoPE positions
    /// are computed once from the position anchor (cache offset plus the rope delta in `state`),
    /// so a new image's t/h/w indices start there rather than at zero. The language model is then
    /// driven in `prefill`-sized chunks, bounding the attention scratch to `[heads, chunk, L]`. The
    /// visual mask and per-layer deepstack tensors slice in lockstep, deepstack rows by
    /// visual-token count rather than by token index.
    ///
    /// The returned state carries `getRopeIndex` delta − cache offset.
    private func prepareContinuation(
        _ input: LMInput,
        inputIds: MLXArray,
        cache: [any KVCache],
        cacheOffset: Int,
        positionOffset: Int,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let remainderLength = inputIds.dim(-1)
        precondition(remainderLength > 0, "prepareContinuation needs a non-empty remainder")

        let vision = try visionInputs(input)

        let (positionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: vision.imageFrames,
            videoGridTHW: vision.videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenIndex,
            videoTokenId: config.videoTokenIndex,
            visionStartTokenId: config.visionStartTokenId,
            // The single-shot path forwards no attention mask; retain the same
            // M-RoPE coordinates regardless of whether prefill is windowed.
            attentionMask: nil,
            positionOffset: positionOffset)

        let visualIndices = vision.visualIndices ?? []
        let typedCache = castCache(cache)
        var visualStart = 0

        /// One forward over `range`, slicing positions, embeddings, and the per-layer deepstack
        /// features in lockstep. Deepstack rows are indexed by visual-token count rather than by
        /// token index, so `visualStart` advances with the ranges — which arrive in order, first
        /// from `forEachChunk` and then the reserved tail.
        func forward(_ range: Range<Int>) -> LMOutput {
            let chunkEmbeds = vision.inputEmbeddings?[0..., range, 0...]
            let chunkIds = chunkEmbeds == nil ? inputIds[0..., range] : nil

            var visualEnd = visualStart
            while visualEnd < visualIndices.count, visualIndices[visualEnd] < range.upperBound {
                visualEnd += 1
            }

            let chunkVisualIndices: [Int]?
            let chunkDeepstack: [MLXArray]?
            if visualEnd > visualStart {
                chunkVisualIndices = visualIndices[visualStart ..< visualEnd].map {
                    $0 - range.lowerBound
                }
                chunkDeepstack = vision.deepstackEmbeds?.map {
                    $0[visualStart ..< visualEnd, 0...]
                }
            } else {
                chunkVisualIndices = nil
                chunkDeepstack = nil
            }
            visualStart = visualEnd

            return languageModel(
                chunkIds,
                cache: typedCache,
                state: nil,
                inputEmbeddings: chunkEmbeds,
                mask: nil,
                positionIds: positionIds[0..., 0..., range],
                visualIndices: chunkVisualIndices,
                deepstackEmbeds: chunkDeepstack,
                // Never the pixels: a non-nil value here clears the carried
                // anchor and restarts positions at zero.
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil)
        }

        let processed = try prefill.forEachChunk(total: remainderLength) { range in
            _ = forward(range)
            if let typedCache {
                asyncEval(typedCache)
            }
        }
        if processed > 0, let typedCache {
            eval(typedCache)
        }

        let lastLogits = forward(processed ..< remainderLength).logits
        prefill.progress?(remainderLength, remainderLength)

        return .logits(
            LMOutput(
                logits: lastLogits,
                state: QwenVL.continuationResumeState(
                    ropeDeltas: ropeDeltas, cacheOffset: cacheOffset, key: ropeDeltasKey)))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        // CBv2 attending caches carry their own positions; they never use `state`.
        if let cbv2 = castCBv2Caches(cache) {
            return languageModel.cbv2Forward(input.tokens, cache: cbv2, positionIds: nil)
        }
        precondition(
            (cache?.first?.offset ?? 0) == 0 || state?[ropeDeltasKey] != nil,
            "Qwen3VL cannot continue a warm prompt cache without \(ropeDeltasKey.id)")
        let typedCache = castCacheOptional(cache)

        let result = languageModel(
            input.tokens,
            cache: typedCache,
            state: state,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualIndices: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result
    }

    /// Stateless forward. CBv2 attending caches take the CBv2 path. Other
    /// caches continue a warm text sequence at the cache offset (text rope
    /// deltas are zero) or compute positions from the tokens on a fresh cache.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        if let cbv2 = castCBv2Caches(cache) {
            return languageModel.cbv2Forward(
                inputs, cache: cbv2, positionIds: nil
            ).logits
        }
        let typedCache = castCacheOptional(cache)

        return languageModel(
            inputs,
            cache: typedCache,
            state: nil,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualIndices: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        ).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = key

            if newKey.contains("model") {
                if newKey.contains("model.visual") {
                    newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if newKey.contains("model.language_model") {
                    newKey = newKey.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                }
            } else if newKey.contains("lm_head") {
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if config.textConfiguration.tieWordEmbeddings && newKey.contains(".lm_head.") {
                // if using tieWordEmbeddings omit these keys as they will not be consumed
                continue
            }

            adjusted[newKey] = value
        }

        let fused = fuseSwitchGLUGateUpWeights(
            weights: adjusted,
            perLayerQuantization: checkpointPerLayerQuantization,
            quantizationAliases: qwen3VLGateUpQuantizationAliases,
            setFused: { path, fused in
                setSwitchGLUGateUpFused(
                    fused, at: path,
                    aliases: qwen3VLPolicyPathCandidates(for: path),
                    in: self)
            })
        return visionModel.sanitize(weights: fused)
    }
}

extension Qwen3VL: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen3VLGateUpQuantizationAliases(for: path)
    }
}

extension Qwen3VL: QuantizationPolicyReceiving {}

extension Qwen3VL: PreparedInputSplitting {

    public func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
        -> LMInput?
    {
        QwenVL.splitPreparedInput(
            input,
            droppingFirst: prefixTokenCount,
            imageTokenId: config.imageTokenIndex,
            videoTokenId: config.videoTokenIndex,
            mergeSize: config.visionConfiguration.spatialMergeSize)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen3VL {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }

    fileprivate func castCBv2Caches(
        _ cache: [any KVCache]?
    ) -> [any CBv2AttendingLayerCache]? {
        guard let cache, !cache.isEmpty else { return nil }
        let cbv2 = cache.compactMap { $0 as? any CBv2AttendingLayerCache }
        guard !cbv2.isEmpty else { return nil }
        precondition(
            cbv2.count == cache.count,
            "Qwen3-VL cannot mix ordinary and CBv2 layer caches")
        return cbv2
    }
}

extension Qwen3VL: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { 3 }
}

extension Qwen3VL: CBv2ModelCapabilityProviding {}

extension Qwen3VL: CBv2PositionedLanguageModelForwardable {
    public func cbv2Forward(
        _ inputs: MLXArray, cache: [KVCache]?, positionIds: MLXArray?
    ) -> MLXArray {
        guard let caches = castCBv2Caches(cache) else {
            preconditionFailure("Qwen3-VL positioned CBv2 forward requires v2 caches")
        }
        return languageModel.cbv2Forward(
            inputs, cache: caches, positionIds: positionIds
        ).logits
    }
}

extension Qwen3VL: CBv2EmbeddingForwardable {
    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.model.embedTokens(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        guard let caches = castCBv2Caches(cache) else {
            preconditionFailure("Qwen3-VL CBv2 embedding forward requires v2 caches")
        }
        return languageModel.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding,
            cache: caches, positionIds: nil
        ).logits
    }
}

extension Qwen3VL: CBv2PositionedEmbeddingForwardable {
    public func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let caches = castCBv2Caches(cache) else {
            preconditionFailure(
                "Qwen3-VL positioned embedding forward requires v2 caches")
        }
        return languageModel.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding,
            cache: caches, positionIds: positionIds
        ).logits
    }
}

extension Qwen3VL: CBv2DeepstackEmbeddingForwardable {
    public var deepstackLayerCount: Int {
        config.visionConfiguration.deepstackVisualIndexes.count
    }

    public func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        deepstackEmbeddings: [MLXArray],
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let caches = castCBv2Caches(cache) else {
            preconditionFailure("Qwen3-VL DeepStack forward requires v2 caches")
        }
        precondition(
            deepstackEmbeddings.count == deepstackLayerCount,
            "Qwen3-VL DeepStack layer count mismatch")
        return languageModel.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding,
            cache: caches, positionIds: positionIds,
            deepstackEmbeds: deepstackEmbeddings
        ).logits
    }
}

public struct Qwen3VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var message = message
        var reasoning: String?
        if message.role == .assistant {
            let reasoningProtocol = QwenReasoningProtocol.tagged
            let startDelimiter = reasoningProtocol.startDelimiter
            let endDelimiter = reasoningProtocol.endDelimiter
            let reasoningAndContent: Substring?
            if message.prefilledReasoningStartDelimiter == startDelimiter {
                reasoningAndContent = message.content[...]
            } else if message.content.hasPrefix(startDelimiter) {
                reasoningAndContent = message.content.dropFirst(startDelimiter.count)
            } else {
                reasoningAndContent = nil
            }
            if let reasoningAndContent,
                let end = reasoningAndContent.range(of: endDelimiter)
            {
                reasoning = String(reasoningAndContent[..<end.lowerBound])
                message.content = String(reasoningAndContent[end.upperBound...])
            }
        }

        var dictionary: MLXLMCommon.Message = [
            "role": message.role.rawValue,
            "content": contentParts(for: message, layout: .imagesThenVideosThenText),
        ]
        if let reasoning {
            dictionary["reasoning_content"] = reasoning
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Chat conventions

extension Qwen3VL {
    // Qwen3-VL shares Qwen's tags and tool-call boundary, but does not declare
    // the original Qwen3 family's model-specific hard-budget transition.
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
