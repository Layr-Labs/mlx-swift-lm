// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// VLM wrapper for Flash-Next `qwen4_exp`.
//
// Fusion `mlx_vlm/models/qwen4_exp/qwen4_exp.py`: `Model(Qwen3_5Model)` owns
// `vision_tower = VisionModel(config.vision_config)` — the published Qwen3-VL
// ViT with `deepstack_visual_indexes` forced empty (`config.py`) — plus the
// Qwen4 language model. Image features replace `<|image_pad|>` / `<|video_pad|>`
// embeddings and positions are Qwen3.5-style interleaved M-RoPE (`get_rope_index`).
// The language tower is served directly through CBv2; vision runs through the
// shared `QwenVisionSeamModel` surface the provider already drives for Qwen3.5.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// `config.json` for `Qwen4ExpForConditionalGeneration`: the Qwen4 text
/// configuration plus the vision tower and media token ids.
///
/// Token ids default to the Flash-Next vocabulary (Fusion `config.py`):
/// `<|image_pad|>` 248056, `<|video_pad|>` 248057, `<|vision_start|>` 248053,
/// `<|vision_end|>` 248054. Note Qwen3.5 checkpoints use different
/// `vision_start/end` ids; these must come from the Qwen4 config, never from
/// the Qwen3.5 defaults.
public struct Qwen4ExpVLMConfiguration: Codable, Sendable {
    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public var text: Qwen4ExpConfiguration
    public var visionConfiguration: VisionConfiguration?
    public var languageModelOnly: Bool
    private let _imageTokenId: Int?
    private let _videoTokenId: Int?
    private let _imageTokenIndex: Int?
    private let _videoTokenIndex: Int?
    private let _visionStartTokenId: Int?
    private let _visionEndTokenId: Int?

    public var modelType: String { text.modelType }
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_053 }
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_054 }

    /// A vision tower is instantiated only when the checkpoint declares one
    /// and does not opt out via `language_model_only`.
    public var servesVision: Bool { visionConfiguration != nil && !languageModelOnly }

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
        case languageModelOnly = "language_model_only"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
    }

    public init(
        text: Qwen4ExpConfiguration = Qwen4ExpConfiguration(),
        visionConfiguration: VisionConfiguration? = nil,
        languageModelOnly: Bool = false,
        imageTokenId: Int? = nil,
        videoTokenId: Int? = nil,
        visionStartTokenId: Int? = nil,
        visionEndTokenId: Int? = nil
    ) {
        self.text = text
        self.visionConfiguration = visionConfiguration
        self.languageModelOnly = languageModelOnly
        self._imageTokenId = imageTokenId
        self._videoTokenId = videoTokenId
        self._imageTokenIndex = nil
        self._videoTokenIndex = nil
        self._visionStartTokenId = visionStartTokenId
        self._visionEndTokenId = visionEndTokenId
    }

    public init(from decoder: Decoder) throws {
        text = try Qwen4ExpConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        visionConfiguration = try container.decodeIfPresent(
            VisionConfiguration.self, forKey: .visionConfiguration)
        languageModelOnly =
            try container.decodeIfPresent(Bool.self, forKey: .languageModelOnly) ?? false
        _imageTokenId = try container.decodeIfPresent(Int.self, forKey: ._imageTokenId)
        _videoTokenId = try container.decodeIfPresent(Int.self, forKey: ._videoTokenId)
        _imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _videoTokenIndex = try container.decodeIfPresent(Int.self, forKey: ._videoTokenIndex)
        _visionStartTokenId = try container.decodeIfPresent(
            Int.self, forKey: ._visionStartTokenId)
        _visionEndTokenId = try container.decodeIfPresent(Int.self, forKey: ._visionEndTokenId)
    }

    public func encode(to encoder: Encoder) throws {
        try text.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(visionConfiguration, forKey: .visionConfiguration)
        try container.encode(languageModelOnly, forKey: .languageModelOnly)
        try container.encodeIfPresent(_imageTokenId, forKey: ._imageTokenId)
        try container.encodeIfPresent(_videoTokenId, forKey: ._videoTokenId)
        try container.encodeIfPresent(_imageTokenIndex, forKey: ._imageTokenIndex)
        try container.encodeIfPresent(_videoTokenIndex, forKey: ._videoTokenIndex)
        try container.encodeIfPresent(_visionStartTokenId, forKey: ._visionStartTokenId)
        try container.encodeIfPresent(_visionEndTokenId, forKey: ._visionEndTokenId)
    }
}

public final class Qwen4Exp:
    Module, VLMModel, KVCacheDimensionProvider, QwenVisionSeamModel,
    CBv2TargetAuxiliaryAllocationProviding, GenericGenerationValidating
{
    public let configuration: Qwen4ExpConfiguration
    public let vlmConfiguration: Qwen4ExpVLMConfiguration

    @ModuleInfo(key: "language_model") var languageModel: Qwen4ExpTextModel
    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel?

    public init(_ configuration: Qwen4ExpVLMConfiguration) {
        self.configuration = configuration.text
        self.vlmConfiguration = configuration
        _languageModel.wrappedValue = Qwen4ExpTextModel(configuration.text.textConfig)
        if let vision = configuration.visionConfiguration, configuration.servesVision {
            precondition(
                vision.deepstackVisualIndexes.isEmpty,
                "Qwen4-Exp does not use deepstack visual features (Fusion config.py)")
            _visionModel.wrappedValue = Qwen3VLVision.VisionModel(vision)
        } else {
            _visionModel.wrappedValue = nil
        }
        super.init()
    }

    /// Text-only wrapper (no tower); media fails closed. Kept for callers that
    /// construct the wrapper from the bare Qwen4 text configuration.
    public convenience init(_ configuration: Qwen4ExpConfiguration) {
        self.init(Qwen4ExpVLMConfiguration(text: configuration))
    }

    /// Whether this wrapper instantiated the vision tower.
    public var servesVision: Bool { visionModel != nil }

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        languageModel.cbv2TargetAuxiliaryAllocationSpecs
    }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }
    public var cbv2Capabilities: CBv2ModelCapabilities { languageModel.cbv2Capabilities }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func validateGenericGeneration() throws {
        try languageModel.validateGenericGeneration()
    }

    public static let mediaRejectedMessage =
        "Qwen4-Exp checkpoint has no servable vision tower (language_model_only or no vision_config); image/video is rejected"

    /// Fusion `engine/vlm.py`: Qwen4-Exp serves text and image input (the
    /// Qwen3-VL tower also carries video) but never audio. A wrapper built
    /// without a tower fails closed on any media; it must not approximate pixels.
    public static func rejectMediaIfPresent(_ input: LMInput) throws {
        if input.image != nil || input.video != nil {
            throw VLMError.processing(mediaRejectedMessage)
        }
    }

    /// Legacy (non-CBv2) `prepare` stays text-only. Production media runs
    /// through the CBv2 vision prefill path (`QwenVisionSeamModel`), which
    /// splices `visionFeatures` over the placeholder spans and drives the
    /// language tower with `positionResult` M-RoPE positions.
    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        if input.image != nil || input.video != nil {
            throw VLMError.processing(
                servesVision
                    ? "Qwen4-Exp media is served through the CBv2 vision prefill path, not legacy prepare"
                    : Self.mediaRejectedMessage)
        }
        return try languageModel.prepare(input, cache: cache, windowSize: windowSize)
    }

    // MARK: - QwenVisionSeamModel

    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        let vision = vlmConfiguration.visionConfiguration
        return Qwen35VisionSeamConfiguration(
            imagePlaceholderTokenId: vlmConfiguration.imageTokenIndex,
            videoPlaceholderTokenId: vlmConfiguration.videoTokenIndex,
            imagePositionTokenId: vlmConfiguration.imageTokenId,
            videoPositionTokenId: vlmConfiguration.videoTokenId,
            visionStartTokenId: vlmConfiguration.visionStartTokenId,
            visionEndTokenId: vlmConfiguration.visionEndTokenId,
            spatialMergeSize: vision?.spatialMergeSize ?? 2,
            temporalPatchSize: vision?.temporalPatchSize ?? 2,
            attention: .causal)
    }

    public var imagePlaceholderTokenId: Int { vlmConfiguration.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { vlmConfiguration.videoTokenIndex }

    public var visionTowerAttentionGeometry: (hiddenSize: Int, numHeads: Int) {
        guard let vision = vlmConfiguration.visionConfiguration else { return (0, 0) }
        return (vision.hiddenSize, vision.numHeads)
    }

    /// Qwen3.5-compatible M-RoPE positions (`Qwen3_5Model.get_rope_index` via
    /// Fusion's `Model(Qwen3_5Model)`), computed with the Qwen4 token ids.
    public func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]? = nil,
        videoGrids: [THW]? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> Qwen35PositionResult {
        try QwenVisionSeamSupport.positionResult(
            tokens: tokens,
            imageGrids: imageGrids,
            videoGrids: Qwen4ExpVideoPrompt.positionGrids(videoGrids),
            attentionMask: attentionMask,
            seam: visionSeamConfiguration)
    }

    /// Final Qwen3-VL tower output (merger, `out_hidden_size` = text hidden)
    /// in the text embedding dtype; no DeepStack levels for Qwen4.
    public func visionFeatures(
        imagePixels: MLXArray? = nil,
        imageGrids: [THW]? = nil,
        videoPixels: MLXArray? = nil,
        videoGrids: [THW]? = nil
    ) throws -> Qwen35VisionFeatures {
        guard let visionModel, let vision = vlmConfiguration.visionConfiguration else {
            throw VLMError.processing(Self.mediaRejectedMessage)
        }
        let textDType = languageModel.scaledInputEmbeddings(
            MLXArray([Int32(0)]).reshaped(1, 1)
        ).dtype
        return try QwenVisionSeamSupport.visionFeatures(
            imagePixels: imagePixels,
            imageGrids: imageGrids,
            videoPixels: videoPixels,
            videoGrids: videoGrids,
            tower: visionModel,
            spatialMergeSize: vision.spatialMergeSize,
            textHiddenSize: languageModel.configuration.hiddenSize,
            textDType: textDType)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var textWeights: [String: MLXArray] = [:]
        var visionWeights: [String: MLXArray] = [:]
        textWeights.reserveCapacity(weights.count)
        let keepVisionTower = servesVision
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("model.visual") {
                key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
            }
            if Qwen4ExpWeightSanitizer.shouldDrop(
                key, mmapPLE: languageModel.mmapPLE, keepVisionTower: keepVisionTower)
            {
                continue
            }
            if key.hasPrefix("vision_tower") {
                visionWeights[key] = value
                continue
            }
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                // Match the text wrapper: converted checkpoints keep lm_head
                // at the root, while this wrapper owns it under language_model.
                key = "language_model." + key
            }
            textWeights[key] = value
        }
        if languageModel.configuration.numExperts > 0 {
            textWeights = qwen35FuseSwitchMLPGateUp(
                weights: textWeights,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
        }
        // The text sanitizer filters with the text-only drop list, so the
        // tower tensors are sanitized separately and merged back afterwards.
        var sanitized = languageModel.sanitize(weights: textWeights)
        if let visionModel, !visionWeights.isEmpty {
            // Idempotent: only re-lays out an HF-order `patch_embed.proj.weight`.
            for (key, value) in visionModel.sanitize(weights: visionWeights) {
                sanitized[key] = value
            }
        }
        return sanitized
    }
}

extension Qwen4Exp: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen4Exp: QuantizationPolicyReceiving {
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? {
        get { languageModel.checkpointPerLayerQuantization }
        set { languageModel.checkpointPerLayerQuantization = newValue }
    }
}

extension Qwen4Exp: CheckpointWeightLoadFiltering {
    public var checkpointWeightLoadFilter: CheckpointWeightLoadFilter {
        Qwen4ExpCheckpointLoad.filter(
            mmapPLE: languageModel.mmapPLE,
            keepVisionTower: servesVision)
    }

    public var skipWholeShardPrefetch: Bool { languageModel.mmapPLE }
}

extension Qwen4Exp: IncrementalCheckpointMaterializing {
    public var needsIncrementalCheckpointMaterialization: Bool {
        languageModel.needsIncrementalCheckpointMaterialization
    }

    public func materializeCheckpointWeightsIncrementally() throws {
        try languageModel.materializeCheckpointWeightsIncrementally()
    }
}

extension Qwen4Exp: Qwen4ExpExternalPLEReleasing {
    public func releaseExternalPLEResources() {
        languageModel.releaseExternalPLEResources()
    }
}

extension Qwen4Exp: Qwen4ExpExternalPLEValidating {
    public func validateExternalPLEResources() throws {
        try languageModel.validateExternalPLEResources()
    }
}

extension Qwen4Exp: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { languageModel.cbv2PositionAxisCount }
}

extension Qwen4Exp: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.scaledInputEmbeddings(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension Qwen4Exp: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { languageModel.cbv2SupportsPackedPrefill }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds,
            requirement: requirement)
    }
}

extension Qwen4Exp: Qwen4ExpMTPTargeting {
    public var qwen4ExpTextTarget: Qwen4ExpTextModel { languageModel }
}

extension Qwen4Exp: CBv2CompleteCheckpointKVTypeProviding, CBv2Qwen4CheckpointGeometryProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { languageModel.cbv2CompleteCheckpointKVDTypes }
    public var cbv2Qwen4CheckpointGeometries: [CBv2Qwen4CheckpointGeometry]? { languageModel.cbv2Qwen4CheckpointGeometries }
}

extension Qwen4Exp: CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(languageModel)
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen4Exp: CBv2RecurrentCaptureMTPForwardable, CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }

    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenForPrefill(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, requirement: requirement)
    }
}

extension Qwen4Exp: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        languageModel.cbv2MTPTopTwo(logits)
    }
}
