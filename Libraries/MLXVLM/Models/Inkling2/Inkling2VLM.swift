// Copyright © 2026 Eigen Labs.
//
// `inkling2_loop` as a vision-language model: the native Inkling 2 text model
// (MLXLLM `Inkling2Model`) plus the vision tower. Image features replace the
// `<|unused_200054|>` rows of the normed text embeddings and run as ordinary
// causal tokens at scalar positions (transformers `Inkling2Model.forward`).
// Media prefill is served by CBv2 (`CBv2MultimodalInput` with causal spans and
// no position state). Like the text model, the legacy `prepare` path keeps no
// contiguous KV and stays text-only.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// `config.json` of an Inkling 2 checkpoint: the text model's configuration
/// (including `image_token_id`) plus `vision_config`.
public struct Inkling2VLMConfiguration: Codable, Sendable {
    public var model: Inkling2Configuration
    public var vision: Inkling2VisionConfiguration?

    enum CodingKeys: String, CodingKey { case vision = "vision_config" }

    public init(model: Inkling2Configuration, vision: Inkling2VisionConfiguration?) {
        self.model = model
        self.vision = vision
    }

    public init(from decoder: Decoder) throws {
        model = try Inkling2Configuration(from: decoder)
        vision = try decoder.container(keyedBy: CodingKeys.self)
            .decodeIfPresent(Inkling2VisionConfiguration.self, forKey: .vision)
    }

    public func encode(to encoder: Encoder) throws {
        try model.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(vision, forKey: .vision)
    }
}

extension Inkling2VLMConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard let vision else { return }
        try vision.validate()
        guard vision.textHiddenSize == model.text.hiddenSize else {
            throw VLMError.processing("vision_config.text_hidden_size differs from the text model")
        }
    }
}

public final class Inkling2VLM: Module, VLMModel {
    public let configuration: Inkling2VLMConfiguration
    @ModuleInfo(key: "language_model") public var languageModel: Inkling2Model
    @ModuleInfo(key: "vision_tower") public var visionTower: Inkling2VisionModel?

    public init(_ configuration: Inkling2VLMConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = Inkling2Model(configuration.model)
        _visionTower.wrappedValue = configuration.vision.map(Inkling2VisionModel.init)
        super.init()
    }

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var loraLayers: [Module] { languageModel.loraLayers }
    /// `<|unused_200054|>`: every image-token row carries this id.
    public var imagePlaceholderTokenId: Int { configuration.model.imageTokenId }
    public var servesVision: Bool { visionTower != nil }

    public static let mediaRejectedMessage =
        "Inkling 2 checkpoint has no vision_config; image input is rejected"

    /// One `[1, rows * cols, hidden]` array per image, in the text embedding
    /// dtype: exactly the rows that replace that image's placeholders. Each
    /// image runs through the tower on its own.
    public func perImageVisionFeatures(pixels: MLXArray, frames: [THW]?) throws -> [MLXArray] {
        guard let visionTower else { throw VLMError.processing(Self.mediaRejectedMessage) }
        let dtype = languageModel.scaledInputEmbeddings(MLXArray([Int32(0)]).reshaped(1, 1)).dtype
        return try visionTower.imageFeatures(pixels: pixels, grids: frames ?? [])
            .map { $0.asType(dtype).expandedDimensions(axis: 0) }
    }

    /// Normed text embeddings with each image's features over its placeholder
    /// rows, `[1, L, hidden]`.
    public func inputEmbeddings(tokens: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let ids = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        let text = languageModel.scaledInputEmbeddings(ids)
        guard let image else { return text }
        let features = try perImageVisionFeatures(pixels: image.pixels, frames: image.frames)
        return try inkling2SpliceImageFeatures(
            textEmbeddings: text, tokens: ids.asArray(Int32.self).map(Int.init),
            imageTokenId: imagePlaceholderTokenId, features: features.map { $0[0] })
    }

    /// Full-recompute logits over a prompt with images: the oracle path.
    public func forwardNoCache(tokens: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let embeddings = try inputEmbeddings(tokens: tokens, image: image)
        return languageModel.forwardNoCache(nil, inputEmbedding: embeddings).logits
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil else {
            throw VLMError.processing(
                servesVision
                    ? "Inkling 2 media is served through the CBv2 engine, not legacy prepare"
                    : Self.mediaRejectedMessage)
        }
        return try languageModel.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    // MARK: Checkpoint

    public var excludedCheckpointNamespaces: [String] { servesVision ? [] : ["model.visual"] }

    /// Vision tensors (`model.visual.*`) move under `vision_tower.`; the text
    /// model prepares everything else in its own namespace.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var text = checkpoint
        let (textWeights, vision) = split(checkpoint.weights)
        text.weights = textWeights
        var prepared = try languageModel.prepareCheckpoint(text)
        prepared.weights = merge(text: prepared.weights, vision: vision)
        return prepared
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let (text, vision) = split(weights)
        return merge(text: try languageModel.sanitize(weights: text), vision: vision)
    }

    private func split(_ weights: [String: MLXArray]) -> (
        text: [String: MLXArray], vision: [String: MLXArray]
    ) {
        var text: [String: MLXArray] = [:]
        var vision: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("model.visual.") {
                vision["vision_tower." + key.dropFirst("model.visual.".count)] = value
            } else if key.hasPrefix("vision_tower.") {
                vision[key] = value
            } else {
                text[key] = value
            }
        }
        return (text, servesVision ? vision : [:])
    }

    private func merge(text: [String: MLXArray], vision: [String: MLXArray]) -> [String: MLXArray] {
        var result = vision
        result.reserveCapacity(text.count + vision.count)
        for (key, value) in text {
            result["language_model." + key] = value
        }
        return result
    }

    // MARK: CBv2 surface (the text model's, plus causal image rows)

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }
}

extension Inkling2VLM: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable, CBv2CausalPositionRequirementProviding
{
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { languageModel.cbv2Capabilities }

    /// Image rows are ordinary causal tokens at the row's scalar cache offset.
    public var cbv2CausalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { servesVision }

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

extension Inkling2VLM: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { languageModel.cbv2SupportsPackedPrefill }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
    }
}

extension Inkling2VLM: CBv2DecodeCommandBufferLimitsProviding {
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        languageModel.decodeCommandBufferLimits
    }
    public var speculativeCommandBufferLimits: GPU.CommandBufferLimits? {
        languageModel.speculativeCommandBufferLimits
    }
}

/// The DSpark drafter binds the text model
/// (`Inkling2DSparkDrafter.load(directory:target: languageModel)`), so the
/// wrapper reports the text model's identity and hands every MTP forward to it.
/// Image requests never reach these paths: the engine keeps any request with
/// media spans target-only for a persistent-history drafter, so it never
/// drafts, verifies or captures features for it.
extension Inkling2VLM: CBv2RecurrentMTPForwardable, CBv2RecurrentCaptureMTPForwardable,
    CBv2MTPPolicyTopTwoProviding
{
    public var cbv2MTPTargetIdentity: ObjectIdentifier { languageModel.cbv2MTPTargetIdentity }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        languageModel.cbv2MTPTopTwo(logits)
    }
}

/// Token prompts only: there is no embedding argument, so this path computes
/// the plain text-token forward and can never stand in for image rows.
extension Inkling2VLM: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenForPrefill(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
            requirement: requirement)
    }
}
