// Copyright © 2026 Eigen Labs.
// Native Qwen3.8-27B towers with Prism's packed signed-Hadamard projections.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class PrismHadamardQwen35: Module, VLMModel, KVCacheDimensionProvider,
    QwenVisionSeamModel, PrismHadamardLoading
{
    private struct TextEnvelope: Decodable { let text_config: Qwen35TextConfiguration }
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration
    public let config: Qwen35Configuration
    private let textConfiguration: Qwen35TextConfiguration
    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel
    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel

    public init(configurationData: Data) throws {
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData)
        textConfiguration = try JSONDecoder.json5().decode(TextEnvelope.self, from: configurationData).text_config
        guard prismCheckpoint.hasVision, config.visionConfiguration.deepstackVisualIndexes.isEmpty,
            config.visionConfiguration.outHiddenSize == config.textConfiguration.hiddenSize
        else { throw PrismCheckpointError.invalid("unsupported vision declaration") }
        _languageModel.wrappedValue = Qwen35TextModel(textConfiguration)
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        super.init()
    }

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }
    public var cbv2Capabilities: CBv2ModelCapabilities {
        var value = languageModel.cbv2Capabilities
        value.supportsMTP = false
        return value
    }
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
    public func newCacheV2(makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache)
        rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Published MLX names and normalization are already converted.
        weights
    }
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        guard input.image == nil && input.video == nil else {
            throw VLMError.processing("Prism media requires native CBv2 vision prefill")
        }
        return try languageModel.prepare(input, cache: cache, windowSize: windowSize)
    }

    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        .init(imagePlaceholderTokenId: config.imageTokenIndex,
              videoPlaceholderTokenId: config.videoTokenIndex,
              imagePositionTokenId: config.imageTokenId, videoPositionTokenId: config.videoTokenId,
              visionStartTokenId: config.visionStartTokenId, visionEndTokenId: config.visionEndTokenId,
              spatialMergeSize: config.visionConfiguration.spatialMergeSize,
              temporalPatchSize: config.visionConfiguration.temporalPatchSize, attention: .causal)
    }
    public var imagePlaceholderTokenId: Int { config.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { config.videoTokenIndex }
    public var visionTowerAttentionGeometry: (hiddenSize: Int, numHeads: Int) {
        (config.visionConfiguration.hiddenSize, config.visionConfiguration.numHeads)
    }
    public func positionResult(tokens: MLXArray, imageGrids: [THW]? = nil,
        videoGrids: [THW]? = nil, attentionMask: MLXArray? = nil) throws -> Qwen35PositionResult {
        try QwenVisionSeamSupport.positionResult(tokens: tokens, imageGrids: imageGrids,
            videoGrids: videoGrids, attentionMask: attentionMask, seam: visionSeamConfiguration)
    }
    public func visionFeatures(imagePixels: MLXArray? = nil, imageGrids: [THW]? = nil,
        videoPixels: MLXArray? = nil, videoGrids: [THW]? = nil) throws -> Qwen35VisionFeatures {
        try QwenVisionSeamSupport.visionFeatures(imagePixels: imagePixels, imageGrids: imageGrids,
            videoPixels: videoPixels, videoGrids: videoGrids, tower: visionModel,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            textHiddenSize: config.textConfiguration.hiddenSize, textDType: .float16)
    }
}

extension PrismHadamardQwen35: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable {
    public func cbv2Forward(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }
    public func cbv2Forward(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?) -> MLXArray {
        languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }
    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }
    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.scaledInputEmbeddings(inputs)
    }
    public func embeddingForward(_ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }
    public func embeddingForward(_ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}
extension PrismHadamardQwen35: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { languageModel.cbv2SupportsPackedPrefill }
    public func cbv2RecurrentPrefill(_ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
    }
}
extension PrismHadamardQwen35: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { languageModel.cbv2CompleteCheckpointKVDTypes }
}
