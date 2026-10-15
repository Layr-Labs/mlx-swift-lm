import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Experimental, unregistered Super VL composition. Construction does not load a
/// checkpoint or establish serving readiness. No MTP or vision-prefix-reuse
/// capability is advertised. The caller owns weight loading, memory admission,
/// request state, evaluation/commit, cancellation and scheduling.
public final class NemotronSuperModel: Module {
    @ModuleInfo(key: "language_model") public var languageModel: NemotronHModel
    @ModuleInfo(key: "image_composition") public var imageComposition: NemotronSuperImageComposition
    /// Exact caller-supplied language JSON, before deriving the redundant layer count.
    /// Nil for injected fixtures. Checkpoint identity remains the loader's responsibility.
    public let sourceLanguageConfigurationJSON: Data?

    public enum Failure: Error {
        case invalidPrompt, incompatibleComponents, wrongModel, invalidPrefillRange
    }

    /// Prepared features are request-local and bound to this model instance.
    /// No global feature cache or text-only prefix identity is inferred here.
    fileprivate final class PreparationIdentity {}
    private let preparationIdentity = PreparationIdentity()

    public struct PreparedInput {
        public let prompt: [Int32]
        public let embeddings: MLXArray
        fileprivate let owner: PreparationIdentity
    }

    /// Allocates the uninitialized architecture, not a loaded or qualified model.
    /// Uses NemotronHModel directly: Lightning execution overrides are inapplicable.
    public init(unloadedConfiguration configuration: NemotronSuperImageConfiguration) throws {
        let language = try Self.validatedLanguageConfiguration(
            configuration.languageConfigurationJSON)
        guard language.hiddenSize == configuration.languageWidth,
            language.vocabSize > Int(configuration.imageTokenID)
        else { throw Failure.incompatibleComponents }
        self.sourceLanguageConfigurationJSON = configuration.languageConfigurationJSON
        self._languageModel.wrappedValue = NemotronHModel(language)
        self._imageComposition.wrappedValue = NemotronSuperImageComposition(
            configuration: configuration)
    }

    /// Strict pinned language boundary, independently testable without allocating
    /// the full model. Original JSON remains unchanged for checkpoint identity.
    static func validatedLanguageConfiguration(_ data: Data) throws -> NemotronHConfiguration {
        guard let original = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Failure.incompatibleComponents
        }
        let required = [
            "model_type", "vocab_size", "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups",
            "conv_kernel",
            "intermediate_size", "moe_intermediate_size", "moe_shared_expert_intermediate_size",
            "moe_latent_size", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
            "layers_block_type", "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor",
            "layer_norm_epsilon", "mlp_bias", "attention_bias", "mamba_proj_bias", "use_bias",
            "use_conv_bias", "tie_word_embeddings", "time_step_limit", "chunk_size",
            "mamba_ssm_cache_dtype", "num_nextn_predict_layers", "mtp_layers_block_type",
            "dtype", "mlp_hidden_act", "mamba_hidden_act",
        ]
        guard required.allSatisfy({ original[$0] != nil && !(original[$0] is NSNull) }),
            original["dtype"] as? String == "bfloat16",
            original["mlp_hidden_act"] as? String == "relu2",
            original["mamba_hidden_act"] as? String == "silu",
            let blocks = original["layers_block_type"] as? [String]
        else { throw Failure.incompatibleComponents }
        let symbols = ["mamba": "M", "moe": "E", "attention": "*"]
        let pattern = blocks.map { symbols[$0] ?? "?" }.joined()
        // Exact order from revision 7d4bc747, not merely the same layer counts.
        let expected =
            "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
        guard pattern == expected, blocks.count == 88 else { throw Failure.incompatibleComponents }
        var normalized = original
        if let count = original["num_hidden_layers"] {
            guard let number = count as? NSNumber, String(cString: number.objCType) != "c",
                number.doubleValue == Double(blocks.count)
            else { throw Failure.incompatibleComponents }
        } else {
            // The pinned config omits this redundant field; the trunk decoder needs it.
            normalized["num_hidden_layers"] = blocks.count
        }
        let language = try JSONDecoder().decode(
            NemotronHConfiguration.self,
            from: JSONSerialization.data(withJSONObject: normalized, options: [.sortedKeys]))
        guard language.modelType == "nemotron_h", language.vocabSize == 131072,
            language.hiddenSize == 4096, language.numHiddenLayers == 88,
            language.hybridOverridePattern == expected, language.moeLatentSize == 1024,
            language.nRoutedExperts == 512, language.numExpertsPerTok == 22,
            language.nSharedExperts == 1, language.nGroup == 1, language.topkGroup == 1,
            language.normTopkProb, language.routedScalingFactor == 5,
            language.intermediateSize == 2688, language.moeIntermediateSize == 2688,
            language.moeSharedExpertIntermediateSize == 5376,
            language.numAttentionHeads == 32, language.numKeyValueHeads == 2,
            language.headDim == 128,
            language.mambaNumHeads == 128, language.mambaHeadDim == 64,
            language.ssmStateSize == 128, language.nGroups == 8, language.convKernel == 4,
            language.mambaSSMCacheDType == "float32", language.layerNormEpsilon == 1e-5,
            !language.mlpBias, !language.attentionBias, !language.mambaProjBias,
            !language.useBias, language.useConvBias, !language.tieWordEmbeddings,
            language.timeStepLimitMin == 0, language.timeStepLimitMax == Float.infinity,
            language.chunkSize == 128, language.numNextnPredictLayers == 1,
            language.mtpLayersBlockType == ["attention", "moe"]
        else { throw Failure.incompatibleComponents }
        return language
    }

    /// Injection boundary for synthetic components. The normal loader must still
    /// account for every checkpoint tensor; this initializer makes no loading claim.
    init(languageModel: NemotronHModel, imageComposition: NemotronSuperImageComposition) throws {
        guard imageComposition.imageTokenID >= 0,
            Int(imageComposition.imageTokenID) < languageModel.vocabularySize,
            languageModel.inputEmbeddings(MLXArray([Int32(0)])).dim(-1) == imageComposition.width
        else { throw Failure.incompatibleComponents }
        self.sourceLanguageConfigurationJSON = nil
        self._languageModel.wrappedValue = languageModel
        self._imageComposition.wrappedValue = imageComposition
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }

    private func validatePrompt(_ prompt: [Int32]) throws {
        guard !prompt.isEmpty,
            prompt.allSatisfy({ $0 >= 0 && Int($0) < languageModel.vocabularySize })
        else { throw Failure.invalidPrompt }
    }

    /// Validates token IDs before embedding lookup. Image features are supplied in
    /// placeholder order; the composition validates their shape and exact count.
    public func prepare(prompt: [Int32], imageFeatures: [MLXArray] = []) throws -> PreparedInput {
        try validatePrompt(prompt)
        let tokens = MLXArray(prompt).reshaped(1, prompt.count)
        let embeddings = try imageComposition.substitute(
            prompt: prompt,
            textEmbeddings: languageModel.inputEmbeddings(tokens), imageFeatures: imageFeatures)
        try languageModel.validateInputEmbeddings(embeddings, tokens: tokens)
        return PreparedInput(prompt: prompt, embeddings: embeddings, owner: preparationIdentity)
    }

    /// Pixels are normalized NCHW tensors, not compressed media. Vision dtype
    /// casting is owned by imageComposition. Preparation never binds request state.
    public func prepare(prompt: [Int32], normalizedImages: [MLXArray]) throws -> PreparedInput {
        try validatePrompt(prompt)
        let features = try normalizedImages.map {
            try imageComposition.imageFeatures(normalizedPixels: $0)
        }
        return try prepare(prompt: prompt, imageFeatures: features)
    }

    /// Allows chunk boundaries inside visual spans after the complete prompt has
    /// been prepared. Validation is repeated before touching recurrent or KV state.
    public func prefill(
        _ prepared: PreparedInput, range: Range<Int>? = nil,
        cache: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        requirement: CBv2PrefillRequirement
    ) throws -> MLXArray {
        guard prepared.owner === preparationIdentity else { throw Failure.wrongModel }
        try validatePrompt(prepared.prompt)
        let fullTokens = MLXArray(prepared.prompt).reshaped(1, prepared.prompt.count)
        try languageModel.validateInputEmbeddings(prepared.embeddings, tokens: fullTokens)
        let selected = range ?? 0 ..< prepared.prompt.count
        guard selected.lowerBound >= 0, selected.upperBound <= prepared.prompt.count,
            !selected.isEmpty
        else { throw Failure.invalidPrefillRange }
        let tokens = MLXArray(Array(prepared.prompt[selected])).reshaped(1, selected.count)
        return try languageModel.prefillWithEmbeddings(
            tokens,
            embeddings: prepared.embeddings[0..., selected, 0...], cache: cache,
            recurrentState: recurrentState, requirement: requirement)
    }

    /// One request's next text token, with externally owned state. Generated token
    /// IDs are validated before native forward constructs or stages any state.
    public func decode(
        token: Int32, cache: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try validatePrompt([token])
        return languageModel.cbv2Forward(
            MLXArray([token]).reshaped(1, 1),
            caches: cache, recurrentState: recurrentState)
    }
}
