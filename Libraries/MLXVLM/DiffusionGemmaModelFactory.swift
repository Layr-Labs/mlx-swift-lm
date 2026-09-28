import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Native diffusion context, deliberately distinct from the autoregressive
/// ModelContext. A caller cannot accidentally feed this model to TokenIterator.
public struct DiffusionGemmaContext {
    public let configuration: ResolvedModelConfiguration
    public let model: DiffusionGemma
    public let generationConfiguration: DiffusionGemmaGenerationConfiguration
    public let tokenizer: any Tokenizer
    public let chatTemplate: String
    public let processor: DiffusionGemmaProcessor?

    /// Apply the checkpoint's exact template with syntax-only Jinja compatibility.
    /// Tools, reasoning and history remain template inputs, never string repairs.
    /// Media expansion is a separate processor operation, not performed here.
    public func renderTokens(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int32] {
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate,
            tools: tools, additionalContext: additionalContext)
        guard !tokens.isEmpty, tokens.count < model.configuration.textConfig.maxPositionEmbeddings,
            tokens.allSatisfy({ $0 >= 0 && $0 < model.configuration.textConfig.vocabularySize })
        else { throw DiffusionGemmaModelError.invalidInput("rendered token range/capacity") }
        return tokens.map(Int32.init)
    }
}

/// The same exclusive-access primitive used by the ordinary model container.
/// Serialization here is not a claim of native continuous-batching support.
public final class DiffusionGemmaContainer: Sendable {
    private let context: SerialAccessContainer<DiffusionGemmaContext>

    public init(context: consuming DiffusionGemmaContext) {
        self.context = .init(context)
    }

    public func perform<R: Sendable>(
        _ action: @Sendable (DiffusionGemmaContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read { try await action($0) }
    }

    /// Linear transfer of decoded media into the container's serialized scope.
    /// As with ModelContainer, evaluate arrays before returning them as owned
    /// request data; never share mutable UserInput across concurrent requests.
    public func perform<V, R: Sendable>(nonSendable values: consuming V,
        _ action: @Sendable (DiffusionGemmaContext, V) async throws -> R) async rethrows -> sending R
    {
        let transferred = SendableBox(values)
        return try await context.read { try await action($0, transferred.consume()) }
    }
}

/// Explicit native factory. Generic AR factories continue to reject this type;
/// serving integrations must select this factory and the block-diffusion runner.
public final class DiffusionGemmaModelFactory: GenericModelFactory, Sendable {
    public static let shared = DiffusionGemmaModelFactory()
    public let modelRegistry = AbstractModelRegistry()
    public init() {}

    public func _wrap(_ context: DiffusionGemmaContext) -> DiffusionGemmaContainer {
        .init(context: context)
    }

    public func _load(
        configuration: ResolvedModelConfiguration, tokenizerLoader: any TokenizerLoader
    ) async throws -> sending DiffusionGemmaContext {
        let directory = configuration.modelDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { throw DiffusionGemmaModelError.invalidInput("missing model directory") }
        // No unchecked AR stopping overrides. Native callers select an explicit
        // DiffusionGemmaGenerationConfiguration per request instead.
        guard configuration.extraEOSTokens.isEmpty, configuration.eosTokenIds.isEmpty else {
            throw DiffusionGemmaModelError.invalidInput("AR stopping overrides on diffusion load")
        }
        let decoder = JSONDecoder.json5()
        let configBytes = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try decoder.decode(DiffusionGemmaConfiguration.self, from: configBytes)
        let generation = try decoder.decode(
            DiffusionGemmaGenerationConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))
        let specialIds =
            (generation.eosTokenIds ?? config.base.eosTokenIds?.values ?? config.textConfig
                .eosTokenIds
                ?? []) + [generation.bosTokenId, generation.padTokenId].compactMap { $0 }
        guard specialIds.allSatisfy({ $0 >= 0 && $0 < config.textConfig.vocabularySize }) else {
            throw DiffusionGemmaModelError.invalidInput("generation special token range")
        }
        let template = SwiftJinjaSyntaxCompatibility.normalize(
            try String(
                contentsOf: configuration.tokenizerDirectory.appendingPathComponent(
                    "chat_template.jinja"),
                encoding: .utf8))
        guard !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DiffusionGemmaModelError.invalidInput("empty chat template")
        }
        // Resolve small metadata/tokenizer failures before constructing weights.
        // This consumes the bytes read above, never a second config/template read.
        let tokenizer = try await tokenizerLoader.load(from: configuration.tokenizerDirectory)
        let processor: DiffusionGemmaProcessor?
        if config.visionConfig != nil {
            let settings = try decoder.decode(DiffusionGemmaProcessorConfiguration.self,
                from: Data(contentsOf: directory.appendingPathComponent("processor_config.json")))
            processor = try DiffusionGemmaProcessor(configuration: settings, model: config,
                tokenizer: tokenizer, template: template)
        } else { processor = nil }
        try Task.checkCancellation()
        let model = DiffusionGemma(config)
        try MLX.withError { errors in
            try loadWeights(modelDirectory: directory, model: model,
                perLayerQuantization: config.base.perLayerQuantization)
            model.train(false)
            eval(model)
            try errors.check()
        }
        try Task.checkCancellation()
        return DiffusionGemmaContext(
            configuration: configuration, model: model, generationConfiguration: generation,
            tokenizer: tokenizer, chatTemplate: template, processor: processor)
    }
}
