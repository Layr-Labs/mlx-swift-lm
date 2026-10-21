import MLX
import MLXLLM
import MLXLMCommon

/// Explicit assembly of text-only Super speculation. Nothing here is selected
/// automatically or registered; the host decides when to call it and owns
/// LOAD admission, construction fences and engine retirement.
public enum NemotronSuperMTPServing {
    public enum Failure: Error { case retiredInstallation }

    /// LOAD supplement for the embedded head, computed before allocation.
    public static func loadQuote(
        configuration: NemotronSuperImageConfiguration, source: ModelOptCheckpointSource
    ) throws -> NemotronSuperMTPModule.LoadQuote {
        try NemotronSuperMTPModule.loadQuote(
            configuration: try NemotronSuperModel.validatedLanguageConfiguration(
                configuration.languageConfigurationJSON),
            source: source)
    }

    /// The installed language model's head, loaded in its own fenced epoch.
    public static func loadHead(
        installed: NemotronSuperInstallation.Installed, source: ModelOptCheckpointSource,
        quote: NemotronSuperMTPModule.LoadQuote, construction: NativeConstructionScope
    ) throws -> NemotronSuperMTPModule {
        guard let model = installed.model else { throw Failure.retiredInstallation }
        return try NemotronSuperMTPModule.load(
            for: model.languageModel, source: source, quote: quote, construction: construction)
    }
}

extension NemotronSuperMTPTextTarget {
    /// Assistant bound to this wrapper's exact language model, so the engine's
    /// target-identity check matches only this target. Several speculative
    /// rows need the captured target serving at least that many requests.
    public func makeAssistant(
        head: NemotronSuperMTPModule, settings: NemotronSuperMTPAssistant.Settings
    ) throws -> NemotronSuperMTPAssistant {
        guard settings.maximumSpeculativeRows <= maximumSpeculativeRows else {
            throw NemotronSuperMTPAssistant.Failure.rows
        }
        return try NemotronSuperMTPAssistant(
            target: base.composite.languageModel, head: head, settings: settings)
    }
}
