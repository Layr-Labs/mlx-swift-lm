import MLXLMCommon

/// Factory-local external-resource operations. Keep the concrete Qwen4
/// downcast inside the owning module; other architecture instances are no-ops.
public enum Qwen4ExpFactoryResources {
    public static func validate(_ model: any LanguageModel) throws {
        try (model as? any Qwen4ExpExternalPLEValidating)?.validateExternalPLEResources()
    }

    public static func release(_ model: any LanguageModel) {
        (model as? any Qwen4ExpExternalPLEReleasing)?.releaseExternalPLEResources()
    }
}
