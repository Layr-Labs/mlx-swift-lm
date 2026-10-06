import MLXLMCommon

extension Qwen4ExpTextModel: GenericGenerationValidating {
    public func validateGenericGeneration() throws {
        throw GenericGenerationError.nativeCBv2Required(modelType: configuration.modelType)
    }
}

extension Qwen4ExpModel: GenericGenerationValidating {
    public func validateGenericGeneration() throws {
        try languageModel.validateGenericGeneration()
    }
}
