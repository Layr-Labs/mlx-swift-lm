import Foundation
import MLXLLM
import MLXLMCommon

protocol ServerGenerationEngine: Sendable {
    var defaultTemperature: Float { get }

    func generateWithResult(
        prompt: String,
        samplingParams: SamplingParams
    ) async throws -> RequestOutput

    func streamOutputs(
        prompt: String,
        samplingParams: SamplingParams
    ) -> AsyncStream<RequestOutput>

    func buildPrompt(messages: [[String: String]]) -> String
    func getStats() -> [String: Any]
}

extension BatchedEngine: ServerGenerationEngine {
    var defaultTemperature: Float { 0.7 }
}

extension DFlashBatchedEngine: ServerGenerationEngine {
    var defaultTemperature: Float { 0 }
}
