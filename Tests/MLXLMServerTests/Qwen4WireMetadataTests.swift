import Foundation
import MLXLMServer
import Testing

@Suite("Qwen4 additive listing and cache accounting")
struct Qwen4WireMetadataTests {
    private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
    }

    @Test func contextAliasesRoundTripAndRemainAbsentWhenUnknown() throws {
        let known = MLXServerModel(id: "fixture", contextLength: 82_000)
        let encoded = try object(known)
        #expect(encoded["context_length"] as? Int == 82_000)
        #expect(encoded["max_model_len"] as? Int == 82_000)
        #expect(try JSONDecoder().decode(MLXServerModel.self, from: JSONEncoder().encode(known)) == known)
        let unknown = try object(MLXServerModel(id: "legacy"))
        #expect(unknown["context_length"] == nil)
        #expect(unknown["max_model_len"] == nil)
    }

    @Test func cacheUsageUsesReportedBoundedTokensAndPreservesUnknown() throws {
        let unknown = OpenAIUsage(promptTokens: 10, completionTokens: 2)
        #expect(try object(unknown)["prompt_tokens_details"] == nil)
        for (reported, expected) in [(-1, 0), (0, 0), (6, 6), (11, 10)] {
            let info = ServerGenerationInfo(promptTokens: 10, completionTokens: 2,
                promptTime: 0, generationTime: 0, stopReason: "stop", cachedPromptTokens: reported)
            #expect(info.cachedPromptTokens == expected)
            let chat = OpenAIUsage(promptTokens: 10, completionTokens: 2,
                cachedPromptTokens: info.cachedPromptTokens)
            #expect(chat.promptTokensDetails?.cachedTokens == expected)
            let response = OpenAIResponseUsage(chatUsage: chat)
            #expect(response.inputTokensDetails?.cachedTokens == expected)
            #expect(response.chatUsage == chat)
        }
        #expect(OpenAIResponseUsage(chatUsage: unknown).inputTokensDetails == nil)
    }
}
