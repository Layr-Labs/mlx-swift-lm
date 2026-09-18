import Foundation
import Testing
@testable import MLXLMServer

@Suite("Responses preserves typed reasoning effort")
struct Qwen4ResponsesReasoningTests {
    @Test func responsesEffortReachesChatTranslation() throws {
        for effort in ["none", "minimal", "high", "future-effort"] {
            let request = OpenAIResponseRequest(
                model: "owned", input: .text("hello"),
                reasoning: .init(effort: effort), maxOutputTokens: 16)
            let chat = request.chatCompletionRequest
            #expect(chat.reasoning?.effort == effort)
            #expect(chat.reasoning?.enabled == nil)
            #expect(chat.maxTokens == 16)
        }
    }

    @Test func absentEffortKeepsTheLegacyNilControl() {
        let request = OpenAIResponseRequest(model: "owned", input: .text("hello"),
            reasoning: .init())
        #expect(request.chatCompletionRequest.reasoning == nil)
    }

    @Test func chatReasoningRoundTripsWithoutInventingFields() throws {
        let legacy = OpenAIReasoningConfig(enabled: false)
        let json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(legacy))
            as? [String: Any])
        #expect(json["effort"] == nil)
        let explicit = OpenAIReasoningConfig(enabled: true, effort: "none")
        #expect(try JSONDecoder().decode(OpenAIReasoningConfig.self,
            from: JSONEncoder().encode(explicit)) == explicit)
    }
}
