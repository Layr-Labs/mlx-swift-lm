import Foundation
import MLXLMServer
import Testing

@Suite("OpenAI assistant tool-only history decoding")
struct OpenAIToolHistoryDecodingTests {
    @Test func omittedAssistantContentPreservesToolCallAndReasoning() throws {
        let data = Data(#"{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}],"reasoning_content":"Use the tool."}"#.utf8)
        let value = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)
        #expect(value.content == .null && value.toolCalls?.count == 1)
        #expect(value.toolCalls?.first?.function.arguments == #"{"city":"Paris"}"#)
        #expect(value.reasoningContent == "Use the tool.")
        #expect(try JSONDecoder().decode(OpenAIChatMessage.self, from: JSONEncoder().encode(value)) == value)
    }

    @Test(arguments: [
        #"{"role":"assistant"}"#, #"{"role":"assistant","tool_calls":[]}"#,
        #"{"role":"user"}"#, #"{"role":"system"}"#, #"{"role":"tool","tool_call_id":"call-1"}"#,
    ])
    func absentRequiredContentStillRejects(_ body: String) {
        #expect(throws: (any Error).self) { try JSONDecoder().decode(OpenAIChatMessage.self, from: Data(body.utf8)) }
    }
}
