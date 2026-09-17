import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

@Suite("Qwen4 opaque argument parser and wire boundary")
struct Qwen4OpaqueArgumentTests {
    @Test func preservesLiteralAndAlreadyEscapedStringsInBothDialects() throws {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "record_text",
                "parameters": ["type": "object", "properties": ["text": ["type": "string"]]] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let parser = try #require(ToolCallFormat.infer(from: "qwen4_exp")).createParser()
        for value in [
            #"A "quoted" value, a backslash \ and Unicode café 雪"#,
            #"A \\"quoted\\" value, a backslash \\\\ and Unicode café 雪"#,
        ] {
            let json = try JSONSerialization.data(withJSONObject: [
                "name": "record_text", "arguments": ["text": value],
            ])
            for payload in [
                "<function=record_text><parameter=text>\(value)</parameter></function>",
                String(decoding: json, as: UTF8.self),
            ] {
                let call = try #require(parser.parse(content: payload, tools: tools))
                #expect(call.function.arguments["text"] == .string(value))
                let wire = try OpenAIToolCall(toolCall: call, id: "synthetic_call")
                let decoded = try JSONDecoder().decode(
                    [String: String].self, from: Data(wire.function.arguments.utf8))
                #expect(decoded == ["text": value])
            }
        }
    }
}
