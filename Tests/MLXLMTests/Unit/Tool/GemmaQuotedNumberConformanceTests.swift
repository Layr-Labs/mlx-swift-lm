import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct GemmaQuotedNumberConformanceTests {
        // Run in separate processes with the switch both absent and set to 1.
        private let enabled =
            ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA_SCHEMA_COERCION"] == "1"

        private let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "count_items",
                    "parameters": [
                        "type": "object",
                        "properties": ["count": ["type": "integer"]],
                        "required": ["count"],
                        "additionalProperties": false,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        @Test("a marker-quoted integer closes as the schema's number before exposure")
        func quotedIntegerAtFrameClose() throws {
            let processor = ToolCallProcessor(
                format: .gemma, tools: tools,
                streaming: .init(exposesName: { _ in true }))
            let prefix = #"<|tool_call>call:count_items{count:<|"|>3<|"|>}"#
            var beforeClose: [ToolCallStreamSegment] = []
            for character in prefix {
                beforeClose += processor.processChunkSegments(String(character))
            }
            let fragmentsBeforeClose = beforeClose.compactMap { segment -> String? in
                if case .callArguments(let text) = segment { return text }
                return nil
            }.joined()
            if enabled { #expect(fragmentsBeforeClose.isEmpty) }
            let segments =
                beforeClose + processor.processChunkSegments("<tool_call|>")
                + processor.processEOSSegments()
            var arguments = ""
            var calls: [ToolCall] = []
            for segment in segments {
                switch segment {
                case .callArguments(let text): arguments += text
                case .callCompleted(let call, let remainder):
                    calls.append(call)
                    arguments += remainder
                case .callFailed, .call, .text:
                    Issue.record("expected one successful streamed Gemma call")
                case .callStarted: break
                }
            }
            let expected: [String: JSONValue] = ["count": enabled ? .int(3) : .string("3")]
            #expect(calls.count == 1)
            let completed = try #require(calls.first)
            #expect(completed.function.arguments == expected)
            #expect(
                try JSONDecoder().decode([String: JSONValue].self, from: Data(arguments.utf8))
                    == expected)
            if enabled {
                #expect(
                    ToolSchemaValidator.validate(
                        arguments: completed.function.arguments, forToolNamed: "count_items",
                        in: tools) == .valid)
            }
        }
    }
}
