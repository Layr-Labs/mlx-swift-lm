// Copyright © 2026 Eigen Labs.

import MLXLMCommon
import Testing

extension UnitTests {
    @Suite("GLM5 response-wide tool parsing")
    struct GLM5ToolStreamingTests {
        private var tools: [[String: any Sendable]] {
            [
                [
                    "type": "function",
                    "function": [
                        "name": "f",
                        "parameters": [
                            "type": "object", "properties": ["x": ["type": "object"]],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]
        }

        @Test func everySplitPreservesQuotedDelimitersAndWaitsForEOS() throws {
            let frame =
                #"<tool_call>f<arg_key>x</arg_key><arg_value>{"text":"</tool_call><tool_call>","n":9007199254740993}</arg_value></tool_call>"#
            let output = "Before 雪 " + frame + "\n<tool_call>f</tool_call>"
            for offset in 0 ... output.count {
                let processor = ToolCallProcessor(format: .glm5, tools: tools)
                let split = output.index(output.startIndex, offsetBy: offset)
                var visible = processor.processChunk(String(output[..<split])) ?? ""
                #expect(processor.drainToolCalls().isEmpty)
                visible += processor.processChunk(String(output[split...])) ?? ""
                #expect(processor.drainToolCalls().isEmpty)
                visible += processor.processEOS(returnBufferedText: true) ?? ""
                #expect(visible == "Before 雪 ")
                let calls = processor.drainToolCalls()
                #expect(calls.count == 2)
                #expect(
                    calls.first?.function.arguments["x"]
                        == .object([
                            "text": .string("</tool_call><tool_call>"),
                            "n": .int(9_007_199_254_740_993),
                        ]))
                #expect(processor.parseFailureCount == 0)
                #expect(processor.drainToolCalls().isEmpty)
                #expect(processor.processEOS(returnBufferedText: true) == nil)
            }
        }

        @Test func malformedLaterCallCannotCommitEarlierCall() {
            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            let first = "<tool_call>f</tool_call>"
            let broken = "<tool_call>f<arg_key>x</arg_key><arg_value>{"
            #expect(processor.processChunk(first) == nil)
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.processChunk(broken) == nil)
            #expect(processor.processEOS(returnBufferedText: true) == first + broken)
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.parseFailureCount == 1)
        }

        @Test func ordinaryTextAndIncompleteOpeningRemainExact() {
            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            var visible = ""
            for character in "normal < text 雪 <tool_" {
                visible += processor.processChunk(String(character)) ?? ""
            }
            visible += processor.processEOS(returnBufferedText: true) ?? ""
            #expect(visible == "normal < text 雪 <tool_")
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.parseFailureCount == 0)
        }

        @Test func oversizedSuffixDisablesExtractionForRestOfResponse() {
            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            let first = "<tool_call>f"
            #expect(processor.processChunk(first) == nil)
            let oversized = String(repeating: "x", count: 1 << 20)
            #expect(processor.processChunk(oversized) == first + oversized)
            #expect(
                processor.processChunk("<tool_call>f</tool_call>") == "<tool_call>f</tool_call>")
            #expect(processor.processEOS(returnBufferedText: true) == nil)
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.parseFailureCount == 1)
        }
    }
}
