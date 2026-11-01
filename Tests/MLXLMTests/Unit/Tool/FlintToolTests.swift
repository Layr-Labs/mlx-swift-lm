import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct FlintToolTests {
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "f",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "s": ["type": "string"], "v": ["type": "object"],
                            "n": ["type": "integer"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        @Test func completeFramesPreserveOpaqueValuesAtEveryChunkBoundary() {
            let raw = "  雪 </tool_call><think>literal</think>  "
            let frame =
                "<tool_call>f<arg_key>s</arg_key><arg_value>\(raw)</arg_value>"
                + #"<arg_key>v</arg_key><arg_value>{"a":[{"b":"</arg_value></tool_call>"}]}</arg_value></tool_call>"#
            for width in [1, 2, 7, frame.count] {
                let processor = ToolCallProcessor(format: .flint, tools: tools)
                let chars = Array("before" + frame + frame + "after")
                var visible = ""
                for offset in stride(from: 0, to: chars.count, by: width) {
                    visible +=
                        processor.processChunk(
                            String(chars[offset ..< min(offset + width, chars.count)])) ?? ""
                }
                visible += processor.processEOS() ?? ""
                #expect(visible == "beforeafter")
                #expect(processor.toolCalls.count == 2)
                #expect(
                    processor.toolCalls.allSatisfy { $0.function.arguments["s"] == .string(raw) })
                #expect(processor.parseFailureCount == 0)
            }
        }

        @Test func malformedFramesAreNeverRepairedOrExecuted() {
            for frame in [
                "<tool_call>f<arg_key>n</arg_key><arg_value>7</arg_value>",
                "<tool_call>f<arg_key>n</arg_key><arg_value>oops</arg_value></tool_call>",
                "<tool_call>f<tool_call>g</tool_call>",
                "<tool_call>f<arg_key>n</arg_key><arg_value>1</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call>",
                #"<tool_call>f<arg_key>v</arg_key><arg_value>{"a": [1,}</arg_value></tool_call>"#,
            ] {
                #expect(FlintToolCallParser().parse(content: frame, tools: tools) == nil)
                let processor = ToolCallProcessor(format: .flint, tools: tools)
                for char in frame { _ = processor.processChunk(String(char)) }
                _ = processor.processEOS()
                #expect(processor.toolCalls.isEmpty)
                #expect(processor.parseFailureCount > 0)
            }
        }

        @Test func formatIsArchitectureScoped() {
            #expect(ToolCallFormat.infer(from: "flint") == .flint)
            #expect(ToolCallFormat.infer(from: "glm4") != .flint)
        }

        @Test func scalarArgumentsRetainTheirJSONTypes() {
            let properties: [String: any Sendable] = [
                "zero": ["type": "integer"], "one": ["type": "integer"],
                "fraction": ["type": "number"], "yes": ["type": "boolean"],
                "no": ["type": "boolean"], "empty": ["type": "null"],
            ]
            let scalarTools: [[String: any Sendable]] = [
                [
                    "type": "function",
                    "function": [
                        "name": "f",
                        "parameters": ["type": "object", "properties": properties]
                            as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]
            let values = [
                ("zero", "0"), ("one", "1"), ("fraction", "1.5"),
                ("yes", "true"), ("no", "false"), ("empty", "null"),
            ]
            let frame =
                "<tool_call>f"
                + values.map {
                    "<arg_key>\($0.0)</arg_key><arg_value> \($0.1) </arg_value>"
                }.joined() + "</tool_call>"
            let expected: [String: JSONValue] = [
                "zero": .int(0), "one": .int(1), "fraction": .double(1.5),
                "yes": .bool(true), "no": .bool(false), "empty": .null,
            ]
            #expect(
                FlintToolCallParser().parse(content: frame, tools: scalarTools)?.function.arguments
                    == expected)
            let processor = ToolCallProcessor(format: .flint, tools: scalarTools)
            for scalar in frame.unicodeScalars { _ = processor.processChunk(String(scalar)) }
            #expect(processor.toolCalls.first?.function.arguments == expected)
            #expect(processor.parseFailureCount == 0)
        }

        @Test func malformedStructuredValuesCannotFallBackToStringsWithoutSchemas() {
            for value in ["{bad}", " {bad}", "\n[1,]", "\t{\"a\":}", "{\"a\":1,}"] {
                let frame =
                    "<tool_call>f<arg_key>v</arg_key><arg_value>\(value)</arg_value></tool_call>"
                #expect(FlintToolCallParser().parse(content: frame, tools: nil) == nil)
                let processor = ToolCallProcessor(format: .flint)
                #expect(processor.processChunk(frame) == frame)
                #expect(processor.toolCalls.isEmpty)
                #expect(processor.parseFailureCount == 1)
            }
        }

        @Test func escapedQuotesAndUnicodeShareCompleteAndStreamedBoundaries() {
            let value = #"{"s":"雪 é \\\" </arg_value></tool_call>","a":[1,{"b":true}]}"#
            let frame =
                "<tool_call>f<arg_key>v</arg_key><arg_value>\(value)</arg_value></tool_call>"
            let expected = FlintToolCallParser().parse(content: frame, tools: tools)
            #expect(expected != nil)
            let scalars = Array(frame.unicodeScalars)
            for split in 0 ... scalars.count {
                let processor = ToolCallProcessor(format: .flint, tools: tools)
                let prefix = String(String.UnicodeScalarView(scalars[..<split]))
                let suffix = String(String.UnicodeScalarView(scalars[split...]))
                #expect((processor.processChunk(prefix) ?? "").isEmpty)
                #expect((processor.processChunk(suffix) ?? "").isEmpty)
                #expect(processor.toolCalls == (expected.map { [$0] } ?? []))
                #expect(processor.parseFailureCount == 0)
            }
        }

        @Test func declaredStringsStartingWithJSONMarkersRemainRawAtEveryBoundary() {
            for raw in [
                "{", " [todo\n", "\t{雪 </tool_call><think>literal</think>  ", " [\"unterminated",
            ] {
                let frame =
                    "<tool_call> f \n<arg_key>s</arg_key><arg_value>\(raw)</arg_value>"
                    + #"<arg_key>v</arg_key><arg_value>{"data":"</arg_value></tool_call><think>"}</arg_value></tool_call>"#
                let expected = FlintToolCallParser().parse(content: frame, tools: tools)
                #expect(expected?.function.arguments["s"] == .string(raw))
                let scalars = Array(frame.unicodeScalars)
                for split in 0 ... scalars.count {
                    let processor = ToolCallProcessor(format: .flint, tools: tools)
                    for part in [scalars[..<split], scalars[split...]] {
                        #expect(
                            (processor.processChunk(String(String.UnicodeScalarView(part))) ?? "")
                                .isEmpty)
                    }
                    #expect(processor.toolCalls == (expected.map { [$0] } ?? []))
                    #expect(processor.parseFailureCount == 0)
                }
            }
        }

        @Test func rawValueModesFollowFunctionAndArgumentAndResetBetweenFrames() {
            let other: [String: any Sendable] = [
                "type": "function",
                "function": [
                    "name": "g",
                    "parameters": ["type": "object", "properties": ["s": ["type": "object"]]]
                        as [String: any Sendable],
                ] as [String: any Sendable],
            ]
            let raw = "[unclosed </tool_call><think>"
            let first = "<tool_call>f<arg_key>s</arg_key><arg_value>\(raw)</arg_value></tool_call>"
            let second =
                #"<tool_call>g<arg_key>s</arg_key><arg_value>{"tag":"</arg_value></tool_call>"}</arg_value></tool_call>"#
            let processor = ToolCallProcessor(format: .flint, tools: tools + [other])
            for scalar in (first + second + first).unicodeScalars {
                #expect((processor.processChunk(String(scalar)) ?? "").isEmpty)
            }
            #expect(processor.toolCalls.map(\.function.name) == ["f", "g", "f"])
            #expect(processor.toolCalls.first?.function.arguments["s"] == .string(raw))
            #expect(
                processor.toolCalls.dropFirst().first?.function.arguments["s"]
                    == .object(["tag": .string("</arg_value></tool_call>")]))
            #expect(processor.parseFailureCount == 0)
        }

        @Test func ambiguousRawDelimiterIsNotRepairedIntoAString() {
            let frame =
                "<tool_call>f<arg_key>s</arg_key><arg_value> [raw </arg_value> tail</arg_value></tool_call>"
            #expect(FlintToolCallParser().parse(content: frame, tools: tools) == nil)
            for width in [1, 3, frame.count] {
                let processor = ToolCallProcessor(format: .flint, tools: tools)
                let chars = Array(frame)
                for offset in stride(from: 0, to: chars.count, by: width) {
                    _ = processor.processChunk(
                        String(chars[offset ..< min(offset + width, chars.count)]))
                }
                _ = processor.processEOS()
                #expect(processor.toolCalls.isEmpty)
                #expect(processor.parseFailureCount == 1)
            }
        }

        @Test func scannerMemoryDoesNotGrowWithRawPayload() {
            var scanner = FlintToolFrameScanner(tools: tools)
            for scalar in "f<arg_key>s</arg_key><arg_value> {".unicodeScalars {
                let ended = scanner.consume(scalar)
                #expect(!ended)
            }
            var prematureEnd = false
            for _ in 0 ..< 100_000 { prematureEnd = scanner.consume("x") || prematureEnd }
            #expect(!prematureEnd)
            #expect(scanner.bufferedCharacterCount <= 32)
            var endings = 0
            for scalar in "</arg_value></tool_call>".unicodeScalars {
                if scanner.consume(scalar) { endings += 1 }
            }
            #expect(endings == 1)
        }
    }
}
