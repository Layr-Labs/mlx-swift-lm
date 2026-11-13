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

        @Test("numeric schema conversion preserves strings, ambiguity, invalid values and nesting")
        func schemaCases() throws {
            let cases: [(String, String, JSONValue)] = [
                (#"{"type":"integer"}"#, "3", .int(3)),
                (#"{"type":"number"}"#, "3.25", .double(3.25)),
                (#"{"type":"integer"}"#, "3e2", .int(300)),
                (#"{"type":"integer"}"#, "3.5", .string("3.5")),
                (#"{"type":"number"}"#, "NaN", .string("NaN")),
                (#"{"type":"number"}"#, "1e9999", .string("1e9999")),
                (#"{"type":"integer"}"#, "9223372036854775808", .string("9223372036854775808")),
                (#"{"type":"string"}"#, "003", .string("003")),
                (#"{"type":["string","integer"]}"#, "003", .string("003")),
                (#"{"anyOf":[{"type":"integer"},{"type":"null"}]}"#, "3", .int(3)),
                (#"{"anyOf":[{"type":"integer"},{"type":"boolean"}]}"#, "3", .string("3")),
                (##"{"$ref":"#/$defs/count"}"##, "3", .string("3")),
                (#"{}"#, "3", .string("3")),
                (#"{"type":"boolean"}"#, "true", .string("true")),
            ]
            for (schema, text, converted) in cases {
                let specs = try #require(
                    JSONSerialization.jsonObject(with: Data(schema.utf8)) as? [String: any Sendable]
                )
                let tools: [[String: any Sendable]] = [
                    [
                        "type": "function",
                        "function": [
                            "name": "count_items",
                            "parameters": [
                                "type": "object", "properties": ["count": specs],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ]
                ]
                let frame = "<|tool_call>call:count_items{count:<|\"|>\(text)<|\"|>}<tool_call|>"
                for strict in [false, true] {
                    let call = try #require(
                        GemmaFunctionParser(strict: strict).parse(content: frame, tools: tools))
                    #expect(
                        call.function.arguments["count"] == (enabled ? converted : .string(text)))
                }
            }
        }

        @Test("nested numeric marker strings follow properties and items")
        func nestedNumbers() throws {
            let tools = try toolSpecs(
                #"""
                [{"function":{"name":"store","parameters":{"type":"object","properties":{
                  "counts":{"type":"array","items":{"type":"integer"}},
                  "meta":{"type":"object","additionalProperties":{"type":"number"}}
                }}}}]
                """#)
            let call = try #require(
                GemmaFunctionParser().parse(
                    content:
                        #"<|tool_call>call:store{counts:[<|"|>3<|"|>],meta:{weight:<|"|>1.5<|"|>}}<tool_call|>"#,
                    tools: tools))
            #expect(
                call.function.arguments == [
                    "counts": .array([enabled ? .int(3) : .string("3")]),
                    "meta": .object(["weight": enabled ? .double(1.5) : .string("1.5")]),
                ])
        }

        @Test("a fallback bare-string neighbor does not prevent numeric marker conversion")
        func fallbackNeighbor() throws {
            let tools = try toolSpecs(
                #"""
                [{"function":{"name":"store","parameters":{"type":"object","properties":{
                  "count":{"type":"integer"},"city":{"type":"string"}
                }}}}]
                """#)
            let call = try #require(
                GemmaFunctionParser().parse(
                    content:
                        #"<|tool_call>call:store{city:Boston, MA,count:<|"|>3<|"|>}<tool_call|>"#,
                    tools: tools))
            #expect(
                call.function.arguments == [
                    "city": .string("Boston, MA"), "count": enabled ? .int(3) : .string("3"),
                ])
        }
        @Test("unsupported container schemas preserve quoted descendants")
        func unsupportedContainers() throws {
            let cases: [(String, String, [String: JSONValue])] = [
                (
                    #"{"properties":{"id":true},"additionalProperties":{"type":"integer"}}"#,
                    #"id:<|"|>003<|"|>"#, ["id": .string("003")]
                ),
                (
                    #"{"patternProperties":{"^id$":{"type":"string"}},"additionalProperties":{"type":"integer"}}"#,
                    #"id:<|"|>003<|"|>"#, ["id": .string("003")]
                ),
                (
                    #"{"properties":{"values":{"type":"array","prefixItems":[{"type":"string"}],"items":{"type":"integer"}}}}"#,
                    #"values:[<|"|>003<|"|>,<|"|>4<|"|>]"#,
                    ["values": .array([.string("003"), .string("4")])]
                ),
                (
                    ##"{"$ref":"#/$defs/args","properties":{"id":{"type":"integer"}}}"##,
                    #"id:<|"|>003<|"|>"#, ["id": .string("003")]
                ),
                (
                    ##"{"properties":{"meta":{"$ref":"#/$defs/args","properties":{"id":{"type":"integer"}}}}}"##,
                    #"meta:{id:<|"|>003<|"|>}"#, ["meta": .object(["id": .string("003")])]
                ),
            ]
            for (schema, arguments, expected) in cases {
                let tools = try toolSpecs(
                    "[{\"function\":{\"name\":\"store\",\"parameters\":" + schema + "}}]")
                let frame = "<|tool_call>call:store{" + arguments + "}<tool_call|>"
                let call = try #require(GemmaFunctionParser().parse(content: frame, tools: tools))
                #expect(call.function.arguments == expected)
                let processor = ToolCallProcessor(
                    format: .gemma, tools: tools, streaming: .init(exposesName: { _ in true }))
                var segments: [ToolCallStreamSegment] = []
                for character in frame {
                    segments += processor.processChunkSegments(String(character))
                }
                segments += processor.processEOSSegments()
                let calls = segments.compactMap { segment -> ToolCall? in
                    if case .callCompleted(let call, _) = segment { return call }
                    return nil
                }
                #expect(calls.count == 1)
                #expect(calls.first?.function.arguments == expected)
            }
        }

        private func toolSpecs(_ json: String) throws -> [[String: any Sendable]] {
            try #require(
                JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: any Sendable]])
        }
    }
}
