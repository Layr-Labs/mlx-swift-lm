// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    /// Every served tool format, on a call that keeps its format's grammar but
    /// breaks the caller's schema: parsing is not schema validation, so the
    /// call is returned as generated (the caller validates it), whole or
    /// streamed, alone or beside valid calls.
    @Suite("Calls that break only their schema are returned as generated")
    struct SchemaOnlyToolCallTests {
        enum Shape: String, CaseIterable, Sendable {
            case missingRequired, undeclaredWithoutAdditionalProperties
            case undeclaredWithAdditionalPropertiesFalse, untypedParameter
            /// `i` is optional the way Pydantic writes it: string or null.
            case nullableUnionParameter
            /// `i` is declared an integer; the model writes text.
            case mismatchedType
        }

        static let formats: [ToolCallFormat] = [
            .gemma, .harmony, .qwen35, .xmlFunction, .nemotron, .mimoV2, .glm5, .json,
        ]

        /// `read(path, i)`, both required, as an agent such as oh-my-pi declares it.
        static func tools(_ shape: Shape) -> [[String: any Sendable]] {
            let intent: [String: any Sendable] =
                switch shape {
                case .untypedParameter: ["description": "intent"]
                case .nullableUnionParameter:
                    ["anyOf": [["type": "string"], ["type": "null"]], "default": NSNull()]
                case .mismatchedType: ["type": "integer"]
                default: ["type": "string"]
                }
            var parameters: [String: any Sendable] = [
                "type": "object",
                "properties": ["path": ["type": "string"], "i": intent] as [String: any Sendable],
                "required": ["path", "i"],
            ]
            if shape == .undeclaredWithAdditionalPropertiesFalse {
                parameters["additionalProperties"] = false
            }
            let function: [String: any Sendable] = [
                "name": "read", "description": "Read a file", "parameters": parameters,
            ]
            return [["type": "function", "function": function]]
        }

        /// The arguments of the call that breaks the schema in `shape`.
        static func arguments(_ shape: Shape) -> [(String, String)] {
            switch shape {
            case .missingRequired: [("path", "CONTEXT.md")]
            case .undeclaredWithoutAdditionalProperties, .undeclaredWithAdditionalPropertiesFalse:
                [("path", "CONTEXT.md"), ("i", "read it"), ("extra", "more")]
            case .untypedParameter, .nullableUnionParameter, .mismatchedType:
                [("path", "CONTEXT.md"), ("i", "read it")]
            }
        }

        /// One call in `format`'s native frame; every value a string.
        static func frame(_ format: ToolCallFormat, _ arguments: [(String, String)]) -> String {
            let json =
                "{" + arguments.map { "\"\($0.0)\":\"\($0.1)\"" }.joined(separator: ",") + "}"
            switch format {
            case .gemma:
                return "<|tool_call>call:read{"
                    + arguments.map { "\($0.0):<|\"|>\($0.1)<|\"|>" }.joined(separator: ",")
                    + "}<tool_call|>"
            case .harmony:
                return "<|start|>assistant to=functions.read<|channel|>commentary json<|message|>"
                    + json + "<|call|>"
            case .glm5:
                return "<tool_call>read"
                    + arguments.map { "<arg_key>\($0.0)</arg_key><arg_value>\($0.1)</arg_value>" }
                    .joined() + "</tool_call>"
            case .json:
                return "<tool_call>\n{\"name\": \"read\", \"arguments\": \(json)}\n</tool_call>"
            case .mimoV2:
                return "<tool_call><function=read>"
                    + arguments.map { "<parameter=\($0.0)>\($0.1)</parameter>" }.joined()
                    + "</function></tool_call>"
            default:
                return "<tool_call>\n<function=read>\n"
                    + arguments.map { "<parameter=\($0.0)>\n\($0.1)\n</parameter>\n" }.joined()
                    + "</function>\n</tool_call>"
            }
        }

        /// The calls a processor returns for `output` and the text it shows,
        /// fed one character at a time; streamed through the segment API when
        /// the format streams calls.
        static func run(_ format: ToolCallFormat, _ output: String, tools: [[String: any Sendable]])
            -> (calls: [ToolCall], text: String)
        {
            let streams = ToolCallStreamingOptions.supports(format)
            let processor = ToolCallProcessor(
                format: format, tools: tools,
                streaming: streams ? ToolCallStreamingOptions(exposesName: { _ in true }) : nil)
            var calls: [ToolCall] = []
            var text = ""
            func take(_ segments: [ToolCallStreamSegment]) {
                for segment in segments {
                    switch segment {
                    case .text(let piece): text += piece
                    case .call(let call), .callCompleted(let call, _): calls.append(call)
                    default: break
                    }
                }
            }
            for character in output {
                if streams {
                    take(processor.processChunkSegments(String(character)))
                } else {
                    text += processor.processChunk(String(character)) ?? ""
                }
            }
            if streams {
                take(processor.processEOSSegments())
            } else {
                text += processor.processEOS(returnBufferedText: true) ?? ""
            }
            return (calls + processor.drainToolCalls(), text)
        }

        @Test(arguments: formats, Shape.allCases)
        func aCallThatBreaksOnlyItsSchemaIsACall(format: ToolCallFormat, shape: Shape) {
            let result = Self.run(
                format, Self.frame(format, Self.arguments(shape)), tools: Self.tools(shape))
            #expect(result.calls.map(\.function.name) == ["read"], "\(format) \(shape)")
            #expect(
                result.calls.first?.function.arguments["path"] == .string("CONTEXT.md"),
                "\(format) \(shape)")
            #expect(
                result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(format) \(shape): \(result.text)")
        }

        /// `DARKBLOOM_GRAMMAR_ONLY_TOOL_PARSING=0` restores each parser's
        /// schema refusals, and only those.
        @Test(arguments: Shape.allCases)
        func theSwitchRestoresTheSchemaRefusals(shape: Shape) {
            let refused: [(any ToolCallParser, ToolCallFormat, Set<Shape>)] = [
                (GLM5ToolCallParser(grammarOnly: false), .glm5, Set(Shape.allCases)),
                (
                    GemmaFunctionParser(grammarOnly: false), .gemma,
                    [.undeclaredWithAdditionalPropertiesFalse]
                ),
                (
                    MiMoV2ToolCallParser(grammarOnly: false), .mimoV2,
                    [.nullableUnionParameter, .mismatchedType]
                ),
            ]
            for (parser, format, shapes) in refused {
                let frame = Self.frame(format, Self.arguments(shape))
                let call = parser.parse(content: frame, tools: Self.tools(shape))
                #expect((call == nil) == shapes.contains(shape), "\(format) \(shape)")
            }
        }

        @Test func theSwitchIsOnUnlessItsVariableTurnsItOff() {
            #expect(GrammarOnlyToolParsing.enabled(environment: [:]))
            for value in ["0", "off", " FALSE ", "no"] {
                #expect(
                    !GrammarOnlyToolParsing.enabled(
                        environment: [GrammarOnlyToolParsing.environmentKey: value]))
            }
            #expect(
                GrammarOnlyToolParsing.enabled(
                    environment: [GrammarOnlyToolParsing.environmentKey: "1"]))
        }

        @Test(arguments: formats, Shape.allCases)
        func aParallelBatchKeepsEveryCall(format: ToolCallFormat, shape: Shape) {
            let valid = Self.frame(
                format, [("path", "a.md"), ("i", shape == .mismatchedType ? "1" : "first")])
            let invalid = Self.frame(format, Self.arguments(shape))
            let result = Self.run(
                format, valid + "\n" + invalid + "\n" + valid, tools: Self.tools(shape))
            #expect(result.calls.count == 3, "\(format) \(shape): \(result.text)")
            #expect(
                result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(format) \(shape): \(result.text)")
        }
    }
}
