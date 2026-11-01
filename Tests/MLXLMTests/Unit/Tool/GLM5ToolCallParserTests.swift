// Copyright © 2026 Eigen Labs.
// Fixtures follow zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda
// chat_template.jinja, SHA-256:
// 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
// The assistant-history loop writes strings verbatim and all other arguments
// with tojson(ensure_ascii=False), inside repeated arg_key/arg_value tags.
// The shipped EigenLabs/GLM-5.3-Flash-MLX-NVFP4-mtp template, SHA-256
// 46b2e2137e2759355b3e0e3e90cf4bd172b568437a75588af5c457cc8c095b2e, keeps
// that loop byte-identical; it only writes `.0.` attribute accesses as `[0].`.

import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite("GLM5 complete native tool frames")
    struct GLM5ToolCallParserTests {
        private let parser = GLM5ToolCallParser()

        private func tool(
            _ name: String = "f", properties: [String: any Sendable] = [:],
            required: [String] = [], flat: Bool = false
        ) -> [String: any Sendable] {
            let function: [String: any Sendable] = [
                "name": name,
                "parameters": [
                    "type": "object", "properties": properties, "required": required,
                ] as [String: any Sendable],
            ]
            return flat ? function : ["type": "function", "function": function]
        }

        private func frame(_ value: String, key: String = "x", name: String = "f") -> String {
            "<tool_call>\(name)<arg_key>\(key)</arg_key><arg_value>\(value)</arg_value></tool_call>"
        }

        private func parse(_ value: String, type: String) -> ToolCall? {
            parser.parse(content: frame(value), tools: [tool(properties: ["x": ["type": type]])])
        }

        @Test("Only the exact native namespaces select the new parser")
        func formatSelection() {
            for type in ["glm5_next", "glm5_next_text", "glm5_next_mtp"] {
                #expect(ToolCallFormat.infer(from: type) == .glm5)
            }
            for type in ["glm5", "glm5_next_vision", "glm5_next_future", "glm53", " glm5_next"] {
                #expect(ToolCallFormat.infer(from: type) == nil)
            }
            #expect(ToolCallFormat.infer(from: "glm4") == .glm4)
            #expect(ToolCallFormat.infer(from: "glm4_moe_lite") == .glm4)
            #expect(ToolCallFormat.glm5.createParser() is GLM5ToolCallParser)
            #expect(ToolCallFormat.glm4.createParser() is GLM4ToolCallParser)
        }

        @Test("Raw strings and JSON types are never coerced into each other")
        func exactArgumentTypes() throws {
            for text in [
                "null", "true", "1", "1.25", "\"quoted\"", "[1,2]", "", " \n雪 e\u{301}\t ",
            ] {
                let call = try #require(parse(text, type: "string"))
                #expect(call.function.arguments["x"] == .string(text))
                let string = try #require(call.function.arguments["x"])
                if case .string(let actual) = string {
                    #expect(Array(actual.utf8) == Array(text.utf8))
                }
            }
            #expect(parse("null", type: "null")?.function.arguments["x"] == .null)
            #expect(parse("true", type: "boolean")?.function.arguments["x"] == .bool(true))
            #expect(parse("false", type: "boolean")?.function.arguments["x"] == .bool(false))
            #expect(parse("0", type: "integer")?.function.arguments["x"] == .int(0))
            #expect(parse("1", type: "integer")?.function.arguments["x"] == .int(1))
            #expect(parse("1.25", type: "number")?.function.arguments["x"] == .double(1.25))
            for (type, value) in [
                ("integer", "true"), ("integer", "1.25"), ("boolean", "1"),
                ("boolean", "yes"), ("null", "nil"), ("number", "\"1\""),
                ("array", "{}"), ("object", "[]"),
            ] {
                #expect(parse(value, type: type) == nil)
            }
        }

        @Test("Nested JSON preserves nulls, booleans, numbers and quoted strings")
        func nestedJSON() throws {
            let raw =
                #"{"nested":[0,1,true,false,null,"null",{"n":9007199254740993}],"markup":"</arg_value></tool_call><arg_key>","escaped":"\"雪\\"}"#
            let call = try #require(parse(raw, type: "object"))
            #expect(
                call.function.arguments["x"]
                    == .object([
                        "nested": .array([
                            .int(0), .int(1), .bool(true), .bool(false), .null, .string("null"),
                            .object(["n": .int(9_007_199_254_740_993)]),
                        ]),
                        "markup": .string("</arg_value></tool_call><arg_key>"),
                        "escaped": .string("\"雪\\"),
                    ]))
            #expect(
                parse(#"[1,1.25,null,"1",{"x":false}]"#, type: "array")?
                    .function.arguments["x"]
                    == .array([
                        .int(1), .double(1.25), .null, .string("1"), .object(["x": .bool(false)]),
                    ]))
            #expect(
                parser.parseEOS(
                    frame(raw) + "<tool_call>f</tool_call>",
                    tools: [
                        tool(properties: ["x": ["type": "object"]])
                    ]
                ).count == 2)
        }

        @Test("Numeric decoding preserves the encoded decimal value or fails closed")
        func numericPrecision() throws {
            for number in [Int.min, Int.max, 9_007_199_254_740_993] {
                #expect(
                    parse(String(number), type: "integer")?.function.arguments["x"] == .int(number))
            }
            for raw in [
                "18446744073709551615", "9007199254740993.25", "0.10000000000000000000001",
                "1e309", "1e-400", "NaN", "Infinity", "+1", "01", "1.", ".5", "1e",
            ] {
                #expect(parse(raw, type: "number") == nil)
            }
            let negativeZero = try #require(parse("-0.0", type: "number")?.function.arguments["x"])
            guard case .double(let zero) = negativeZero else {
                Issue.record("negative zero lost its sign")
                return
            }
            #expect(zero == 0 && zero.sign == .minus)
            #expect(parse("0.125", type: "number")?.function.arguments["x"] == .double(0.125))
            #expect(parse("1e100", type: "number")?.function.arguments["x"] == .double(1e100))
            #expect(parse(#"{"bad":18446744073709551615}"#, type: "object") == nil)
        }

        @Test("Type ambiguity and missing schema evidence fail closed")
        func schemaEvidence() {
            let ambiguous: [[String: any Sendable]] = [
                [:], ["enum": ["null", 1] as [any Sendable]], ["type": ["string", "null"]],
                ["type": "string", "nullable": true], ["type": "future"],
                ["anyOf": [["type": "string"], ["type": "null"]]],
                ["type": "string", "$ref": "#/$defs/x"], ["type": "number", "nullable": 1],
            ]
            for schema in ambiguous {
                #expect(
                    parser.parse(content: frame("null"), tools: [tool(properties: ["x": schema])])
                        == nil)
            }
            #expect(parser.parse(content: frame("null"), tools: nil) == nil)
            #expect(parser.parse(content: frame("null"), tools: []) == nil)
            #expect(parser.parse(content: frame("null"), tools: [tool()]) == nil)
            let nullable = tool(properties: ["x": ["type": ["integer", "null"]]])
            #expect(
                parser.parse(content: frame("null"), tools: [nullable])?
                    .function.arguments["x"] == .null)
            #expect(
                parser.parse(content: frame("7"), tools: [nullable])?
                    .function.arguments["x"] == .int(7))
            let additional: [String: any Sendable] = [
                "function": [
                    "name": "f",
                    "parameters": [
                        "type": "object", "additionalProperties": ["type": "string"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
            #expect(
                parser.parse(content: frame("null"), tools: [additional])?
                    .function.arguments["x"] == .string("null"))
        }

        @Test("No-argument calls and parallel calls retain order and distinct instances")
        func noArgumentsAndParallelCalls() throws {
            let tools = [
                tool("ping"), tool("f", properties: ["x": ["type": "integer"]], flat: true),
            ]
            let calls = parser.parseEOS(
                " \n<tool_call>ping</tool_call>\t" + frame("1") + frame("2") + "\r\n", tools: tools)
            #expect(calls.map(\.function.name) == ["ping", "f", "f"])
            #expect(calls.map(\.function.arguments) == [[:], ["x": .int(1)], ["x": .int(2)]])
            #expect(
                parser.parse(content: "<tool_call>ping</tool_call>", tools: tools)?
                    .function.arguments.isEmpty == true)
            #expect(
                parser.parse(content: "<tool_call>ping</tool_call>", tools: [["name": "ping"]])?
                    .function.arguments.isEmpty == true)
            #expect(
                parser.parse(
                    content: "<tool_call>f</tool_call>",
                    tools: [
                        tool(properties: ["x": ["type": "integer"]], required: ["x"])
                    ]) == nil)
            #expect(
                parser.parse(content: "<tool_call>ping</tool_call>", tools: [tool(), tool()]) == nil
            )
            #expect(
                parser.parse(
                    content: "<tool_call>f</tool_call>", tools: [tool(), tool(flat: true)])
                    == nil)
            #expect(parser.parse(content: "<tool_call>unknown</tool_call>", tools: tools) == nil)

            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            for character in frame("1") + frame("2") + "<tool_call>ping</tool_call>" {
                _ = processor.processChunk(String(character))
            }
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.processEOS(returnBufferedText: true) == nil)
            let streamed = processor.drainToolCalls()
            #expect(streamed.map(\.function.name) == ["f", "f", "ping"])
            #expect(streamed.map(\.function.arguments) == [["x": .int(1)], ["x": .int(2)], [:]])
            #expect(processor.drainToolCalls().isEmpty)
        }

        @Test("The native wire has no call-ID field; ID-like argument bytes remain arguments")
        func callIDsAreNotInventedOrDiscarded() {
            let tools = [tool(properties: ["call_id": ["type": "string"]])]
            #expect(
                parser.parse(content: frame("call_123", key: "call_id"), tools: tools)?
                    .function.arguments["call_id"] == .string("call_123"))
            for text in [
                "<tool_call id=\"call_123\">f</tool_call>",
                "<tool_call>f<id>call_123</id></tool_call>",
                "<tool_call>f:call_123</tool_call>",
            ] {
                #expect(parser.parse(content: text, tools: tools) == nil)
            }
        }

        @Test(
            "Malformed framing, duplicate arguments and incomplete batches never partially succeed")
        func malformedFrames() {
            let tools = [tool(properties: ["x": ["type": "string"]])]
            let valid = frame("value")
            let malformed = [
                "f<arg_key>x</arg_key><arg_value>v</arg_value>",
                "<tool_call>f", String(valid.dropLast()), valid + "trailing",
                "<tool_call>f<arg_key>x</arg_key><arg_value>v</tool_call>",
                "<tool_call>f<arg_value>v</arg_value></tool_call>",
                "<tool_call>f<arg_key>x</arg_key>junk<arg_value>v</arg_value></tool_call>",
                "<tool_call>f<arg_key>x</arg_key><arg_value>v</arg_value>junk</tool_call>",
                "<tool_call>f<arg_key>x</arg_key><arg_value>a</arg_value><arg_key>x</arg_key><arg_value>b</arg_value></tool_call>",
                frame("<tool_call>nested"), frame("<arg_key>nested"),
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
            ]
            for text in malformed {
                #expect(parser.parse(content: text, tools: tools) == nil)
                #expect(parser.parseEOS(text, tools: tools).isEmpty)
                #expect(parser.parseEOS(valid + text, tools: tools).isEmpty)
            }
            #expect(parser.parse(content: valid + valid, tools: tools) == nil)
            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            for character in String(valid.dropLast()) {
                _ = processor.processChunk(String(character))
                #expect(processor.toolCalls.isEmpty)
            }
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
        }

        @Test("Nested JSON rejects duplicate keys, invalid syntax and excessive depth")
        func malformedJSON() {
            for text in [
                #"{"x":1,"x":2}"#, #"{"a":1,"\u0061":2}"#,
                #"{"x":1,}"#, #"{"x":tru}"#, #"{"x":01}"#,
                #"{"x":"\uD800"}"#, #"{"x":1} trailing"#,
            ] {
                #expect(parse(text, type: "object") == nil)
            }
            #expect(parse("[1,]", type: "array") == nil)
            #expect(
                parse(
                    String(repeating: "[", count: 66) + "0" + String(repeating: "]", count: 66),
                    type: "array") == nil)
            #expect(parse(String(repeating: "x", count: 1 << 20), type: "string") == nil)
        }

        @Test("Argument names preserve leading combining scalars and reject semantic duplicates")
        func unicodeKeys() throws {
            let key = "\u{301}name"
            let call = try #require(
                parser.parse(
                    content: frame("value", key: key),
                    tools: [
                        tool(properties: [key: ["type": "string"]])
                    ]))
            let actualKey = try #require(call.function.arguments.keys.first)
            #expect(Array(actualKey.utf8) == Array(key.utf8))
            #expect(parse("{\"é\":1,\"e\\u0301\":2}", type: "object") == nil)
            #expect(
                parser.parse(
                    content: frame("1", key: "e\u{301}"),
                    tools: [
                        tool(properties: ["é": ["type": "integer"]])
                    ]) == nil)
        }
    }
}
