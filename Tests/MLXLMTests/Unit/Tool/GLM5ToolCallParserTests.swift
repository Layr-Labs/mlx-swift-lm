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

        @Test("Numeric decoding preserves the encoded decimal value or keeps the raw text")
        func numericPrecision() throws {
            for number in [Int.min, Int.max, 9_007_199_254_740_993] {
                #expect(
                    parse(String(number), type: "integer")?.function.arguments["x"] == .int(number))
            }
            for raw in [
                "18446744073709551615", "9007199254740993.25", "0.10000000000000000000001",
                "1e309", "1e-400", "NaN", "Infinity", "+1", "01", "1.", ".5", "1e",
            ] {
                #expect(parse(raw, type: "number")?.function.arguments["x"] == .string(raw))
            }
            let negativeZero = try #require(parse("-0.0", type: "number")?.function.arguments["x"])
            guard case .double(let zero) = negativeZero else {
                Issue.record("negative zero lost its sign")
                return
            }
            #expect(zero == 0 && zero.sign == .minus)
            #expect(parse("0.125", type: "number")?.function.arguments["x"] == .double(0.125))
            #expect(parse("1e100", type: "number")?.function.arguments["x"] == .double(1e100))
            let bad = #"{"bad":18446744073709551615}"#
            #expect(parse(bad, type: "object")?.function.arguments["x"] == .string(bad))
        }

        @Test("Type evidence decides only between a raw string and JSON")
        func schemaEvidence() {
            let nullValued: [[String: any Sendable]] = [
                [:], ["enum": ["null", 1] as [any Sendable]], ["type": ["string", "null"]],
                ["type": "string", "nullable": true], ["type": "future"],
                ["anyOf": [["type": "string"], ["type": "null"]]],
                ["type": "number", "nullable": 1],
            ]
            for schema in nullValued {
                #expect(
                    parser.parse(content: frame("null"), tools: [tool(properties: ["x": schema])])?
                        .function.arguments["x"] == .null)
            }
            let typedString = tool(properties: ["x": ["type": "string", "$ref": "#/$defs/x"]])
            #expect(
                parser.parse(content: frame("null"), tools: [typedString])?
                    .function.arguments["x"] == .string("null"))
            #expect(parser.parse(content: frame("null"), tools: nil) == nil)
            #expect(parser.parse(content: frame("null"), tools: []) == nil)
            #expect(
                parser.parse(content: frame("null"), tools: [tool()])?.function.arguments["x"]
                    == .null)
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
                    ])?.function.arguments.isEmpty == true)
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

        @Test("JSON with duplicate keys, invalid syntax or excessive depth stays raw text")
        func malformedJSON() {
            let deep = String(repeating: "[", count: 66) + "0" + String(repeating: "]", count: 66)
            for (type, text) in [
                ("object", #"{"x":1,"x":2}"#), ("object", #"{"a":1,"\u0061":2}"#),
                ("object", #"{"x":1,}"#), ("object", #"{"x":tru}"#), ("object", #"{"x":01}"#),
                ("object", #"{"x":"\uD800"}"#), ("object", #"{"x":1} trailing"#),
                ("array", "[1,]"), ("array", deep),
            ] {
                #expect(parse(text, type: type)?.function.arguments["x"] == .string(text))
            }
            #expect(parse(String(repeating: "x", count: 1 << 20), type: "string") == nil)
        }

        private func readTool(
            additionalProperties: (any Sendable)? = nil
        ) -> [[String: any Sendable]] {
            var parameters: [String: any Sendable] = [
                "type": "object",
                "properties": [
                    "i": ["type": "string"], "path": ["type": "string"],
                ] as [String: any Sendable],
                "required": ["path", "i"],
            ]
            if let additionalProperties {
                parameters["additionalProperties"] = additionalProperties
            }
            let function: [String: any Sendable] = ["name": "read", "parameters": parameters]
            return [["type": "function", "function": function]]
        }

        private func readFrame(_ path: String, intent: String? = nil) -> String {
            let intentArgument = intent.map {
                "<arg_key>i</arg_key><arg_value>\($0)</arg_value>"
            }
            return "<tool_call>read<arg_key>path</arg_key><arg_value>\(path)</arg_value>"
                + (intentArgument ?? "") + "</tool_call>"
        }

        @Test("A call that omits a required argument is returned as generated")
        func missingRequiredArguments() throws {
            let tools = readTool()
            let call = try #require(parser.parse(content: readFrame("CONTEXT.md"), tools: tools))
            #expect(call.function.name == "read")
            #expect(call.function.arguments == ["path": .string("CONTEXT.md")])
            #expect(
                parser.parse(content: "<tool_call>read</tool_call>", tools: tools)?
                    .function.arguments.isEmpty == true)

            let batch =
                readFrame("a.md", intent: "first") + "\n" + readFrame("b.md") + "\n"
                + readFrame("c.md", intent: "third")
            let calls = parser.parseEOS(batch, tools: tools)
            #expect(
                calls.map { $0.function.arguments["path"] } == [
                    .string("a.md"), .string("b.md"), .string("c.md"),
                ])
            #expect(
                calls.map { $0.function.arguments["i"] } == [
                    .string("first"), nil, .string("third"),
                ])

            let processor = ToolCallProcessor(format: .glm5, tools: tools)
            for character in "Reading.\n" + batch {
                _ = processor.processChunk(String(character))
            }
            #expect(processor.processEOS(returnBufferedText: true) == nil)
            #expect(processor.drainToolCalls().count == 3)
            #expect(processor.parseFailureCount == 0)
        }

        @Test(
            "An undeclared argument is returned whatever additionalProperties says",
            arguments: [nil, false, true, [:] as [String: any Sendable]] as [(any Sendable)?])
        func undeclaredArguments(additionalProperties: (any Sendable)?) throws {
            let tools = readTool(additionalProperties: additionalProperties)
            for (raw, value) in [
                ("plain text", JSONValue.string("plain text")), ("42", .int(42)),
                (#"{"a":[1,null]}"#, .object(["a": .array([.int(1), .null])])),
                ("{not json", .string("{not json")),
            ] {
                let frame =
                    "<tool_call>read<arg_key>path</arg_key><arg_value>a.md</arg_value>"
                    + "<arg_key>extra</arg_key><arg_value>\(raw)</arg_value></tool_call>"
                let call = try #require(parser.parse(content: frame, tools: tools), "\(raw)")
                #expect(call.function.arguments["extra"] == value, "\(raw)")
                #expect(call.function.arguments["path"] == .string("a.md"))
            }
            // A key that only canonically equals a declared one is another key.
            let accent = [tool(properties: ["\u{e9}": ["type": "string"]])]
            let call = try #require(
                parser.parse(content: frame("1", key: "e\u{301}"), tools: accent))
            let key = try #require(call.function.arguments.keys.first)
            #expect(Array(key.utf8) == Array("e\u{301}".utf8))
            #expect(call.function.arguments[key] == .int(1))
        }

        @Test(
            "An argument without usable type evidence is JSON when it parses, else its raw text",
            arguments: [
                [:], ["description": "intent"], ["enum": ["null", 1] as [any Sendable]],
                ["type": "future"], ["allOf": [["type": "integer"]]], ["$ref": "#/$defs/x"],
            ] as [[String: any Sendable]])
        func untypedArguments(schema: [String: any Sendable]) throws {
            let tools = [tool(properties: ["x": schema])]
            for (raw, value) in [
                ("null", JSONValue.null), (" 7 ", .int(7)),
                (#"[1,"a"]"#, .array([.int(1), .string("a")])),
                (#""quoted""#, .string("quoted")), ("hello world", .string("hello world")),
                (#"{"a":"#, .string(#"{"a":"#)), ("1 2", .string("1 2")), ("", .string("")),
            ] {
                #expect(
                    parser.parse(content: frame(raw), tools: tools)?.function.arguments["x"]
                        == value,
                    "\(raw)")
            }
            // Its value ends at its first `</arg_value>`, as a raw string's does.
            let calls = parser.parseEOS(
                frame(#""a"#) + frame(#"b""#), tools: tools)
            #expect(calls.map { $0.function.arguments["x"] } == [.string(#""a"#), .string(#"b""#)])
        }

        @Test("Pydantic's optional string keeps text that looks like JSON as text")
        func optionalStrings() {
            let schema: [String: any Sendable] = [
                "anyOf": [["type": "string"], ["type": "null"]], "default": NSNull(),
            ]
            let optional = [tool(properties: ["x": schema])]
            for (raw, value) in [
                ("123", JSONValue.string("123")), ("1.0", .string("1.0")),
                ("true", .string("true")),
                (#""q""#, .string(#""q""#)), ("null", .null),
            ] {
                #expect(
                    parser.parse(content: frame(raw), tools: optional)?.function.arguments["x"]
                        == value,
                    "\(raw)")
            }
        }

        @Test("A value of a non-string type keeps its JSON reading, but never a top-level string")
        func nonStringTypesKeepJSON() {
            let integer = [tool(properties: ["n": ["type": "integer"]])]
            #expect(
                parser.parse(
                    content: frame(#"{"cmd":"x </arg_value> y"}"#, key: "n"), tools: integer)?
                    .function.arguments["n"] == .object(["cmd": .string("x </arg_value> y")]))
            let calls = parser.parseEOS(
                frame(#""12"#, key: "n") + frame(#"14""#, key: "n"), tools: integer)
            #expect(
                calls.map { $0.function.arguments["n"] } == [.string(#""12"#), .string(#"14""#)])
            // A string union reads an object only where `object` is declared.
            let optional: [String: any Sendable] = ["type": ["string", "null"]]
            let split = parser.parseEOS(
                frame(#"{"a":""#) + frame(#""}"#), tools: [tool(properties: ["x": optional])])
            #expect(
                split.map { $0.function.arguments["x"] } == [.string(#"{"a":""#), .string(#""}"#)])
        }

        @Test("Schema-checked parsing (the switch off) keeps every refusal it had")
        func schemaChecksRestoreTheOldRefusals() {
            let checked = GLM5ToolCallParser(grammarOnly: false)
            func refused(_ value: String, schema: [String: any Sendable], key: String = "x") -> Bool
            {
                checked.parse(
                    content: frame(value, key: key), tools: [tool(properties: ["x": schema])])
                    == nil
            }
            for (type, value) in [
                ("integer", "true"), ("integer", "1.25"), ("boolean", "1"), ("boolean", "yes"),
                ("null", "nil"), ("number", "\"1\""), ("array", "{}"), ("object", "[]"),
                ("number", "18446744073709551615"), ("number", "9007199254740993.25"),
                ("number", "0.10000000000000000000001"), ("number", "1e309"), ("number", "1e-400"),
                ("number", "NaN"), ("number", "Infinity"), ("number", "+1"), ("number", "01"),
                ("number", "1."), ("number", ".5"), ("number", "1e"),
                ("object", #"{"bad":18446744073709551615}"#), ("object", #"{"x":1,"x":2}"#),
                ("object", #"{"a":1,"\u0061":2}"#), ("object", #"{"x":1,}"#),
                ("object", #"{"x":tru}"#),
                ("object", #"{"x":01}"#), ("object", #"{"x":"\uD800"}"#),
                ("object", #"{"x":1} trailing"#),
                ("object", "{\"é\":1,\"e\\u0301\":2}"), ("array", "[1,]"),
                (
                    "array",
                    String(repeating: "[", count: 66) + "0" + String(repeating: "]", count: 66)
                ),
            ] {
                #expect(refused(value, schema: ["type": type]), "\(type) \(value)")
            }
            for schema: [String: any Sendable] in [
                [:], ["enum": ["null", 1] as [any Sendable]], ["type": ["string", "null"]],
                ["type": "string", "nullable": true], ["type": "future"],
                ["anyOf": [["type": "string"], ["type": "null"]]],
                ["type": "string", "$ref": "#/$defs/x"], ["type": "number", "nullable": 1],
            ] {
                #expect(refused("null", schema: schema), "\(schema)")
            }
            #expect(checked.parse(content: frame("null"), tools: [tool()]) == nil)
            #expect(
                checked.parse(
                    content: "<tool_call>f</tool_call>",
                    tools: [tool(properties: ["x": ["type": "integer"]], required: ["x"])]) == nil)
            #expect(
                checked.parse(
                    content: frame("1", key: "e\u{301}"),
                    tools: [tool(properties: ["é": ["type": "integer"]])]) == nil)
        }

        @Test("A union with string takes JSON only for its other declared types")
        func stringUnions() throws {
            for schema in [
                ["type": ["string", "null"]], ["type": "string", "nullable": true],
            ] as [[String: any Sendable]] {
                for (raw, value) in [
                    ("null", JSONValue.null), ("123", .string("123")), ("hello", .string("hello")),
                    (#""q""#, .string(#""q""#)),
                ] {
                    #expect(
                        parser.parse(content: frame(raw), tools: [tool(properties: ["x": schema])])?
                            .function.arguments["x"] == value, "\(schema) \(raw)")
                }
            }
            let objectOrString = [tool(properties: ["x": ["type": ["string", "object"]]])]
            #expect(
                parser.parse(content: frame(#"{"m":"</arg_value>"}"#), tools: objectOrString)?
                    .function.arguments["x"] == .object(["m": .string("</arg_value>")]))
            #expect(
                parser.parse(content: frame("{draft"), tools: objectOrString)?
                    .function.arguments["x"] == .string("{draft"))
        }

        @Test("A typed value of another type is returned as generated")
        func typeMismatches() {
            for (type, raw, value) in [
                ("integer", "true", JSONValue.bool(true)), ("integer", "1.25", .double(1.25)),
                ("boolean", "1", .int(1)), ("boolean", "yes", .string("yes")),
                ("null", "nil", .string("nil")), ("number", #""1""#, .string("1")),
                ("array", "{}", .object([:])), ("object", "[]", .array([])),
                ("number", "1e309", .string("1e309")), ("number", "01", .string("01")),
                ("integer", "18446744073709551615", .string("18446744073709551615")),
                ("object", #"{"x":1,"x":2}"#, .string(#"{"x":1,"x":2}"#)),
            ] {
                #expect(parse(raw, type: type)?.function.arguments["x"] == value, "\(type) \(raw)")
            }
        }

        @Test("Argument types come from the exact declared key, else additionalProperties")
        func argumentTypes() {
            let parameters: [String: any Sendable] = [
                "properties": [
                    "\u{e9}": ["type": "integer"],
                    "n": ["type": ["string", "future"], "nullable": true] as [String: any Sendable],
                    "u": ["description": "no type"],
                ] as [String: any Sendable],
                "additionalProperties": ["type": "object"],
            ]
            #expect(
                GLM5ToolCallParser.argumentTypes("\u{e9}", parameters: parameters) == ["integer"])
            #expect(
                GLM5ToolCallParser.argumentTypes("e\u{301}", parameters: parameters) == ["object"])
            #expect(
                GLM5ToolCallParser.argumentTypes("n", parameters: parameters) == ["string", "null"])
            #expect(GLM5ToolCallParser.argumentTypes("u", parameters: parameters).isEmpty)
            #expect(GLM5ToolCallParser.argumentTypes("x", parameters: [:]).isEmpty)
        }

        @Test("Frames that break the grammar still fail closed")
        func grammarFailuresStayClosed() {
            let tools = [tool(properties: ["x": ["type": "integer"]])]
            for text in [
                frame("1<tool_call>f</tool_call>"), frame("{\"a\":<arg_key>"),
                "<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value>",
                "<tool_call>f<arg_key></arg_key><arg_value>1</arg_value></tool_call>",
                "<tool_call>g<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            ] {
                #expect(parser.parse(content: text, tools: tools) == nil, "\(text)")
                #expect(parser.parseEOS(frame("2") + text, tools: tools).isEmpty, "\(text)")
            }
        }

        @Test(
            "Argument names preserve leading combining scalars and stay distinct from canonical equivalents"
        )
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
            let duplicates = "{\"é\":1,\"e\\u0301\":2}"
            #expect(
                parse(duplicates, type: "object")?.function.arguments["x"] == .string(duplicates))
            // A key that only canonically equals a declared one takes no type from it.
            #expect(
                parser.parse(
                    content: frame("1", key: "e\u{301}"),
                    tools: [
                        tool(properties: ["é": ["type": "string"]])
                    ])?.function.arguments["e\u{301}"] == .int(1))
        }
    }
}
