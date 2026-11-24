// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

struct ToolArgumentNormalizationTests {
    private func tools(_ schema: [String: any Sendable]) -> [[String: any Sendable]] {
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["value": schema], "required": ["value"],
        ]
        return [["function": ["name": "read", "parameters": parameters] as [String: any Sendable]]]
    }

    @Test("JSON, native XML and recovered XML read the same declared values")
    func dialectParity() throws {
        let cases: [(String, String, JSONValue)] = [
            ("integer", "6", .int(6)), ("integer", "1.0", .int(1)),
            ("int64", "1.0", .int(1)), ("float64", "0.25", .double(0.25)),
            ("integer", "1.20e2", .int(120)), ("integer", "-0.0", .int(0)),
            ("integer", "9007199254740993.0", .int(9_007_199_254_740_993)),
            ("integer", "9223372036854775807.0", .int(Int.max)),
            ("integer", "-9223372036854775808.0", .int(Int.min)),
            ("number", "1e100", .double(1e100)), ("number", "0.25", .double(0.25)),
            ("boolean", "false", .bool(false)), ("boolean", "yes", .bool(true)),
            ("array", "['SF', 'LA']", .array([.string("SF"), .string("LA")])),
            ("array", "\"SF\", \"LA\"", .array([.string("SF"), .string("LA")])),
            ("array", "('SF', 'LA')", .array([.string("SF"), .string("LA")])),
            ("object", "{'enabled': False}", .object(["enabled": .bool(false)])),
        ]
        for (type, text, expected) in cases {
            let json = String(
                decoding: try JSONSerialization.data(withJSONObject: [
                    "name": "read", "arguments": ["value": text],
                ]), as: UTF8.self)
            let xml = "<function=read><parameter=value>\(text)</parameter></function>"
            for (format, payload): (ToolCallFormat, String) in [
                (.json, "<tool_call>\(json)</tool_call>"),
                (.xmlFunction, "<tool_call>\(xml)</tool_call>"),
                (.lfm2, xml),
            ] {
                let processor = ToolCallProcessor(
                    format: format, tools: tools(["type": type]),
                    toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
                _ = processor.processChunk(payload)
                processor.processEOS()
                #expect(
                    processor.toolCalls.first?.function.arguments["value"] == expected,
                    "\(type): \(text), \(format)")
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
    }

    @Test("Unconvertible values survive normalization and strict validation rejects them")
    func invalidValues() {
        let cases = [
            ("integer", "1.5"), ("integer", "1.00000000000000000000000000000000000000001"),
            ("integer", "9223372036854775808"), ("number", "nan"), ("number", "inf"),
            ("boolean", "maybe"), ("array", "SF, LA"), ("array", "{'a': 1}"),
            ("object", "[1, 2]"), ("array", "__import__('os').system('x')"),
        ]
        for (type, text) in cases {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": type])
                    == .string(text))
            for policy in ToolCallValidationPolicy.allCases {
                let processor = ToolCallProcessor(
                    format: .lfm2, tools: tools(["type": type]),
                    toolCallPolicy: .init(validation: policy, returnsFailedCallsAsText: false))
                _ = processor.processChunk(
                    "<function=read><parameter=value>\(text)</parameter></function>")
                processor.processEOS()
                if policy == .strict {
                    #expect(processor.toolCalls.isEmpty)
                    #expect(processor.rejectedToolCalls.first?.reason == .invalidArguments)
                    #expect(processor.recoveredToolCallCount == 0)
                } else {
                    #expect(processor.toolCalls.first?.function.arguments["value"] == .string(text))
                    #expect(processor.recoveredToolCallCount == 1)
                }
            }
        }
    }

    @Test("Strings and ambiguous unions retain their spelling")
    func ambiguity() {
        let schemas: [[String: any Sendable]] = [
            ["type": "string"], ["type": ["string", "integer"]],
            ["type": ["boolean", "integer"]],
            ["anyOf": [["type": "integer"], ["$ref": "#/$defs/text"]]],
            ["type": "integer", "$ref": "#/$defs/other"],
            ["description": "opaque"],
        ]
        for schema in schemas {
            #expect(
                ToolArgumentNormalization.normalize(.string("001"), schema: schema)
                    == .string("001"))
        }
        #expect(
            ToolArgumentNormalization.normalize(
                .string("null"), schema: ["type": ["string", "null"]]) == .string("null"))
    }

    @Test("Numeric unions and intersections respect integer subtyping")
    func numericCombinators() {
        let intersection: [String: any Sendable] = [
            "allOf": [["type": "number"], ["type": "integer"]]
        ]
        #expect(
            ToolArgumentNormalization.normalize(.string("6.0"), schema: intersection) == .int(6))
        let union: [String: any Sendable] = ["type": ["integer", "number"]]
        #expect(ToolArgumentNormalization.normalize(.string("1.5"), schema: union) == .double(1.5))
    }

    @Test("Normalization descends into declared objects and arrays and preserves IDs")
    func nestedArguments() {
        let item: [String: any Sendable] = [
            "type": "object", "properties": ["count": ["type": "integer"]],
        ]
        let schema: [String: any Sendable] = ["type": "array", "items": item]
        let call = ToolCall(
            function: .init(name: "read", arguments: ["value": .string("[{'count': '2'}]")]),
            id: "keep-me")
        let normalized = ToolArgumentNormalization.normalize(call, tools: tools(schema))
        #expect(normalized.id == call.id)
        #expect(normalized.function.arguments["value"] == .array([.object(["count": .int(2)])]))
        #expect(ToolArgumentNormalization.normalize(normalized, tools: tools(schema)) == normalized)
    }

    @Test("Permissive schema validation never bypasses name authorization or syntax")
    func authorization() {
        for recovery in ToolCallRecoveryPolicy.allCases {
            let processor = ToolCallProcessor(
                format: .json, tools: tools(["type": "integer"]),
                toolCallPolicy: .init(
                    recovery: recovery, validation: .permissive, returnsFailedCallsAsText: false))
            _ = processor.processChunk(
                #"<tool_call>{"name":"other","arguments":{"value":"6"}}</tool_call>"#)
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.rejectedToolCalls.first?.reason == .undeclaredTool)
        }
    }

    @Test("Extreme exponents and non-integral decimals cannot overflow or round into integers")
    func extremeIntegers() {
        for text in [
            "1e9223372036854775807", "1e-9223372036854775808",
            "1.0e-9223372036854775808", "1e-999999999999999999999999",
            "-9223372036854775809.0", "99999999999999999999.1", "0.00000000000000000001",
        ] {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": "integer"])
                    == .string(text))
        }
        for (text, value) in [("0e10000", 0), ("1200e-2", 12), (".000e-300", 0)] {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": "integer"])
                    == .int(value))
        }
    }

    private static let notString: [String: any Sendable] = ["not": ["type": "string"]]
    private static let integerOrText: [String: any Sendable] = [
        "if": ["type": "integer"],
        "then": ["type": "integer", "minimum": -100_000_000] as [String: any Sendable],
        "else": ["type": "string"],
    ]

    /// The calls `format` reads from one `value` parameter written as text.
    private func xmlCall(_ text: String, _ schema: [String: any Sendable], format: ToolCallFormat)
        -> JSONValue?
    {
        let processor = ToolCallProcessor(
            format: format, tools: tools(schema),
            toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
        _ = processor.processChunk(
            "<tool_call>\n<function=read>\n<parameter=value>\n\(text)\n</parameter>\n</function>\n</tool_call>"
        )
        processor.processEOS()
        return processor.toolCalls.first?.function.arguments["value"]
    }

    @Test("XML text reads as the JSON literal a combinator or untyped schema admits")
    func xmlLiteralsUnderCombinators() {
        let cases: [([String: any Sendable], String, JSONValue)] = [
            (Self.notString, "-2500", .int(-2500)),
            (Self.notString, "15000", .int(15000)),
            (Self.notString, "true", .bool(true)),
            (Self.notString, "null", .null),
            (Self.integerOrText, "-2500", .int(-2500)),
            (["type": ["integer", "string"]] as [String: any Sendable], "-2500", .int(-2500)),
            (
                ["anyOf": [["type": "string"], ["type": "number"]] as [[String: any Sendable]]],
                "2.5", .double(2.5)
            ),
            (["description": "untyped"], "-2500", .int(-2500)),
            ([:], "42", .int(42)),
            // A literal a non-string type admits wins over the string reading,
            // as vLLM reads `null` for an optional string; Python's spellings too.
            (["type": ["boolean", "string"]], "true", .bool(true)),
            (["type": ["null", "string"]], "null", .null),
            (["type": ["null", "string"]], "None", .null),
            (["type": ["boolean", "string"]], "True", .bool(true)),
            (["type": ["boolean", "string"]], "False", .bool(false)),
            (["type": ["boolean", "string"]], "TRUE", .bool(true)),
            (["type": ["null", "string"]], "NULL", .null),
            (Self.notString, "None", .null),
            ([:], "true", .bool(true)),
        ]
        for (schema, text, expected) in cases {
            for format: ToolCallFormat in [.xmlFunction, .nemotron, .qwen35] {
                #expect(xmlCall(text, schema, format: format) == expected, "\(text), \(format)")
            }
        }
    }

    @Test("XML text keeps its spelling when no admitted literal reads it")
    func xmlTextKeptUnderCombinators() {
        let cases: [([String: any Sendable], String)] = [
            (Self.integerOrText, "12.50"),
            (Self.integerOrText, "-200000000"),
            (Self.integerOrText, "001"),
            (["type": ["integer", "string"]], "+5"),
            (["type": ["integer", "string"]], "true"),
            (["type": ["integer", "string"]], "None"),
            (["type": ["boolean", "string"]], "yes"),
            (["type": ["null", "string"]], "nil"),
            (["not": ["type": "integer"]], "7"),
            (["type": "string"], "-2500"),
            (["type": "string"], "None"),
            (
                [
                    "anyOf": [["type": "integer"], ["$ref": "#/$defs/missing"]]
                        as [[String: any Sendable]]
                ], "-2500"
            ),
        ]
        for (schema, text) in cases {
            for format: ToolCallFormat in [.xmlFunction, .nemotron, .qwen35] {
                #expect(
                    xmlCall(text, schema, format: format) == .string(text), "\(text), \(format)")
            }
        }
    }

    @Test("A quoted string becomes a literal only when the schema rejects every string")
    func quotedLiteralsOnlyWhenTextIsRejected() {
        let json: (String, [String: any Sendable]) -> JSONValue? = { text, schema in
            let processor = ToolCallProcessor(
                format: .json, tools: self.tools(schema),
                toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
            _ = processor.processChunk(
                #"<tool_call>{"name":"read","arguments":{"value":"\#(text)"}}</tool_call>"#)
            processor.processEOS()
            return processor.toolCalls.first?.function.arguments["value"]
        }
        #expect(json("-2500", Self.notString) == .int(-2500))
        #expect(json("-2500", Self.integerOrText) == .string("-2500"))
        #expect(json("-2500", ["type": ["integer", "string"]]) == .string("-2500"))
        #expect(json("-2500", ["description": "untyped"]) == .string("-2500"))
        #expect(json("abc", Self.notString) == .string("abc"))
        #expect(json("null", ["type": ["null", "string"]]) == .string("null"))
        #expect(json("None", Self.notString) == .string("None"))
    }

    @Test("Gemma keeps its marker-quoted and bare text under string-rejecting schemas")
    func gemmaValuesUnchanged() throws {
        let schemas: [[String: any Sendable]] = [
            Self.notString,
            ["anyOf": [["type": "integer"], ["type": "boolean"]] as [[String: any Sendable]]],
        ]
        for schema in schemas {
            let tools = tools(schema)
            for strict in [false, true] {
                let call = try #require(
                    GemmaFunctionParser(strict: strict).parse(
                        content: #"<|tool_call>call:read{value:<|"|>-2500<|"|>}<tool_call|>"#,
                        tools: tools))
                #expect(call.function.arguments["value"] == .string("-2500"))
            }
            let older = GemmaFunctionParser(
                startTag: "<start_function_call>", endTag: "<end_function_call>",
                escapeMarker: "<escape>")
            for field in ["<escape>-2500<escape>", "-2500"] {
                let call = try #require(
                    older.parse(
                        content:
                            "<start_function_call>call:read{value:\(field)}<end_function_call>",
                        tools: tools))
                #expect(call.function.arguments["value"] == .string("-2500"), "\(field)")
            }
        }
    }

    @Test("A number no integer or double holds exactly keeps its digits as text")
    func inexactNumbersKeepTheirText() {
        let schemas: [[String: any Sendable]] = [
            Self.notString, [:], ["type": ["integer", "string"]],
            ["anyOf": [["type": "string"], ["type": "number"]] as [[String: any Sendable]]],
        ]
        for schema in schemas {
            for text in [
                "12345678901234567890", "-98765432109876543210", "0.12345678901234567891",
                "123456789012345678901234567890.5", "1e400",
            ] {
                for format: ToolCallFormat in [.xmlFunction, .nemotron, .qwen35] {
                    #expect(
                        xmlCall(text, schema, format: format) == .string(text), "\(text), \(format)"
                    )
                }
            }
        }
        for (text, value): (String, JSONValue) in [
            ("9223372036854775807", .int(Int.max)), ("-9223372036854775808", .int(Int.min)),
            ("1.5", .double(1.5)), ("0.1", .double(0.1)), ("1e3", .int(1000)),
            ("2.50", .double(2.5)),
        ] {
            #expect(xmlCall(text, [:], format: .qwen35) == value, "\(text)")
        }
    }

    @Test("XML JSON object or array text under an untyped parameter reads as that value")
    func untypedContainers() throws {
        let untyped: [[String: any Sendable]] = [[:], ["description": "any value"]]
        for schema in untyped {
            for format: ToolCallFormat in [.xmlFunction, .nemotron, .qwen35] {
                #expect(
                    xmlCall(#"{"a":1,"b":[true,null,"x"]}"#, schema, format: format)
                        == .object(["a": .int(1), "b": .array([.bool(true), .null, .string("x")])]))
                #expect(
                    xmlCall(#"[1,"two",{"c":2.5}]"#, schema, format: format)
                        == .array([.int(1), .string("two"), .object(["c": .double(2.5)])]))
            }
        }
        // Partial or trailing text, inexact numbers, and typed or combinator
        // schemas keep the text.
        for text in [
            #"{"a":"#, #"{"a":1} and more"#, #"{"n":12345678901234567890}"#,
            #"[0.12345678901234567]"#,
        ] {
            #expect(xmlCall(text, [:], format: .qwen35) == .string(text), "\(text)")
        }
        let typed: [[String: any Sendable]] = [
            Self.notString, ["type": ["object", "string"]], ["type": "string"],
            ["anyOf": [["type": "string"], ["type": "object"]] as [[String: any Sendable]]],
        ]
        for schema in typed {
            #expect(xmlCall(#"{"a":1}"#, schema, format: .qwen35) == .string(#"{"a":1}"#))
        }
        // A quoted JSON string and Gemma's marker-quoted text are unchanged.
        let processor = ToolCallProcessor(
            format: .json, tools: tools([:]),
            toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
        _ = processor.processChunk(
            #"<tool_call>{"name":"read","arguments":{"value":"{\"a\":1}"}}</tool_call>"#)
        processor.processEOS()
        #expect(processor.toolCalls.first?.function.arguments["value"] == .string(#"{"a":1}"#))
        let gemma = try #require(
            GemmaFunctionParser().parse(
                content: #"<|tool_call>call:read{value:<|"|>{"a":1}<|"|>}<tool_call|>"#,
                tools: tools([:])))
        #expect(gemma.function.arguments["value"] == .string(#"{"a":1}"#))
    }

    /// The schemas as the coordinator forwards them: ingress normalization
    /// writes the first branch's type beside a union (and `nullable`), and a
    /// render type beside a reference.
    private static func forwarded(_ json: String) throws -> [String: any Sendable] {
        try #require(tryParseJSON(json) as? [String: any Sendable])
    }

    private func forwardedTools(
        _ property: [String: any Sendable], defs: [String: any Sendable] = [:]
    )
        -> [[String: any Sendable]]
    {
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["value": property], "$defs": defs,
        ]
        return [["function": ["name": "read", "parameters": parameters] as [String: any Sendable]]]
    }

    @Test("A render type beside a union or reference does not hide the branch a value reads as")
    func renderTypeBesideUnions() throws {
        let cases: [(String, [String: any Sendable], String, JSONValue)] = [
            (
                #"{"anyOf":[{"type":"string"},{"type":"integer"}],"type":"string"}"#, [:], "-4200",
                .int(-4200)
            ),
            (
                #"{"anyOf":[{"type":"string"},{"type":"boolean"}],"type":"string"}"#, [:], "true",
                .bool(true)
            ),
            (
                #"{"anyOf":[{"type":"string"},{"type":"boolean"}],"type":"string"}"#, [:], "False",
                .bool(false)
            ),
            (
                #"{"oneOf":[{"type":"string"},{"type":"number"}],"type":"string"}"#, [:], "2.5",
                .double(2.5)
            ),
            (
                #"{"anyOf":[{"type":"integer"},{"type":"boolean"}],"type":"integer"}"#, [:],
                "false", .bool(false)
            ),
            (
                #"{"anyOf":[{"type":"integer"},{"type":"null"}],"nullable":true,"type":"integer"}"#,
                [:], "null", .null
            ),
            (
                #"{"anyOf":[{"type":"integer"},{"type":"null"}],"nullable":true,"type":"integer"}"#,
                [:], "42", .int(42)
            ),
            (
                ##"{"anyOf":[{"$ref":"#/$defs/AutoRenewFlag","type":"string"},{"type":"null"}],"nullable":true,"type":"string"}"##,
                ["AutoRenewFlag": ["type": "boolean"] as [String: any Sendable]], "true",
                .bool(true)
            ),
            (
                ##"{"$ref":"#/$defs/Count","type":"string"}"##,
                ["Count": ["type": "integer"] as [String: any Sendable]], "7", .int(7)
            ),
            (
                #"{"anyOf":[{"type":"string"},{"type":"integer"}],"type":"string"}"#, [:], "abc",
                .string("abc")
            ),
            (#"{"type":"string","nullable":true}"#, [:], "null", .null),
            (#"{"type":"string","nullable":true}"#, [:], "x", .string("x")),
            (#"{"enum":["default","inherit","archive",-1],"type":"string"}"#, [:], "-1", .int(-1)),
            (
                #"{"enum":["default","inherit","archive",-1],"type":"string"}"#, [:], "archive",
                .string("archive")
            ),
            (#"{"not":{"type":"string"},"type":"string"}"#, [:], "-2500", .int(-2500)),
            (
                #"{"if":{"type":"integer"},"then":{"type":"integer","minimum":-100000000},"else":{"type":"string"},"type":"string"}"#,
                [:], "-2500", .int(-2500)
            ),
            // A client's type that agrees with every branch keeps asserting.
            (
                #"{"type":"string","anyOf":[{"minLength":1},{"pattern":"^a"}]}"#, [:], "5",
                .string("5")
            ),
            (
                #"{"type":"integer","anyOf":[{"type":"integer"},{"type":"null"}]}"#, [:], "null",
                .string("null")
            ),
        ]
        for (json, defs, text, expected) in cases {
            let tools = forwardedTools(try Self.forwarded(json), defs: defs)
            for format: ToolCallFormat in [.xmlFunction, .nemotron, .qwen35] {
                let processor = ToolCallProcessor(
                    format: format, tools: tools,
                    toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
                _ = processor.processChunk(
                    "<tool_call>\n<function=read>\n<parameter=value>\n\(text)\n</parameter>\n</function>\n</tool_call>"
                )
                processor.processEOS()
                #expect(
                    processor.toolCalls.first?.function.arguments["value"] == expected,
                    "\(json) \(text) \(format)")
            }
            // The SDK's strict check accepts what the reading returns whenever
            // the reading changed the text.
            if expected != .string(text) {
                let schema = try Self.forwarded(json)
                if case .invalid(let violations) = ToolSchemaValidator.validate(
                    expected, against: schema),
                    defs.isEmpty
                {
                    Issue.record("\(json) \(text): \(ToolSchemaValidator.describe(violations))")
                }
            }
        }
    }

    @Test("JSON and Gemma calls under forwarded union schemas")
    func forwardedUnionsInJSONAndGemma() throws {
        let union = try Self.forwarded(
            #"{"anyOf":[{"type":"string"},{"type":"integer"}],"type":"string"}"#)
        let tools = forwardedTools(union)
        // A JSON integer stays an integer and a JSON string a string (GPT-OSS
        // and the JSON dialects write typed values).
        for (payload, expected): (String, JSONValue) in [
            (#"{"value":-4200}"#, .int(-4200)), (#"{"value":"-4200"}"#, .string("-4200")),
        ] {
            let json = ToolCallProcessor(
                format: .json, tools: tools,
                toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
            _ = json.processChunk(
                #"<tool_call>{"name":"read","arguments":"# + payload + "}</tool_call>")
            json.processEOS()
            #expect(json.toolCalls.first?.function.arguments["value"] == expected, "\(payload)")
            let harmony = ToolCallProcessor(
                format: .harmony, tools: tools,
                toolCallPolicy: ToolCallPolicy(returnsFailedCallsAsText: false))
            _ = harmony.processChunk(
                "<|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json<|message|>"
                    + payload + "<|call|>")
            harmony.processEOS()
            #expect(
                harmony.toolCalls.first?.function.arguments["value"] == expected,
                "harmony \(payload)")
        }
        // Gemma: a marker-quoted value keeps its text; a bare value reads as
        // the union branch its JSON type matches, as under `["string","integer"]`.
        let quoted = try #require(
            GemmaFunctionParser().parse(
                content: #"<|tool_call>call:read{value:<|"|>-4200<|"|>}<tool_call|>"#, tools: tools)
        )
        #expect(quoted.function.arguments["value"] == .string("-4200"))
        let bare = try #require(
            GemmaFunctionParser().parse(
                content: "<|tool_call>call:read{value:-4200}<tool_call|>", tools: tools))
        #expect(bare.function.arguments["value"] == .int(-4200))
    }

    @Test("A streamed call keeps the digits its complete parse keeps")
    func streamedInexactNumbersAgree() {
        let text = "12345678901234567890"
        let processor = ToolCallProcessor(
            format: .qwen35, tools: tools([:]),
            streaming: .init(exposesName: { _ in true }))
        let frame =
            "<tool_call>\n<function=read>\n<parameter=value>\n\(text)\n</parameter>\n</function>\n</tool_call>"
        var arguments = ""
        var completed: [ToolCall] = []
        for character in frame {
            for segment in processor.processChunkSegments(String(character)) {
                switch segment {
                case .callArguments(let fragment): arguments += fragment
                case .callCompleted(let call, let remainder):
                    completed.append(call)
                    arguments += remainder
                default: break
                }
            }
        }
        for segment in processor.processEOSSegments() {
            if case .callCompleted(let call, let remainder) = segment {
                completed.append(call)
                arguments += remainder
            }
        }
        #expect(completed.first?.function.arguments["value"] == .string(text))
        #expect(arguments == #"{"value":"12345678901234567890"}"#)
    }

    @Test("Streaming exposes no text a literal reading would replace")
    func keepsTextAgreesWithLiteralReading() {
        let parameters: (([String: any Sendable]) -> [[String: any Sendable]]) = { self.tools($0) }
        for schema: [String: any Sendable] in [
            Self.notString, Self.integerOrText, ["type": ["integer", "string"]], [:],
            ["type": ["string", "null"]], ["type": ["string", "boolean"]],
        ] {
            #expect(
                !parameterKeepsText(paramName: "value", funcName: "read", tools: parameters(schema))
            )
        }
        for schema: [String: any Sendable] in [
            ["type": "string"], ["enum": ["a", "b"]],
            [
                "anyOf": [["type": "integer"], ["$ref": "#/$defs/missing"]]
                    as [[String: any Sendable]]
            ],
        ] {
            #expect(
                parameterKeepsText(paramName: "value", funcName: "read", tools: parameters(schema)))
        }
    }

    @Test("Python container parsing is bounded and rejects incomplete structure")
    func boundedLiterals() {
        #expect(
            tryParsePythonLiteral(
                String(repeating: "[", count: 64) + "0" + String(repeating: "]", count: 64)) == nil)
        #expect(tryParsePythonLiteral("['SF',") == nil)
        #expect(tryParsePythonLiteral("[f()]") == nil)
    }
}
