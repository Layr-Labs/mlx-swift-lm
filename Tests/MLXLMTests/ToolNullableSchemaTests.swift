// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// OpenAPI 3.0 `nullable: true` adds null to the type declared on the same
/// schema node. Fixtures are decoded from JSON so markers arrive as on the wire.
struct ToolNullableSchemaTests {
    private let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")

    /// What the provider's schema normalizer emits for nullable integer,
    /// boolean and object parameters, each with a genuine value.
    private static let shapes: [(json: String, text: String, value: JSONValue)] = [
        (#"{"type":"integer","nullable":true}"#, "42", .int(42)),
        (
            #"{"type":"integer","nullable":true,"anyOf":[{"type":"integer"},{"type":"null"}]}"#,
            "42", .int(42)
        ),
        (
            #"{"type":"integer","nullable":true,"oneOf":[{"type":"integer"},{"type":"null"}]}"#,
            "42", .int(42)
        ),
        (#"{"type":"boolean","nullable":true}"#, "false", .bool(false)),
        (
            #"{"type":"boolean","nullable":true,"anyOf":[{"type":"boolean"},{"type":"null"}]}"#,
            "false", .bool(false)
        ),
        (
            #"{"type":"object","nullable":true,"properties":{},"anyOf":["#
                + #"{"type":"object","properties":{"n":{"type":"integer"}}},{"type":"null"}]}"#,
            #"{"n":3}"#, .object(["n": .int(3)])
        ),
    ]

    private func schema(_ json: String) throws -> [String: any Sendable] {
        try #require(tryParseJSON(json) as? [String: any Sendable])
    }

    private func marked(json markers: [String]) throws -> [[String: any Sendable]] {
        try markers.map { try schema(#"{"type":"integer","nullable":\#($0)}"#) }
    }

    private func marked(_ markers: [any Sendable]) -> [[String: any Sendable]] {
        markers.map { ["type": "integer", "nullable": $0] }
    }

    private func tools(_ property: [String: any Sendable]) -> [[String: any Sendable]] {
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["p": property],
        ]
        let function: [String: any Sendable] = ["name": "probe", "parameters": parameters]
        return [["function": function]]
    }

    private func frame(_ text: String) -> String {
        "<tool_call>\n<function=probe>\n<parameter=p>\n\(text)\n</parameter>\n"
            + "</function>\n</tool_call>"
    }

    /// Reads `text` as the `p` parameter of a framed Qwen XML call.
    private func read(_ text: String, _ property: [String: any Sendable]) throws -> JSONValue {
        let call = try #require(parser.parse(content: frame(text), tools: tools(property)))
        return try #require(call.function.arguments["p"])
    }

    private func convert(_ text: String, _ property: [String: any Sendable]) -> JSONValue {
        ToolArgumentNormalization.normalize(.string(text), schema: property)
    }

    /// "valid", "unknown", or the violations.
    private func outcome(_ value: JSONValue, _ property: [String: any Sendable]) -> String {
        switch ToolSchemaValidator.validate(value, against: property) {
        case .valid: "valid"
        case .unknown: "unknown"
        case .invalid(let violations): violations.map(\.description).joined(separator: "; ")
        }
    }

    // MARK: - Normalized shapes

    @Test("Normalized nullable shapes read XML null as a valid null")
    func normalizedShapesAdmitNull() throws {
        for shape in Self.shapes {
            let property = try schema(shape.json)
            let value = try read("null", property)
            #expect(value == .null, "\(shape.json)")
            #expect(
                ToolSchemaValidator.validate(
                    arguments: ["p": value], forToolNamed: "probe", in: tools(property)) == .valid,
                "\(shape.json)")
        }
    }

    @Test("Normalized nullable shapes still read genuine values")
    func normalizedShapesKeepGenuineValues() throws {
        for shape in Self.shapes {
            let property = try schema(shape.json)
            let value = try read(shape.text, property)
            #expect(value == shape.value, "\(shape.json)")
            #expect(outcome(value, property) == "valid", "\(shape.json)")
        }
    }

    @Test("Strict validation dispatches a nullable null instead of rejecting it")
    func strictProcessorAcceptsNull() throws {
        for shape in Self.shapes {
            let property = try schema(shape.json)
            let processor = ToolCallProcessor(
                format: .qwen35, tools: tools(property),
                toolCallPolicy: .init(validation: .strict))
            _ = processor.processChunk(frame("null"))
            processor.processEOS()
            #expect(processor.rejectedToolCalls.isEmpty, "\(shape.json)")
            #expect(
                processor.toolCalls.first?.function.arguments["p"] == .null, "\(shape.json)")
        }
    }

    // MARK: - Values that must not change

    @Test("Typed nulls pass through and string schemas keep literal text")
    func typedNullAndStringLiterals() throws {
        let nullable = try schema(#"{"type":"integer","nullable":true}"#)
        #expect(ToolArgumentNormalization.normalize(.null, schema: nullable) == .null)
        #expect(ToolArgumentNormalization.normalize(.null, schema: ["type": "integer"]) == .null)

        for json in [
            #"{"type":"string"}"#, #"{"type":"string","nullable":true}"#,
            #"{"type":["string"],"nullable":true}"#,
            #"{"type":"string","nullable":true,"anyOf":[{"type":"string"},{"type":"null"}]}"#,
        ] {
            let property = try schema(json)
            for text in ["null", "None", "true", "001", #"{"a":1}"#] {
                let value = try read(text, property)
                #expect(value == .string(text), "\(json): \(text)")
                #expect(outcome(value, property) == "valid", "\(json): \(text)")
            }
        }
    }

    @Test("A sibling type without the marker still excludes null")
    func conjunctionWithoutMarker() throws {
        let nullBranch = try schema(
            #"{"type":"integer","anyOf":[{"type":"integer"},{"type":"null"}]}"#)
        let text = try read("null", nullBranch)
        #expect(text == .string("null"))
        #expect(outcome(text, nullBranch) == "$ must be an integer")
        #expect(outcome(.null, nullBranch) == "$ must be an integer")

        let stringBranch = try schema(
            #"{"type":"integer","anyOf":[{"type":"integer"},{"type":"string"}]}"#)
        let word = try read("abc", stringBranch)
        #expect(word == .string("abc"))
        #expect(outcome(word, stringBranch) == "$ must be an integer")
        let number = try read("42", stringBranch)
        #expect(number == .int(42))
        #expect(outcome(number, stringBranch) == "valid")
    }

    // MARK: - The marker

    @Test("Only a real Boolean true admits null")
    func strictMarker() throws {
        for property in marked([true, NSNumber(value: true)]) {
            #expect(convert("null", property) == .null)
            #expect(outcome(.null, property) == "valid")
        }

        let off = try marked(json: ["false"]) + marked([false, NSNumber(value: false)])
        let malformed =
            try marked(json: ["1", "0", #""true""#, "null", "[true]"])
            + marked([1, NSNumber(value: 1), "true", NSNull()])
        for property in off + malformed {
            let marker = String(describing: property["nullable"])
            #expect(convert("null", property) == .string("null"), "\(marker)")
            #expect(convert("42", property) == .int(42), "\(marker)")
            #expect(outcome(.null, property) == "$ must be an integer", "\(marker)")
        }
        for property in off {
            #expect(outcome(.int(42), property) == "valid")
        }
        for property in malformed {
            let marker = String(describing: property["nullable"])
            #expect(outcome(.int(42), property) == "unknown", "\(marker)")
        }

        // `false` does not remove a null the type itself declares.
        let declared = try schema(#"{"type":["integer","null"],"nullable":false}"#)
        #expect(convert("null", declared) == .null)
        #expect(outcome(.null, declared) == "valid")
    }

    @Test("The marker widens only a type declared on the same node")
    func sameNodeType() throws {
        let typeless = try schema(#"{"nullable":true,"anyOf":[{"type":"integer"}]}"#)
        #expect(convert("null", typeless) == .string("null"))
        #expect(
            outcome(.null, typeless) == "$ must satisfy at least one of the allowed schemas")

        let bare = try schema(#"{"nullable":true}"#)
        #expect(convert("null", bare) == .string("null"))
        #expect(outcome(.null, bare) == "valid")

        let childMarker = try schema(
            #"{"type":"integer","anyOf":[{"type":"integer","nullable":true}]}"#)
        #expect(convert("null", childMarker) == .string("null"))
        #expect(outcome(.null, childMarker) == "$ must be an integer")

        let siblingMarker = try schema(#"{"type":"integer","allOf":[{"nullable":true}]}"#)
        #expect(convert("null", siblingMarker) == .string("null"))
        #expect(outcome(.null, siblingMarker) == "$ must be an integer")

        let branch = try schema(#"{"anyOf":[{"type":"integer","nullable":true}]}"#)
        #expect(convert("null", branch) == .null)
        #expect(outcome(.null, branch) == "valid")
    }

    // MARK: - Other assertions still apply

    @Test("enum and const still decide whether null is allowed")
    func enumAndConst() throws {
        let cases = [
            (
                #"{"type":"integer","nullable":true,"enum":[1,2]}"#,
                "$ must be one of the allowed values"
            ),
            (#"{"type":"integer","nullable":true,"enum":[1,null]}"#, "valid"),
            (
                #"{"type":"integer","nullable":true,"const":1}"#,
                "$ must equal the declared constant"
            ),
            (#"{"type":"integer","nullable":true,"const":null}"#, "valid"),
        ]
        for (json, expected) in cases {
            let property = try schema(json)
            #expect(convert("null", property) == .null, "\(json)")
            #expect(outcome(.null, property) == expected, "\(json)")
        }
    }

    @Test("Combinators can still exclude an admitted null")
    func combinators() throws {
        let oneOfRequirement = "$ must satisfy exactly one of the allowed schemas"
        let cases: [(String, JSONValue, String)] = [
            (#""allOf":[{"type":"integer","nullable":true}]"#, .null, "valid"),
            (#""allOf":[{"type":"integer"}]"#, .string("null"), "$ must be an integer"),
            (#""anyOf":[{"type":"integer","nullable":true}]"#, .null, "valid"),
            (
                #""anyOf":[{"type":"integer"},{"type":"boolean"}]"#, .string("null"),
                "$ must satisfy at least one of the allowed schemas"
            ),
            (
                #""oneOf":[{"type":"integer"},{"type":"boolean"}]"#, .string("null"),
                oneOfRequirement
            ),
            (
                #""oneOf":[{"type":"integer","nullable":true},{"type":"null"}]"#, .null,
                oneOfRequirement
            ),
        ]
        for (combinator, converted, expected) in cases {
            let property = try schema(#"{"type":"integer","nullable":true,\#(combinator)}"#)
            #expect(convert("null", property) == converted, "\(combinator)")
            #expect(outcome(.null, property) == expected, "\(combinator)")
        }
    }

    @Test("References, unsupported keywords and depth limits stay unproven")
    func unknownStaysUnknown() throws {
        for json in [
            ##"{"type":"integer","nullable":true,"$ref":"#/$defs/value"}"##,
            ##"{"type":"integer","nullable":true,"anyOf":[{"$ref":"#/$defs/value"}]}"##,
            #"{"type":"integer","nullable":true,"not":{"type":"null"}}"#,
        ] {
            let property = try schema(json)
            #expect(convert("null", property) == .string("null"), "\(json)")
            #expect(outcome(.null, property) == "unknown", "\(json)")
        }

        var deep = try schema(#"{"type":"integer","nullable":true}"#)
        for _ in 0 ... 64 {
            deep = ["allOf": [deep]]
        }
        #expect(convert("null", deep) == .string("null"))
        #expect(outcome(.null, deep) == "unknown")
    }

    // MARK: - Nesting

    @Test("Nullable members of objects and arrays admit null")
    func nestedMembers() throws {
        let member = #"{"type":"integer","nullable":true}"#
        let union =
            #"{"type":"integer","nullable":true,"anyOf":[{"type":"integer"},{"type":"null"}]}"#
        for nested in [member, union] {
            let object = try schema(#"{"type":"object","properties":{"n":\#(nested)}}"#)
            for text in [#"{"n": null}"#, #"{"n": "null"}"#] {
                let value = try read(text, object)
                #expect(value == .object(["n": .null]), "\(nested): \(text)")
                #expect(outcome(value, object) == "valid", "\(nested): \(text)")
            }
        }

        let array = try schema(#"{"type":"array","items":\#(member)}"#)
        let items = try read("[1, null]", array)
        #expect(items == .array([.int(1), .null]))
        #expect(outcome(items, array) == "valid")

        let plain = try schema(#"{"type":"object","properties":{"n":{"type":"integer"}}}"#)
        let rejected = try read(#"{"n": null}"#, plain)
        #expect(rejected == .object(["n": .null]))
        #expect(outcome(rejected, plain) == "$.n must be an integer")
    }

    // MARK: - Known gap

    /// The normalizer collapses `anyOf(integer, string, null)` to a sibling
    /// integer type, so the conjunction no longer shows the string branch.
    @Test("A string branch hidden by the collapsed type no longer keeps literal null")
    func hiddenStringBranch() throws {
        let property = try schema(
            #"{"type":"integer","nullable":true,"anyOf":["#
                + #"{"type":"integer"},{"type":"string"},{"type":"null"}]}"#)
        withKnownIssue("The collapsed sibling type hides the string branch") {
            #expect(convert("null", property) == .string("null"))
        }
    }
}
