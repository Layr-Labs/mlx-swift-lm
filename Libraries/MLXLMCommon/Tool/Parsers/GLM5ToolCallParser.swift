// Copyright © 2026 Eigen Labs.
// GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda/chat_template.jinja
// SHA-256: 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
// The shipped EigenLabs/GLM-5.3-Flash-MLX-NVFP4-mtp template, SHA-256
// 46b2e2137e2759355b3e0e3e90cf4bd172b568437a75588af5c457cc8c095b2e, only
// writes its `.0.` attribute accesses as `[0].`; its tool-call frames are identical.

import Foundation

/// Parses complete native GLM5 frames, not GLM4's permissive recovery dialect.
/// The template emits top-level strings verbatim and other values as JSON. A
/// value whose declared type is only `string` is raw text. A JSON value of a
/// declared non-string type, or any JSON but a string where no `string` is
/// declared, keeps its JSON reading (quoted delimiters in it are data); any
/// other value is JSON when its raw text parses (for a union with `string`,
/// only as one of the other types), else that raw text, as vLLM and SGLang
/// read it. Types come from `argumentTypes`.
///
/// Only the frame grammar and the declared function name are checked: a frame
/// missing a required argument, with an undeclared argument or with a value of
/// another type is returned as generated (`grammarOnly`, on unless
/// `GrammarOnlyToolParsing` is switched off). Output frames contain no call IDs;
/// the response owner assigns them. Parsing is not constrained decoding or
/// JSON-Schema validation. The caller must still validate schema assertions,
/// selection and call cardinality. Raw strings have no delimiter-escaping
/// mechanism in the pinned template; literal structural-tag collisions are
/// not losslessly representable.
public struct GLM5ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    /// UTF-8 bound of one parsed response tool suffix.
    static let maximumBytes = 1 << 20
    /// False restores the schema refusals: a call that misses a required
    /// argument, or has an argument without a declared type, of an ambiguous
    /// type or of another type, does not parse.
    public let grammarOnly: Bool

    public init(grammarOnly: Bool = GrammarOnlyToolParsing.isEnabled) {
        self.grammarOnly = grammarOnly
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard content.utf8.count <= Self.maximumBytes else { return nil }
        var reader = GLM5ToolOutputReader(content, grammarOnly: grammarOnly)
        guard let call = reader.call(tools: tools), reader.isAtEnd else { return nil }
        return call
    }

    /// A malformed or truncated member invalidates the supplied batch. Never
    /// return an earlier successful prefix or manufacture a missing end tag.
    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard buffer.utf8.count <= Self.maximumBytes else { return [] }
        var reader = GLM5ToolOutputReader(buffer, grammarOnly: grammarOnly)
        var calls: [ToolCall] = []
        while !reader.isAtEnd {
            guard let call = reader.call(tools: tools) else { return [] }
            calls.append(call)
        }
        return calls
    }

    /// The JSON types `parameters` declares for the argument `key`: its
    /// property schema's (with its `anyOf`/`oneOf` branches'), else those of
    /// `additionalProperties`. Empty when nothing declares a type. They choose
    /// between a value's raw and JSON readings, never whether a call parses.
    public static func argumentTypes(
        _ key: String, parameters: [String: any Sendable]
    ) -> Set<String> {
        var schema = parameters["additionalProperties"]
        if let properties = parameters["properties"] as? [String: any Sendable],
            let declared = properties.index(forKey: key),
            // Swift dictionary lookup equates canonically equivalent Unicode
            // keys; JSON property names do not.
            properties[declared].key.utf8.elementsEqual(key.utf8)
        {
            schema = properties[declared].value
        }
        guard let schema = schema as? [String: any Sendable] else { return [] }
        var types: Set<String> = []
        let branches = (schema["anyOf"] ?? schema["oneOf"]) as? [[String: any Sendable]] ?? []
        for node in [schema] + branches {
            if let type = node["type"] as? String {
                types.insert(type)
            } else if let values = node["type"] as? [String] {
                types.formUnion(values)
            }
        }
        if let nullable = schema["nullable"], JSONValue.from(nullable) == .bool(true) {
            types.insert("null")
        }
        return types.intersection(jsonTypeNames)
    }

    fileprivate static let jsonTypeNames: Set<String> = [
        "string", "integer", "number", "boolean", "null", "object", "array",
    ]
}

private struct GLM5ToolOutputReader {
    private let bytes: [UInt8]
    private let grammarOnly: Bool
    private var index = 0

    init(_ text: String, grammarOnly: Bool = true) {
        bytes = Array(text.utf8)
        self.grammarOnly = grammarOnly
    }

    var isAtEnd: Bool {
        mutating get {
            skipWhitespace()
            return index == bytes.count
        }
    }

    mutating func call(tools: [[String: any Sendable]]?) -> ToolCall? {
        skipWhitespace()
        guard consume("<tool_call>"), let rawName = take(until: "<", consumeEnd: false) else {
            return nil
        }
        let name = rawName.trimmingCharacters(in: CharacterSet(charactersIn: " \t\r\n"))
        guard Self.validFunctionName(name),
            let parameters = Self.parameters(name, tools: tools, grammarOnly: grammarOnly)
        else { return nil }

        var arguments: [String: JSONValue] = [:]
        while true {
            skipWhitespace()
            if consume("</tool_call>") {
                if !grammarOnly, Self.schemaRefuses(arguments, parameters: parameters) {
                    return nil
                }
                return ToolCall(function: .init(name: name, arguments: arguments))
            }
            guard consume("<arg_key>"), let key = take(until: "</arg_key>"),
                !key.isEmpty, !key.utf8.contains(60), arguments[key] == nil,
                consume("<arg_value>"),
                let value = argument(
                    types: GLM5ToolCallParser.argumentTypes(key, parameters: parameters))
            else { return nil }
            arguments[key] = value
        }
    }

    /// The parameter schema of the one function declared as `name`, used only
    /// as type evidence; empty when it is omitted or not an object.
    private static func parameters(
        _ name: String, tools: [[String: any Sendable]]?, grammarOnly: Bool
    ) -> [String: any Sendable]? {
        guard let tools else { return nil }
        var matches: [[String: any Sendable]] = []
        for tool in tools {
            let function: [String: any Sendable]
            if let wrapped = tool["function"] {
                guard let value = wrapped as? [String: any Sendable] else { return nil }
                function = value
            } else {
                function = tool
            }
            if function["name"] as? String == name { matches.append(function) }
        }
        guard matches.count == 1 else { return nil }
        guard let rawParameters = matches[0]["parameters"] else { return [:] }
        guard let parameters = rawParameters as? [String: any Sendable] else {
            return grammarOnly ? [:] : nil
        }
        return parameters
    }

    /// The refusals of schema-checked parsing (`grammarOnly` false), on a
    /// call the grammar accepts: a parameter object that is not a plain
    /// object schema, a missing required argument, an argument without a
    /// declared type (none, unknown, indirect, or a key that only canonically
    /// equals a declared one), a union with `string`, or a value of another type.
    private static func schemaRefuses(
        _ arguments: [String: JSONValue], parameters: [String: any Sendable]
    ) -> Bool {
        let indirect = [
            "$ref", "$dynamicRef", "anyOf", "oneOf", "allOf", "not", "if", "then", "else",
        ]
        guard parameters["type"] == nil || parameters["type"] as? String == "object",
            !indirect.contains(where: { parameters[$0] != nil }),
            parameters["properties"].map({ $0 is [String: any Sendable] }) ?? true
        else { return true }
        if let rawRequired = parameters["required"] {
            guard let required = rawRequired as? [String],
                required.allSatisfy({ name in
                    arguments.keys.contains { $0.utf8.elementsEqual(name.utf8) }
                })
            else { return true }
        }
        let properties = parameters["properties"] as? [String: any Sendable] ?? [:]
        for (key, value) in arguments {
            var schema = parameters["additionalProperties"]
            if let declared = properties.index(forKey: key) {
                guard properties[declared].key.utf8.elementsEqual(key.utf8) else { return true }
                schema = properties[declared].value
            }
            guard let schema = schema as? [String: any Sendable],
                !indirect.contains(where: { schema[$0] != nil })
            else { return true }
            var types: Set<String>
            if let type = schema["type"] as? String {
                types = [type]
            } else if let values = schema["type"] as? [String], !values.isEmpty {
                types = Set(values)
            } else {
                return true
            }
            if let nullable = schema["nullable"] {
                guard case .bool(let enabled) = JSONValue.from(nullable) else { return true }
                if enabled { types.insert("null") }
            }
            guard types.isSubset(of: GLM5ToolCallParser.jsonTypeNames) else { return true }
            if types.contains("string") ? types.count != 1 : !matches(value, types: types) {
                return true
            }
        }
        return false
    }

    private mutating func argument(types: Set<String>) -> JSONValue? {
        let start = index
        // Quoted delimiters inside a typed JSON value are data. A JSON string
        // is kept only where a string is not written raw, never at the top.
        if !types.isEmpty, types != ["string"], let value = jsonValue(depth: 0),
            Self.matches(value, types: types)
                || (!types.contains("string") && !Self.isString(value))
        {
            skipWhitespace()
            if consume("</arg_value>") { return value }
        }
        index = start
        guard let raw = take(until: "</arg_value>"),
            !["<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>"]
                .contains(where: raw.contains)
        else { return nil }
        // Even `null` is the literal string "null" for a string parameter.
        if types != ["string"], let value = Self.json(raw),
            !types.contains("string") || Self.matches(value, types: types)
        {
            return value
        }
        return .string(raw)
    }

    private static func isString(_ value: JSONValue) -> Bool {
        if case .string = value { return true }
        return false
    }

    /// `text` as one JSON value, surrounding whitespace allowed.
    private static func json(_ text: String) -> JSONValue? {
        var reader = GLM5ToolOutputReader(text)
        guard let value = reader.jsonValue(depth: 0), reader.isAtEnd else { return nil }
        return value
    }

    private static func matches(_ value: JSONValue, types: Set<String>) -> Bool {
        switch value {
        case .null: return types.contains("null")
        case .bool: return types.contains("boolean")
        case .int: return types.contains("integer") || types.contains("number")
        case .double(let number):
            return types.contains("number")
                || (types.contains("integer") && number.rounded(.towardZero) == number)
        case .object: return types.contains("object")
        case .array: return types.contains("array")
        case .string: return false
        }
    }

    // Decode JSON structurally rather than through a dictionary decoder, which
    // silently overwrites duplicate keys. Strings use Foundation's strict escape
    // decoder; numbers must preserve their decimal value through JSONValue's
    // actual encoder, not just parse successfully as a rounded Double.
    private mutating func jsonValue(depth: Int) -> JSONValue? {
        guard depth <= 64 else { return nil }
        skipWhitespace()
        guard index < bytes.count else { return nil }
        switch bytes[index] {
        case 34: return jsonString().map(JSONValue.string)
        case 123:
            index += 1
            skipWhitespace()
            var values: [String: JSONValue] = [:]
            if consume("}") { return .object(values) }
            while true {
                skipWhitespace()
                guard let key = jsonString(), values[key] == nil else { return nil }
                skipWhitespace()
                guard consume(":"), let value = jsonValue(depth: depth + 1) else { return nil }
                values[key] = value
                skipWhitespace()
                if consume("}") { return .object(values) }
                guard consume(",") else { return nil }
            }
        case 91:
            index += 1
            skipWhitespace()
            var values: [JSONValue] = []
            if consume("]") { return .array(values) }
            while true {
                guard let value = jsonValue(depth: depth + 1) else { return nil }
                values.append(value)
                skipWhitespace()
                if consume("]") { return .array(values) }
                guard consume(",") else { return nil }
            }
        case 116: return consume("true") ? .bool(true) : nil
        case 102: return consume("false") ? .bool(false) : nil
        case 110: return consume("null") ? .null : nil
        default: return jsonNumber()
        }
    }

    private mutating func jsonString() -> String? {
        let start = index
        guard consume("\"") else { return nil }
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if byte < 32 { return nil }
            if byte == 34 {
                return try? JSONDecoder().decode(String.self, from: Data(bytes[start ..< index]))
            }
            if byte == 92 {
                guard index < bytes.count else { return nil }
                index += 1
            }
        }
        return nil
    }

    private mutating func jsonNumber() -> JSONValue? {
        let start = index
        while index < bytes.count,
            (48 ... 57).contains(bytes[index]) || [43, 45, 46, 69, 101].contains(bytes[index])
        {
            index += 1
        }
        guard index > start, index - start <= 1024 else { return nil }
        let data = Data(bytes[start ..< index])
        guard let original = GLM5DecimalIdentity(data),
            var value = try? JSONDecoder().decode(JSONValue.self, from: data)
        else { return nil }
        switch value {
        case .int(0) where bytes[start] == 45: value = .double(-0.0)
        case .int: break
        case .double(let number): guard number.isFinite else { return nil }
        default: return nil
        }
        guard let encoded = try? JSONEncoder().encode(value),
            let roundTrip = GLM5DecimalIdentity(encoded), original == roundTrip
        else { return nil }
        return value
    }

    private static func validFunctionName(_ name: String) -> Bool {
        (1 ... 64).contains(name.utf8.count)
            && name.utf8.allSatisfy {
                (48 ... 57).contains($0) || (65 ... 90).contains($0) || (97 ... 122).contains($0)
                    || $0 == 45 || $0 == 95
            }
    }

    private mutating func skipWhitespace() {
        while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 }
    }

    private mutating func consume(_ literal: String) -> Bool {
        let marker = Array(literal.utf8)
        guard bytes[index...].starts(with: marker) else { return false }
        index += marker.count
        return true
    }

    private mutating func take(until literal: String, consumeEnd: Bool = true) -> String? {
        let marker = Array(literal.utf8)
        let start = index
        while index < bytes.count {
            if bytes[index...].starts(with: marker) {
                let value = String(decoding: bytes[start ..< index], as: UTF8.self)
                if consumeEnd { index += marker.count }
                return value
            }
            index += 1
        }
        return nil
    }
}

/// Syntax validation and exact normalized decimal identity for JSON numbers.
/// No floating-point/Decimal conversion is used in the comparison itself.
private struct GLM5DecimalIdentity: Equatable {
    let negative: Bool
    let digits: String
    let exponent: Int

    init?(_ data: Data) {
        guard Self.isJSONNumber(Array(data)) else { return nil }
        var text = String(decoding: data, as: UTF8.self).lowercased()
        negative = text.hasPrefix("-")
        if negative { text.removeFirst() }
        let parts = text.split(separator: "e", omittingEmptySubsequences: false)
        guard parts.count <= 2 else { return nil }
        let power: Int
        if parts.count == 2 {
            guard let parsed = Int(parts[1]) else { return nil }
            power = parsed
        } else {
            power = 0
        }
        let decimal = parts[0].split(separator: ".", omittingEmptySubsequences: false)
        guard decimal.count <= 2 else { return nil }
        var coefficient = decimal.joined()
        let fractionalDigits = decimal.count == 2 ? decimal[1].count : 0
        let (initialPower, overflow) = power.subtractingReportingOverflow(fractionalDigits)
        guard !overflow else { return nil }
        var adjustedPower = initialPower
        while coefficient.first == "0" { coefficient.removeFirst() }
        if coefficient.isEmpty {
            digits = "0"
            exponent = 0
            return
        }
        while coefficient.last == "0" {
            coefficient.removeLast()
            let (nextPower, overflow) = adjustedPower.addingReportingOverflow(1)
            guard !overflow else { return nil }
            adjustedPower = nextPower
        }
        digits = coefficient
        exponent = adjustedPower
    }

    private static func isJSONNumber(_ bytes: [UInt8]) -> Bool {
        var index = 0
        if bytes.first == 45 { index += 1 }
        guard index < bytes.count else { return false }
        if bytes[index] == 48 {
            index += 1
        } else {
            guard (49 ... 57).contains(bytes[index]) else { return false }
            while index < bytes.count, (48 ... 57).contains(bytes[index]) { index += 1 }
        }
        if index < bytes.count, bytes[index] == 46 {
            index += 1
            let start = index
            while index < bytes.count, (48 ... 57).contains(bytes[index]) { index += 1 }
            guard index > start else { return false }
        }
        if index < bytes.count, bytes[index] == 69 || bytes[index] == 101 {
            index += 1
            if index < bytes.count, bytes[index] == 43 || bytes[index] == 45 { index += 1 }
            let start = index
            while index < bytes.count, (48 ... 57).contains(bytes[index]) { index += 1 }
            guard index > start else { return false }
        }
        return index == bytes.count
    }
}
