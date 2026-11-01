// Copyright © 2026 Eigen Labs.
// GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda/chat_template.jinja
// SHA-256: 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
// The shipped EigenLabs/GLM-5.3-Flash-MLX-NVFP4-mtp template, SHA-256
// 46b2e2137e2759355b3e0e3e90cf4bd172b568437a75588af5c457cc8c095b2e, only
// writes its `.0.` attribute accesses as `[0].`; its tool-call frames are identical.

import Foundation

/// Parses complete native GLM5 frames, not GLM4's permissive recovery dialect.
/// The template emits top-level strings verbatim and other values as JSON, so
/// conversion requires an unambiguous declared parameter type. Missing type
/// evidence and string/non-string unions fail closed rather than guessing.
///
/// Output frames contain no call IDs; the response owner assigns them. Parsing
/// is not constrained decoding or complete JSON-Schema validation. The caller
/// must still validate schema assertions, selection and call cardinality.
/// Raw strings have no delimiter-escaping mechanism in the pinned template;
/// literal structural-tag collisions are not losslessly representable.
public struct GLM5ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    /// UTF-8 bound of one parsed response tool suffix.
    static let maximumBytes = 1 << 20

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard content.utf8.count <= Self.maximumBytes else { return nil }
        var reader = GLM5ToolOutputReader(content)
        guard let call = reader.call(tools: tools), reader.isAtEnd else { return nil }
        return call
    }

    /// A malformed or truncated member invalidates the supplied batch. Never
    /// return an earlier successful prefix or manufacture a missing end tag.
    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard buffer.utf8.count <= Self.maximumBytes else { return [] }
        var reader = GLM5ToolOutputReader(buffer)
        var calls: [ToolCall] = []
        while !reader.isAtEnd {
            guard let call = reader.call(tools: tools) else { return [] }
            calls.append(call)
        }
        return calls
    }
}

private struct GLM5ToolOutputReader {
    private let bytes: [UInt8]
    private var index = 0

    init(_ text: String) { bytes = Array(text.utf8) }

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
        guard Self.validFunctionName(name), let parameters = Self.parameters(name, tools: tools)
        else { return nil }

        var arguments: [String: JSONValue] = [:]
        while true {
            skipWhitespace()
            if consume("</tool_call>") {
                if let rawRequired = parameters["required"] {
                    guard let required = rawRequired as? [String],
                        required.allSatisfy({ name in
                            arguments.keys.contains { $0.utf8.elementsEqual(name.utf8) }
                        })
                    else { return nil }
                }
                return ToolCall(function: .init(name: name, arguments: arguments))
            }
            guard consume("<arg_key>"), let key = take(until: "</arg_key>"),
                !key.isEmpty, !key.utf8.contains(60), arguments[key] == nil,
                consume("<arg_value>"),
                let schema = Self.argumentSchema(key, parameters: parameters),
                let value = argument(schema: schema)
            else { return nil }
            arguments[key] = value
        }
    }

    private static func parameters(
        _ name: String, tools: [[String: any Sendable]]?
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
        // An omitted parameter schema still permits an unambiguous no-arg
        // call, but provides no evidence for converting any emitted argument.
        guard let rawParameters = matches[0]["parameters"] else { return [:] }
        guard let parameters = rawParameters as? [String: any Sendable],
            parameters["type"] == nil || parameters["type"] as? String == "object",
            !hasIndirectType(parameters)
        else { return nil }
        if let properties = parameters["properties"],
            !(properties is [String: any Sendable])
        {
            return nil
        }
        return parameters
    }

    private static func argumentSchema(
        _ key: String, parameters: [String: any Sendable]
    ) -> [String: any Sendable]? {
        if let properties = parameters["properties"] as? [String: any Sendable],
            let property = properties[key]
        {
            // Swift dictionary lookup equates canonically equivalent Unicode
            // keys; JSON property names do not. Never borrow type evidence from
            // a differently encoded key, even if the rendered glyph is equal.
            guard let declaredKey = properties.keys.first(where: { $0 == key }),
                declaredKey.utf8.elementsEqual(key.utf8)
            else { return nil }
            return property as? [String: any Sendable]
        }
        return parameters["additionalProperties"] as? [String: any Sendable]
    }

    private mutating func argument(schema: [String: any Sendable]) -> JSONValue? {
        guard !Self.hasIndirectType(schema) else { return nil }
        var types: Set<String>
        if let type = schema["type"] as? String {
            types = [type]
        } else if let values = schema["type"] as? [String], !values.isEmpty {
            types = Set(values)
        } else {
            return nil
        }
        if let nullable = schema["nullable"] {
            guard case .bool(let enabled) = JSONValue.from(nullable) else { return nil }
            if enabled { types.insert("null") }
        }
        guard
            types.isSubset(of: [
                "string", "integer", "number", "boolean", "null", "object", "array",
            ])
        else { return nil }
        if types.contains("string") {
            // Even `null` is the literal string "null" for a string parameter.
            guard types.count == 1, let raw = take(until: "</arg_value>"),
                !["<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>"]
                    .contains(where: raw.contains)
            else { return nil }
            return .string(raw)
        }
        guard let value = jsonValue(depth: 0), Self.matches(value, types: types) else { return nil }
        skipWhitespace()
        guard consume("</arg_value>") else { return nil }
        return value
    }

    private static func hasIndirectType(_ schema: [String: any Sendable]) -> Bool {
        ["$ref", "$dynamicRef", "anyOf", "oneOf", "allOf", "not", "if", "then", "else"]
            .contains { schema[$0] != nil }
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
