// Copyright © 2026 Apple Inc.

import Foundation

/// Schema-guided reading of textual arguments, shared by all call dialects.
/// This is intentionally separate from validation: unconvertible or ambiguous
/// values keep their original spelling so the caller can reject or inspect them.
enum ToolArgumentNormalization {
    static func normalize(_ call: ToolCall, tools: [[String: any Sendable]]?) -> ToolCall {
        guard !call.function.arguments.isEmpty, let tools,
            let schema = ToolSchemaValidator.parametersSchema(
                ofToolNamed: call.function.name, in: tools),
            let properties = schema["properties"] as? [String: any Sendable]
        else { return call }
        var arguments = call.function.arguments
        var changed = false
        for (key, value) in call.function.arguments {
            let converted = normalize(value, schema: properties[key] as? [String: any Sendable])
            if converted != value {
                arguments[key] = converted
                changed = true
            }
        }
        guard changed else { return call }
        return ToolCall(
            function: .init(name: call.function.name, arguments: arguments), id: call.id)
    }

    /// `root` is the parameters schema the value's `schema` belongs to. With
    /// it, a local `$ref` reads as the schema it names (`resolved`); without
    /// it a reference is unknown and the value keeps its spelling.
    static func normalize(
        _ value: JSONValue, schema: [String: any Sendable]?, depth: Int = 0,
        root: [String: any Sendable]? = nil
    ) -> JSONValue {
        switch value {
        case .string, .array, .object: break
        default: return value
        }
        return normalize(
            value, reading: reading(schema, depth: depth, root: root), depth: depth, root: root)
    }

    /// `normalize` under a reading already taken: an array's items, and an
    /// object's members outside `properties`, share one reading.
    private static func normalize(
        _ value: JSONValue, reading: Reading?, depth: Int, root: [String: any Sendable]?
    ) -> JSONValue {
        switch value {
        case .string, .array, .object: break
        default: return value
        }
        guard let reading else { return value }
        let (schema, types, nonNull) = (reading.schema, reading.types, reading.nonNull)
        var result = value
        if case .string(let text) = value {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if types.contains("null"), ["null", "none", "nil"].contains(trimmed.lowercased()) {
                return .null
            }
            switch nonNull.first {
            case "integer", "number":
                result = number(trimmed, integer: nonNull.first == "integer") ?? value
            case "boolean":
                switch trimmed.lowercased() {
                case "true", "1", "yes", "on": result = .bool(true)
                case "false", "0", "no", "off": result = .bool(false)
                default: break
                }
            case "object", "array":
                let parsed: (any Sendable)? =
                    tryParseJSON(trimmed) ?? tryParsePythonLiteral(trimmed)
                if let parsed {
                    let json = JSONValue.from(parsed)
                    switch (nonNull.first, json) {
                    case ("object", .object), ("array", .array): result = json
                    default: break
                    }
                }
            default: break
            }
        }
        switch result {
        case .object(let object):
            let properties = schema["properties"] as? [String: any Sendable] ?? [:]
            let extra = schema["additionalProperties"] as? [String: any Sendable]
            guard !properties.isEmpty || extra != nil else { return result }
            var normalized = object
            var extraReading: Reading??
            for (key, value) in object {
                let converted: JSONValue
                if let property = properties[key] as? [String: any Sendable] {
                    converted = normalize(value, schema: property, depth: depth + 1, root: root)
                } else {
                    let shared =
                        extraReading ?? Self.reading(extra, depth: depth + 1, root: root)
                    extraReading = shared
                    converted = normalize(value, reading: shared, depth: depth + 1, root: root)
                }
                if converted != value { normalized[key] = converted }
            }
            return .object(normalized)
        case .array(let array):
            guard let items = schema["items"] as? [String: any Sendable] else { return result }
            let itemReading = Self.reading(items, depth: depth + 1, root: root)
            var normalized = array
            for index in array.indices {
                let converted = normalize(
                    array[index], reading: itemReading, depth: depth + 1, root: root)
                if converted != array[index] { normalized[index] = converted }
            }
            return .array(normalized)
        default: return result
        }
    }

    /// Whether `normalize` keeps every text written for `schema` as it is: a
    /// value it would never read as another JSON type, so a streamed call can
    /// expose it while it is generated and the complete parse still agrees.
    static func keepsText(
        schema: [String: any Sendable]?, root: [String: any Sendable]? = nil
    ) -> Bool {
        reading(schema, depth: 0, root: root) == nil
    }

    private struct Reading {
        let schema: [String: any Sendable]
        let types: Set<String>
        let nonNull: Set<String>
    }

    /// The local references one reading may follow, those of `root`, and a
    /// budget of steps spent once the reading has followed one: each hop costs
    /// by its pointer's length, each schema node entered one step, a `type`
    /// list one step per name. Union branches that name the same definitions
    /// again (a diamond, a cycle, or many references to a wide or costly node)
    /// would otherwise cost time exponential in their depth, or the node's
    /// size times its references; past the budget the reading is unknown, so
    /// the value keeps its text, as for a reference that names nothing (a
    /// sibling `type` hint is not read then). Before any hop a reading reads
    /// only the schema as written, which bounds it, so it is not budgeted.
    private struct References {
        let root: [String: any Sendable]?
        private var remaining = 256
        private var followed = false

        init(root: [String: any Sendable]?) { self.root = root }

        mutating func spend(_ cost: Int = 1) -> Bool {
            guard followed else { return true }
            guard cost <= remaining else {
                remaining = 0
                return false
            }
            remaining -= cost
            return true
        }

        mutating func follow(_ pointer: String) -> Bool {
            followed = true
            return spend(1 + pointer.utf8.count / 64)
        }
    }

    /// The one non-string reading `normalize` gives a value under `schema`, or
    /// nil when every value keeps its spelling: no schema, a string schema,
    /// an unknown one, or one that admits a string or several types.
    private static func reading(
        _ schema: [String: any Sendable]?, depth: Int, root: [String: any Sendable]?
    ) -> Reading? {
        var references = References(root: root)
        guard depth < 32, let declared = schema, let schema = resolved(declared, &references),
            schema["type"] as? String != "string",
            let types = declaredTypes(schema, depth: depth, &references)
        else { return nil }
        var nonNull = types.subtracting(["null"])
        if nonNull.contains("number") { nonNull.remove("integer") }
        // An admitted string or multiple interpretations is not an invitation
        // to guess which union branch the model meant (for example "001").
        guard !types.contains("string"), nonNull.count <= 1 else { return nil }
        return Reading(schema: schema, types: types, nonNull: nonNull)
    }

    private static let maximumReferenceHops = 8

    /// The keywords a reading and `normalize` look at: a hinted node keeps
    /// only these, so its other keywords are never copied.
    private static let readKeywords = [
        "type", "nullable", "anyOf", "oneOf", "allOf", "not", "if", "then", "else", "$dynamicRef",
        "properties", "additionalProperties", "items",
    ]

    /// `schema` with a local `$ref` read for conversion: the node it names in
    /// the references' root (a JSON pointer such as `#/$defs/Point` or
    /// `#/definitions/Point`, followed through a few chained references), else
    /// the node's `readKeywords` without its reference when a sibling `type`
    /// says what it holds (the coordinator writes its target's type beside a
    /// reference it could not inline), else nil. Without a root the schema is
    /// returned as written, so a reference stays unknown (`declaredTypes`);
    /// past the budget it is unknown too.
    private static func resolved(
        _ schema: [String: any Sendable], _ references: inout References
    ) -> [String: any Sendable]? {
        guard let root = references.root, schema["$ref"] != nil else { return schema }
        var node = schema
        var hops = 0
        while let reference = node["$ref"] {
            let pointer = reference as? String
            if let pointer, !references.follow(pointer) { return nil }
            guard hops < maximumReferenceHops, let pointer,
                let target = localTarget(pointer, in: root)
            else {
                guard schema["type"] != nil else { return nil }
                return schema.filter(keys: readKeywords)
            }
            node = target
            hops += 1
        }
        return node
    }

    /// The schema object a same-document JSON pointer (RFC 6901) names, read as
    /// the coordinator's `$ref` resolver reads it: at most 1,024 bytes and 32
    /// tokens, each token percent-decoded (a malformed escape names nothing),
    /// then `~1` and `~0` unescaped, and arrays walked by index
    /// (`#/$defs/U/anyOf/0`).
    private static func localTarget(
        _ pointer: String, in root: [String: any Sendable]
    ) -> [String: any Sendable]? {
        guard pointer.hasPrefix("#"), pointer.utf8.count <= 1_024 else { return nil }
        let path = pointer.dropFirst()
        guard path.isEmpty || path.hasPrefix("/") else { return nil }
        let tokens = path.split(separator: "/", omittingEmptySubsequences: false).dropFirst()
        guard tokens.count <= 32 else { return nil }
        var node: any Sendable = root
        for raw in tokens {
            guard let decoded = String(raw).removingPercentEncoding else { return nil }
            let token = decoded.replacingOccurrences(of: "~1", with: "/")
                .replacingOccurrences(of: "~0", with: "~")
            if let object = node as? [String: any Sendable], let next = object[token] {
                node = next
            } else if let array = node as? [any Sendable], let index = Int(token),
                String(index) == token, array.indices.contains(index)
            {
                node = array[index]
            } else {
                return nil
            }
        }
        return node as? [String: any Sendable]
    }

    /// Reads an unquoted scalar the model wrote, such as Gemma's `zip:01000`,
    /// as vLLM's `convert_value` does: its text when the schema admits a
    /// string but not the type of its JSON `reading`, else the reading. An
    /// unknown or ambiguous schema keeps the reading.
    static func bareLiteral(
        _ text: String, reading: JSONValue, schema: [String: any Sendable]?
    ) -> JSONValue {
        guard let schema, let types = declaredTypes(schema, depth: 0), types.contains("string")
        else { return reading }
        let type: String
        switch reading {
        case .null: type = "null"
        case .bool: type = "boolean"
        case .int: type = "integer"
        case .double: type = "number"
        default: return reading
        }
        return types.contains(type) ? reading : .string(text)
    }

    private static func declaredTypes(_ schema: [String: any Sendable], depth: Int) -> Set<String>?
    {
        var references = References(root: nil)
        return declaredTypes(schema, depth: depth, &references)
    }

    /// Returns only interpretations every branch describes. Unknown branches
    /// (notably references) prevent coercion rather than being silently ignored.
    private static func declaredTypes(
        _ schema: [String: any Sendable], depth: Int, _ references: inout References
    ) -> Set<String>? {
        guard depth < 32, references.spend(),
            ["$ref", "$dynamicRef", "not", "if", "then", "else"].allSatisfy({ schema[$0] == nil })
        else { return nil }
        var result: Set<String>?
        if let raw = schema["type"] {
            let names = (raw as? String).map { [$0] } ?? (raw as? [String]) ?? []
            let bytes = names.reduce(0) { $0 + $1.utf8.count }
            guard references.spend(names.count + bytes / 64) else { return nil }
            let mapped = names.compactMap(canonicalType)
            guard !names.isEmpty, mapped.count == names.count else { return nil }
            result = Set(mapped)
            // Integer is a subtype of number, so intersections such as
            // allOf(number, integer) still describe an integer.
            if result?.contains("number") == true { result?.insert("integer") }
            // OpenAPI 3.0 `nullable: true` adds null to this node's own type.
            // Only a real Boolean counts: a boxed 1 is not `true`, and a value
            // that is no scalar is not converted to find out.
            if let marker = schema["nullable"], marker is Bool || marker is NSNumber,
                JSONValue.from(marker) == .bool(true)
            {
                result?.insert("null")
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] where schema[key] != nil {
            guard let branches = schema[key] as? [[String: any Sendable]], !branches.isEmpty else {
                return nil
            }
            var types: [Set<String>] = []
            for branch in branches {
                guard let target = resolved(branch, &references),
                    let branchTypes = declaredTypes(target, depth: depth + 1, &references)
                else { return nil }
                types.append(branchTypes)
            }
            let combined = types.dropFirst().reduce(types[0]) {
                key == "allOf" ? $0.intersection($1) : $0.union($1)
            }
            result = result.map { $0.intersection(combined) } ?? combined
        }
        return result
    }

    private static func canonicalType(_ name: String) -> String? {
        let type = name.lowercased()
        switch type {
        case "string", "str", "text", "varchar", "char", "enum": return "string"
        case "integer", "int", "uint", "long", "short", "unsigned": return "integer"
        case "number", "float": return "number"
        case "boolean", "bool", "binary": return "boolean"
        case "object", "dict": return "object"
        case "array", "arr", "list": return "array"
        case "null": return "null"
        default:
            // Preserve the aliases already accepted by the textual parsers.
            if ["int", "uint", "long", "short", "unsigned"].contains(where: type.hasPrefix) {
                return "integer"
            }
            if type.hasPrefix("num") || type.hasPrefix("float") { return "number" }
            if type.hasPrefix("dict") { return "object" }
            if type.hasPrefix("list") { return "array" }
            return nil
        }
    }

    private static func number(_ text: String, integer: Bool) -> JSONValue? {
        if let value = Int(text) { return .int(value) }
        guard let value = Double(text), value.isFinite else { return nil }
        return decimalInteger(text).map(JSONValue.int) ?? (integer ? nil : .double(value))
    }

    /// Recognize integral decimal/exponent spellings without floating-point
    /// rounding. No fixed-width conversion occurs until all digits are known.
    private static func decimalInteger(_ text: String) -> Int? {
        var body = text[...]
        let negative = body.first == "-"
        if body.first == "-" || body.first == "+" { body.removeFirst() }
        let parts = body.split(
            omittingEmptySubsequences: false, whereSeparator: { $0 == "e" || $0 == "E" })
        guard parts.count <= 2 else { return nil }
        let exponent: Int
        if parts.count == 2 {
            guard let parsed = Int(parts[1]) else { return nil }
            exponent = parsed
        } else {
            exponent = 0
        }
        let mantissa = parts[0].split(separator: ".", omittingEmptySubsequences: false)
        guard mantissa.count <= 2 else { return nil }
        let fractionalCount = mantissa.count == 2 ? mantissa[1].count : 0
        var digits = mantissa.joined()
        guard !digits.isEmpty, digits.utf8.allSatisfy({ (48 ... 57).contains($0) }) else {
            return nil
        }
        digits = String(digits.drop(while: { $0 == "0" }))
        if digits.isEmpty { return 0 }
        let (scale, overflow) = exponent.subtractingReportingOverflow(fractionalCount)
        guard !overflow else { return nil }
        if scale >= 0 {
            guard scale <= 19, digits.count <= 19 - scale else { return nil }
            digits += String(repeating: "0", count: scale)
        } else {
            guard scale > -digits.count, digits.suffix(-scale).allSatisfy({ $0 == "0" }) else {
                return nil
            }
            digits.removeLast(-scale)
        }
        return Int((negative ? "-" : "") + digits)
    }
}

extension Dictionary where Key == String {
    /// The entries whose keys are in `keys`, read one key at a time.
    fileprivate func filter(keys: [String]) -> [String: Value] {
        var kept: [String: Value] = [:]
        for key in keys { if let value = self[key] { kept[key] = value } }
        return kept
    }
}
