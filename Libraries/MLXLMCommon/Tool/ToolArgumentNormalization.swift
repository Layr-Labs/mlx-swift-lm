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

    /// When a text that spells a JSON literal (a number, `true`, `false` or
    /// `null`) becomes that literal under a schema with no single non-string
    /// reading (`not`, `if`/`then`/`else`, a union, no `type`).
    enum LiteralReading {
        /// Only when the schema admits the literal and rejects the text as a
        /// string, so the text could never be valid as written.
        case whenTextRejected
        /// Also when both are admitted and the text is the literal's own
        /// spelling, in any case (`True`, `FALSE`, `Null`) or Python's `None`:
        /// the dialects
        /// whose every value is unquoted text (the Qwen XML parameters), where a
        /// string and a literal are written alike, as vLLM's `convert_value`
        /// reads them (`null` for an optional string).
        case preferred
        /// Never: the text keeps its spelling (Gemma's marker-quoted strings).
        case never
    }

    /// `root` is the parameters schema the value's `schema` belongs to. With
    /// it, a local `$ref` reads as the schema it names (`resolved`); without
    /// it a reference is unknown and the value keeps its spelling.
    static func normalize(
        _ value: JSONValue, schema: [String: any Sendable]?, depth: Int = 0,
        root: [String: any Sendable]? = nil, literals: LiteralReading = .whenTextRejected
    ) -> JSONValue {
        switch value {
        case .string, .array, .object: break
        default: return value
        }
        let reading = reading(schema, depth: depth, root: root)
        if reading == nil, literals != .never, case .string(let text) = value, let schema,
            let literal = literal(
                text, schema: schema, depth: depth, root: root, literals: literals)
        {
            return literal
        }
        if reading == nil, literals == .preferred, case .string(let text) = value, let schema,
            let container = untypedContainer(text, schema: schema, root: root)
        {
            return container
        }
        return normalize(value, reading: reading, depth: depth, root: root)
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
        guard reading(schema, depth: 0, root: root) == nil else { return false }
        guard let schema else { return true }
        return !admitsLiteralType(schema, depth: 0, root: root)
    }

    private struct Reading {
        let schema: [String: any Sendable]
        let types: Set<String>
        let nonNull: Set<String>
    }

    /// The local references one reading may follow, those of `root`, and a
    /// budget of steps spent once the reading has followed one: each hop costs
    /// by its pointer's length, each schema node entered one step, a `type`
    /// list one step per name and per 64 bytes of names. Union branches that
    /// name the same definitions
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
        /// A step was refused: some part of the schema went unread.
        private(set) var exhausted = false

        init(root: [String: any Sendable]?) { self.root = root }

        mutating func spend(_ cost: Int = 1) -> Bool {
            guard followed else { return true }
            guard cost <= remaining else {
                remaining = 0
                exhausted = true
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
            schema["type"] as? String != "string" || rendersSiblingType(schema),
            let types = declaredTypes(schema, depth: depth, &references)
        else { return nil }
        var nonNull = types.subtracting(["null"])
        if nonNull.contains("number") { nonNull.remove("integer") }
        // An admitted string or multiple interpretations is not an invitation
        // to guess which union branch the model meant (for example "001").
        guard !types.contains("string"), nonNull.count <= 1 else { return nil }
        return Reading(schema: schema, types: types, nonNull: nonNull)
    }

    /// The JSON literal `text` spells, when `schema` admits it and the policy
    /// lets it replace the text (`LiteralReading`). A schema this reader
    /// cannot decide (an unknown reference, `$dynamicRef`, an unreadable
    /// pattern, past the reference budget) keeps the text.
    private static func literal(
        _ text: String, schema: [String: any Sendable], depth: Int,
        root: [String: any Sendable]?, literals: LiteralReading
    ) -> JSONValue? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let literal = jsonLiteral(trimmed)
                ?? (literals == .preferred ? spelledLiteral(trimmed) : nil),
            admitsLiteralType(schema, depth: depth, root: root)
        else { return nil }
        var references = References(root: root)
        guard admission(of: .value(literal), schema, depth: depth, &references) == .yes else {
            return nil
        }
        if literals == .preferred && trimmed == text { return literal }
        var textReferences = References(root: root)
        return admission(of: .value(.string(text)), schema, depth: depth, &textReferences) == .no
            ? literal : nil
    }

    /// Keywords that say nothing about a value: a schema of these alone (`{}`
    /// included) declares no type.
    private static let annotationKeywords: Set<String> = [
        "description", "title", "default", "examples", "$comment", "deprecated", "readOnly",
        "writeOnly",
    ]

    /// The JSON object or array an XML text writes under a parameter with no
    /// declared type (`{}`, annotations only, or a local reference to such a
    /// node): only a text that parses completely as one, and whose numbers an
    /// integer or a double holds exactly. Anything else keeps its text.
    private static func untypedContainer(
        _ text: String, schema: [String: any Sendable], root: [String: any Sendable]?
    ) -> JSONValue? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "{" || trimmed.first == "[" else { return nil }
        var references = References(root: root)
        guard let node = resolved(schema, &references), node["$ref"] == nil,
            node.keys.allSatisfy(annotationKeywords.contains),
            numbersAreExact(trimmed), let parsed = tryParseJSON(trimmed)
        else { return nil }
        let value = JSONValue.from(parsed)
        switch value {
        case .object, .array: return value
        default: return nil
        }
    }

    /// Whether every number outside the strings of a JSON text decodes
    /// exactly: an integer of at most 18 digits, or a decimal of at most 15
    /// significant digits.
    private static func numbersAreExact(_ text: String) -> Bool {
        var inString = false
        var escaped = false
        var token = ""
        func exact(_ token: String) -> Bool {
            guard !token.isEmpty else { return true }
            let mantissa = token.prefix { $0 != "e" && $0 != "E" }
            let digits = mantissa.filter(\.isNumber).drop { $0 == "0" }
            let integral = mantissa.count == token.count && !token.contains(".")
            return digits.count <= (integral ? 18 : 15)
        }
        for character in text {
            if inString {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    inString = false
                }
                continue
            }
            if character.isNumber || "+-.eE".contains(character) {
                token.append(character)
                continue
            }
            guard exact(token) else { return false }
            token = ""
            if character == "\"" { inString = true }
        }
        return exact(token)
    }

    /// The same scalars in any case, and Python's `None`, as vLLM reads
    /// XML text (`coerce_to_schema_type`, `_convert_param_value`).
    private static func spelledLiteral(_ text: String) -> JSONValue? {
        switch text.lowercased() {
        case "true": .bool(true)
        case "false": .bool(false)
        case "null", "none": .null
        default: nil
        }
    }

    /// A scalar spelled exactly as JSON writes it: `true`, `false`, `null`
    /// or a JSON number (no sign but `-`, no leading zeros, no bare point)
    /// that an integer or a double holds exactly. A number neither holds
    /// (`12345678901234567890`, `0.12345678901234567891`) keeps its text.
    private static func jsonLiteral(_ text: String) -> JSONValue? {
        switch text {
        case "true": return .bool(true)
        case "false": return .bool(false)
        case "null": return .null
        default: break
        }
        var scalars = text.unicodeScalars[...]
        if scalars.first == "-" { scalars.removeFirst() }
        func digits() -> Int {
            let count = scalars.prefix(while: { ("0" ... "9").contains($0) }).count
            scalars.removeFirst(count)
            return count
        }
        let leading = scalars.first
        let whole = digits()
        guard whole > 0, leading != "0" || whole == 1 else { return nil }
        if scalars.first == "." {
            scalars.removeFirst()
            guard digits() > 0 else { return nil }
        }
        if scalars.first == "e" || scalars.first == "E" {
            scalars.removeFirst()
            if scalars.first == "+" || scalars.first == "-" { scalars.removeFirst() }
            guard digits() > 0 else { return nil }
        }
        guard scalars.isEmpty else { return nil }
        let value = number(text, integer: false)
        if case .double(let double) = value, !holdsExactly(text, double) { return nil }
        return value
    }

    /// Whether `double` is the number `text` writes, digit for digit: the
    /// decimal reading of both agree, within the 38 digits a `Decimal` keeps.
    private static func holdsExactly(_ text: String, _ double: Double) -> Bool {
        let mantissa = text.prefix { $0 != "e" && $0 != "E" }
        let digits = mantissa.filter(\.isNumber).drop { $0 == "0" }
        let posix = Locale(identifier: "en_US_POSIX")
        guard digits.count <= 38, let written = Decimal(string: text, locale: posix),
            let read = Decimal(string: "\(double)", locale: posix)
        else { return false }
        return written == read
    }

    /// What one reading of a scalar against a schema can tell: admitted,
    /// rejected, admitted for some values of a type only (`maybe`, when the
    /// probe is a type), or undecided by this reader (`unknown`).
    private enum Admission { case yes, no, maybe, unknown }

    /// A concrete scalar, or any value of a JSON type.
    private enum Probe {
        case value(JSONValue)
        case type(String)

        var type: String {
            switch self {
            case .type(let type): return type
            case .value(.null): return "null"
            case .value(.bool): return "boolean"
            case .value(.int): return "integer"
            case .value(.double): return "number"
            case .value(.string): return "string"
            case .value(.array): return "array"
            case .value(.object): return "object"
            }
        }
    }

    /// Whether `literal` may read some text under `schema`, read decidedly:
    /// a number, Boolean or null the schema may admit.
    private static func admitsLiteralType(
        _ schema: [String: any Sendable], depth: Int, root: [String: any Sendable]?
    ) -> Bool {
        var references = References(root: root)
        var readings: [String: Admission] = [:]
        for type in ["integer", "number", "boolean", "null", "string"] {
            let reading = admission(of: .type(type), schema, depth: depth, &references)
            guard reading != .unknown else { return false }
            readings[type] = reading
        }
        guard !references.exhausted else { return false }
        return ["integer", "number", "boolean", "null"].contains { readings[$0] != .no }
    }

    /// Reads the keywords that decide a scalar: `type`, `nullable`, `enum`,
    /// `const`, the numeric and string bounds, `pattern`, `not`, `allOf`,
    /// `anyOf`, `oneOf`, `if`/`then`/`else` and local references. Object and
    /// array keywords hold vacuously for a scalar; other keywords annotate.
    private static func admission(
        of probe: Probe, _ node: any Sendable, depth: Int, _ references: inout References
    ) -> Admission {
        if let flag = node as? Bool { return flag ? .yes : .no }
        guard depth < 32, references.spend(), let declared = node as? [String: any Sendable],
            declared["$dynamicRef"] == nil
        else { return .unknown }
        guard let schema = resolved(declared, &references), schema["$ref"] == nil else {
            return .unknown
        }
        var parts: [Admission] = []
        let hinted = rendersSiblingType(schema)
        if hinted, probe.type == "null", !marksNullable(schema) { parts.append(.no) }
        if !hinted, let raw = schema["type"] {
            let names = (raw as? String).map { [$0] } ?? (raw as? [String]) ?? []
            let types = names.compactMap(canonicalType)
            guard !names.isEmpty, types.count == names.count else { return .unknown }
            var admitted = Set(types)
            if let marker = schema["nullable"], marker is Bool || marker is NSNumber,
                JSONValue.from(marker) == .bool(true)
            {
                admitted.insert("null")
            }
            let type = probe.type
            parts.append(
                admitted.contains(type) || (type == "integer" && admitted.contains("number"))
                    ? .yes : .no)
        }
        for key in ["enum", "const"] where schema[key] != nil {
            let options = key == "enum" ? (schema[key] as? [any Sendable]) : [schema[key]!]
            guard let options else { return .unknown }
            let values = options.map { JSONValue.from($0) }
            switch probe {
            case .value(let value): parts.append(values.contains(value) ? .yes : .no)
            case .type(let type):
                parts.append(
                    values.contains {
                        Probe.value($0).type == type
                            || (type == "number" && Probe.value($0).type == "integer")
                    }
                        ? .maybe : .no)
            }
        }
        parts.append(bounds(of: probe, schema))
        if let not = schema["not"] {
            switch admission(of: probe, not, depth: depth + 1, &references) {
            case .yes: parts.append(.no)
            case .no: parts.append(.yes)
            case let other: parts.append(other)
            }
        }
        for key in ["allOf", "anyOf", "oneOf"] where schema[key] != nil {
            guard let branches = schema[key] as? [any Sendable], !branches.isEmpty else {
                return .unknown
            }
            let readings = branches.map {
                admission(of: probe, $0, depth: depth + 1, &references)
            }
            switch key {
            case "allOf": parts.append(all(readings))
            case "anyOf": parts.append(any(readings))
            default:
                let admitted = readings.filter { $0 == .yes }.count
                if readings.contains(.unknown) {
                    parts.append(.unknown)
                } else if admitted > 1 {
                    parts.append(.no)
                } else if readings.contains(.maybe) {
                    parts.append(.maybe)
                } else {
                    parts.append(admitted == 1 ? .yes : .no)
                }
            }
        }
        if let condition = schema["if"] {
            let then = schema["then"].map {
                admission(of: probe, $0, depth: depth + 1, &references)
            }
            let otherwise = schema["else"].map {
                admission(of: probe, $0, depth: depth + 1, &references)
            }
            switch admission(of: probe, condition, depth: depth + 1, &references) {
            case .yes: parts.append(then ?? .yes)
            case .no: parts.append(otherwise ?? .yes)
            case .maybe:
                let both = [then ?? .yes, otherwise ?? .yes]
                parts.append(
                    both.contains(.unknown)
                        ? .unknown
                        : all(both) == .yes
                            ? .yes
                            : any(both) == .no ? .no : .maybe)
            case .unknown: parts.append(.unknown)
            }
        }
        return all(parts)
    }

    private static func all(_ readings: [Admission]) -> Admission {
        if readings.contains(.no) { return .no }
        if readings.contains(.unknown) { return .unknown }
        return readings.contains(.maybe) ? .maybe : .yes
    }

    private static func any(_ readings: [Admission]) -> Admission {
        if readings.contains(.yes) { return .yes }
        if readings.contains(.unknown) { return .unknown }
        return readings.contains(.maybe) ? .maybe : .no
    }

    /// The numeric bounds for a number, the length bounds and `pattern` for
    /// a string; a type probe only learns that some values may fail them.
    private static func bounds(of probe: Probe, _ schema: [String: any Sendable]) -> Admission {
        let numeric = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]
        let textual = ["minLength", "maxLength", "pattern"]
        let keys =
            ["integer", "number"].contains(probe.type)
            ? numeric
            : probe.type == "string" ? textual : []
        let present = keys.filter { schema[$0] != nil }
        guard !present.isEmpty else { return .yes }
        guard case .value(let value) = probe else { return .maybe }
        func bound(_ key: String) -> Double? {
            guard let raw = schema[key], raw is NSNumber || raw is Int || raw is Double else {
                return nil
            }
            switch JSONValue.from(raw) {
            case .int(let int): return Double(int)
            case .double(let double): return double
            default: return nil
            }
        }
        switch value {
        case .int, .double:
            let number: Double
            if case .int(let int) = value {
                number = Double(int)
            } else if case .double(let double) = value {
                number = double
            } else {
                return .unknown
            }
            let exclusiveFlags = (
                minimum: schema["exclusiveMinimum"] as? Bool == true,
                maximum: schema["exclusiveMaximum"] as? Bool == true
            )
            for key in present {
                if ["exclusiveMinimum", "exclusiveMaximum"].contains(key), schema[key] is Bool {
                    continue
                }
                guard let limit = bound(key) else { return .unknown }
                let holds: Bool
                switch key {
                case "minimum": holds = exclusiveFlags.minimum ? number > limit : number >= limit
                case "maximum": holds = exclusiveFlags.maximum ? number < limit : number <= limit
                case "exclusiveMinimum": holds = number > limit
                case "exclusiveMaximum": holds = number < limit
                default:
                    guard limit > 0 else { return .unknown }
                    let quotient = number / limit
                    holds = quotient.isFinite && quotient.rounded() == quotient
                }
                if !holds { return .no }
            }
            return .yes
        case .string(let text):
            let length = text.unicodeScalars.count
            for key in present {
                if key == "pattern" {
                    guard let pattern = schema[key] as? String,
                        let regex = try? NSRegularExpression(pattern: pattern)
                    else { return .unknown }
                    let range = NSRange(text.startIndex..., in: text)
                    if regex.firstMatch(in: text, range: range) == nil { return .no }
                } else {
                    guard let limit = bound(key) else { return .unknown }
                    if key == "minLength" ? Double(length) < limit : Double(length) > limit {
                        return .no
                    }
                }
            }
            return .yes
        default:
            return .yes
        }
    }

    private static let maximumReferenceHops = 8

    /// The keywords a reading and `normalize` look at: a hinted node keeps
    /// only these, so its other keywords are never copied. A keyword either
    /// starts to read belongs here (`qwenXMLReferenceHintsKeepWhatTheReaderReads`).
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
        let hinted = rendersSiblingType(schema)
        if !hinted, let raw = schema["type"] {
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
        if hinted, !marksNullable(schema) { result?.remove("null") }
        return result
    }

    /// Whether a node's `type` is a render hint, not the caller's assertion:
    /// the single type ingress normalization (the coordinator's and the
    /// provider's `ToolSchemaNormalization`) writes for the chat template
    /// beside a construct that decides the value itself, where that type is
    /// evidently the template's:
    /// - `anyOf`/`oneOf` whose branches include that type: the first branch's
    ///   type, as the provider's validator reads it (`ToolConstraintValidation`:
    ///   a satisfied union is not held to the sibling type);
    /// - `enum` with a member that type does not admit (`-1` beside `string`);
    /// - `not` of that same type (`{not: {type: string}, type: string}` admits
    ///   nothing as written);
    /// - `if`/`then`/`else` with a branch typed otherwise.
    /// A type that agrees with every branch (a client's `{type: string,
    /// anyOf: [{minLength: 1}, {pattern: …}]}`) keeps asserting. A null still
    /// needs `nullable: true` beside a hinted type, as the normalizer writes it.
    /// A `$ref`'s siblings are read as draft 7 reads them: a followed
    /// reference reads its target alone (`resolved`).
    static func rendersSiblingType(_ schema: [String: any Sendable]) -> Bool {
        guard let raw = schema["type"] as? String, let type = canonicalType(raw) else {
            return false
        }
        func typeName(_ node: any Sendable) -> String? {
            ((node as? [String: any Sendable])?["type"] as? String).flatMap(canonicalType)
        }
        for key in ["anyOf", "oneOf"] {
            if let branches = schema[key] as? [any Sendable],
                branches.contains(where: { typeName($0) == type })
            {
                return true
            }
        }
        if let members = schema["enum"] as? [any Sendable],
            members.contains(where: { !typeAdmits(type, JSONValue.from($0)) })
        {
            return true
        }
        if let negated = schema["not"], typeName(negated) == type { return true }
        for key in ["if", "then", "else"] {
            if let branch = schema[key], let name = typeName(branch), name != type { return true }
        }
        return false
    }

    /// Whether a JSON type name admits `value`.
    static func typeAdmits(_ type: String, _ value: JSONValue) -> Bool {
        switch value {
        case .null: type == "null"
        case .bool: type == "boolean"
        case .int: type == "integer" || type == "number"
        case .double: type == "number"
        case .string: type == "string"
        case .array: type == "array"
        case .object: type == "object"
        }
    }

    /// The `nullable: true` marker, read as `declaredTypes` reads it.
    private static func marksNullable(_ schema: [String: any Sendable]) -> Bool {
        guard let marker = schema["nullable"], marker is Bool || marker is NSNumber else {
            return false
        }
        return JSONValue.from(marker) == .bool(true)
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
