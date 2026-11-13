// Copyright © 2026 Eigen Labs.

import CoreFoundation
import Foundation

/// Reads the ATEM calls of one Onyx tool frame the way the reference servers
/// (vLLM, SGLang, llama.cpp) do. `ATEMToolCallParser` keeps upstream's
/// stricter reading of a single invoke.
///
/// - Every complete `<atem:invoke>` element is one call, in order. The
///   `<atem:function_calls>` wrapper and the text around the elements are
///   not required and not kept.
/// - A value runs to the first `</atem:parameter>`, so it can hold any other
///   markup, an `</atem:invoke>` included.
/// - A doubled name `X.X` is the declared tool `X`. Any other name that was
///   not declared is rejected, not passed on.
/// - A parameter that the schema types as a string keeps its text byte for
///   byte. Other values are decoded as JSON when they parse.
/// - An invoke that never closes is not a call.
struct ATEMReferenceScanner {
    /// One `<atem:invoke>` element before its name and values are checked.
    struct Invoke {
        var name: String?
        /// In the order written. A repeated name keeps its last value.
        var parameters: [(name: String, value: Substring)]
        var text: Substring
    }

    struct Result {
        var invokes: [Invoke] = []
        /// The text ends inside an invoke: its close, or the close of one of
        /// its values, is missing.
        var endsInsideInvoke = false
        /// The text of the invoke that was cut off.
        var unfinished: Substring?
    }

    private static let invokeOpen = "<atem:invoke"
    private static let invokeClose = "</atem:invoke>"
    private static let parameterOpen = "<atem:parameter"
    private static let parameterClose = "</atem:parameter>"

    /// - Parameter cutOff: generation ended inside this frame, so a marker
    ///   may be cut in the middle.
    static func scan(_ body: String, cutOff: Bool = false) -> Result {
        var result = Result()
        var cursor = body.startIndex
        scanning: while true {
            switch openTag(invokeOpen, in: body, from: cursor) {
            case .none:
                break scanning
            case .cutOff(let start):
                result.endsInsideInvoke = true
                result.unfinished = body[start...]
                return result
            case .tag(let tag):
                guard let element = element(in: body, after: tag.upperBound) else {
                    result.endsInsideInvoke = true
                    result.unfinished = body[tag.lowerBound...]
                    return result
                }
                result.invokes.append(
                    Invoke(
                        name: attribute("name", in: body[tag]),
                        parameters: element.parameters,
                        text: body[tag.lowerBound ..< element.end]))
                cursor = element.end
            }
        }
        // A marker cut off by the token limit, such as `<atem:inv`.
        if cutOff, let start = partialMarkerStart(invokeOpen, in: body, from: cursor) {
            result.endsInsideInvoke = true
            result.unfinished = body[start...]
        }
        return result
    }

    // MARK: - Structure

    private enum OpenTag {
        case none
        /// The whole tag, from `<` to `>`.
        case tag(Range<String.Index>)
        /// The text ends inside the tag that starts here.
        case cutOff(String.Index)
    }

    /// The next `prefix` tag. A tag name ends at whitespace or `>`, so
    /// `<atem:invokeExtension` is not `<atem:invoke`. A `<` before the closing
    /// `>` means the text only mentions the tag.
    private static func openTag(
        _ prefix: String, in body: String, from start: String.Index
    ) -> OpenTag {
        var searchStart = start
        while let candidate = body.range(of: prefix, range: searchStart ..< body.endIndex) {
            searchStart = candidate.upperBound
            guard candidate.upperBound < body.endIndex else {
                return .cutOff(candidate.lowerBound)
            }
            let next = body[candidate.upperBound]
            guard next.isWhitespace || next == ">" else { continue }
            var index = candidate.upperBound
            while index < body.endIndex, body[index] != ">", body[index] != "<" {
                index = body.index(after: index)
            }
            guard index < body.endIndex else { return .cutOff(candidate.lowerBound) }
            if body[index] == ">" {
                return .tag(candidate.lowerBound ..< body.index(after: index))
            }
        }
        return .none
    }

    /// The parameters of an invoke and the end of its close tag, or `nil`
    /// when the text ends first.
    private static func element(
        in body: String, after start: String.Index
    ) -> (parameters: [(name: String, value: Substring)], end: String.Index)? {
        var parameters: [(name: String, value: Substring)] = []
        var position = start
        var close = body.range(of: invokeClose, range: position ..< body.endIndex)
        while true {
            // A close tag inside a value that was just read is part of that value.
            if let found = close, found.lowerBound < position {
                close = body.range(of: invokeClose, range: position ..< body.endIndex)
            }
            switch openTag(parameterOpen, in: body, from: position) {
            case .tag(let tag) where close.map({ tag.lowerBound < $0.lowerBound }) ?? true:
                guard
                    let valueEnd = body.range(
                        of: parameterClose, range: tag.upperBound ..< body.endIndex)
                else { return nil }
                if let name = attribute("name", in: body[tag]), !name.isEmpty {
                    parameters.append((name, body[tag.upperBound ..< valueEnd.lowerBound]))
                }
                position = valueEnd.upperBound
            case .cutOff(let cut) where close.map({ cut < $0.lowerBound }) ?? true:
                return nil
            default:
                guard let close else { return nil }
                return (parameters, close.upperBound)
            }
        }
    }

    private static func attribute(_ name: String, in tag: Substring) -> String? {
        let needle = name + "="
        var searchStart = tag.startIndex
        while let found = tag.range(of: needle, range: searchStart ..< tag.endIndex) {
            searchStart = found.upperBound
            guard found.lowerBound > tag.startIndex,
                tag[tag.index(before: found.lowerBound)].isWhitespace,
                found.upperBound < tag.endIndex
            else { continue }
            let quote = tag[found.upperBound]
            guard quote == "\"" || quote == "'" else { return nil }
            let valueStart = tag.index(after: found.upperBound)
            guard let valueEnd = tag[valueStart...].firstIndex(of: quote) else { return nil }
            return String(tag[valueStart ..< valueEnd])
        }
        return nil
    }

    /// Where a proper prefix of `marker` starts when the text ends with one.
    private static func partialMarkerStart(
        _ marker: String, in body: String, from start: String.Index
    ) -> String.Index? {
        let tail = body[start...].suffix(marker.count - 1)
        var index = tail.startIndex
        while index < tail.endIndex {
            if marker.hasPrefix(tail[index...]) { return index }
            index = tail.index(after: index)
        }
        return nil
    }
}

// MARK: - Names and values

/// The tools of a request, as the reference servers bind a generated call to them.
struct ATEMDeclaredTools: Sendable {
    private var parameters: [String: [String: any Sendable]] = [:]

    init(_ tools: [[String: any Sendable]]?) {
        for tool in tools ?? [] {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            guard let name = function["name"] as? String, !name.isEmpty,
                parameters[name] == nil
            else { continue }
            parameters[name] = function["parameters"] as? [String: any Sendable] ?? [:]
        }
    }

    /// The declared tool a generated name means, or `nil`.
    ///
    /// The chat template lists a tool `X` that has no namespace as the
    /// recipient `"X.*"`, and the model then writes `X.X`. Only that doubled
    /// form is mapped. Matching on the last segment alone could call a tool
    /// of another namespace.
    func resolve(_ generated: String) -> String? {
        if parameters[generated] != nil { return generated }
        guard let dot = generated.firstIndex(of: ".") else { return nil }
        let head = String(generated[..<dot])
        guard generated[generated.index(after: dot)...] == head, parameters[head] != nil
        else { return nil }
        return head
    }

    /// A parameter value typed by the tool's schema.
    func value(_ raw: Substring, parameter: String, ofTool tool: String) -> JSONValue {
        let root = parameters[tool] ?? [:]
        let properties = root["properties"] as? [String: any Sendable]
        let schema = properties?[parameter] ?? root["additionalProperties"]
        var references: [String: Set<String>?] = [:]
        return ATEMReferenceValue.convert(
            String(raw),
            admitted: ATEMReferenceValue.admittedTypes(
                schema, root: root, references: &references))
    }
}

/// Value typing of the reference servers: vLLM's `convert_value` over the
/// JSON types a parameter schema admits.
enum ATEMReferenceValue {
    static let maximumDepth = 32

    /// Keeps the text when the schema admits only a string. Otherwise decodes
    /// JSON; when the schema also admits a string, the decoded value is used
    /// only if its type is one of the other admitted types.
    static func convert(_ raw: String, admitted: Set<String>?) -> JSONValue {
        if admitted == ["string"] { return .string(raw) }
        guard var decoded = decode(raw) else { return .string(raw) }
        // A schema that asks for an integer gets one from `6.0`.
        if case .double(let number) = decoded, let admitted,
            admitted.contains("integer"), !admitted.contains("number"),
            let integer = Int(exactly: number)
        {
            decoded = .int(integer)
        }
        guard let admitted, admitted.contains("string") else { return decoded }
        return admitted.subtracting(["string"]).contains(type(of: decoded))
            ? decoded : .string(raw)
    }

    /// The JSON types `schema` admits, or `nil` when it admits any. Reads
    /// `type`, `const`, `enum`, a local `$ref`, `anyOf` and `oneOf` (union)
    /// and `allOf` (intersection).
    ///
    /// The schema comes from the client. Nesting deeper than `maximumDepth`
    /// is read as untyped, so no schema can exhaust the stack.
    static func admittedTypes(
        _ schema: Any?, root: [String: any Sendable],
        references: inout [String: Set<String>?], depth: Int = 0
    ) -> Set<String>? {
        guard depth < maximumDepth, let schema = schema as? [String: any Sendable] else {
            return nil
        }
        var constraints: [Set<String>?] = []
        if let name = schema["type"] as? String {
            constraints.append([canonical(name)])
        } else if let names = schema["type"] as? [any Sendable] {
            constraints.append(Set(names.compactMap { ($0 as? String).map(canonical) }))
        }
        // OpenAPI 3.0 marks a nullable parameter beside its type.
        if let marker = schema["nullable"], JSONValue.from(marker) == .bool(true),
            let declared = constraints.last ?? nil
        {
            constraints[constraints.count - 1] = declared.union(["null"])
        }
        if let constant = schema["const"] {
            constraints.append([type(of: JSONValue.from(constant))])
        }
        if let values = schema["enum"] as? [any Sendable] {
            constraints.append(Set(values.map { type(of: JSONValue.from($0)) }))
        }
        if let reference = schema["$ref"] as? String {
            if references[reference] == nil {
                // A reference that is still being resolved admits any type.
                references[reference] = .some(nil)
                var node: Any? = reference.hasPrefix("#/") ? root : nil
                for part in reference.dropFirst(2).split(separator: "/") {
                    let key = part.replacingOccurrences(of: "~1", with: "/")
                        .replacingOccurrences(of: "~0", with: "~")
                    node = (node as? [String: any Sendable])?[key]
                }
                references[reference] = .some(
                    admittedTypes(node, root: root, references: &references, depth: depth + 1))
            }
            constraints.append(references[reference] ?? nil)
        }
        for key in ["anyOf", "oneOf"] {
            guard let branches = schema[key] as? [any Sendable], !branches.isEmpty else {
                continue
            }
            var union: Set<String>? = []
            for branch in branches {
                guard
                    let types = admittedTypes(
                        branch, root: root, references: &references, depth: depth + 1)
                else {
                    // A branch without a type admits any type.
                    union = nil
                    break
                }
                union?.formUnion(types)
            }
            if let union { constraints.append(union) }
        }
        if let branches = schema["allOf"] as? [any Sendable] {
            for branch in branches {
                constraints.append(
                    admittedTypes(branch, root: root, references: &references, depth: depth + 1))
            }
        }
        // An integer is also a number.
        let known = constraints.compactMap { $0 }.map {
            $0.contains("number") ? $0.union(["integer"]) : $0
        }
        guard var admitted = known.first else { return nil }
        for other in known.dropFirst() { admitted.formIntersection(other) }
        return admitted
    }

    /// Decodes any JSON value, a bare scalar included. A number written with
    /// a fraction or an exponent stays a floating-point number.
    static func decode(_ raw: String) -> JSONValue? {
        guard let data = raw.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
        else { return nil }
        return value(object)
    }

    private static func value(_ object: Any) -> JSONValue {
        switch object {
        case is NSNull:
            return .null
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return .bool(number.boolValue) }
            if !CFNumberIsFloatType(number), let integer = Int(exactly: number) {
                return .int(integer)
            }
            return .double(number.doubleValue)
        case let string as String:
            return .string(string)
        case let array as [Any]:
            return .array(array.map(value))
        case let dictionary as [String: Any]:
            return .object(dictionary.mapValues(value))
        default:
            return .string(String(describing: object))
        }
    }

    private static func type(of value: JSONValue) -> String {
        switch value {
        case .null: "null"
        case .bool: "boolean"
        case .int: "integer"
        case .double: "number"
        case .string: "string"
        case .array: "array"
        case .object: "object"
        }
    }

    private static func canonical(_ name: String) -> String {
        switch name.lowercased() {
        case "str", "text": "string"
        case "int": "integer"
        case "float", "double": "number"
        case "bool": "boolean"
        case "list": "array"
        case "dict": "object"
        case let lowercased: lowercased
        }
    }
}
