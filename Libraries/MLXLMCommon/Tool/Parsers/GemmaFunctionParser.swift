// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let escapeMarker: String?

    /// Values are either wrapped in the escape marker or written as bare JSON,
    /// so both spans have to be opaque while the argument list is split.
    private let scanner: StructuredTextScanner

    /// Nested objects and arrays are written in the dialect's brace form, whose keys are unquoted.
    private let structuredValues = BareKeyJSONParser()

    /// Legacy Gemma parser defaults retained for callers that predate the
    /// configurable Gemma/Gemma4 tag formats.
    public init() {
        self.init(
            startTag: "<start_function_call>",
            endTag: "<end_function_call>",
            escapeMarker: "<escape>")
    }

    public init(startTag: String, endTag: String, escapeMarker: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.escapeMarker = escapeMarker
        self.scanner = StructuredTextScanner(quotes: ["\""], escapeMarker: escapeMarker)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let startTag, let endTag, let marker = escapeMarker else { return nil }

        // Strip tags if present
        var text = content[...]
        if let startRange = text.range(of: startTag) {
            text = text[startRange.upperBound...]
        }
        if let endRange = text.range(of: endTag) {
            text = text[..<endRange.lowerBound]
        }

        // Pattern: call:(\w+)\{(.*)\}, where the closing brace is the one that
        // balances the opening brace rather than the first or last in the text.
        guard let callRange = text.range(of: "call:") else { return nil }
        let remaining = text[callRange.upperBound...]

        guard let braceStart = scanner.firstTopLevelIndex(of: "{", in: remaining),
            let braceEnd = scanner.endOfGroup(in: remaining, openedAt: braceStart)
        else { return nil }

        let rawFunctionName = extractName(String(remaining[..<braceStart]))
        guard let funcName = resolvedFunctionName(rawFunctionName, tools: tools) else { return nil }

        let body = remaining[remaining.index(after: braceStart) ..< braceEnd]
        var fields: [(key: String, value: String)] = []

        // A model may emit an unescaped comma inside a schema-declared string
        // (for example `location:Boston, MA`). Keep the continuation attached
        // to its preceding value rather than treating it as an empty argument.
        for field in scanner.splitTopLevel(body, separator: ",") {
            if let colon = scanner.firstTopLevelIndex(of: ":", in: field) {
                let key = extractName(String(field[..<colon]).trimmingCharacters(in: .whitespaces))
                guard !key.isEmpty else { continue }
                let value = String(field[field.index(after: colon)...])
                fields.append((key, value))
            } else if !fields.isEmpty {
                fields[fields.count - 1].value += "," + field
            }
        }

        var arguments: [String: any Sendable] = [:]

        for field in fields {
            guard permitsParameter(field.key, for: funcName, tools: tools) else { return nil }
            let rawValue = field.value[...].trimmingWhitespace()
            arguments[field.key] = value(
                of: rawValue, key: field.key, funcName: funcName, marker: marker, tools: tools)
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Decodes one field value.
    ///
    /// A marker-wrapped value is a string the model escaped precisely because it
    /// may contain protocol punctuation, so it is taken verbatim. Everything else
    /// is typed from the schema when the parameter is declared, and otherwise
    /// decoded as JSON, falling back to the literal text.
    ///
    /// A parameter the schema declares structured is read as a brace-form literal first: the
    /// dialect writes those without quoting their keys, which strict JSON refuses.
    private func value(
        of rawValue: Substring,
        key: String,
        funcName: String,
        marker: String,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        if rawValue.hasPrefix(marker) {
            let contentStart = rawValue.index(rawValue.startIndex, offsetBy: marker.count)
            let unescaped =
                rawValue[contentStart...].range(of: marker)
                .map { String(rawValue[contentStart ..< $0.lowerBound]) }
                ?? String(rawValue[contentStart...])
            return convertParameterValue(
                unescaped, paramName: key, funcName: funcName, tools: tools)
        }

        let literal = String(rawValue)
        if let quoted = decodedJSONString(literal) {
            return convertParameterValue(quoted, paramName: key, funcName: funcName, tools: tools)
        }
        // Gemma4's constrained grammar uses its quote marker inside nested
        // arrays/objects too. Convert that dialect quoting before handing a
        // structured literal to the JSON-like parser; top-level marker-wrapped
        // strings are handled by the fast path above.
        let structuredLiteral = literal.replacingOccurrences(of: marker, with: "\"")
        guard let declaredType = getParameterType(funcName: funcName, paramName: key, tools: tools)
        else {
            return structuredValues.parse(structuredLiteral) ?? literal
        }

        if Self.isStructured(declaredType), let value = structuredValues.parse(structuredLiteral) {
            return value
        }
        return convertParameterValue(literal, paramName: key, funcName: funcName, tools: tools)
    }

    /// Whether a declared schema type is one the dialect writes in brace form.
    private static func isStructured(_ type: String) -> Bool {
        let type = type.lowercased()
        return type == "object" || type == "array" || type.hasPrefix("dict")
            || type.hasPrefix("list")
    }

    /// Resolve a model-emitted name against the declared schema. Exact matches
    /// are required unless a single sufficiently descriptive declaration is an
    /// obvious recovery target; short names are never fuzzed because `run` and
    /// `sum` would otherwise authorize unrelated calls.
    private func resolvedFunctionName(_ rawName: String, tools: [[String: any Sendable]]?) -> String? {
        guard let tools, !tools.isEmpty else { return rawName.isEmpty ? nil : rawName }
        let declared = tools.compactMap { tool in
            (tool["function"] as? [String: any Sendable])?["name"] as? String
        }
        if declared.contains(rawName) { return rawName }

        let candidates = declared.filter { name in
            guard name.count >= 8 else { return false }
            if rawName.contains(name) { return true }
            let distance = editDistance(rawName, name)
            return Double(distance) / Double(max(rawName.count, name.count)) <= 0.25
        }
        return candidates.count == 1 ? candidates[0] : nil
    }

    private func permitsParameter(
        _ parameter: String, for functionName: String, tools: [[String: any Sendable]]?
    ) -> Bool {
        guard let tools, !tools.isEmpty else { return true }
        guard let function = tools.compactMap({ $0["function"] as? [String: any Sendable] }).first(where: {
            $0["name"] as? String == functionName
        }), let parameters = function["parameters"] as? [String: any Sendable]
        else { return true }

        let properties = parameters["properties"] as? [String: any Sendable] ?? [:]
        return properties[parameter] != nil || parameters["additionalProperties"] as? Bool != false
    }

    private func decodedJSONString(_ literal: String) -> String? {
        guard literal.first == "\"", literal.last == "\"", let data = literal.data(using: .utf8)
        else { return nil }
        return try? JSONDecoder().decode(String.self, from: data)
    }

    private func editDistance(_ lhs: String, _ rhs: String) -> Int {
        let lhs = Array(lhs)
        let rhs = Array(rhs)
        var previous = Array(0 ... rhs.count)
        for (i, left) in lhs.enumerated() {
            var current = [i + 1]
            current.reserveCapacity(rhs.count + 1)
            for (j, right) in rhs.enumerated() {
                current.append(min(
                    previous[j + 1] + 1,
                    current[j] + 1,
                    previous[j] + (left == right ? 0 : 1)))
            }
            previous = current
        }
        return previous[rhs.count]
    }
}
