// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    private struct ArgumentSchema {
        let names: Set<String>
        let allowsAdditional: Bool
    }

    public let startTag: String?
    public let endTag: String?
    /// The escape marker of a single-dialect parser. `nil` selects the
    /// superset parser, which accepts both Gemma tag styles and both markers.
    public let escapeMarker: String?
    public var alternateStartTags: [String] {
        strict || escapeMarker != nil ? [] : ["<start_function_call>"]
    }
    public var alternateEndTags: [String] {
        strict || escapeMarker != nil ? [] : ["<end_function_call>"]
    }
    private let strict: Bool

    private let quoteMarkers = ["<|\"|>", "<escape>"]
    private let wrapperTags = [
        "<|tool_call>",
        "<tool_call|>",
        "<start_function_call>",
        "<end_function_call>",
    ]

    /// Values are either wrapped in the escape marker or written as bare JSON,
    /// so both spans have to be opaque while the argument list is split.
    private let scanner: StructuredTextScanner

    /// Nested objects and arrays are written in the dialect's brace form, whose keys are unquoted.
    private let structuredValues = BareKeyJSONParser()

    /// The superset parser used by the fork's server. It accepts the Gemma 4
    /// `<|tool_call>` frame and the older `<start_function_call>` frame.
    /// `strict` accepts only the exact Gemma 4 template language.
    public init(strict: Bool = false) {
        self.startTag = "<|tool_call>"
        self.endTag = "<tool_call|>"
        self.escapeMarker = nil
        self.strict = strict
        self.scanner = StructuredTextScanner(quotes: ["\""])
    }

    /// A parser for one Gemma dialect with explicit tags and escape marker.
    public init(startTag: String, endTag: String, escapeMarker: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.escapeMarker = escapeMarker
        self.strict = false
        self.scanner = StructuredTextScanner(quotes: ["\""], escapeMarker: escapeMarker)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        if let escapeMarker {
            return parseSingleDialect(content: content, marker: escapeMarker, tools: tools)
        }

        // Strip tags if present. Gemma 4 emits the newer `<|tool_call>` form,
        // while older Gemma-family templates use `<start_function_call>`.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if strict {
            guard text.hasPrefix("<|tool_call>"), text.hasSuffix("<tool_call|>") else { return nil }
            text = String(text.dropFirst("<|tool_call>".count).dropLast("<tool_call|>".count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.hasPrefix("call:"), text.hasSuffix("}") else { return nil }
        } else {
            for tag in wrapperTags { text = text.replacingOccurrences(of: tag, with: "") }
        }

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name (word characters until {)
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let rawFuncName = String(remaining[..<braceStart])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let funcName = resolveFunctionName(rawFuncName, tools: tools) else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        let argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])
        guard let arguments = parseArguments(argsStr, funcName: funcName, tools: tools)
        else { return nil }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Shared with the incremental streamer so an exposed name is exactly the
    /// name this parser will later resolve for the complete frame.
    func resolveFunctionName(
        _ rawName: String,
        tools: [[String: any Sendable]]?
    ) -> String? {
        guard !rawName.isEmpty else { return nil }
        guard let tools, !tools.isEmpty else { return rawName }

        let declaredNames = tools.compactMap { tool in
            (tool["function"] as? [String: any Sendable])?["name"] as? String
        }
        guard !declaredNames.isEmpty else { return rawName }
        if declaredNames.contains(rawName) { return rawName }
        if strict { return nil }

        // Repair only a unique, nearby declared name. This covers observed
        // Gemma token glitches without turning arbitrary text into a tool call.
        let likelyMatches = declaredNames.filter { declaredName in
            let maxDistance = declaredName.count < 5 ? 0 : max(1, declaredName.count / 4)
            return (declaredName.count >= 5 && rawName.hasPrefix(declaredName + " "))
                || editDistance(rawName, declaredName) <= maxDistance
        }
        return likelyMatches.count == 1 ? likelyMatches[0] : nil
    }

    private func editDistance(_ lhs: String, _ rhs: String) -> Int {
        let left = Array(lhs)
        let right = Array(rhs)
        var previous = Array(0 ... right.count)

        for (leftIndex, leftCharacter) in left.enumerated() {
            var current = [leftIndex + 1]
            current.reserveCapacity(right.count + 1)
            for (rightIndex, rightCharacter) in right.enumerated() {
                current.append(
                    min(
                        current[rightIndex] + 1,
                        previous[rightIndex + 1] + 1,
                        previous[rightIndex] + (leftCharacter == rightCharacter ? 0 : 1)
                    ))
            }
            previous = current
        }
        return previous[right.count]
    }

    private func parseArguments(
        _ text: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: JSONValue]? {
        let schema = argumentSchema(funcName: funcName, tools: tools)
        var parser = GemmaArgumentValueParser(text, rejectDuplicates: strict)
        if let parsed = parser.parseTopLevelArguments(),
            parser.isAtEnd,
            schema.allowsAdditional || parsed.keys.allSatisfy(schema.names.contains)
        {
            return parsed
        }
        if strict { return nil }

        // Historical unconstrained auto output is intentionally accepted by
        // the bounded compatibility parser below (missing markers, comma-
        // bearing bare strings). Required/named output never needs this path:
        // its sampler emits only the strict template language above.
        var arguments: [String: JSONValue] = [:]
        // Gemma occasionally omits its string markers. The schema lets us keep
        // a comma in `Boston, MA` while still splitting before `unit:`.
        for pair in splitTopLevel(text, separator: ",") {
            guard let colon = pair.firstIndex(of: ":") else { continue }
            let key = parseArgumentKey(pair[..<colon])
            let rawValue = pair[pair.index(after: colon)...].trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            guard schema.allowsAdditional || schema.names.contains(key) else { return nil }
            arguments[key] = JSONValue.from(parseValue(rawValue))
        }
        return arguments
    }

    private func argumentSchema(
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> ArgumentSchema {
        guard
            let function = tools?.lazy.compactMap({ tool in
                tool["function"] as? [String: any Sendable]
            }).first(where: { $0["name"] as? String == funcName }),
            let parameters = function["parameters"] as? [String: any Sendable]
        else {
            return ArgumentSchema(names: [], allowsAdditional: true)
        }
        let properties = parameters["properties"] as? [String: any Sendable]
        return ArgumentSchema(
            names: Set(properties?.keys.map { $0 } ?? []),
            allowsAdditional: parameters["additionalProperties"] as? Bool != false)
    }

    public func parseEOS(_ content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if strict { return parse(content: content, tools: tools).map { [$0] } ?? [] }
        var normalized = content
        for tag in alternateStartTags {
            normalized = normalized.replacingOccurrences(of: tag, with: startTag!)
        }
        return normalized.components(separatedBy: startTag!).filter { !$0.isEmpty }
            .compactMap { parse(content: $0, tools: tools) }
    }

    private func parseArgumentKey(_ rawKey: Substring) -> String {
        let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return decodeJSONString(key) ?? key
    }

    private func parseValue(_ rawValue: String) -> any Sendable {
        for marker in quoteMarkers where rawValue.hasPrefix(marker) && rawValue.hasSuffix(marker) {
            let start = rawValue.index(rawValue.startIndex, offsetBy: marker.count)
            let end = rawValue.index(rawValue.endIndex, offsetBy: -marker.count)
            return String(rawValue[start ..< end])
        }
        if let string = decodeJSONString(rawValue) {
            return string
        }
        if let data = rawValue.data(using: .utf8),
            let json = deserializeJSON(data)
        {
            return json
        }
        return rawValue
    }

    private func decodeJSONString(_ rawValue: String) -> String? {
        guard rawValue.first == "\"", rawValue.last == "\"",
            let data = rawValue.data(using: .utf8)
        else { return nil }
        return try? JSONDecoder().decode(String.self, from: data)
    }

    private func splitTopLevel(
        _ text: String,
        separator: Character
    ) -> [String] {
        var result: [String] = []
        var start = text.startIndex
        var index = text.startIndex
        var depth = 0
        var inJSONString = false
        var isEscaped = false

        while index < text.endIndex {
            if !inJSONString,
                let marker = quoteMarkers.first(where: { text[index...].hasPrefix($0) })
            {
                let afterStart = text.index(index, offsetBy: marker.count)
                if let end = text.range(of: marker, range: afterStart ..< text.endIndex) {
                    index = end.upperBound
                    continue
                }
            }

            if inJSONString {
                if isEscaped {
                    isEscaped = false
                } else if text[index] == "\\" {
                    isEscaped = true
                } else if text[index] == "\"" {
                    inJSONString = false
                }
                index = text.index(after: index)
                continue
            }

            if text[index] == "\"" {
                inJSONString = true
                index = text.index(after: index)
                continue
            }

            switch text[index] {
            case "{", "[":
                depth += 1
            case "}", "]":
                depth = max(0, depth - 1)
            case separator where depth == 0:
                let next = text.index(after: index)
                if startsArgument(in: text, at: next) {
                    result.append(String(text[start ..< index]))
                    start = next
                }
            default:
                break
            }
            index = text.index(after: index)
        }

        result.append(String(text[start ..< text.endIndex]))
        return result
    }

    private func startsArgument(
        in text: String,
        at start: String.Index
    ) -> Bool {
        var keyStart = start
        while keyStart < text.endIndex, text[keyStart].isWhitespace {
            keyStart = text.index(after: keyStart)
        }

        var index = keyStart
        var inJSONString = false
        var isEscaped = false
        while index < text.endIndex {
            let character = text[index]
            if inJSONString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    inJSONString = false
                }
            } else if character == "\"" {
                inJSONString = true
            } else if character == ":" {
                let key = parseArgumentKey(text[keyStart ..< index])
                return !key.isEmpty
            } else if character == "," || character == "{" || character == "}" {
                return false
            }
            index = text.index(after: index)
        }
        return false
    }

    // MARK: - Single-dialect parser

    private func parseSingleDialect(
        content: String, marker: String, tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard let startTag, let endTag else { return nil }

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

        let funcName = String(remaining[..<braceStart])
        guard !funcName.isEmpty else { return nil }

        let body = remaining[remaining.index(after: braceStart) ..< braceEnd]
        var arguments: [String: any Sendable] = [:]

        // Each field is `key:value`. Splitting at top level keeps a comma that
        // belongs to an escaped string or a nested object out of the split, so a
        // value is never truncated and its remainder never becomes a stray key.
        for field in scanner.splitTopLevel(body, separator: ",") {
            guard let colon = scanner.firstTopLevelIndex(of: ":", in: field) else { continue }
            let key = Self.unmarked(field[..<colon].trimmingWhitespace(), marker: marker)
            guard !key.isEmpty else { continue }

            let rawValue = field[field.index(after: colon)...].trimmingWhitespace()
            arguments[key] = value(
                of: rawValue, key: key, funcName: funcName, marker: marker, tools: tools)
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
        guard let declaredType = getParameterType(funcName: funcName, paramName: key, tools: tools)
        else {
            return parseStructured(literal, marker: marker) ?? literal
        }

        if Self.isStructured(declaredType), let value = parseStructured(literal, marker: marker) {
            return value
        }
        return convertParameterValue(literal, paramName: key, funcName: funcName, tools: tools)
    }

    /// Strips a marker pair around a key, which the model sometimes writes.
    /// Nested keys get this from `quotingMarkedStrings`; a top-level key is read directly.
    private static func unmarked(_ key: Substring, marker: String) -> String {
        guard key.count >= 2 * marker.count, key.hasPrefix(marker), key.hasSuffix(marker)
        else { return String(key) }
        return String(key.dropFirst(marker.count).dropLast(marker.count))
    }

    /// Parses a brace-form literal, then retries with nested marker strings quoted as JSON.
    /// The original text goes first: FunctionGemma's `<escape>` can sit inside a JSON-quoted
    /// string, as in `{a:"<escape>hi<escape>"}`, which the rewrite would break.
    private func parseStructured(_ literal: String, marker: String) -> (any Sendable)? {
        if let value = structuredValues.parse(literal) { return value }
        guard literal.contains(marker) else { return nil }
        return structuredValues.parse(Self.quotingMarkedStrings(literal, marker: marker))
    }

    /// Rewrites each `marker…marker` span as a JSON string literal.
    /// An unterminated marker leaves the rest of the text unchanged.
    private static func quotingMarkedStrings(_ literal: String, marker: String) -> String {
        var out = ""
        var rest = literal[...]
        while let open = rest.range(of: marker) {
            out += rest[..<open.lowerBound]
            let afterOpen = rest[open.upperBound...]
            guard let close = afterOpen.range(of: marker) else {
                out += rest[open.lowerBound...]
                return out
            }
            out += jsonStringLiteral(afterOpen[..<close.lowerBound])
            rest = afterOpen[close.upperBound...]
        }
        out += rest
        return out
    }

    /// Quotes `text` as a JSON string, escaping backslash, double quote, and control characters.
    private static func jsonStringLiteral(_ text: Substring) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where scalar.value < 0x20:
                let hex = String(scalar.value, radix: 16, uppercase: true)
                out += (scalar.value < 0x10 ? "\\u000" : "\\u00") + hex
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    /// Whether a declared schema type is one the dialect writes in brace form.
    private static func isStructured(_ type: String) -> Bool {
        let type = type.lowercased()
        return type == "object" || type == "array" || type.hasPrefix("dict")
            || type.hasPrefix("list")
    }
}

/// Strict recursive parser for the exact value syntax produced by Gemma's
/// `format_argument(..., escape_keys=False)` macro. The older permissive
/// parser remains as a fallback only for historical unconstrained auto output.
struct GemmaArgumentValueParser {
    private let text: String
    private let rejectDuplicates: Bool
    private var index: String.Index

    init(_ text: String, rejectDuplicates: Bool = false) {
        self.text = text
        self.rejectDuplicates = rejectDuplicates
        self.index = text.startIndex
    }

    /// One complete value with nothing but whitespace around it, or nil.
    static func parseCompleteValue(_ text: String) -> JSONValue? {
        var parser = Self(text)
        guard let value = parser.parseValue(), parser.isAtEnd else { return nil }
        return value
    }

    var isAtEnd: Bool {
        var copy = self
        copy.skipWhitespace()
        return copy.index == text.endIndex
    }

    mutating func parseTopLevelArguments() -> [String: JSONValue]? {
        skipWhitespace()
        if index == text.endIndex { return [:] }
        var output: [String: JSONValue] = [:]
        while index < text.endIndex {
            guard let key = parseKey(), consume(":"),
                let value = parseValue()
            else { return nil }
            if rejectDuplicates && output[key] != nil { return nil }
            output[key] = value
            skipWhitespace()
            if index == text.endIndex { return output }
            guard consume(",") else { return nil }
        }
        return output
    }

    private mutating func parseValue() -> JSONValue? {
        skipWhitespace()
        for marker in ["<|\"|>", "<escape>"] where text[index...].hasPrefix(marker) {
            advance(marker.count)
            guard let range = text.range(of: marker, range: index ..< text.endIndex) else {
                return nil
            }
            let value = String(text[index ..< range.lowerBound])
            index = range.upperBound
            return .string(value)
        }
        if consume("{") {
            var object: [String: JSONValue] = [:]
            skipWhitespace()
            if consume("}") { return .object(object) }
            while true {
                guard let key = parseKey(), consume(":"),
                    let value = parseValue()
                else { return nil }
                if rejectDuplicates && object[key] != nil { return nil }
                object[key] = value
                if consume("}") { return .object(object) }
                guard consume(",") else { return nil }
            }
        }
        if consume("[") {
            var array: [JSONValue] = []
            skipWhitespace()
            if consume("]") { return .array(array) }
            while true {
                guard let value = parseValue() else { return nil }
                array.append(value)
                if consume("]") { return .array(array) }
                guard consume(",") else { return nil }
            }
        }
        if consume("true") { return .bool(true) }
        if consume("false") { return .bool(false) }
        if consume("null") { return .null }
        if text[index...].hasPrefix("\""), let string = parseJSONString() {
            return .string(string)
        }
        return parseNumber()
    }

    private mutating func parseKey() -> String? {
        skipWhitespace()
        if text[index...].hasPrefix("\"") { return parseJSONString() }
        let start = index
        while index < text.endIndex {
            let character = text[index]
            if character == ":" { break }
            if character == "," || character == "{" || character == "}" || character == "["
                || character == "]"
            {
                return nil
            }
            index = text.index(after: index)
        }
        let key = text[start ..< index].trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    private mutating func parseJSONString() -> String? {
        let start = index
        guard consume("\"") else { return nil }
        var escaped = false
        while index < text.endIndex {
            let character = text[index]
            index = text.index(after: index)
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                let raw = String(text[start ..< index])
                return try? JSONDecoder().decode(String.self, from: Data(raw.utf8))
            }
        }
        return nil
    }

    private mutating func parseNumber() -> JSONValue? {
        let start = index
        while index < text.endIndex {
            let character = text[index]
            guard
                character.isNumber || character == "-" || character == "+"
                    || character == "." || character == "e" || character == "E"
            else { break }
            index = text.index(after: index)
        }
        guard start != index else { return nil }
        let raw = String(text[start ..< index])
        if let integer = Int(raw) { return .int(integer) }
        if let number = Double(raw), number.isFinite { return .double(number) }
        return nil
    }

    private mutating func consume(_ literal: String) -> Bool {
        skipWhitespace()
        guard text[index...].hasPrefix(literal) else { return false }
        advance(literal.count)
        return true
    }

    private mutating func skipWhitespace() {
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
    }

    private mutating func advance(_ count: Int) {
        index = text.index(index, offsetBy: count)
    }
}
