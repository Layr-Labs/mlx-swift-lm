// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// LFM2 / LFM2.5 pythonic tool frames:
/// `<|tool_call_start|>[f(a='x', b=2, c=True, d=None, e=[1, 2]), g()]<|tool_call_end|>`.
///
/// The LFM2.5 chat template renders string arguments as single-quoted Python
/// literals (escaping `\`, `'`, `\n`, `\r`) and every other value with `tojson`
/// (collections) or `string` (`True`/`False`/`None`, numbers), and it places ALL
/// calls of one assistant turn in a single bracketed list. This parser accepts
/// exactly that grammar: one frame may carry several calls, and commas, quotes
/// or brackets inside values never split or truncate them.
///
/// Ported from upstream ml-explore/mlx-swift-lm `PythonicToolCallParser`
/// (fa652f9e, 7871b09b) as a separate `.lfm2` parser so Llama 3's inline
/// pythonic fallback keeps the historical `PythonicToolCallParser` behavior.
public struct LFM2ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call_start|>"
    public let endTag: String? = "<|tool_call_end|>"

    /// Python string literals use either quote character, and values may nest
    /// any bracket kind, so every scan of this dialect shares one configuration.
    private static let scanner = StructuredTextScanner(quotes: ["'", "\""])

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseFrame(content: content, tools: tools).first
    }

    /// Every call in one `[...]` frame, in order. Empty when the list is
    /// unbalanced (truncated) or any call body is malformed, so the processor
    /// treats the whole frame as a parse failure instead of exposing a subset.
    public func parseFrame(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard let bodies = callBodies(in: unwrapped(content)), !bodies.isEmpty else { return [] }
        var calls: [ToolCall] = []
        for body in bodies {
            guard let call = parseCall(body, tools: tools) else { return [] }
            calls.append(call)
        }
        return calls
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        toolCallBuffer
            .components(separatedBy: startTag!)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .flatMap { parseFrame(content: $0, tools: tools) }
    }

    /// Strips the protocol tags and surrounding whitespace, leaving the call list.
    private func unwrapped(_ content: String) -> Substring {
        var text = content[...]
        if let startTag, let startRange = text.range(of: startTag) {
            text = text[startRange.upperBound...]
        }
        if let endTag, let endRange = text.range(of: endTag) {
            text = text[..<endRange.lowerBound]
        }
        return text.trimmingWhitespace()
    }

    /// Splits a call list into one body per call. A `[` left open means the
    /// payload is truncated (nil); bodies are separated by top-level commas.
    private func callBodies(in text: Substring) -> [Substring]? {
        var list = text
        if list.first == "[" {
            guard let end = Self.scanner.endOfGroup(in: list, openedAt: list.startIndex),
                list[list.index(after: end)...].allSatisfy(\.isWhitespace)
            else { return nil }
            list = list[list.index(after: list.startIndex) ..< end]
        }
        return Self.scanner.splitTopLevel(list, separator: ",")
    }

    /// Parses one `name(arguments)` body; the argument list ends at the
    /// parenthesis balancing the one that opened it.
    private func parseCall(_ body: Substring, tools: [[String: any Sendable]]?) -> ToolCall? {
        let body = body.trimmingWhitespace()
        guard let open = Self.scanner.firstTopLevelIndex(of: "(", in: body),
            let close = Self.scanner.endOfGroup(in: body, openedAt: open),
            body.index(after: close) == body.endIndex
        else { return nil }
        let name = body[..<open]
        // OpenAI function names: [A-Za-z0-9_-]; the template emits them raw.
        guard !name.isEmpty,
            name.allSatisfy({
                ($0.isASCII && ($0.isLetter || $0.isNumber)) || $0 == "_" || $0 == "-"
            })
        else { return nil }
        let funcName = String(name)
        guard
            let arguments = parseArguments(
                body[body.index(after: open) ..< close], funcName: funcName, tools: tools)
        else { return nil }
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Keyword arguments only (`key=value`); a positional or empty field is
    /// malformed. Values are Python/JSON literals.
    private func parseArguments(
        _ argsString: Substring, funcName: String, tools: [[String: any Sendable]]?
    ) -> [String: any Sendable]? {
        var arguments: [String: any Sendable] = [:]
        for pair in Self.scanner.splitTopLevel(argsString, separator: ",") {
            guard let eq = Self.scanner.firstTopLevelIndex(of: "=", in: pair) else { return nil }
            let key = String(pair[..<eq]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty, key.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }),
                arguments[key] == nil
            else { return nil }
            let value = pair[pair.index(after: eq)...].trimmingWhitespace()
            guard let parsed = parseArgumentValue(value, key: key, funcName: funcName, tools: tools)
            else { return nil }
            arguments[key] = parsed
        }
        return unwrapArgumentWrapper(arguments, funcName: funcName, tools: tools)
    }

    /// Converts one argument without letting inferred scalar types override a
    /// declared schema. Unparseable literals are malformed (nil), never raw text.
    private func parseArgumentValue(
        _ value: Substring, key: String, funcName: String, tools: [[String: any Sendable]]?
    ) -> (any Sendable)? {
        let literal = String(value)
        guard !literal.isEmpty else { return nil }
        if value.first == "{" || value.first == "[" {
            return tryParseJSON(literal) ?? tryParsePythonLiteral(literal)
        }
        if value.first == "'" || value.first == "\"" {
            guard let string = tryParsePythonLiteral(literal) as? String else { return nil }
            return convertParameterValue(string, paramName: key, funcName: funcName, tools: tools)
        }
        return tryParsePythonLiteral(literal)
    }

    /// LFM2 sometimes wraps all arguments in one schema-shaped key, e.g.
    /// `get_weather(properties={"location": "Tokyo"})`. Unwrap only recognized
    /// wrapper names that are not declared parameters (upstream 7871b09).
    private func unwrapArgumentWrapper(
        _ arguments: [String: any Sendable], funcName: String, tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        guard arguments.count == 1,
            let (key, value) = arguments.first,
            let object = value as? [String: any Sendable],
            ["properties", "parameters", "arguments", "args", "kwargs"].contains(key.lowercased()),
            getParameterType(funcName: funcName, paramName: key, tools: tools) == nil
        else { return arguments }
        return object
    }
}
