import Foundation

/// Recursive Flint's arg_key/arg_value protocol. Unlike permissive GLM parsing,
/// require a complete frame, retain raw string whitespace, and never repair JSON.
public struct FlintToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = FlintToolMarkers.start
    public let endTag: String? = FlintToolMarkers.end
    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let frame = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard frame.hasPrefix(FlintToolMarkers.start), frame.hasSuffix(FlintToolMarkers.end) else {
            return nil
        }
        let body = String(
            frame.dropFirst(FlintToolMarkers.start.count).dropLast(FlintToolMarkers.end.count))
        let nameEnd = body.firstIndex(of: "<") ?? body.endIndex
        let name = String(body[..<nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty,
            name.range(of: "^[A-Za-z_][A-Za-z0-9_.-]*$", options: .regularExpression) != nil
        else { return nil }
        var remaining = body[nameEnd...]
        var arguments: [String: any Sendable] = [:]
        while !remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            remaining = remaining.drop(while: {
                $0 == " " || $0 == "\n" || $0 == "\r" || $0 == "\t"
            })
            guard remaining.hasPrefix(FlintToolMarkers.keyStart),
                let keyEnd = remaining.range(of: FlintToolMarkers.keyEnd)
            else { return nil }
            let key = String(
                remaining.dropFirst(FlintToolMarkers.keyStart.count)[..<keyEnd.lowerBound])
            guard !key.isEmpty, !key.contains("<"), arguments[key] == nil else { return nil }
            remaining = remaining[keyEnd.upperBound...]
            guard remaining.hasPrefix(FlintToolMarkers.valueStart) else { return nil }
            remaining = remaining.dropFirst(FlintToolMarkers.valueStart.count)
            // Find the argument terminator using the same JSON-aware lexer as
            // streaming. A quoted terminator inside an object is data.
            let rawString = isStringType(funcName: name, argName: key, tools: tools)
            guard let valueEnd = argumentEnd(in: remaining, rawString: rawString) else {
                return nil
            }
            let value = String(remaining[..<valueEnd.lowerBound])
            if rawString {
                arguments[key] = value
            } else if let parsed = parseJSONArgument(value) {
                // Unlike a whole JSON tool payload, each Flint argument can be
                // a scalar (including false, zero, and null).
                arguments[key] = parsed
            } else if tools == nil, !startsStructuredValue(value) {
                arguments[key] = value
            } else {
                return nil
            }
            remaining = remaining[valueEnd.upperBound...]
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parse(content: buffer, tools: tools).map { [$0] } ?? []
    }

    private func startsStructuredValue(_ value: String) -> Bool {
        let first = value.first(where: { !$0.isWhitespace })
        return first == "{" || first == "["
    }

    private func parseJSONArgument(_ value: String) -> (any Sendable)? {
        // Foundation accepts trailing commas even though Flint requires valid
        // JSON. Reject them outside strings without changing literal payloads.
        var quoted = false
        var escaped = false
        var previous: Unicode.Scalar?
        for scalar in value.unicodeScalars {
            if quoted {
                if escaped {
                    escaped = false
                } else if scalar == "\\" {
                    escaped = true
                } else if scalar == "\"" {
                    quoted = false
                }
                continue
            }
            if scalar == "\"" { quoted = true }
            if scalar.properties.isWhitespace { continue }
            if previous == ",", scalar == "}" || scalar == "]" { return nil }
            previous = scalar
        }
        guard
            let parsed = try? JSONSerialization.jsonObject(
                with: Data(value.utf8), options: .fragmentsAllowed)
        else { return nil }
        return asSendable(parsed)
    }

    private func argumentEnd(in text: Substring, rawString: Bool) -> Range<String.Index>? {
        // The first delimiter ends a declared raw string. The pinned template
        // has no escape for a literal delimiter; trailing non-frame data fails
        // parsing rather than being repaired or appended to a truncated value.
        var scanner = FlintArgumentValueScanner(rawString: rawString)
        let scalars = text.unicodeScalars
        for index in scalars.indices where scanner.consume(scalars[index]) {
            let end = scalars.index(after: index)
            return scalars.index(end, offsetBy: -FlintToolMarkers.valueEnd.count) ..< end
        }
        return nil
    }
}
