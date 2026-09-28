// Copyright © 2026 Eigen Labs.
// MiMo native XML wire dialect. Template reference: XiaomiMiMo/
// MiMo-V2.6-Flash-RL@5711b268169967567844e1e560e8a3966da959b1.
import Foundation

/// Native mapping arguments are raw string spans or JSON nonstring values.
/// Neither a Qwen architecture alias nor its JSON/EOS-recovery dialect.
///
/// A literal `</parameter>` has no escape mechanism in the pinned template.
/// Its byte stream can be identical to a real delimiter followed by additional
/// structure. Such values are not generally representable losslessly: this
/// parser rejects malformed/duplicate structure, never guesses a repair, and
/// cannot infer the caller's intended value from an otherwise valid collision.
/// Declaration/schema/cardinality validation remains mandatory at the caller.
public struct MiMoV2ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let frame = Self.trimFramingWhitespace(content)
        guard let startTag, let endTag, frame.unicodeScalars.starts(with: startTag.unicodeScalars),
              let end = Qwen35ToolFrameScanner.endRange(in: frame, startTag: startTag, endTag: endTag),
              end.upperBound == frame.endIndex else { return nil }
        var body = frame[frame.unicodeScalars.index(frame.unicodeScalars.startIndex,
            offsetBy: startTag.unicodeScalars.count)..<end.lowerBound]
        body = Self.dropFramingWhitespace(body)
        guard let function = Self.opening("<function=", from: &body), Self.validFunctionName(function),
              Self.declared(function, tools: tools) else { return nil }
        var arguments: [String: JSONValue] = [:]
        while true {
            body = Self.dropFramingWhitespace(body)
            if body.unicodeScalars.starts(with: "</function>".unicodeScalars) {
                body = body[body.unicodeScalars.index(body.unicodeScalars.startIndex,
                    offsetBy: "</function>".unicodeScalars.count)...]
                guard body.unicodeScalars.allSatisfy(Self.isFramingWhitespace) else { return nil }
                return ToolCall(function: .init(name: function, arguments: arguments))
            }
            guard let parameter = Self.opening("<parameter=", from: &body),
                  arguments[parameter] == nil,
                  let closing = body.range(of: "</parameter>") else { return nil }
            let raw = String(body[..<closing.lowerBound])
            guard let value = Self.value(raw, function: function, parameter: parameter, tools: tools) else { return nil }
            arguments[parameter] = value
            body = body[closing.upperBound...]
        }
    }

    /// True EOS never authorizes inventing a missing native outer terminator.
    /// Direct callers may supply several adjacent COMPLETE frames. A malformed
    /// or incomplete suffix invalidates this supplied batch rather than being
    /// repaired, split on literal argument markers, or silently skipped.
    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard let startTag, let endTag else { return [] }
        var remaining = Self.trimFramingWhitespace(buffer)
        var result: [ToolCall] = []
        while !remaining.isEmpty {
            guard remaining.unicodeScalars.starts(with: startTag.unicodeScalars),
                  let end = Qwen35ToolFrameScanner.endRange(in: remaining, startTag: startTag, endTag: endTag),
                  let call = parse(content: String(remaining[..<end.upperBound]), tools: tools) else { return [] }
            result.append(call)
            remaining = Self.trimFramingWhitespace(String(remaining[end.upperBound...]))
        }
        return result
    }

    private static func value(_ raw: String, function: String, parameter: String,
                              tools: [[String: any Sendable]]?) -> JSONValue? {
        let properties = getParameterConfig(funcName: function, tools: tools)
        guard let schema = properties[parameter] as? [String: any Sendable] else {
            return .string(raw) // No type evidence: retain bytes; caller validates schema.
        }
        // Do not infer a type from a string-looking JSON literal or ambiguous
        // union. The native template has no discriminator for string "1" vs1.
        if schema["anyOf"] != nil || schema["oneOf"] != nil || schema["allOf"] != nil { return nil }
        guard let type = schema["type"] as? String else {
            return schema["type"] == nil ? .string(raw) : nil
        }
        if type == "string" { return .string(raw) }
        guard ["integer", "number", "boolean", "object", "array", "null"].contains(type),
              let decoded = try? JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8)) else { return nil }
        switch (type, decoded) {
        case ("integer", .int), ("number", .int), ("number", .double),
             ("boolean", .bool), ("object", .object), ("array", .array), ("null", .null):
            return decoded
        default: return nil
        }
    }

    private static func declared(_ name: String, tools: [[String: any Sendable]]?) -> Bool {
        guard let tools else { return true }
        let matches = tools.filter { ($0["function"] as? [String: any Sendable])?["name"] as? String == name }
        return matches.count == 1
    }
    private static func opening(_ prefix: String, from body: inout Substring) -> String? {
        // Native keys may begin with a combining scalar. Advancing a Character
        // count can consume that scalar together with the ASCII '=' grapheme.
        // Match and advance the exact delimiter scalars; never normalize keys.
        guard body.unicodeScalars.starts(with: prefix.unicodeScalars) else { return nil }
        body = body[body.unicodeScalars.index(body.unicodeScalars.startIndex,
            offsetBy: prefix.unicodeScalars.count)...]
        guard let angle = body.unicodeScalars.firstIndex(where: { $0.value == 62 }) else { return nil }
        let name = String(body[..<angle])
        guard !name.isEmpty, !name.contains("<") else { return nil }
        body = body[body.unicodeScalars.index(after: angle)...]
        return name
    }
    private static func validFunctionName(_ name: String) -> Bool {
        let bytes = name.utf8
        return (1...64).contains(bytes.count) && bytes.allSatisfy {
            (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45 || $0 == 95
        }
    }
    private static func isFramingWhitespace(_ value: Unicode.Scalar) -> Bool {
        [UInt32(0x20), 0x09, 0x0A, 0x0D].contains(value.value)
    }
    private static func dropFramingWhitespace(_ value: Substring) -> Substring {
        var text = value
        while let first = text.unicodeScalars.first, isFramingWhitespace(first) {
            text = text[text.unicodeScalars.index(after: text.unicodeScalars.startIndex)...]
        }
        return text
    }
    private static func trimFramingWhitespace(_ value: String) -> String {
        var text = dropFramingWhitespace(value[...])
        while let last = text.unicodeScalars.last, isFramingWhitespace(last) {
            text = text[..<text.unicodeScalars.index(before: text.unicodeScalars.endIndex)]
        }
        return String(text)
    }
}
