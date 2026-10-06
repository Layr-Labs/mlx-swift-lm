// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for Qwen 3.5's two observed tool-call payload dialects.
///
/// Qwen 3.5's chat template requests the XML function dialect, but the model can
/// occasionally emit the Qwen/Hermes JSON dialect used by earlier Qwen releases.
/// Both payloads are framed by `<tool_call>` tags. Dialect selection is structural
/// and deterministic: XML must begin with `<function=`, while JSON must begin with
/// `{`. Malformed or ambiguous payloads are never repaired into executable calls.
///
/// Ported from upstream 3260260.
public struct Qwen35ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let payload = payloadBody(in: content) else { return nil }

        let call: ToolCall?
        if payload.hasPrefix("<function=") {
            call = parseXMLPayload(payload, tools: tools)
        } else if payload.hasPrefix("{") {
            // The outer wrapper is already resolved structurally. Do not let
            // a second parser search for literal wrapper strings in arguments.
            call = payload.data(using: .utf8).flatMap {
                try? JSONDecoder().decode(ToolCall.Function.self, from: $0)
            }.map { ToolCall(function: $0) }
        } else {
            return nil
        }

        guard let call, isDeclaredTool(call.function.name, tools: tools) else {
            return nil
        }
        return call
    }

    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        // Direct parser consumers may hand us several complete frames even
        // though ToolCallProcessor usually drains those before EOS. Walk outer
        // boundaries; never split on wrapper spellings inside argument values.
        guard let startTag, let endTag else { return [] }
        var remaining = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard remaining.hasPrefix(startTag) else {
            return parse(content: remaining, tools: tools).map { [$0] } ?? []
        }
        var calls: [ToolCall] = []
        while remaining.hasPrefix(startTag) {
            guard let end = Qwen35ToolFrameScanner.endRange(in: remaining, startTag: startTag, endTag: endTag) else {
                // Preserve complete-payload EOS handling when an upstream stop
                // intercepted the outer end tag. Malformed tails yield no call.
                if let call = parse(content: remaining, tools: tools) { calls.append(call) }
                break
            }
            if let call = parse(content: String(remaining[..<end.upperBound]), tools: tools) {
                calls.append(call)
            }
            remaining = String(remaining[end.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return calls
    }

    /// Returns the body used only to select a parser. The concrete parser remains
    /// responsible for validating the entire payload.
    private func payloadBody(in content: String) -> String? {
        var payload = content.trimmingCharacters(in: .whitespacesAndNewlines)

        if let startTag, let endTag, payload.hasPrefix(startTag) {
            if let end = Qwen35ToolFrameScanner.endRange(in: payload, startTag: startTag, endTag: endTag) {
                guard payload[end.upperBound...].allSatisfy(\.isWhitespace) else { return nil }
                payload = String(payload[..<end.lowerBound])
            }
            payload.removeFirst(startTag.count)
            payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let endTag, payload.hasSuffix(endTag) {
            // Preserve the historical direct-parser body+closing-tag form.
            payload.removeLast(endTag.count)
            payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return payload
    }

    /// Validate the whole XML body before delegating value conversion to the
    /// existing parser. This prevents a recognizable prefix plus unrelated or
    /// mixed-dialect text from becoming an executable call.
    private func parseXMLPayload(_ payload: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var remainder = payload[...]
        guard let function = consumeNamedOpeningTag("<function=", from: &remainder) else { return nil }
        var arguments: [String: any Sendable] = [:]

        while true {
            remainder = remainder.drop(while: { $0.isWhitespace })
            if remainder.hasPrefix("</function>") {
                remainder = remainder.dropFirst("</function>".count)
                guard remainder.allSatisfy(\.isWhitespace) else { return nil }
                return ToolCall(function: .init(name: function, arguments: arguments))
            }

            guard let parameter = consumeNamedOpeningTag("<parameter=", from: &remainder),
                let parameterEnd = remainder.range(of: "</parameter>")
            else { return nil }
            var value = String(remainder[..<parameterEnd.lowerBound])
            // Preserve the existing XML value contract: strip one framing LF
            // at either edge, and use the schema-aware converter unchanged.
            if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\n") { value.removeLast() }
            arguments[parameter] = convertParameterValue(
                value, paramName: parameter, funcName: function, tools: tools)
            remainder = remainder[parameterEnd.upperBound...]
        }
    }

    private func consumeNamedOpeningTag(
        _ prefix: String, from text: inout Substring
    ) -> String? {
        guard text.hasPrefix(prefix) else { return nil }
        text = text.dropFirst(prefix.count)
        guard let closingAngle = text.unicodeScalars.firstIndex(where: { $0.value == 62 }) else { return nil }

        let name = text[..<closingAngle]
        guard !name.isEmpty, !name.contains(where: \.isWhitespace) else { return nil }
        text = text[text.unicodeScalars.index(after: closingAngle)...]
        return String(name)
    }
}
