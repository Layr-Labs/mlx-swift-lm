// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for Qwen 3.5's two observed tool-call payload dialects.
///
/// Qwen 3.5's chat template requests the XML function dialect, but the model can
/// occasionally emit the Qwen/Hermes JSON dialect used by earlier Qwen releases.
/// Both payloads are framed by `<tool_call>` tags. Dialect selection is structural
/// and deterministic: XML must begin with `<function=`, while JSON must begin with
/// `{`. Malformed or ambiguous payloads are never repaired into executable calls.
public struct Qwen35ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    private let jsonParser: JSONToolCallParser

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.jsonParser = JSONToolCallParser(startTag: startTag, endTag: endTag)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let payload = payloadBody(in: content) else { return nil }

        let call: ToolCall?
        if payload.hasPrefix("<function=") {
            call = parseCanonicalXML(payload, tools: tools)
        } else if payload.hasPrefix("{") {
            // The outer wrapper is already resolved structurally. Do not let
            // a second parser search for literal wrapper strings in arguments.
            call = jsonParser.parsePayload(payload)
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
            guard
                let end = Qwen35ToolFrameScanner.endRange(
                    in: remaining, startTag: startTag, endTag: endTag)
            else {
                // Preserve complete-payload EOS handling when an upstream stop
                // intercepted the outer end tag. Malformed tails yield no call.
                if let call = parse(content: remaining, tools: tools) { calls.append(call) }
                break
            }
            if let call = parse(content: String(remaining[..<end.upperBound]), tools: tools) {
                calls.append(call)
            }
            remaining = String(remaining[end.upperBound...]).trimmingCharacters(
                in: .whitespacesAndNewlines)
        }
        return calls
    }

    /// Returns the body used only to select a parser. The concrete parser remains
    /// responsible for validating the entire payload.
    ///
    /// The outer frame end is found structurally, so a literal end tag inside
    /// a JSON string or an XML parameter value never ends the frame. A frame
    /// whose end tag was intercepted by a stop sequence keeps its whole body.
    private func payloadBody(in content: String) -> String? {
        var payload = content.trimmingCharacters(in: .whitespacesAndNewlines)

        if let startTag, let endTag, payload.hasPrefix(startTag) {
            if let end = Qwen35ToolFrameScanner.endRange(
                in: payload, startTag: startTag, endTag: endTag)
            {
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

    /// Validate and extract a canonical XML payload with the shared structural
    /// scanner. Validation and value extraction use the same grammar, so a
    /// parameter value containing a literal `</function>` cannot truncate the
    /// parse into silently incomplete arguments, and a recognizable prefix plus
    /// unrelated or mixed-dialect text never becomes an executable call.
    private func parseCanonicalXML(
        _ payload: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        QwenXMLPayloadScanner.parseCanonical(payload[...], tools: tools)
    }
}
