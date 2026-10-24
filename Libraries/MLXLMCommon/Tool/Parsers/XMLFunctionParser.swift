// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
///
/// Structure is validated with the shared `QwenXMLPayloadScanner`, the same
/// grammar used by ``Qwen35ToolCallParser`` and by cross-dialect recovery. A
/// payload becomes a call only when it is canonical in full:
///
/// - the function close is the *structural* close rather than the first textual
///   `</function>`, so a parameter value may contain that literal verbatim;
/// - every `<parameter=` must close with `</parameter>`;
/// - nothing but whitespace may follow the payload inside its frame.
///
/// A malformed payload yields `nil` instead of a partially populated call. This
/// matters most at end of stream, where a truncated frame must be rejected
/// rather than executed with silently missing arguments — structural validity
/// must never depend on whether the caller's schema declares `required`.
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    private let acceptBareFunction: Bool

    public init(startTag: String, endTag: String, acceptBareFunction: Bool = false) {
        self.startTag = startTag
        self.endTag = endTag
        self.acceptBareFunction = acceptBareFunction
    }

    public var alternateStartTags: [String] { acceptBareFunction ? ["<function="] : [] }

    public func endTags(forStartTag tag: String) -> [String] {
        if acceptBareFunction, tag == "<function=" { return ["</function>"] }
        return [endTag].compactMap { $0 }
    }

    public func parseEOS(_ buffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if !acceptBareFunction {
            return buffer.components(separatedBy: startTag ?? "<tool_call>")
                .filter { !$0.isEmpty }.compactMap { parse(content: $0, tools: tools) }
        }
        guard let regex = try? NSRegularExpression(pattern: #"<function=[\s\S]*?</function>"#)
        else { return [] }
        return regex.matches(in: buffer, range: NSRange(buffer.startIndex..., in: buffer))
            .compactMap { match in
                guard let range = Range(match.range, in: buffer) else { return nil }
                return parse(content: String(buffer[range]), tools: tools)
            }
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard
            let payload = QwenXMLPayloadScanner.framedPayload(
                in: content, startTag: startTag, endTag: endTag),
            payload.hasPrefix(QwenXMLPayloadScanner.functionOpen)
        else { return nil }

        // Nemotron's bare `<function=...>` frame has no outer wrapper to bound
        // it, so a repeated parameter name is ambiguous and refused.
        if acceptBareFunction,
            case .complete(let parsed) = QwenXMLPayloadScanner.scan(payload[...]),
            Set(parsed.parameters.map(\.name)).count != parsed.parameters.count
        {
            return nil
        }

        return QwenXMLPayloadScanner.parseCanonical(payload[...], tools: tools)
    }
}
