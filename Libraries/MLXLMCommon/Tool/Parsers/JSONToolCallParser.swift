// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    private struct Payload: Decodable {
        struct FunctionPayload: Decodable {
            let id: String?
            let name: String
            let arguments: JSONValue?
        }

        let id: String?
        let name: String?
        let arguments: JSONValue?
        let function: FunctionPayload?
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        // Find the JSON content between tags
        var text = content

        // Strip tags if present
        if let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let data = jsonStr.data(using: .utf8),
            let payload = try? JSONDecoder().decode(Payload.self, from: data)
        else { return nil }

        let functionPayload = payload.function
        guard let name = functionPayload?.name ?? payload.name,
            let arguments = normalizedArguments(functionPayload?.arguments ?? payload.arguments)
        else { return nil }

        let function = ToolCall.Function(name: name, arguments: arguments)

        // If tool schemas are provided, only accept calls to declared tools.
        // Bare JSON objects that merely look like {"name":..,"arguments":..}
        // should not be misparsed as tool calls. Upstream 1335fb5.
        if let tools, !tools.isEmpty {
            var isDeclaredTool = false
            for tool in tools {
                let functionSpec = tool["function"] as? [String: any Sendable]
                if functionSpec?["name"] as? String == function.name {
                    isDeclaredTool = true
                    break
                }
            }

            guard isDeclaredTool else {
                return nil
            }
        }

        return ToolCall(function: function, id: payload.id ?? functionPayload?.id)
    }

    /// OpenAI serializes `function.arguments` as a JSON string, while the
    /// native tool dialect uses an object. Accept both wire-compatible forms.
    private func normalizedArguments(_ value: JSONValue?) -> [String: JSONValue]? {
        guard let value else { return nil }
        switch value {
        case .object(let arguments):
            return arguments
        case .string(let encoded):
            guard let data = encoded.data(using: .utf8) else { return nil }
            return try? JSONDecoder().decode([String: JSONValue].self, from: data)
        default:
            return nil
        }
    }
}
