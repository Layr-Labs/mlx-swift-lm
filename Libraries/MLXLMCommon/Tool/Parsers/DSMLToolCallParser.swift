// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// Parser for DeepSeek-V4's DSML tool-call format.
///
/// Reference: `encoding_dsv4.py` (DeepSeek-V4 reference encoder) `parse_tool_calls`.
///
/// A single `tool_calls` block wraps one or more `invoke` elements, each
/// carrying zero or more `parameter` elements:
///
/// ```
/// <｜DSML｜tool_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
/// <｜DSML｜parameter name="unit" string="false">1</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜tool_calls>
/// ```
///
/// `string="true"` parameters carry a raw string value; `string="false"`
/// parameters carry a JSON-encoded value (number, boolean, array, or object).
public struct DSMLToolCallParser: ToolCallParser, Sendable {
    /// The fullwidth-pipe DSML markup token used by all DSML special tags.
    static let dsmlToken = "｜DSML｜"

    public let startTag: String? = "<\(DSMLToolCallParser.dsmlToken)tool_calls>"
    public let endTag: String? = "</\(DSMLToolCallParser.dsmlToken)tool_calls>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseMultiple(content: content, tools: tools).first
    }

    public func parseMultiple(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        let invokeOpenTag = "<\(Self.dsmlToken)invoke"
        let invokeCloseTag = "</\(Self.dsmlToken)invoke>"

        var results: [ToolCall] = []
        var searchRange = text.startIndex ..< text.endIndex

        while let invokeStart = text.range(of: invokeOpenTag, range: searchRange) {
            guard
                let headerEnd = text.range(
                    of: ">", range: invokeStart.upperBound ..< text.endIndex)
            else { break }
            guard
                let invokeEnd = text.range(
                    of: invokeCloseTag, range: headerEnd.upperBound ..< text.endIndex)
            else { break }

            let header = String(text[invokeStart.upperBound ..< headerEnd.lowerBound])
            let name = extractName(header)

            if !name.isEmpty {
                let body = String(text[headerEnd.upperBound ..< invokeEnd.lowerBound])
                let arguments = parseParameters(body)
                results.append(ToolCall(function: .init(name: name, arguments: arguments)))
            }

            searchRange = invokeEnd.upperBound ..< text.endIndex
        }

        return results
    }

    /// Extract the `name="..."` attribute value from an invoke's opening-tag
    /// header (the text between `<｜DSML｜invoke` and the following `>`).
    private func extractName(_ header: String) -> String {
        guard let nameKeyRange = header.range(of: "name=\"") else { return "" }
        let rest = header[nameKeyRange.upperBound...]
        guard let closingQuote = rest.range(of: "\"") else { return "" }
        return String(rest[rest.startIndex ..< closingQuote.lowerBound])
    }

    /// Parse all `<｜DSML｜parameter name="..." string="true|false">value</｜DSML｜parameter>`
    /// elements within an invoke body.
    private func parseParameters(_ body: String) -> [String: any Sendable] {
        let paramOpenTag = "<\(Self.dsmlToken)parameter"
        let paramCloseTag = "</\(Self.dsmlToken)parameter>"

        var arguments: [String: any Sendable] = [:]
        var searchRange = body.startIndex ..< body.endIndex

        while let paramStart = body.range(of: paramOpenTag, range: searchRange) {
            guard
                let headerEnd = body.range(
                    of: ">", range: paramStart.upperBound ..< body.endIndex)
            else { break }
            guard
                let paramEnd = body.range(
                    of: paramCloseTag, range: headerEnd.upperBound ..< body.endIndex)
            else { break }

            let header = String(body[paramStart.upperBound ..< headerEnd.lowerBound])
            let rawValue = String(body[headerEnd.upperBound ..< paramEnd.lowerBound])

            if let name = attributeValue(named: "name", in: header) {
                let isString = attributeValue(named: "string", in: header) != "false"
                arguments[name] = isString ? rawValue : parseJSONValue(rawValue)
            }

            searchRange = paramEnd.upperBound ..< body.endIndex
        }

        return arguments
    }

    /// Extract an `attr="value"` attribute value from a tag header fragment.
    private func attributeValue(named attribute: String, in header: String) -> String? {
        let marker = "\(attribute)=\""
        guard let keyRange = header.range(of: marker) else { return nil }
        let rest = header[keyRange.upperBound...]
        guard let closingQuote = rest.range(of: "\"") else { return nil }
        return String(rest[rest.startIndex ..< closingQuote.lowerBound])
    }

    /// Parse a `string="false"` parameter value as JSON (number, boolean,
    /// array, or object), matching `json.loads` in the reference Python
    /// implementation.
    ///
    /// `JSONSerialization.jsonObject(with:)` (used by `tryParseJSON`) does not
    /// accept bare top-level scalar fragments (e.g. `5`, `true`, `null`) on
    /// this platform, so those are handled explicitly before falling back to
    /// `tryParseJSON` for arrays/objects.
    private func parseJSONValue(_ raw: String) -> any Sendable {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case "true": return true
        case "false": return false
        case "null": return NSNull()
        default: break
        }
        if let intValue = Int(trimmed) {
            return intValue
        }
        if let doubleValue = Double(trimmed) {
            return doubleValue
        }
        return tryParseJSON(raw) ?? raw
    }
}
