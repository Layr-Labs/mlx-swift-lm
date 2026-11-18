// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// MiMo V2 frames, read as `MiMoV2ToolCallParser` reads them:
/// `<function=NAME>` and `<parameter=KEY>RAW</parameter>` members, framing
/// whitespace (space, tab, LF, CR) between tags and nothing else. A value
/// the parser keeps as its raw text (a `string` parameter, one without a
/// declared type, or one the declarations do not list) streams as generated,
/// byte for byte, up to the first `</parameter>`, which ends it in the parser
/// too. Any other value waits for its closing tag and leaves converted by the
/// parser's own reading (`MiMoV2ToolCallParser.value`), so the complete parse
/// never disagrees with what left.
struct MiMoV2ArgumentStreamer: ToolCallArgumentStreamer {
    private enum Phase {
        case functionTag
        case betweenParameters
        case parameterTag
        case rawValue
        case bufferedValue(String)
        case finished
    }

    private static let functionOpen = Array("<function=".unicodeScalars)
    private static let functionClose = Array("</function>".unicodeScalars)
    private static let parameterOpen = Array("<parameter=".unicodeScalars)
    private static let parameterClose = Array("</parameter>".unicodeScalars)
    private static let tagLimit = 256

    private let tools: [[String: any Sendable]]?
    private let grammarOnly: Bool
    private(set) var name: String?
    private(set) var abandoned = false
    private var phase = Phase.functionTag
    private var token = [Unicode.Scalar]()
    private var exposed = ExposedArgumentObject()

    init(tools: [[String: any Sendable]]?, grammarOnly: Bool = GrammarOnlyToolParsing.isEnabled) {
        self.tools = tools
        self.grammarOnly = grammarOnly
    }

    mutating func append(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars { out += consume(scalar) }
        return name == nil ? "" : out
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        exposed.remainder(completing: arguments)
    }

    private mutating func consume(_ scalar: Unicode.Scalar) -> String {
        switch phase {
        case .functionTag:
            if token.isEmpty, MiMoV2ToolCallParser.isFramingWhitespace(scalar) { return "" }
            token.append(scalar)
            if token.count <= Self.functionOpen.count {
                if !Self.functionOpen.starts(with: token) { abandon() }
                return ""
            }
            guard scalar == ">" else {
                if scalar == "<" || token.count > Self.tagLimit { abandon() }
                return ""
            }
            let function = scalarString(token.dropFirst(Self.functionOpen.count).dropLast())
            guard MiMoV2ToolCallParser.validFunctionName(function),
                MiMoV2ToolCallParser.declared(function, tools: tools)
            else {
                abandon()
                return ""
            }
            name = function
            token = []
            phase = .betweenParameters
            return ""
        case .betweenParameters:
            if token.isEmpty, MiMoV2ToolCallParser.isFramingWhitespace(scalar) { return "" }
            token.append(scalar)
            if token == Self.functionClose {
                phase = .finished
            } else if token == Self.parameterOpen {
                token = []
                phase = .parameterTag
            } else if !Self.functionClose.starts(with: token), !Self.parameterOpen.starts(with: token) {
                phase = .finished
            }
            return ""
        case .parameterTag:
            guard scalar == ">" else {
                token.append(scalar)
                if scalar == "<" || token.count > Self.tagLimit { phase = .finished }
                return ""
            }
            let key = scalarString(token)
            token = []
            guard !key.isEmpty, !exposed.contains(key) else {
                phase = .finished
                return ""
            }
            guard keepsRawText(key) else {
                phase = .bufferedValue(key)
                return ""
            }
            phase = .rawValue
            return exposed.beginString(key)
        case .rawValue:
            token.append(scalar)
            if token.count >= Self.parameterClose.count,
                Array(token.suffix(Self.parameterClose.count)) == Self.parameterClose
            {
                let content = token.dropLast(Self.parameterClose.count)
                token = []
                phase = .betweenParameters
                return exposed.appendString(content) + exposed.endString()
            }
            let released = releasedPrefix(of: &token, before: [Self.parameterClose])
            return released.isEmpty ? "" : exposed.appendString(released)
        case .bufferedValue(let key):
            token.append(scalar)
            guard token.count >= Self.parameterClose.count,
                Array(token.suffix(Self.parameterClose.count)) == Self.parameterClose
            else { return "" }
            let raw = scalarString(token.dropLast(Self.parameterClose.count))
            token = []
            phase = .betweenParameters
            guard let function = name,
                let value = MiMoV2ToolCallParser.value(
                    raw, function: function, parameter: key, tools: tools, grammarOnly: grammarOnly),
                let text = exposed.member(key, value)
            else {
                phase = .finished
                return ""
            }
            return text
        case .finished:
            return ""
        }
    }

    /// The parser keeps the raw text of a parameter the declarations do not
    /// list, of one without a declared type, and of a `string` one; a union
    /// or any other type is read only once the value is complete.
    private func keepsRawText(_ key: String) -> Bool {
        guard let function = name else { return true }
        let properties = getParameterConfig(funcName: function, tools: tools)
        guard let schema = properties[key] as? [String: any Sendable] else { return true }
        if schema["anyOf"] != nil || schema["oneOf"] != nil || schema["allOf"] != nil { return false }
        guard let type = schema["type"] else { return true }
        return type as? String == "string"
    }

    private mutating func abandon() {
        if name == nil { abandoned = true }
        phase = .finished
    }
}
