// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// Qwen 3.5 frames: the XML dialect `<function=NAME><parameter=K>V</parameter>`
/// streams string parameters as they are generated and exposes converted
/// values once their parameter closes; the framed JSON dialect delegates to
/// `JSONFrameArgumentStreamer`.
struct Qwen35ArgumentStreamer: ToolCallArgumentStreamer {
    private enum Phase {
        case dialect
        case json
        case functionTag
        case betweenParameters
        case parameterTag
        case stringValue(started: Bool)
        case bufferedValue(String)
        case finished
    }

    private static let functionOpen = Array("<function=".unicodeScalars)
    private static let functionClose = Array("</function>".unicodeScalars)
    private static let parameterOpen = Array("<parameter=".unicodeScalars)
    private static let parameterClose = Array("</parameter>".unicodeScalars)
    private static let framedParameterClose = Array("\n</parameter>".unicodeScalars)
    private static let tagLimit = 256

    private let tools: [[String: any Sendable]]?
    private var jsonDialect: JSONFrameArgumentStreamer
    private var xmlName: String?
    private(set) var abandoned = false
    private var phase = Phase.dialect
    private var token = [Unicode.Scalar]()
    private var lastStringScalar: Unicode.Scalar?
    private var exposed = ExposedArgumentObject()

    init(tools: [[String: any Sendable]]?) {
        self.tools = tools
        self.jsonDialect = JSONFrameArgumentStreamer(tools: tools)
    }

    var name: String? {
        if case .json = phase { return jsonDialect.name }
        return xmlName
    }

    mutating func append(_ text: String) -> String {
        if case .json = phase { return appendJSON(text) }
        var out = ""
        var scalars = text.unicodeScalars[...]
        while let scalar = scalars.first {
            scalars = scalars.dropFirst()
            out += consume(scalar)
            if case .json = phase {
                // The opening brace selected the JSON dialect.
                return appendJSON("{" + scalarString(scalars))
            }
        }
        return xmlName == nil ? "" : out
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        if case .json = phase { return jsonDialect.remainder(completing: arguments) }
        return exposed.remainder(completing: arguments)
    }

    private mutating func appendJSON(_ text: String) -> String {
        let out = jsonDialect.append(text)
        abandoned = jsonDialect.abandoned
        return out
    }

    private mutating func consume(_ scalar: Unicode.Scalar) -> String {
        switch phase {
        case .dialect:
            if scalar.isFramingWhitespace { return "" }
            switch scalar {
            case "{": phase = .json
            case "<":
                token = [scalar]
                phase = .functionTag
            default: abandon()
            }
            return ""
        case .json:
            return ""
        case .functionTag:
            token.append(scalar)
            if token.count <= Self.functionOpen.count {
                if !Self.functionOpen.starts(with: token) { abandon() }
                return ""
            }
            guard scalar == ">" else {
                if token.count > Self.tagLimit { abandon() }
                return ""
            }
            let function = scalarString(token.dropFirst(Self.functionOpen.count).dropLast())
            guard !function.isEmpty, !function.contains(where: \.isWhitespace),
                isDeclaredTool(function, tools: tools)
            else {
                abandon()
                return ""
            }
            xmlName = function
            token = []
            phase = .betweenParameters
            return ""
        case .betweenParameters:
            if token.isEmpty, scalar.isFramingWhitespace { return "" }
            token.append(scalar)
            if token == Self.functionClose {
                phase = .finished
            } else if token == Self.parameterOpen {
                token = []
                phase = .parameterTag
            } else if !Self.functionClose.starts(with: token),
                !Self.parameterOpen.starts(with: token)
            {
                phase = .finished
            }
            return ""
        case .parameterTag:
            guard scalar == ">" else {
                token.append(scalar)
                if token.count > Self.tagLimit { phase = .finished }
                return ""
            }
            let key = scalarString(token)
            token = []
            guard !key.isEmpty, !key.contains(where: \.isWhitespace), !exposed.contains(key) else {
                phase = .finished
                return ""
            }
            guard streamsAsString(key) else {
                phase = .bufferedValue(key)
                return ""
            }
            lastStringScalar = nil
            phase = .stringValue(started: false)
            return exposed.beginString(key)
        case .stringValue(let started):
            // One framing LF after the opening tag is never part of the value.
            if !started {
                phase = .stringValue(started: true)
                if scalar == "\n" { return "" }
            }
            token.append(scalar)
            if token.count >= Self.parameterClose.count,
                Array(token.suffix(Self.parameterClose.count)) == Self.parameterClose
            {
                var content = Array(token.dropLast(Self.parameterClose.count))
                // One framing LF before the closing tag is stripped too,
                // unless it is the second half of a CRLF character.
                if content.last == "\n" {
                    let previous =
                        content.count >= 2 ? content[content.count - 2] : lastStringScalar
                    if previous != "\r" { content.removeLast() }
                }
                token = []
                phase = .betweenParameters
                return exposed.appendString(content) + exposed.endString()
            }
            let released = releasedPrefix(
                of: &token, before: [Self.parameterClose, Self.framedParameterClose])
            if let last = released.last { lastStringScalar = last }
            return released.isEmpty ? "" : exposed.appendString(released)
        case .bufferedValue(let key):
            token.append(scalar)
            guard token.count >= Self.parameterClose.count,
                Array(token.suffix(Self.parameterClose.count)) == Self.parameterClose
            else { return "" }
            var value = scalarString(token.dropLast(Self.parameterClose.count))
            token = []
            if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\n") { value.removeLast() }
            phase = .betweenParameters
            guard let function = xmlName,
                let text = exposed.member(
                    key,
                    JSONValue.from(
                        convertParameterValue(
                            value, paramName: key, funcName: function, tools: tools,
                            resolvingReferences: true)))
            else {
                phase = .finished
                return ""
            }
            return text
        case .finished:
            return ""
        }
    }

    /// A parameter streams as text only when the complete parse keeps every
    /// text of it as written (`parameterKeepsText`, the parse's own reading of
    /// unions, type lists, `nullable` and local `$ref`s); any other waits for
    /// its closing tag and leaves converted, so the parse never disagrees.
    private func streamsAsString(_ key: String) -> Bool {
        guard let function = xmlName else { return true }
        return parameterKeepsText(paramName: key, funcName: function, tools: tools)
    }

    private mutating func abandon() {
        if xmlName == nil { abandoned = true }
        phase = .finished
    }
}
