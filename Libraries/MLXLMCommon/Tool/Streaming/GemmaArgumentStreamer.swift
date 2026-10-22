// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// `call:NAME{key:value,...}` in the strict marker grammar. Marker-quoted
/// strings stream as they are generated; every other value is exposed once it
/// is complete. Any deviation stops exposure and the complete parse supplies
/// the rest.
struct GemmaArgumentStreamer: ToolCallArgumentStreamer {
    private enum Phase {
        case header
        case expectKey
        case bareKey
        case jsonKey(escaped: Bool)
        case expectColon(String)
        case expectValue(String)
        case markerProbe(String)
        case markerString(key: String, marker: [Unicode.Scalar])
        case composite(String)
        case afterValue
        case finished
    }

    private static let markers = ["<|\"|>", "<escape>"]
    private static let headerLimit = 256

    private let tools: [[String: any Sendable]]?
    private(set) var name: String?
    private(set) var abandoned = false
    private var phase = Phase.header
    private var token = [Unicode.Scalar]()
    private var lexer = GemmaValueLexer()
    private var exposed = ExposedArgumentObject()

    init(tools: [[String: any Sendable]]?) { self.tools = tools }

    mutating func append(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars {
            out += consume(scalar)
        }
        return name == nil ? "" : out
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        exposed.remainder(completing: arguments)
    }

    private mutating func consume(_ scalar: Unicode.Scalar) -> String {
        switch phase {
        case .header:
            consumeHeader(scalar)
            return ""
        case .expectKey:
            if scalar.isFramingWhitespace { return "" }
            switch scalar {
            case "}": phase = .finished
            case "\"":
                token = [scalar]
                phase = .jsonKey(escaped: false)
            case ",", "{", "[", "]", ":": phase = .finished
            default:
                token = [scalar]
                phase = .bareKey
            }
            return ""
        case .bareKey:
            switch scalar {
            case ":":
                let key = scalarString(token).trimmingCharacters(in: .whitespacesAndNewlines)
                phase = key.isEmpty ? .finished : .expectValue(key)
            case ",", "{", "}", "[", "]": phase = .finished
            default: token.append(scalar)
            }
            return ""
        case .jsonKey(let escaped):
            token.append(scalar)
            if escaped {
                phase = .jsonKey(escaped: false)
            } else if scalar == "\\" {
                phase = .jsonKey(escaped: true)
            } else if scalar == "\"" {
                let key = try? JSONDecoder().decode(
                    String.self, from: Data(scalarString(token).utf8))
                phase = key.map { .expectColon($0) } ?? .finished
            }
            return ""
        case .expectColon(let key):
            if scalar.isFramingWhitespace { return "" }
            phase = scalar == ":" ? .expectValue(key) : .finished
            return ""
        case .expectValue(let key):
            if scalar.isFramingWhitespace { return "" }
            // A repeated key keeps only its last value in the complete parse.
            guard !exposed.contains(key) else {
                phase = .finished
                return ""
            }
            if scalar == "<" {
                token = [scalar]
                phase = .markerProbe(key)
                return ""
            }
            lexer = GemmaValueLexer()
            token = []
            phase = .composite(key)
            return consume(scalar)
        case .markerProbe(let key):
            token.append(scalar)
            let probe = scalarString(token)
            if let marker = Self.markers.first(where: { $0 == probe }) {
                token = []
                phase = .markerString(key: key, marker: Array(marker.unicodeScalars))
                return exposed.beginString(key)
            }
            if !Self.markers.contains(where: { $0.hasPrefix(probe) }) { phase = .finished }
            return ""
        case .markerString(let key, let marker):
            token.append(scalar)
            if token.count >= marker.count, Array(token.suffix(marker.count)) == marker {
                let content = token.dropLast(marker.count)
                token = []
                phase = .afterValue
                return exposed.appendString(content) + exposed.endString()
            }
            let released = releasedPrefix(of: &token, before: [marker])
            phase = .markerString(key: key, marker: marker)
            return released.isEmpty ? "" : exposed.appendString(released)
        case .composite(let key):
            if lexer.isAtTopLevel, scalar == "," || scalar == "}" {
                // A scalar value ends before its separator, which then closes
                // or continues the top-level object.
                phase = .afterValue
                return exposeComposite(key) + consume(scalar)
            }
            let structural = lexer.consume(scalar)
            if lexer.failed {
                phase = .finished
                return ""
            }
            token.append(scalar)
            if lexer.depth == 0, structural == "}" || structural == "]" {
                phase = .afterValue
                return exposeComposite(key)
            }
            return ""
        case .afterValue:
            if scalar.isFramingWhitespace { return "" }
            switch scalar {
            case ",": phase = .expectKey
            default: phase = .finished
            }
            return ""
        case .finished:
            return ""
        }
    }

    private mutating func exposeComposite(_ key: String) -> String {
        let raw = scalarString(token)
        token = []
        guard let value = GemmaArgumentValueParser.parseCompleteValue(raw),
            let text = exposed.member(key, value)
        else {
            phase = .finished
            return ""
        }
        return text
    }

    private mutating func consumeHeader(_ scalar: Unicode.Scalar) {
        guard scalar == "{" else {
            token.append(scalar)
            if token.count > Self.headerLimit { abandon() }
            return
        }
        let header = scalarString(token)
        guard let callRange = header.range(of: "call:") else {
            // The complete parse takes the first brace after `call:`.
            token.append(scalar)
            return
        }
        let rawName = String(header[callRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let resolved = GemmaFunctionParser().resolveFunctionName(rawName, tools: tools) else {
            abandon()
            return
        }
        name = resolved
        token = []
        phase = .expectKey
    }

    private mutating func abandon() {
        abandoned = true
        phase = .finished
    }
}

/// Bracket depth of one Gemma value, ignoring brackets inside JSON strings and
/// marker-quoted strings. A bare `<` that does not begin a marker is outside
/// the strict grammar and fails the value.
private struct GemmaValueLexer {
    private enum Quote {
        case none
        case json(escaped: Bool)
        case native([Unicode.Scalar])
    }

    private static let markers = [Array("<|\"|>".unicodeScalars), Array("<escape>".unicodeScalars)]

    private var quote = Quote.none
    private var probe: [Unicode.Scalar] = []
    private var recent: [Unicode.Scalar] = []
    private(set) var depth = 0
    private(set) var failed = false

    /// Outside every string and bracket: a separator here ends the value.
    var isAtTopLevel: Bool {
        guard case .none = quote else { return false }
        return probe.isEmpty && depth == 0
    }

    /// Returns `scalar` when it is outside every string, else nil.
    mutating func consume(_ scalar: Unicode.Scalar) -> Unicode.Scalar? {
        switch quote {
        case .json(let escaped):
            if escaped {
                quote = .json(escaped: false)
            } else if scalar == "\\" {
                quote = .json(escaped: true)
            } else if scalar == "\"" {
                quote = .none
            }
            return nil
        case .native(let marker):
            recent.append(scalar)
            if recent.count > marker.count { recent.removeFirst(recent.count - marker.count) }
            if recent == marker {
                quote = .none
                recent = []
            }
            return nil
        case .none:
            if !probe.isEmpty || scalar == "<" {
                probe.append(scalar)
                if let marker = Self.markers.first(where: { $0 == probe }) {
                    quote = .native(marker)
                    probe = []
                } else if !Self.markers.contains(where: { $0.starts(with: probe) }) {
                    failed = true
                }
                return nil
            }
            switch scalar {
            case "\"": quote = .json(escaped: false)
            case "{", "[": depth += 1
            case "}", "]":
                depth -= 1
                if depth < 0 { failed = true }
            default: break
            }
            return scalar
        }
    }
}
