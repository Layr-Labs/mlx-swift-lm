// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// `{"name": "...", "arguments": {...}}` (Hermes JSON). Once the name is known
/// the raw arguments object streams verbatim, so the exposed text is exactly
/// what the complete parse decodes.
struct JSONFrameArgumentStreamer: ToolCallArgumentStreamer {
    private enum Phase {
        case start
        case expectKey
        case key
        case expectColon(String)
        case expectValue(String)
        case nameValue
        case arguments
        case otherValue
        case afterValue
        case finished
    }

    private let tools: [[String: any Sendable]]?
    private(set) var name: String?
    private(set) var abandoned = false
    private var phase = Phase.start
    private var token = [Unicode.Scalar]()
    private var lexer = JSONValueLexer()
    private var arguments = [Unicode.Scalar]()
    private var argumentsComplete = false
    private var exposedCount = 0

    init(tools: [[String: any Sendable]]?) { self.tools = tools }

    mutating func append(_ text: String) -> String {
        for scalar in text.unicodeScalars { consume(scalar) }
        guard name != nil, exposedCount < arguments.count else { return "" }
        let fragment = scalarString(arguments[exposedCount...])
        exposedCount = arguments.count
        return fragment
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        if argumentsComplete {
            guard ToolArgumentJSON.decodeObject(scalarString(self.arguments)) == arguments else {
                return nil
            }
            return scalarString(self.arguments[exposedCount...])
        }
        return exposedCount == 0 ? ToolArgumentJSON.encode(.object(arguments)) : nil
    }

    private mutating func consume(_ scalar: Unicode.Scalar) {
        switch phase {
        case .start:
            if scalar.isFramingWhitespace { return }
            if scalar == "{" { phase = .expectKey } else { stop() }
        case .expectKey:
            if scalar.isFramingWhitespace { return }
            if scalar == "\"" {
                token = [scalar]
                lexer = JSONValueLexer()
                _ = lexer.consume(scalar)
                phase = .key
            } else {
                stop()
            }
        case .key:
            token.append(scalar)
            _ = lexer.consume(scalar)
            guard lexer.complete else { return }
            guard
                let key = try? JSONDecoder().decode(
                    String.self, from: Data(scalarString(token).utf8))
            else {
                stop()
                return
            }
            phase = .expectColon(key)
        case .expectColon(let key):
            if scalar.isFramingWhitespace { return }
            if scalar == ":" { phase = .expectValue(key) } else { stop() }
        case .expectValue(let key):
            if scalar.isFramingWhitespace { return }
            lexer = JSONValueLexer()
            switch key {
            case "name" where name == nil:
                guard scalar == "\"" else { return stop() }
                token = [scalar]
                _ = lexer.consume(scalar)
                phase = .nameValue
            case "arguments" where arguments.isEmpty:
                guard scalar == "{" else { return stop() }
                arguments = [scalar]
                _ = lexer.consume(scalar)
                phase = .arguments
            default:
                // Repeated name/arguments keys are not guessed at.
                guard key != "name", key != "arguments" else { return stop() }
                phase = .otherValue
                consume(scalar)
            }
        case .nameValue:
            token.append(scalar)
            _ = lexer.consume(scalar)
            guard lexer.complete else { return }
            guard
                let decoded = try? JSONDecoder().decode(
                    String.self, from: Data(scalarString(token).utf8)),
                isDeclaredTool(decoded, tools: tools)
            else { return stop() }
            name = decoded
            phase = .afterValue
        case .arguments:
            arguments.append(scalar)
            _ = lexer.consume(scalar)
            if lexer.complete {
                argumentsComplete = true
                phase = .afterValue
            }
        case .otherValue:
            let terminator = lexer.consume(scalar)
            if lexer.complete {
                phase = .afterValue
                if let terminator { consume(terminator) }
            }
        case .afterValue:
            if scalar.isFramingWhitespace { return }
            switch scalar {
            case ",": phase = .expectKey
            case "}": phase = .finished
            default: stop()
            }
        case .finished:
            return
        }
    }

    private mutating func stop() {
        if name == nil { abandoned = true }
        phase = .finished
    }
}

/// Completion of one JSON value. Containers and strings complete at their
/// closing scalar; numbers and literals complete at the first scalar after
/// them, which is returned for the caller to reprocess.
private struct JSONValueLexer {
    private var started = false
    private var primitive = false
    private var inString = false
    private var escaped = false
    private var depth = 0
    private(set) var complete = false

    mutating func consume(_ scalar: Unicode.Scalar) -> Unicode.Scalar? {
        guard !complete else { return nil }
        if !started {
            started = true
            switch scalar {
            case "\"": inString = true
            case "{", "[": depth = 1
            default: primitive = true
            }
            return nil
        }
        if primitive {
            if scalar == "," || scalar == "}" || scalar == "]" || scalar.properties.isWhitespace {
                complete = true
                return scalar
            }
            return nil
        }
        if inString {
            if escaped {
                escaped = false
            } else if scalar == "\\" {
                escaped = true
            } else if scalar == "\"" {
                inString = false
                if depth == 0 { complete = true }
            }
            return nil
        }
        switch scalar {
        case "\"": inString = true
        case "{", "[": depth += 1
        case "}", "]":
            depth -= 1
            if depth == 0 { complete = true }
        default: break
        }
        return nil
    }
}
