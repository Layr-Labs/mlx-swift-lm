// Copyright © 2026 Eigen Labs Inc.
import Foundation

/// Preserves request object order: option order and structured-state order are model inputs.
public indirect enum DecisionJSON: Sendable, Equatable {
    case string(String)
    case number(String)
    case bool(Bool)
    case null
    case array([DecisionJSON])
    case object([Field])

    public struct Field: Sendable, Equatable {
        public let key: String
        public let value: DecisionJSON
        public init(_ key: String, _ value: DecisionJSON) {
            self.key = key
            self.value = value
        }
    }
    public subscript(_ key: String) -> DecisionJSON? {
        guard case .object(let fields) = self else { return nil }
        return fields.first { $0.key == key }?.value
    }
    public var string: String? { if case .string(let value) = self { value } else { nil } }
    public var fields: [Field]? { if case .object(let value) = self { value } else { nil } }
    public var elements: [DecisionJSON]? { if case .array(let value) = self { value } else { nil } }

    public static func parse(_ data: Data) throws -> DecisionJSON {
        // Foundation validates number/string grammar; the small traversal retains key order.
        do {
            _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        } catch {
            throw LayaError.invalidJSON("Request body must be valid JSON")
        }
        var parser = Parser(bytes: Array(data))
        return try parser.value(depth: 0)
    }

    public func rendered(ascii: Bool = false) -> String {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .number(let value): return value
        case .string(let value):
            if !ascii {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.withoutEscapingSlashes]
                return String(data: try! encoder.encode(value), encoding: .utf8)!
            }
            var result = "\""
            for unit in value.utf16 {
                switch unit {
                case 34: result += "\\\""
                case 92: result += "\\\\"
                case 8: result += "\\b"
                case 9: result += "\\t"
                case 10: result += "\\n"
                case 12: result += "\\f"
                case 13: result += "\\r"
                case 0 ..< 32: result += String(format: "\\u%04x", unit)
                default:
                    if unit > 127 {
                        result += String(format: "\\u%04x", unit)
                    } else if unit < 128 {
                        result += String(UnicodeScalar(unit)!)
                    }
                }
            }
            return result + "\""
        case .array(let values):
            return "[" + values.map { $0.rendered(ascii: ascii) }.joined(separator: ", ") + "]"
        case .object(let fields):
            return "{"
                + fields.map {
                    DecisionJSON.string($0.key).rendered(ascii: ascii) + ": "
                        + $0.value.rendered(ascii: ascii)
                }.joined(separator: ", ") + "}"
        }
    }

    private struct Parser {
        let bytes: [UInt8]
        var cursor = 0
        mutating func whitespace() {
            while cursor < bytes.count && [9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 }
        }
        mutating func value(depth: Int) throws -> DecisionJSON {
            guard depth < 64 else {
                throw LayaError.invalidRequest("JSON nesting exceeds 64 levels")
            }
            whitespace()
            guard cursor < bytes.count else { throw LayaError.invalidRequest("Incomplete JSON") }
            let start = cursor
            let token = bytes[cursor]
            cursor += 1
            if token == 34 {
                while cursor < bytes.count {
                    if bytes[cursor] == 92 {
                        cursor += 2
                        continue
                    }
                    if bytes[cursor] == 34 {
                        cursor += 1
                        break
                    }
                    cursor += 1
                }
                return .string(
                    try JSONDecoder().decode(String.self, from: Data(bytes[start ..< cursor])))
            }
            if token == 123 || token == 91 {
                var fields: [Field] = []
                var values: [DecisionJSON] = []
                var seenKeys = Set<String>()
                let end: UInt8 = token == 123 ? 125 : 93
                whitespace()
                while bytes[cursor] != end {
                    if token == 123 {
                        guard case .string(let key) = try value(depth: depth + 1) else {
                            throw LayaError.invalidRequest("Invalid object key")
                        }
                        guard seenKeys.insert(key).inserted else {
                            throw LayaError.invalidRequest("Duplicate object key")
                        }
                        whitespace()
                        cursor += 1  // colon (already validated)
                        fields.append(.init(key, try value(depth: depth + 1)))
                    } else {
                        values.append(try value(depth: depth + 1))
                    }
                    whitespace()
                    if bytes[cursor] == 44 {
                        cursor += 1
                        whitespace()
                    } else {
                        break
                    }
                }
                cursor += 1
                return token == 123 ? .object(fields) : .array(values)
            }
            while cursor < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[cursor]) {
                cursor += 1
            }
            let scalar = String(decoding: bytes[start ..< cursor], as: UTF8.self)
            switch scalar {
            case "null": return .null
            case "true": return .bool(true)
            case "false": return .bool(false)
            default:
                guard let number = Double(scalar), number.isFinite else {
                    throw LayaError.invalidRequest("JSON numbers must be finite")
                }
                // Python's request path parses JSON before rendering the structured prompt.
                // Preserve integer precision, but normalize floating lexemes such as 1e2
                // and 1.2300 to the parsed value rather than tokenizing their wire spelling.
                let floating = scalar.contains(".") || scalar.contains("e") || scalar.contains("E")
                return .number(floating ? number.description : scalar == "-0" ? "0" : scalar)
            }
        }
    }
}
