// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// One ordered step of tool-call output when calls are exposed before their
/// native frame closes. Without streaming options a processor produces only
/// `.text`, and complete calls accumulate in `ToolCallProcessor.toolCalls`.
public enum ToolCallStreamSegment: Sendable, Equatable {
    /// Visible text outside every call frame.
    case text(String)
    /// The current frame fixed its function name; argument fragments follow.
    case callStarted(name: String)
    /// Next fragment of the started call's OpenAI `arguments` JSON text.
    case callArguments(String)
    /// The started call's frame closed and parsed. `remainingArguments`
    /// completes the exposed JSON text to exactly `call`'s arguments.
    case callCompleted(call: ToolCall, remainingArguments: String)
    /// The started call cannot be completed consistently with what was
    /// already exposed.
    case callFailed(ToolCallStreamFailure)
    /// A call that was never exposed, complete in one piece.
    case call(ToolCall)
}

public enum ToolCallStreamFailure: String, Sendable, Equatable {
    /// The frame closed but did not parse.
    case malformed
    /// The frame parsed differently from the exposed prefix.
    case diverged
    /// Generation ended inside the frame.
    case unterminated
}

/// Opt-in for exposing calls while they are generated.
public struct ToolCallStreamingOptions: Sendable {
    /// A call is exposed only when this accepts its resolved function name.
    /// Rejected names complete in one piece, so the caller still validates
    /// them before anything is shown.
    public var exposesName: @Sendable (String) -> Bool

    public init(exposesName: @escaping @Sendable (String) -> Bool) {
        self.exposesName = exposesName
    }

    /// Formats with an incremental argument renderer. Others keep the
    /// complete-frame behavior even when streaming is requested.
    public static func supports(_ format: ToolCallFormat) -> Bool {
        switch format {
        case .gemma, .qwen35, .harmony, .json: true
        default: false
        }
    }
}

/// Incremental view of one call frame's payload: the start tag is removed and
/// the end tag is never fed. It exposes only argument JSON that the
/// complete-frame parser must agree with; `remainder(completing:)` checks
/// that agreement once the frame has been parsed.
protocol ToolCallArgumentStreamer {
    /// Function name once the payload has fixed it.
    var name: String? { get }
    /// True when the payload cannot be rendered incrementally and no name
    /// was found. Such a call completes in one piece.
    var abandoned: Bool { get }
    /// Consume more payload. Returns newly certain argument JSON text, which
    /// is always empty until `name` is set.
    mutating func append(_ text: String) -> String
    /// JSON text completing everything returned by `append` to exactly
    /// `arguments`, or nil when the exposed prefix disagrees.
    func remainder(completing arguments: [String: JSONValue]) -> String?
}

func makeToolCallArgumentStreamer(
    format: ToolCallFormat, tools: [[String: any Sendable]]?
) -> (any ToolCallArgumentStreamer)? {
    switch format {
    case .gemma: GemmaArgumentStreamer(tools: tools)
    case .qwen35: Qwen35ArgumentStreamer(tools: tools)
    case .harmony: HarmonyArgumentStreamer()
    case .json: JSONFrameArgumentStreamer(tools: tools)
    default: nil
    }
}

// MARK: - JSON text

enum ToolArgumentJSON {
    static func quoted(_ text: String) -> String {
        "\"" + escaped(text.unicodeScalars) + "\""
    }

    /// JSON string content. Any valid escaping decodes to the same value, so
    /// this need not match `JSONEncoder` byte for byte.
    static func escaped<S: Sequence>(_ scalars: S) -> String where S.Element == Unicode.Scalar {
        var out = String.UnicodeScalarView()
        for scalar in scalars {
            switch scalar {
            case "\"": out.append(contentsOf: #"\""#.unicodeScalars)
            case "\\": out.append(contentsOf: #"\\"#.unicodeScalars)
            case "\n": out.append(contentsOf: #"\n"#.unicodeScalars)
            case "\r": out.append(contentsOf: #"\r"#.unicodeScalars)
            case "\t": out.append(contentsOf: #"\t"#.unicodeScalars)
            default:
                if scalar.value < 0x20 {
                    out.append(contentsOf: String(format: "\\u%04x", scalar.value).unicodeScalars)
                } else {
                    out.append(scalar)
                }
            }
        }
        return String(out)
    }

    /// Compact, key-sorted encoding (the server's tool-argument encoding).
    static func encode(_ value: JSONValue) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value) else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    static func decodeObject(_ text: String) -> [String: JSONValue]? {
        try? JSONDecoder().decode([String: JSONValue].self, from: Data(text.utf8))
    }
}

/// Exposed prefix of an arguments object: complete members plus at most one
/// open string member. `remainder(completing:)` closes it against the
/// authoritative arguments, appending members that were never exposed.
struct ExposedArgumentObject {
    private var opened = false
    private var members: [(key: String, value: JSONValue)] = []
    private var memberKeys: Set<String> = []
    private var openKey: String?
    private var openScalars: [Unicode.Scalar] = []

    func contains(_ key: String) -> Bool {
        memberKeys.contains(key) || openKey == key
    }

    mutating func member(_ key: String, _ value: JSONValue) -> String? {
        guard let json = ToolArgumentJSON.encode(value) else { return nil }
        let text = separator() + ToolArgumentJSON.quoted(key) + ":" + json
        members.append((key, value))
        memberKeys.insert(key)
        return text
    }

    mutating func beginString(_ key: String) -> String {
        let text = separator() + ToolArgumentJSON.quoted(key) + ":\""
        openKey = key
        openScalars = []
        return text
    }

    mutating func appendString<S: Sequence>(_ scalars: S) -> String
    where S.Element == Unicode.Scalar {
        let start = openScalars.count
        openScalars.append(contentsOf: scalars)
        return ToolArgumentJSON.escaped(openScalars[start...])
    }

    mutating func endString() -> String {
        guard let key = openKey else { return "" }
        var value = String.UnicodeScalarView()
        value.append(contentsOf: openScalars)
        members.append((key, .string(String(value))))
        memberKeys.insert(key)
        openKey = nil
        openScalars = []
        return "\""
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        guard memberKeys.count == members.count,
            members.allSatisfy({ arguments[$0.key] == $0.value })
        else { return nil }
        var text = opened ? "" : "{"
        var count = members.count
        if let openKey {
            guard !memberKeys.contains(openKey), case .string(let full)? = arguments[openKey]
            else { return nil }
            let fullScalars = Array(full.unicodeScalars)
            guard fullScalars.starts(with: openScalars) else { return nil }
            text += ToolArgumentJSON.escaped(fullScalars[openScalars.count...]) + "\""
            count += 1
        }
        for key in arguments.keys.sorted() where !contains(key) {
            guard let value = arguments[key], let json = ToolArgumentJSON.encode(value) else {
                return nil
            }
            text += (count > 0 ? "," : "") + ToolArgumentJSON.quoted(key) + ":" + json
            count += 1
        }
        return text + "}"
    }

    private mutating func separator() -> String {
        guard opened else {
            opened = true
            return "{"
        }
        return members.isEmpty ? "" : ","
    }
}

extension Unicode.Scalar {
    var isFramingWhitespace: Bool { properties.isWhitespace }
}

func scalarString(_ scalars: some Sequence<Unicode.Scalar>) -> String {
    var view = String.UnicodeScalarView()
    view.append(contentsOf: scalars)
    return String(view)
}

/// Keeps the suffix of a growing value that may still turn out to be the
/// start of one of `terminators`, releasing everything before it.
func releasedPrefix(
    of pending: inout [Unicode.Scalar], before terminators: [[Unicode.Scalar]]
) -> [Unicode.Scalar] {
    var start = 0
    while start < pending.count,
        !terminators.contains(where: {
            $0.count > pending.count - start && $0.starts(with: pending[start...])
        })
    {
        start += 1
    }
    let released = Array(pending[..<start])
    pending.removeFirst(start)
    return released
}
