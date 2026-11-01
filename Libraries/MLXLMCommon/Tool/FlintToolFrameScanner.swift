import Foundation

enum FlintToolMarkers {
    static let start = "<tool_call>"
    static let end = "</tool_call>"
    static let keyStart = "<arg_key>"
    static let keyEnd = "</arg_key>"
    static let valueStart = "<arg_value>"
    static let valueEnd = "</arg_value>"
}

/// Flint values are opaque until their argument delimiter. In particular a
/// literal tool closing tag inside an argument must never create another call.
/// Declared string arguments are raw even when they start with `{` or `[`.
/// Identifiers are bounded by the supplied schemas; payloads are never retained.
/// A literal `</arg_value>` cannot be escaped in the pinned raw-string protocol.
/// Callers must reject such values before rendering historical tool arguments.
public struct FlintToolFrameScanner: Sendable {
    private enum Phase { case name, tags, key }
    private let rawStringArguments: [String: Set<String>]
    private let maximumNameLength: Int
    private var phase = Phase.name
    private var functionName = ""
    private var nameHasTrailingWhitespace = false
    private var invalidName = false
    private var stringKeys = Set<String>()
    private var maximumKeyLength = 0
    private var key = ""
    private var keyOverflow = false
    private var rawStringValue = false
    private var marker = ""
    private var valueScanner: FlintArgumentValueScanner?

    public init(tools: [[String: any Sendable]]? = nil) {
        var arguments = [String: Set<String>]()
        for tool in tools ?? [] {
            guard let function = tool["function"] as? [String: any Sendable],
                let name = function["name"] as? String,
                let parameters = function["parameters"] as? [String: any Sendable],
                let properties = parameters["properties"] as? [String: any Sendable]
            else { continue }
            for key in properties.keys
            where isStringType(funcName: name, argName: key, tools: tools) {
                arguments[name, default: []].insert(key)
            }
        }
        rawStringArguments = arguments
        maximumNameLength = arguments.keys.map { $0.unicodeScalars.count }.max() ?? 0
    }

    public var bufferedCharacterCount: Int {
        marker.count + functionName.count + key.count + (valueScanner?.bufferedCharacterCount ?? 0)
    }

    public mutating func consume(_ scalar: Unicode.Scalar) -> Bool {
        if var scanner = valueScanner {
            valueScanner = scanner.consume(scalar) ? nil : scanner
            return false
        }
        if phase == .name {
            if scalar == "<" {
                stringKeys = invalidName ? [] : (rawStringArguments[functionName] ?? [])
                maximumKeyLength = stringKeys.map { $0.unicodeScalars.count }.max() ?? 0
                functionName = ""
                phase = .tags
            } else {
                if scalar.properties.isWhitespace {
                    if !functionName.isEmpty { nameHasTrailingWhitespace = true }
                } else if nameHasTrailingWhitespace
                    || functionName.unicodeScalars.count >= maximumNameLength
                {
                    invalidName = true
                } else {
                    functionName.unicodeScalars.append(scalar)
                }
                return false
            }
        }
        if phase == .key {
            if key.unicodeScalars.count < maximumKeyLength + FlintToolMarkers.keyEnd.count {
                key.unicodeScalars.append(scalar)
            } else {
                keyOverflow = true
            }
        }
        marker.unicodeScalars.append(scalar)
        if marker.unicodeScalars.count > FlintToolMarkers.end.count {
            marker.unicodeScalars.removeFirst()
        }
        if phase == .key {
            if marker.hasSuffix(FlintToolMarkers.keyEnd) {
                rawStringValue =
                    !keyOverflow
                    && stringKeys.contains(String(key.dropLast(FlintToolMarkers.keyEnd.count)))
                key = ""
                marker = ""
                phase = .tags
            }
        } else if marker.hasSuffix(FlintToolMarkers.keyStart) {
            key = ""
            keyOverflow = false
            phase = .key
        } else if marker.hasSuffix(FlintToolMarkers.valueStart) {
            valueScanner = FlintArgumentValueScanner(rawString: rawStringValue)
            rawStringValue = false
            marker = ""
        } else if marker.hasSuffix(FlintToolMarkers.end) {
            return true
        }
        return false
    }
}

/// Shared by complete-frame parsing and streaming so JSON string escapes and
/// argument boundaries cannot drift between the two paths. This only finds a
/// boundary; the parser still validates JSON before accepting a structured value.
struct FlintArgumentValueScanner: Sendable {
    private let rawString: Bool
    private var marker = ""
    private var valueStarted = false
    private var structured = false
    private var depth = 0
    private var quoted = false
    private var escaped = false

    init(rawString: Bool = false) { self.rawString = rawString }
    var bufferedCharacterCount: Int { marker.count }

    mutating func consume(_ scalar: Unicode.Scalar) -> Bool {
        if !valueStarted && !scalar.properties.isWhitespace {
            valueStarted = true
            structured = !rawString && (scalar == "{" || scalar == "[")
        }
        if structured {
            if quoted {
                if escaped {
                    escaped = false
                } else if scalar == "\\" {
                    escaped = true
                } else if scalar == "\"" {
                    quoted = false
                }
                return false
            }
            if scalar == "\"" {
                quoted = true
                return false
            }
            if scalar == "{" || scalar == "[" { depth += 1 }
            if scalar == "}" || scalar == "]" { depth -= 1 }
            if depth != 0 { return false }
        }
        marker.unicodeScalars.append(scalar)
        if marker.unicodeScalars.count > FlintToolMarkers.valueEnd.count {
            marker.unicodeScalars.removeFirst()
        }
        return marker.hasSuffix(FlintToolMarkers.valueEnd)
    }
}
