import Foundation

/// Finds the outer Qwen tool delimiter without interpreting argument data.
/// JSON strings use JSON's quote/escape rules; XML parameter values end only
/// at their parameter delimiter. This is framing, not JSON/XML repair.
/// An unescaped `</parameter>` within an XML value remains ambiguous and is
/// not guessed at. The payload parser still validates the complete frame.
public struct Qwen35ToolFrameScanner: Sendable {
    private enum State: Sendable {
        case payload, xmlOpening, xml, parameterOpening, parameter
        case json, malformed
    }

    private var state: State = .payload
    private var marker = ""
    private var inString = false
    private var escaped = false
    private let endTag: String

    /// The opening wrapper has already been consumed by the caller.
    public init(endTag: String = "</tool_call>") { self.endTag = endTag }

    /// Bounded lexer state only; the owner may independently buffer the payload.
    public var bufferedCharacterCount: Int { marker.unicodeScalars.count }

    /// Returns true exactly when an outer closing delimiter is completed.
    public mutating func consume(_ scalar: Unicode.Scalar) -> Bool {
        switch state {
        case .payload:
            if marker.isEmpty, scalar.properties.isWhitespace { return false }
            if marker.isEmpty, scalar.value == 123 {
                state = .json
                return false
            }
            marker.unicodeScalars.append(scalar)
            if "<function=".hasPrefix(marker) {
                if marker == "<function=" { state = .xmlOpening; marker = "" }
                return false
            }
            state = .malformed
            return marker.hasSuffix(endTag)
        case .json:
            if inString {
                if escaped { escaped = false }
                else if scalar.value == 92 { escaped = true }
                else if scalar.value == 34 { inString = false }
                return false
            }
            if scalar.value == 34 {
                inString = true
                marker = ""
                return false
            }
            remember(scalar)
            return marker.hasSuffix(endTag)
        case .xmlOpening, .parameterOpening:
            if scalar.value == 62 {
                state = state == .xmlOpening ? .xml : .parameter
                marker = ""
            }
            return false
        case .xml:
            remember(scalar)
            if marker.hasSuffix("<parameter=") {
                state = .parameterOpening
                marker = ""
                return false
            }
            return marker.hasSuffix(endTag)
        case .parameter:
            remember(scalar)
            if marker.hasSuffix("</parameter>") {
                state = .xml
                marker = ""
            }
            return false
        case .malformed:
            remember(scalar)
            return marker.hasSuffix(endTag)
        }
    }

    private mutating func remember(_ scalar: Unicode.Scalar) {
        marker.unicodeScalars.append(scalar)
        let bound = max(endTag.unicodeScalars.count, "</parameter>".count, "<parameter=".count)
        if marker.unicodeScalars.count > bound {
            marker.unicodeScalars.removeFirst(marker.unicodeScalars.count - bound)
        }
    }

    /// Scan one wrapper-prefixed buffer. Literal wrappers inside JSON strings
    /// or XML parameter values are ignored; trailing text remains untouched.
    static func endRange(in text: String, startTag: String, endTag: String) -> Range<String.Index>? {
        guard text.hasPrefix(startTag) else { return nil }
        var scanner = Self(endTag: endTag)
        let scalars = text.unicodeScalars
        var index = scalars.index(scalars.startIndex, offsetBy: startTag.unicodeScalars.count)
        while index < scalars.endIndex {
            let next = scalars.index(after: index)
            if scanner.consume(scalars[index]) {
                return scalars.index(next, offsetBy: -endTag.unicodeScalars.count)..<next
            }
            index = next
        }
        return nil
    }
}
