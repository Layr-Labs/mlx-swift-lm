// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// `NAME<|channel|>commentary json<|message|>{...}` (or `NAME <|constrain|>json
/// <|message|>{...}`). The arguments text after `<|message|>` streams verbatim
/// with surrounding whitespace withheld, as the complete parse trims it.
struct HarmonyArgumentStreamer: ToolCallArgumentStreamer {
    private static let nameMarkers = ["<|channel|>commentary", "<|constrain|>", "<|message|>", " "]
    private static let messageMarker = "<|message|>"
    private static let headerLimit = 512

    private(set) var name: String?
    private(set) var abandoned = false
    private var header = ""
    private var inArguments = false
    private var arguments = [Unicode.Scalar]()
    private var exposedCount = 0

    mutating func append(_ text: String) -> String {
        guard !abandoned else { return "" }
        var scalars = text.unicodeScalars[...]
        if !inArguments {
            header += text
            if name == nil { resolveName() }
            guard let message = header.range(of: Self.messageMarker) else {
                if header.unicodeScalars.count > Self.headerLimit { abandoned = name == nil }
                return ""
            }
            guard name != nil else {
                abandoned = true
                return ""
            }
            inArguments = true
            scalars = header[message.upperBound...].unicodeScalars[...]
        }
        for scalar in scalars where !(arguments.isEmpty && scalar.isFramingWhitespace) {
            arguments.append(scalar)
        }
        guard name != nil else { return "" }
        var end = arguments.count
        while end > exposedCount, arguments[end - 1].isFramingWhitespace { end -= 1 }
        guard end > exposedCount else { return "" }
        let fragment = scalarString(arguments[exposedCount ..< end])
        exposedCount = end
        return fragment
    }

    func remainder(completing arguments: [String: JSONValue]) -> String? {
        var end = self.arguments.count
        while end > 0, self.arguments[end - 1].isFramingWhitespace { end -= 1 }
        let full = Array(self.arguments[..<end])
        if inArguments, ToolArgumentJSON.decodeObject(scalarString(full)) == arguments {
            return scalarString(full[exposedCount...])
        }
        return exposedCount == 0 ? ToolArgumentJSON.encode(.object(arguments)) : nil
    }

    /// Same rule as the complete parse: the name ends at the earliest complete
    /// marker, and no later scalar can move that marker.
    private mutating func resolveName() {
        var earliest: Range<String.Index>?
        for marker in Self.nameMarkers {
            guard let range = header.range(of: marker) else { continue }
            if earliest == nil || range.lowerBound < earliest!.lowerBound { earliest = range }
        }
        guard let earliest else { return }
        let resolved = header[..<earliest.lowerBound].trimmingCharacters(
            in: .whitespacesAndNewlines)
        if resolved.isEmpty { abandoned = true } else { name = resolved }
    }
}
