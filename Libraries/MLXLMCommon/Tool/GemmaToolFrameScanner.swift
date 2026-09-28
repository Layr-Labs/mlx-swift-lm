/// Incremental native Gemma framing: an end tag inside a marker-quoted or
/// JSON-quoted argument is data, not a frame terminator.
struct GemmaToolFrameScanner {
    private enum Quote { case none, json, native(String) }
    private var quote = Quote.none
    private var escaped = false
    private var candidate = ""
    private let end = "<tool_call|>"
    private let markers = ["<|\"|>", "<escape>"]

    mutating func consume(_ scalar: Unicode.Scalar) -> Bool {
        let character = String(scalar)
        switch quote {
        case .json:
            if escaped { escaped = false }
            else if character == "\\" { escaped = true }
            else if character == "\"" { quote = .none }
            return false
        case .native(let marker):
            candidate += character
            if candidate.hasSuffix(marker) { quote = .none; candidate = "" }
            else if candidate.count > marker.count { candidate = String(candidate.suffix(marker.count)) }
            return false
        case .none:
            if candidate.isEmpty && character != "<" {
                if character == "\"" { quote = .json }
                return false
            }
            candidate += character
            if candidate == end { return true }
            if markers.contains(candidate) { quote = .native(candidate); candidate = ""; return false }
            if ([end] + markers).contains(where: { $0.hasPrefix(candidate) }) { return false }
            candidate = character == "<" ? "<" : ""
            if character == "\"" { quote = .json }
            return false
        }
    }
}
