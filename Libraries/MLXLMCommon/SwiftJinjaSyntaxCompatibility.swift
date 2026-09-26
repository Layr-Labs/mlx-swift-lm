import Foundation

/// Syntax-only compatibility with Python Jinja's whitespace controls.
/// Mirrors the existing DarkBloom ProviderCoreFoundation normalizer's comment
/// and literal-brace rules. Request clock binding remains with the provider.
/// This transforms template syntax, never message content or generated output.
public enum SwiftJinjaSyntaxCompatibility {
    public static func normalize(_ template: String) -> String {
        var result = template
        result = result.replacingOccurrences(
            of: #"-#\}\s*"#, with: "#}", options: .regularExpression)
        result = result.replacingOccurrences(
            of: #"\s*\{#-"#, with: "{#", options: .regularExpression)
        result = result.replacingOccurrences(
            of: #"\{\s+\{\{-"#, with: "{{ '{' -}}{{-", options: .regularExpression)
        result = result.replacingOccurrences(
            of: #"\{\s+\{%-"#, with: "{{ '{' -}}{%-", options: .regularExpression)
        return result
    }
}
