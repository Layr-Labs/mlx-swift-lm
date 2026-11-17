// Copyright © 2026 Eigen Labs.

import Foundation

/// The GLM5, Gemma and MiMo V2 tool parsers return a call that keeps their
/// frame grammar whatever the caller's schema says (a missing `required`
/// argument, an undeclared one, a value without a declared type, of a union or
/// of another type), for the caller to validate, as vLLM and SGLang do.
///
/// `DARKBLOOM_GRAMMAR_ONLY_TOOL_PARSING`: unset or any other value keeps this
/// on; `0`, `off`, `false` or `no` (any case) restores those parsers' schema
/// refusals. Read once per process.
public enum GrammarOnlyToolParsing {
    public static let environmentKey = "DARKBLOOM_GRAMMAR_ONLY_TOOL_PARSING"
    public static let isEnabled = enabled(environment: ProcessInfo.processInfo.environment)

    public static func enabled(environment: [String: String]) -> Bool {
        guard
            let value = environment[environmentKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        else { return true }
        return !["0", "off", "false", "no"].contains(value)
    }
}
