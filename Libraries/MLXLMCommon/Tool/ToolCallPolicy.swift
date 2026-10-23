// Copyright © 2026 Apple Inc.

/// Rules for recovering and validating generated tool calls.
/// Native parser requirements and declared-tool authorization always apply.
public struct ToolCallPolicy: Hashable, Sendable {
    /// Bounded recovery of calls emitted in a different dialect.
    /// Applies to text formats; framed token protocols use their native parser.
    public var recovery: ToolCallRecoveryPolicy

    /// Schema enforcement after unambiguous argument normalization.
    public var validation: ToolCallValidationPolicy

    /// Selects the state machine of ``ToolCallProcessor``.
    ///
    /// `true` (the fork's default) selects the frame state machine: a frame
    /// that does not parse, or that names an undeclared tool, comes back as
    /// visible text. `false` selects upstream's recovery state machine: such
    /// output becomes a ``RejectedToolCall``, and `recovery` and `validation`
    /// apply. Generation callers set it through ``GenerateParameters/toolCallPolicy``.
    public var returnsFailedCallsAsText: Bool

    public init(
        recovery: ToolCallRecoveryPolicy = .conservative,
        validation: ToolCallValidationPolicy = .permissive,
        returnsFailedCallsAsText: Bool = true
    ) {
        self.recovery = recovery
        self.validation = validation
        self.returnsFailedCallsAsText = returnsFailedCallsAsText
    }
}

/// Controls schema enforcement independently of syntax recovery and tool-name
/// authorization. Both modes normalize unambiguous, schema-declared values.
public enum ToolCallValidationPolicy: String, Hashable, Sendable, CaseIterable {
    /// Reject proven schema violations. Unsupported schema assertions remain
    /// unknown. Enable this to check arguments before automatic tool dispatch.
    case strict
    /// Forward parsed arguments after normalization, even if they violate the
    /// schema. This is the default; applications validate arguments themselves.
    /// Native parser requirements and declared-tool authorization still apply.
    case permissive
}
