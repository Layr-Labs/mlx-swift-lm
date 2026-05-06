// Copyright © 2026 Apple Inc.

import Foundation

/// Errors thrown by the Gemma 4 MTP drafter pipeline.
public enum Gemma4MTPError: LocalizedError, Sendable, Equatable {
    /// The target passed to `generateGemma4MTP` wasn't a `Gemma4TextModel`.
    /// Associated value is the actual type name.
    case unsupportedTarget(String)

    /// `bind(target:)` was called a second time with a different target.
    /// The drafter's weights encode assumptions about one specific target;
    /// rebinding to a different target is not supported.
    case rebindForbidden

    /// A drafter/target compatibility check failed at bind time. `field`
    /// names the mismatched field; `drafter` and `target` are the observed
    /// values (stringified).
    case incompatibleDrafter(field: String, drafter: String, target: String)

    /// `generateGemma4MTP` was called with a `blockSize` outside the
    /// supported range (typically 2–16).
    case invalidBlockSize(Int)

    /// A drafter forward was attempted before `bind(target:)`.
    case drafterNotBound

    public var errorDescription: String? {
        switch self {
        case .unsupportedTarget(let typeName):
            return "Gemma4 MTP requires a Gemma4TextModel target; got \(typeName)."
        case .rebindForbidden:
            return
                "Gemma4AssistantDraftModel cannot be rebound to a different target. "
                + "Construct a new drafter instead."
        case .incompatibleDrafter(let field, let drafter, let target):
            return
                "Drafter/target mismatch on \(field): drafter=\(drafter), target=\(target)."
        case .invalidBlockSize(let n):
            return "Invalid blockSize \(n): must be between 2 and 16 inclusive."
        case .drafterNotBound:
            return
                "Gemma4AssistantDraftModel.bind(target:) must be called before "
                + "any forward pass."
        }
    }
}
