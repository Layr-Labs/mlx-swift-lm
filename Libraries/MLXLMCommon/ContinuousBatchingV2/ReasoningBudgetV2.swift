import Foundation

/// An output-token budget for a family's atomic thought markers.
/// Marker tokens do not spend the budget. The request's total output limit still applies.
public struct CBv2ReasoningBudget: Sendable, Equatable {
    public let maximumTokens: Int
    public let openTokenID: Int
    public let closeTokenID: Int
    public let startsInsideThought: Bool

    public init(
        maximumTokens: Int, openTokenID: Int, closeTokenID: Int,
        startsInsideThought: Bool
    ) throws {
        guard maximumTokens > 0, openTokenID >= 0, closeTokenID >= 0,
            openTokenID != closeTokenID,
            openTokenID <= Int(Int32.max), closeTokenID <= Int(Int32.max)
        else { throw CBv2ReasoningBudgetError.invalidConfiguration }
        self.maximumTokens = maximumTokens
        self.openTokenID = openTokenID
        self.closeTokenID = closeTokenID
        self.startsInsideThought = startsInsideThought
    }

    public static func enabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        ["1", "true", "yes", "on"].contains(
            environment["DARKBLOOM_CBV2_REASONING_BUDGET"]?.lowercased() ?? "")
    }
}

public enum CBv2ReasoningBudgetError: Error, Equatable, Sendable {
    case invalidConfiguration
    case unsupported
}
