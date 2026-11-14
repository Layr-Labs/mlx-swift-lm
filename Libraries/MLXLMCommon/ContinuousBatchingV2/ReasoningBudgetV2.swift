import Foundation
import MLX

/// An output-token budget for a family's atomic thought markers.
/// The limit is 1...Int32.max. Markers do not spend it; the total output limit still applies.
public struct CBv2ReasoningBudget: Sendable, Equatable {
    public let maximumTokens: Int
    public let openTokenID: Int
    public let closeTokenID: Int
    public let startsInsideThought: Bool

    public init(
        maximumTokens: Int, openTokenID: Int, closeTokenID: Int,
        startsInsideThought: Bool
    ) throws {
        guard maximumTokens > 0, maximumTokens <= Int(Int32.max),
            openTokenID >= 0, closeTokenID >= 0,
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
            environment["DARKBLOOM_NATIVE_REASONING_BUDGET"]?.lowercased() ?? "")
    }
}

public enum CBv2ReasoningBudgetError: Error, Equatable, Sendable {
    case invalidConfiguration
    case unsupported
}

/// Host state advances only at the existing confirmed-token boundary.
/// Budget rows never chain or speculate; reconfiguration replays confirmed history.
final class CBv2ReasoningBudgetSampler {
    private struct Row {
        let budget: CBv2ReasoningBudget
        var inside: Bool
        var closed = false
        var consumed = 0
        var failure: String?

        mutating func confirm(_ token: Int) {
            guard failure == nil else { return }
            if token == budget.openTokenID {
                if inside || closed {
                    failure = "reasoning budget: repeated thought opener"
                } else {
                    inside = true
                }
            } else if token == budget.closeTokenID {
                inside = false
                closed = true
            } else if inside {
                guard consumed < budget.maximumTokens else {
                    failure = "reasoning budget: token beyond thought limit"
                    return
                }
                consumed += 1
            }
        }
    }

    private var rows: [CBv2RequestID: Row] = [:]
    var hasRows: Bool { !rows.isEmpty }

    func configure(_ descriptions: [CBv2SamplerRow]) {
        for description in descriptions {
            guard let budget = description.reasoningBudget else {
                rows.removeValue(forKey: description.id)
                continue
            }
            var row = Row(budget: budget, inside: budget.startsInsideThought)
            for token in description.outputTokens { row.confirm(token) }
            rows[description.id] = row
        }
    }

    func mask(_ logits: MLXArray, requestIDs: [CBv2RequestID]) -> MLXArray {
        guard requestIDs.contains(where: { rows[$0] != nil }) else { return logits }
        let vocab = logits.dim(-1)
        let vocabulary = MLXArray(0 ..< vocab)
        return concatenated(
            requestIDs.enumerated().map { index, id in
                let original = logits[index ..< index + 1]
                guard var row = rows[id] else { return original }
                guard row.budget.openTokenID < vocab, row.budget.closeTokenID < vocab else {
                    row.failure = "reasoning budget: marker outside model vocabulary"
                    rows[id] = row
                    return original
                }
                if row.inside && row.consumed == row.budget.maximumTokens {
                    // The forced delimiter is deterministic, even if its model logit
                    // was -infinity. Raw logprobs still describe the original logits.
                    return which(
                        vocabulary .== row.budget.closeTokenID,
                        MLXArray(Float(0)), MLXArray(-Float.infinity)
                    ).reshaped([1, vocab])
                }
                if row.inside || row.closed {
                    return which(
                        vocabulary .== row.budget.openTokenID,
                        MLXArray(-Float.infinity), original)
                }
                return original
            }, axis: 0)
    }

    func confirm(tokens: [Int], requestIDs: [CBv2RequestID]) {
        for (id, token) in zip(requestIDs, tokens) { rows[id]?.confirm(token) }
    }

    func failure(for id: CBv2RequestID) -> String? { rows[id]?.failure }
    func requestDidFinish(_ id: CBv2RequestID) { rows.removeValue(forKey: id) }
}
