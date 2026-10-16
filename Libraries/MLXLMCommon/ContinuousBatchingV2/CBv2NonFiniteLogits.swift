// CBv2NonFiniteLogits.swift
//
// Fail-closed handling of sampled logits that hold NaN or infinity.

import MLX

/// Logits with NaN or ±infinity define no token: greedy argmax over NaN picks
/// token 0, and a stochastic sampler draws from an undefined distribution.
/// Every sampled row and every MTP verify position therefore carries one lazy
/// flag, built in the same step graph as its token and read back with it at
/// the step's existing host sync: sampled rows append their flags to the
/// token readback, verify positions replace a flagged target token with
/// `invalidTarget` inside the existing acceptance packet. A flagged request
/// finishes with `finishReason` and commits nothing from the flagged position.
/// Unflagged rows sample and commit exactly as before.
enum CBv2NonFiniteLogits {
    /// An ordinary engine error, so hosts classify it as a generation failure.
    static let failureMessage = "non-finite logits: the model produced NaN or infinity"

    static var finishReason: CBv2FinishReason { .error(failureMessage) }

    /// Verify target token of a non-finite position. Token ids are never
    /// negative, so no draft matches it.
    static let invalidTarget: Int32 = -1

    /// Lazy int32 flags shaped like the first `keptAxes` axes of `logits`:
    /// 1 where any logit of that row (or row position) is NaN or ±infinity.
    /// `x - x` is zero for finite `x` and NaN for NaN and ±infinity, and a sum
    /// keeps NaN, so one elementwise pass and one reduction decide each flag.
    static func flags(_ logits: MLXArray, keptAxes: Int = 1) -> MLXArray {
        precondition(logits.ndim > keptAxes, "CBv2 non-finite flags need a reduced axis")
        let residue = sum(logits - logits, axes: Array(keptAxes ..< logits.ndim))
        return isNaN(residue).asType(.int32)
    }

    /// The step's one finalize readback: `[K]` sampled tokens, then their
    /// `[K]` flags. The tokens array itself stays the chained-decode input.
    static func readback(tokens: MLXArray, flags: MLXArray) -> MLXArray {
        concatenated([tokens.reshaped([-1]), flags.reshaped([-1]).asType(tokens.dtype)], axis: 0)
    }

    /// `targets` (int32) with every flagged position set to `invalidTarget`.
    static func invalidatingTargets(_ targets: MLXArray, flags: MLXArray) -> MLXArray {
        which(flags .!= 0, MLXArray(invalidTarget), targets.asType(.int32))
    }

    /// Splits a host `readback` of `rows` sampled rows into the tokens and the
    /// indices of flagged rows.
    static func split(_ values: [Int32], rows: Int) -> (tokens: [Int32], flagged: Set<Int>) {
        precondition(values.count == 2 * rows, "CBv2 sampled readback must hold tokens and flags")
        var flagged = Set<Int>()
        for index in 0 ..< rows where values[rows + index] != 0 { flagged.insert(index) }
        return (Array(values[..<rows]), flagged)
    }
}
