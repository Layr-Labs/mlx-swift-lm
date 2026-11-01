import Foundation
import MLX
import MLXLMCommon

/// Super's fused decode kernels on the row-exact trunk (`superRowExactForward`:
/// MTP verify windows and batched decode) and on the draft head's graph step,
/// each bit-identical to the op chain it replaces and behind the guard the
/// target's one-token decode uses: the block-boundary residual norm, the Mamba
/// gated norm and (one token per row) SSM step, the MoE expert combine,
/// squared ReLU fused into the shared-expert up projection (and the head's
/// fc1), and the head's fused expert selection and router-index route.
public enum NemotronSuperMTPFusion {
    /// Default true; launching with `DARKBLOOM_NEMOTRON_SUPER_MTP_FUSION=0`
    /// runs the previous op chains (A/B measurement only).
    public static let enabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON_SUPER_MTP_FUSION"] != "0"

    /// Overrides `enabled` within one task (tests).
    @TaskLocal static var taskOverride: Bool?

    static var active: Bool { taskOverride ?? enabled }

    private static let zeroIndexLock = NSLock()
    private nonisolated(unsafe) static var zeroIndices: [Int: MLXArray] = [:]

    /// `rows` uint32 zeros built on the host once per row count, so a gathered
    /// product's weight index needs no device fill per call. Row-exact calls
    /// take at most `ModelOptRowExact.maximumRows` rows, which bounds the
    /// cache; larger counts are not cached.
    static func zeroRowIndices(_ rows: Int) -> MLXArray {
        guard rows <= ModelOptRowExact.maximumRows else {
            return MLXArray.zeros([rows], type: UInt32.self)
        }
        return zeroIndexLock.withLock {
            if let cached = zeroIndices[rows] { return cached }
            let value = MLXArray([UInt32](repeating: 0, count: rows))
            zeroIndices[rows] = value
            return value
        }
    }
}

/// Submission of the row-exact trunk (verify windows, batched decode) in parts
/// of `everyLayers` blocks, so the GPU runs while the rest of the graph is
/// built. Evaluation order only; values are unchanged. 0 builds it whole.
public enum NemotronSuperMTPWindowSubmission {
    /// Default 12; `DARKBLOOM_NEMOTRON_SUPER_MTP_WINDOW_SUBMIT` at launch
    /// overrides it.
    public static let everyLayers =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON_SUPER_MTP_WINDOW_SUBMIT"] ?? "")
        ?? 12

    /// Overrides `everyLayers` within one task (tests).
    @TaskLocal static var taskOverride: Int?

    static var active: Int { taskOverride ?? everyLayers }
}
