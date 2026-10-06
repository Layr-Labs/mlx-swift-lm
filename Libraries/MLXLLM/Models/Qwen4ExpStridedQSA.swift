import Foundation
import MLX
import MLXLMCommon
import os

/// Separate opt-in candidate for the actual contiguous backend's capacity
/// views. MLX's default custom-kernel packing copies their entire logical KV.
/// Read evaluated device strides instead; never infer lazy-array alignment or
/// strides from host metadata (both can change during evaluation).
enum Qwen4ExpStridedQSA {
    static let envFlag = "DARKBLOOM_QWEN4_QSA_STRIDED_KV"

    static func enabled(environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Bool {
        ["1", "true", "yes", "on"].contains(
            environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")
    }

    static func eligible(queries: MLXArray, keys: MLXArray, values: MLXArray,
                         compactLogicalKeyTokens: Int?) -> Bool {
        compactLogicalKeyTokens == nil && queries.ndim == 4 && queries.dim(0) == 1
            && (1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(
                queries.dim(2))
            && Qwen4ExpNativeSparseGQA.canAttend(
                queries: queries, keys: keys, values: values, selectedWidth: 512)
            && Qwen4ExpNativeSparseGQA.steelEnabled()
    }

    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4StridedKV")
    private static let diagnoseFirstCall = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_QSA_STRIDED_KV_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var proven = false
    nonisolated(unsafe) private static var calls = 0
    nonisolated(unsafe) private static var proofs = 0

    static var needsProof: Bool { lock.withLock { !proven } }
    static func markProven() { lock.withLock { proven = true; proofs += 1 } }
    static func recordCall() {
        let first = lock.withLock { calls += 1; return calls == 1 }
        if first && diagnoseFirstCall {
            logger.info("qwen4_strided_kv first_initialized_call=1")
        }
    }
    static func snapshot() -> (calls: Int, proofs: Int) { lock.withLock { (calls, proofs) } }
    static func resetForTesting() { lock.withLock { proven = false; calls = 0; proofs = 0 } }
}

/// Numeric dispatch evidence; successful request completion supplies the GPU
/// outcome. Counts are planned calls, not claims about bytes actually copied.
/// `proofs` counts first-use initialization evaluations, not numerical oracles.
public enum Qwen4ExpStridedKVInvocation {
    public static func snapshot() -> (calls: Int, proofs: Int) { Qwen4ExpStridedQSA.snapshot() }
}
