import Foundation
import MLX
import MLXLMCommon
import os

/// Opt-in layer scheduling, adapted from the oMLX
/// Qwen4 decoder loop's early layer submission. No model arithmetic changes.
/// The initial proof scope is singleton text decode/verify on native paged KV;
/// unknown caches, media/position payloads and wider prefill retain old scheduling.
enum Qwen4ExpLayerSubmission {
    static let flag = "DARKBLOOM_QWEN4_LAYER_ASYNC"

    static func plan(
        batchSize: Int, sequenceWidth: Int, hasEmbeddings: Bool, hasPositions: Bool,
        caches: [any CBv2AttendingLayerCache],
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Plan? {
        guard environment[flag] == "1", batchSize == 1, (1...6).contains(sequenceWidth),
            !hasEmbeddings, !hasPositions, !caches.isEmpty
        else { return nil }
        let native = caches.compactMap { $0 as? PagedLayerCache }
        guard native.count == caches.count,
            native.allSatisfy({ $0.rows.count == 1 && !$0.qwen4HasWriteFault })
        else { return nil }
        return Plan(sequenceWidth: sequenceWidth, caches: native)
    }

    struct Plan {
        let sequenceWidth: Int
        let caches: [PagedLayerCache]

        /// Enqueue only the current valid layer. The engine continues to own
        /// final evaluation roots, recurrent commit, page fences and retirement.
        @discardableResult
        func submit(_ hidden: MLXArray) -> Bool {
            guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) == sequenceWidth,
                caches.allSatisfy({ !$0.qwen4HasWriteFault })
            else { return false }
            // Chained decode can reference unfilled host-backed PLE inputs.
            // Drain registered fills, but leave the engine's scope OPEN so
            // later layers can register more. Never submit a stale placeholder.
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            guard caches.allSatisfy({ !$0.qwen4HasWriteFault }) else { return false }
            asyncEval(hidden)
            Qwen4ExpLayerSubmissionInvocation.record(width: sequenceWidth)
            return true
        }
    }
}

/// Numeric path witness. It contains no prompt, token, tensor or user content.
public enum Qwen4ExpLayerSubmissionInvocation {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4LayerSubmission")
    private static let diagnoseFirst = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_LAYER_ASYNC_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var calls = 0
    static func record(width: Int) {
        let first = lock.withLock { calls += 1; return calls == 1 }
        if first && diagnoseFirst {
            // Numeric enqueue witness only. Completed output/state and request
            // semantics still require their independent qualification gates.
            logger.info("qwen4_layer_async first_submitted_layer=1 width=\(width, privacy: .public)")
        }
    }
    public static func snapshot() -> Int { lock.withLock { calls } }
}
