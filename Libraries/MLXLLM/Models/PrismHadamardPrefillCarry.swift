import Foundation
import MLX
import MLXLMCommon

/// Default-on eligible packed prefill scheduling. The lexical scope borrows caches only
/// while the owning engine builds its forward; no task or array stores it.
/// Final evaluation, commit/rollback and retirement remain engine-owned.
enum PrismHadamardPrefillCarry {
    static let enabled = isEnabled(environmentValue: ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_PREFILL_CARRY_ASYNC"])

    static func isEnabled(environmentValue: String?) -> Bool {
        environmentValue == nil || environmentValue == "1"
    }
    private static let diagnose = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_PREFILL_CARRY_DIAGNOSTICS"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var submitted = 0

    struct Context: @unchecked Sendable {
        let batch: Int
        let caches: [any CBv2AttendingLayerCache]
        var permitsSubmission: Bool {
            caches.allSatisfy { cache in
                guard cache.rows.count == batch else { return false }
                if let paged = cache as? PagedLayerCache {
                    // This existing accessor reports the generic native pool
                    // write validator; no Qwen4 arithmetic is selected here.
                    return !paged.qwen4HasWriteFault
                }
                return cache is CBv2LayerCache
            }
        }
    }
    @TaskLocal static var context: Context?

    static func withScope<T>(
        isPacked: Bool, batch: Int, width: Int, hasEmbeddings: Bool,
        hasPositions: Bool, capturesState: Bool,
        caches: [any CBv2AttendingLayerCache], body: () -> T
    ) -> T {
        guard enabled else { return body() }
        // Short load-time dtype probes and decode keep their old admission/
        // evaluation boundary. Media/verification need separate qualification.
        let eligible = isPacked && batch > 0 && width >= 128 && !hasEmbeddings
            && !hasPositions && !capturesState && !caches.isEmpty
        let candidate = eligible ? Context(batch: batch, caches: caches) : nil
        return $context.withValue(candidate, operation: body)
    }

    @discardableResult
    static func submit(_ carry: MLXArray) -> Bool {
        guard enabled, let context, context.permitsSubmission,
            carry.ndim == 3, carry.dim(0) == context.batch,
            carry.dim(1) > 0, carry.dim(2) > 0
        else { return false }
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        guard context.permitsSubmission else { return false }
        asyncEval(carry)
        if diagnose { lock.withLock { submitted += 1 } }
        return true
    }

    static var submittedCount: Int { lock.withLock { submitted } }
}
