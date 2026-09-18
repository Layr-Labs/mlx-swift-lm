import MLX

/// Load-time serving policy. Providers install this before backend probes or
/// request admission and derive capacity from the installed capability. An
/// already constructed target must not retain a different process-env choice.
public protocol CBv2Qwen4BatchCapabilityConfiguring: AnyObject {
    var cbv2Qwen4BatchedAttentionEnabled: Bool { get }
    var cbv2Qwen4MaximumBatchRows: Int { get }
    func cbv2ConfigureQwen4BatchedAttention(enabled: Bool) throws
    func cbv2InstallQwen4BatchedAttention(enabled: Bool) throws
}

public enum CBv2Qwen4BatchPolicyError: Error, Equatable {
    case alreadySealed
    case unavailable
}

public enum CBv2Qwen4BatchPolicy {
    /// B7/B8 cross a separate MoE dispatch threshold on the real K=10
    /// artifact. They are outside the bounded candidate's qualification.
    public static let maximumCandidateRows = 4
}

extension CBv2Qwen4BatchCapabilityConfiguring {
    public var cbv2Qwen4MaximumBatchRows: Int {
        cbv2Qwen4BatchedAttentionEnabled ? CBv2Qwen4BatchPolicy.maximumCandidateRows : 1
    }
}

/// QSA decomposes attention after the batched projections. These views borrow
/// the actual sequence owners; they never copy KV, rebind the shared bank, or
/// replace the paged pool's write/read fence chain.
public protocol CBv2Qwen4BatchScopeProviding: CBv2AttendingLayerCache {
    func qwen4BeginBatchScope() -> CBv2Qwen4BatchScope
}

public final class CBv2Qwen4BatchScope {
    public let caches: [any CBv2AttendingLayerCache & CBv2Qwen4GatheredCache]
    private let rows: [CBv2SequenceKV]
    private let offsets: [Int]
    private let commit: (Int) -> Void
    private var finished = false

    init(
        rows: [CBv2SequenceKV],
        caches: [any CBv2AttendingLayerCache & CBv2Qwen4GatheredCache],
        commit: @escaping (Int) -> Void
    ) {
        precondition(rows.count > 1 && rows.count == caches.count)
        self.rows = rows
        self.offsets = rows.map(\.absoluteOffset)
        self.caches = caches
        self.commit = commit
    }

    /// The original bank remains bound to its entire cohort. Advance its
    /// device offsets exactly once after the singleton views wrote each row.
    @discardableResult
    public func finish(length: Int) -> Bool {
        precondition(!finished && length > 0)
        finished = true
        // A paged dtype/write failure deliberately stops later row writes.
        // Leave the parent offsets unchanged and retain the original fault;
        // checkedModelForward will reject the graph and retire the cohort.
        guard !caches.contains(where: { $0.qwen4HasWriteFault }) else { return false }
        precondition(zip(rows, offsets).allSatisfy { $0.absoluteOffset == $1 + length },
                     "Qwen4 row scope did not advance every KV owner exactly once")
        commit(length)
        return true
    }
}

/// Capacity is not a committed token count. Exact-width legacy/checkpoint
/// sidecars may establish a frontier; a padded bank always needs an explicit
/// one. The count can temporarily exceed KV while a verify window is built.
public enum CBv2Qwen4IndexerFrontier {
    static func evaluationState(_ row: CBv2SequenceKV) -> [MLXArray] {
        guard let row = row as? any CBv2Qwen4IndexerRow else { return [] }
        return [row.qwen4IndexKeys, row.qwen4IndexPositionIds, row.qwen4PooledIndexKeys].compactMap { $0 }
    }

    public static func covers(
        _ tokens: Int, count: Int?, keys: MLXArray?, positions: MLXArray?
    ) -> Bool {
        guard tokens >= 0 else { return false }
        if tokens == 0 { return true }
        guard let keys, let positions,
            keys.ndim == 3, (positions.ndim == 2 || positions.ndim == 3),
            keys.dim(1) >= tokens, positions.dim(-1) >= tokens
        else { return false }
        if let count {
            return count >= tokens && count <= keys.dim(1) && count <= positions.dim(-1)
        }
        return keys.dim(1) == tokens && positions.dim(-1) == tokens
    }
}
