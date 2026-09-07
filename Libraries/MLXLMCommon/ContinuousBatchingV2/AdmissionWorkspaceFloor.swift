/// Immutable upper bound for one request's overlapping attention workspaces.
/// It must cover every supported chunk/decode shape at the reserved length,
/// including two submitted steps. It is physical scratch, not KV storage.
public struct CBv2RequestWorkspaceProjection: Sendable {
    private let project: @Sendable (Int) -> Int?
    private let projectAggregate: (@Sendable (Int, Int) -> Int?)?

    public init(bytesForTokens: @escaping @Sendable (Int) -> Int?) {
        project = bytesForTokens
        projectAggregate = nil
    }

    /// The aggregate must bound the sum of individual allowances for every
    /// split of totalTokens into at most maximumRequests positive lengths.
    /// Existing and retired owners are charged separately by admission.
    public init(
        bytesForTokens: @escaping @Sendable (Int) -> Int?,
        bytesForAggregate: @escaping @Sendable (Int, Int) -> Int?
    ) {
        project = bytesForTokens
        projectAggregate = bytesForAggregate
    }

    func bytes(forTokens tokens: Int) -> Int? {
        guard tokens >= 0 else { return nil }
        if tokens == 0 { return 0 }
        guard let value = project(tokens), value >= 0 else { return nil }
        return value
    }

    func bytes(totalTokens: Int, maximumRequests: Int) -> Int? {
        guard totalTokens >= 0, maximumRequests >= 0 else { return nil }
        if totalTokens == 0 { return 0 }
        guard maximumRequests > 0 else { return nil }
        let count = min(totalTokens, maximumRequests)
        if let projectAggregate {
            guard let value = projectAggregate(totalTokens, count), value >= 0 else { return nil }
            return value
        }
        // Generic projections have no algebraic contract beyond monotonicity.
        // Each positive request is at most the aggregate length; retain the
        // conservative fallback until a projection supplies a tighter proof.
        guard let single = bytes(forTokens: totalTokens) else { return nil }
        let (result, overflow) = single.multipliedReportingOverflow(by: count)
        return overflow ? nil : result
    }
}

/// Prepaid request allowances and actual leased workspaces are independent
/// owners. Live leases consume existing credit once; retiring credit stays
/// charged until GPU work releases it, so a new request cannot reuse it early.
struct CBv2AdmissionWorkspaceFloor {
    var prepaidBytes = 0
    var leasedBytes = 0
    private(set) var retiredBytes = 0

    /// Credits retired while any workspace is live cannot be reused for a
    /// newly admitted request's future workspace. Conservatively retain that
    /// part of the old credit until outstanding leased bytes fall below it.
    mutating func replacePrepaid(_ bytes: Int) {
        precondition(bytes >= 0)
        if bytes < prepaidBytes {
            let released = prepaidBytes - bytes
            retiredBytes += min(released, max(0, leasedBytes - retiredBytes))
        }
        prepaidBytes = bytes
    }

    func replacingPrepaid(_ bytes: Int) -> Self {
        var result = self
        result.replacePrepaid(bytes)
        return result
    }

    mutating func releaseLease(_ bytes: Int) {
        precondition(bytes >= 0 && bytes <= leasedBytes)
        leasedBytes -= bytes
        retiredBytes = min(retiredBytes, leasedBytes)
    }

    func overhead(prepaid: Int? = nil, leased: Int? = nil) -> Int {
        let state = prepaid.map(replacingPrepaid) ?? self
        return max(state.retiredBytes, (leased ?? state.leasedBytes) - state.prepaidBytes)
    }
}
