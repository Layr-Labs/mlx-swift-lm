import MLX

/// Immutable execution envelope, not a request or advertised context limit.
/// EngineV2 and SchedulerV2 copy their value-type configuration into let
/// properties. Raising either work limit therefore requires a new engine and
/// a new resolution before any work can use the larger envelope.
public struct CBv2MTPAllocationLimits: Sendable, Equatable {
    public let maximumPrefillTokens: Int
    public let maximumDraftTokens: Int
    /// Owner-wide bounds of one speculative round over every decoding
    /// request: the captured window's positions and the draft steps, the
    /// largest totals the depth policy allows at any decode width. One
    /// request at `maximumDraftTokens` when not given.
    public let maximumWindowPositions: Int
    public let maximumRoundDraftSteps: Int
    /// Request-sized history copies one step may hold at once over every
    /// request (`historyCopies`). One request's round when not given.
    public let maximumHistoryCopies: Int
    /// The deepest draft chain any round runs (`roundDepth`), at most
    /// `maximumDraftTokens`; a depth table can keep it lower. That maximum
    /// when not given.
    public let maximumRoundDepth: Int

    public init(
        maximumPrefillTokens: Int, maximumDraftTokens: Int,
        maximumWindowPositions: Int? = nil, maximumRoundDraftSteps: Int? = nil,
        maximumHistoryCopies: Int? = nil, maximumRoundDepth: Int? = nil
    ) {
        self.maximumPrefillTokens = maximumPrefillTokens
        self.maximumDraftTokens = maximumDraftTokens
        self.maximumWindowPositions = maximumWindowPositions ?? maximumDraftTokens + 1
        self.maximumRoundDraftSteps = maximumRoundDraftSteps ?? maximumDraftTokens
        self.maximumHistoryCopies = maximumHistoryCopies ?? maximumDraftTokens + 1
        self.maximumRoundDepth = min(maximumRoundDepth ?? maximumDraftTokens, maximumDraftTokens)
    }

    /// The deepest draft chain a round runs on an owner of `requests`
    /// concurrent requests: the largest depth the policy fixes for a decode
    /// width up to `requests`, the most rows one plan holds.
    public static func roundDepth(_ config: CBv2MTPConfig, requests: Int) -> Int {
        (1 ... max(1, requests)).reduce(0) { max($0, depth(config, rows: $1)) }
    }

    /// The owner-wide bounds of `config`: the largest window positions and
    /// draft steps of any planned round (`plannedRounds`).
    public static func roundBounds(_ config: CBv2MTPConfig) -> (positions: Int, steps: Int) {
        var positions = 0
        var steps = 0
        for round in plannedRounds(config) {
            positions = max(positions, round.drafting * (round.depth + 1))
            steps = max(steps, round.drafting * round.depth)
        }
        return (positions, steps)
    }

    /// Recurrent generations two live captured windows need beyond the
    /// `perRequest` generations every request already holds: a committed
    /// prefix is a view retaining its whole previous window while the next
    /// window is built, so N drafting rows hold 2 x N x (depth + 1); the
    /// largest excess over N x `perRequest` of any planned round.
    public static func capturedWindowGenerations(_ config: CBv2MTPConfig, perRequest: Int) -> Int {
        var generations = 0
        for round in plannedRounds(config) {
            let windows = 2 * round.drafting * (round.depth + 1)
            generations = max(generations, windows - round.drafting * perRequest)
        }
        return generations
    }

    /// The round of each planned decode width N: at most the speculative batch
    /// of its N rows drafts, at the depth the policy fixes for N (the maximum
    /// depth when adaptive). The depth policy counts every decode row, media
    /// rows that do not draft included, so a plan can be wider than the
    /// speculative batch. Widths run to the larger of the batch and the depth
    /// table; every wider plan repeats the last round.
    private static func plannedRounds(_ config: CBv2MTPConfig) -> [(drafting: Int, depth: Int)] {
        let batch = max(1, config.maxSpeculativeBatch)
        let widest = max(batch, config.fixedDraftTokensByDecodeRows?.count ?? 0)
        return (1 ... widest).map { rows in
            (drafting: min(rows, batch), depth: depth(config, rows: rows))
        }
    }

    /// Whole-history copies of request-owned assistant state that MLX may
    /// hold at once when a write cannot donate its buffer (the previous step's
    /// attention, or the engine's retained roots of a pipelined step, still
    /// holds it): each speculating row up to one per draft step plus one, and
    /// every other request (prefilling or decoding plain) one. The largest
    /// total at any decode width up to `requests`, at the depth the policy
    /// fixes for that width.
    public static func historyCopies(_ config: CBv2MTPConfig, requests: Int) -> Int {
        let speculative = max(1, config.maxSpeculativeBatch)
        let requests = max(1, requests)
        var copies = 0
        for rows in 1 ... requests {
            let drafting = min(rows, speculative)
            copies = max(
                copies, drafting * (depth(config, rows: rows) + 1) + (requests - drafting))
        }
        return copies
    }

    private static func depth(_ config: CBv2MTPConfig, rows: Int) -> Int {
        var depth = config.fixedDraftTokens ?? config.maxDraftTokens
        if config.fixedDraftTokens != nil, let table = config.fixedDraftTokensByDecodeRows,
            !table.isEmpty
        {
            depth = min(depth, table[min(rows, table.count) - 1])
        }
        return depth
    }
}

/// Independent allocations of the same maximum logical extent. The allocator
/// bound is applied to ONE buffer, then multiplied by allocationCount.
/// `logicalBytesPerToken` declares a buffer that grows with the request: its
/// extent is `logicalBytes + logicalBytesPerToken * n` for a request holding
/// `n` reserved tokens, and admission charges it per reserved token.
public struct CBv2MTPFixedBufferSpec: Sendable, Equatable {
    public let logicalBytes: Int
    public let allocationCount: Int
    public let logicalBytesPerToken: Int

    public init(logicalBytes: Int, allocationCount: Int = 1, logicalBytesPerToken: Int = 0) {
        self.logicalBytes = logicalBytes
        self.allocationCount = allocationCount
        self.logicalBytesPerToken = logicalBytesPerToken
    }

    /// The concrete extent of one buffer at `tokens` reserved tokens.
    public func logicalBytes(atTokens tokens: Int) -> Int? {
        let (growth, overflow) = logicalBytesPerToken.multipliedReportingOverflow(by: tokens)
        guard !overflow, tokens >= 0 else { return nil }
        return CBv2MTPBoundedAdmission.add(logicalBytes, growth)
    }
}

public struct CBv2MTPBoundedAllocationSpec: Sendable, Equatable {
    public let resident: [CBv2MTPFixedBufferSpec]
    public let working: [CBv2MTPFixedBufferSpec]
    public let hostBytes: Int
    public let hostBytesPerToken: Int
    /// Work of one round or one prompt-history priming that exists once per
    /// engine, whatever the number of requests. Not part of the per-request
    /// charge: the host reserves `CBv2ResolvedMTPAdmission.ownerBytes` once,
    /// before the first request, until retirement.
    public let ownerWorking: [CBv2MTPFixedBufferSpec]

    public init(
        resident: [CBv2MTPFixedBufferSpec], working: [CBv2MTPFixedBufferSpec], hostBytes: Int,
        hostBytesPerToken: Int = 0, ownerWorking: [CBv2MTPFixedBufferSpec] = []
    ) {
        self.resident = resident
        self.working = working
        self.hostBytes = hostBytes
        self.hostBytesPerToken = hostBytesPerToken
        self.ownerWorking = ownerWorking
    }
}

/// Opt-in only. Nil refuses admission; it is not a fallback to an unpriced
/// state. The declaration covers request-owned roots and source-visible work,
/// not weights, target-forward activations or opaque kernel scratch. Existing
/// host activation/scratch safety reserves must remain in force.
public protocol CBv2MTPBoundedAllocationProviding: CBv2MTPDrafter {
    func boundedRequestAllocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec?
}

/// A bounded drafter whose request-owned state also grows with each request's
/// own sequence. `boundedRequestAllocation` then declares only what does not
/// grow, and these buffers are priced per reserved token by the engine's
/// auxiliary ledger, so a request pays for its own length rather than the
/// drafter's configured maximum. Padding covers rows a round stages beyond the
/// reserved tokens. Nil or empty refuses admission.
public protocol CBv2MTPSequenceScaledAllocationProviding: CBv2MTPBoundedAllocationProviding {
    func boundedRequestTokenAllocation(limits: CBv2MTPAllocationLimits)
        -> [CBv2AuxiliaryAllocationSpec]?
}

/// A request's MTP charge is `fixedBytesPerRequest + auxiliaryBytesPerToken *
/// reservedTokens`; each pool's capacity follows the same linear rule.
public struct CBv2ResolvedMTPAdmission: Sendable, Equatable {
    public let limits: CBv2MTPAllocationLimits
    public let residentBytes: Int
    public let workingBytes: Int
    public let hostBytes: Int
    public let residentBytesPerToken: Int
    public let workingBytesPerToken: Int
    public let hostBytesPerToken: Int
    /// This amount is already included in EngineV2.resolvedFixedBytesPerRequest.
    /// Consumers must replace the legacy MTP variable term, never add it twice.
    public let fixedBytesPerRequest: Int
    /// Already included in the engine admission's auxiliary per-token rate:
    /// the growing buffers of the fixed declaration and, for a sequence-scaled
    /// declaration, its projected rate.
    public internal(set) var auxiliaryBytesPerToken: Int
    public var auxiliaryTokenGranularity: Int { 1 }
    public var auxiliaryTokenAllocationPadding: Int { 0 }
    /// Added to the fixed charge, makes `auxiliaryBytesPerToken` x tokens an
    /// upper bound of the projected per-token charge (padding, block rounding
    /// and allocator extra) for consumers that price linearly.
    public internal(set) var auxiliaryLinearSlackBytes: Int = 0
    /// Bytes the host reserves once per engine: the declaration's owner-wide
    /// work and, when EngineV2 sizes captured verify windows owner-wide, their
    /// recurrent generations. Never part of `fixedBytesPerRequest`.
    public internal(set) var ownerBytes: Int

    init(
        limits: CBv2MTPAllocationLimits, residentBytes: Int, workingBytes: Int, hostBytes: Int,
        residentBytesPerToken: Int = 0, workingBytesPerToken: Int = 0, hostBytesPerToken: Int = 0,
        fixedBytesPerRequest: Int, auxiliaryBytesPerToken: Int = 0, ownerBytes: Int = 0
    ) {
        self.limits = limits
        self.residentBytes = residentBytes
        self.workingBytes = workingBytes
        self.hostBytes = hostBytes
        self.residentBytesPerToken = residentBytesPerToken
        self.workingBytesPerToken = workingBytesPerToken
        self.hostBytesPerToken = hostBytesPerToken
        self.fixedBytesPerRequest = fixedBytesPerRequest
        self.auxiliaryBytesPerToken = auxiliaryBytesPerToken
        self.ownerBytes = ownerBytes
    }

    /// The whole charge of a request holding `tokens` reserved tokens; nil on overflow.
    public func bytes(forTokens tokens: Int) -> Int? {
        let (growth, overflow) = auxiliaryBytesPerToken.multipliedReportingOverflow(by: tokens)
        guard !overflow, tokens >= 0, let fixed = CBv2MTPBoundedAdmission.add(
            fixedBytesPerRequest, auxiliaryLinearSlackBytes)
        else { return nil }
        return CBv2MTPBoundedAdmission.add(fixed, growth)
    }

    /// One pool's capacity at `tokens` reserved tokens; nil on overflow.
    func limit(_ pool: CBv2AdmittedMTPContext.Pool, tokens: Int) -> Int? {
        let (base, perToken) =
            switch pool {
            case .resident: (residentBytes, residentBytesPerToken)
            case .working: (workingBytes, workingBytesPerToken)
            case .host: (hostBytes, hostBytesPerToken)
            }
        let (growth, overflow) = perToken.multipliedReportingOverflow(by: tokens)
        guard !overflow, tokens >= 0 else { return nil }
        return CBv2MTPBoundedAdmission.add(base, growth)
    }
}

public enum CBv2MTPAdmissionRefusal: Error, Sendable, Equatable {
    case invalidLimits
    case missingDeclaration
    case invalidDeclaration
    case allocatorPolicyUnavailable
    case unrepresentableAllocation
    case invalidExistingCharge
    case chargeOverflow
}

public enum CBv2MTPAdmissionResolution: Sendable, Equatable {
    case bounded(CBv2ResolvedMTPAdmission)
    case unavailable(CBv2MTPAdmissionRefusal)
}

/// Construction-only arithmetic. No request-token loop, allocation, native
/// evaluation or materialization credit. Both backend policies consume the
/// resulting fixed charge through AdmissionV2's common non-backend ledger.
enum CBv2MTPBoundedAdmission {
    static func resolve(
        spec: CBv2MTPBoundedAllocationSpec?, limits: CBv2MTPAllocationLimits,
        policy: AllocationFootprintPolicy?
    ) -> CBv2MTPAdmissionResolution {
        guard let policy else { return .unavailable(.allocatorPolicyUnavailable) }
        return resolve(
            spec: spec, limits: limits, upperBound: policy.upperBound(byteCount:),
            maximumExtraBytes: policy.maximumExtraBytes)
    }

    // Scalar injection is internal and used only by arithmetic tests. Public
    // consumers receive the engine's captured real allocator resolution.
    /// A buffer without per-token growth is charged its exact allocator
    /// bound. A growing buffer of extent `a + b*n` is charged `a + E` fixed and
    /// `b` per reserved token, where `E` is the allocator's maximum extra bytes
    /// (`B(x) <= x + E`), so the linear charge bounds every request length.
    static func resolve(
        spec: CBv2MTPBoundedAllocationSpec?, limits: CBv2MTPAllocationLimits,
        upperBound: (Int) -> Int?, maximumExtraBytes: Int? = nil
    ) -> CBv2MTPAdmissionResolution {
        guard limits.maximumPrefillTokens > 0, limits.maximumDraftTokens > 0 else {
            return .unavailable(.invalidLimits)
        }
        guard let spec else { return .unavailable(.missingDeclaration) }
        guard !spec.resident.isEmpty, !spec.working.isEmpty || !spec.ownerWorking.isEmpty,
            spec.hostBytes >= 0, spec.hostBytesPerToken >= 0,
            (spec.resident + spec.working).allSatisfy({
                $0.logicalBytesPerToken >= 0 && $0.allocationCount > 0
                    && ($0.logicalBytes > 0 || $0.logicalBytesPerToken > 0)
            }),
            // Owner-wide work exists once per engine and cannot grow per request.
            spec.ownerWorking.allSatisfy({
                $0.logicalBytes > 0 && $0.allocationCount > 0 && $0.logicalBytesPerToken == 0
            })
        else {
            return .unavailable(.invalidDeclaration)
        }
        func project(_ buffers: [CBv2MTPFixedBufferSpec]) -> (fixed: Int, perToken: Int)? {
            var fixed = 0
            var perToken = 0
            for buffer in buffers {
                let base: Int
                if buffer.logicalBytesPerToken == 0 {
                    guard let bound = upperBound(buffer.logicalBytes),
                        bound >= buffer.logicalBytes
                    else { return nil }
                    base = bound
                } else {
                    guard let extra = maximumExtraBytes, extra >= 0,
                        let bound = add(buffer.logicalBytes, extra)
                    else { return nil }
                    base = bound
                }
                let (all, multiplyOverflow) = base.multipliedReportingOverflow(
                    by: buffer.allocationCount)
                let (growth, growthOverflow) = buffer.logicalBytesPerToken
                    .multipliedReportingOverflow(by: buffer.allocationCount)
                guard !multiplyOverflow, !growthOverflow, let sum = add(fixed, all),
                    let rate = add(perToken, growth)
                else { return nil }
                fixed = sum
                perToken = rate
            }
            return (fixed, perToken)
        }
        guard let resident = project(spec.resident), let working = project(spec.working),
            let owner = project(spec.ownerWorking)
        else {
            return .unavailable(.unrepresentableAllocation)
        }
        guard let device = add(resident.fixed, working.fixed),
            let total = add(device, spec.hostBytes),
            let devicePerToken = add(resident.perToken, working.perToken),
            let perToken = add(devicePerToken, spec.hostBytesPerToken)
        else { return .unavailable(.chargeOverflow) }
        return .bounded(
            .init(
                limits: limits, residentBytes: resident.fixed, workingBytes: working.fixed,
                hostBytes: spec.hostBytes, residentBytesPerToken: resident.perToken,
                workingBytesPerToken: working.perToken, hostBytesPerToken: spec.hostBytesPerToken,
                fixedBytesPerRequest: total, auxiliaryBytesPerToken: perToken,
                ownerBytes: owner.fixed))
    }

    @discardableResult
    static func apply(_ resolution: CBv2MTPAdmissionResolution, to config: inout AdmissionV2.Config)
        -> CBv2MTPAdmissionResolution
    {
        guard case .bounded(let value) = resolution else {
            config.fixedBytesPerRequest = Int.max
            return resolution
        }
        guard config.fixedBytesPerRequest >= 0, config.auxiliaryBytesPerToken >= 0,
            config.auxiliaryTokenGranularity > 0, config.auxiliaryTokenAllocationPadding >= 0
        else {
            config.fixedBytesPerRequest = Int.max
            return .unavailable(.invalidExistingCharge)
        }
        guard let total = add(config.fixedBytesPerRequest, value.fixedBytesPerRequest),
            let rate = add(config.auxiliaryBytesPerToken, value.auxiliaryBytesPerToken)
        else {
            config.fixedBytesPerRequest = Int.max
            return .unavailable(.chargeOverflow)
        }
        config.fixedBytesPerRequest = total
        // Per-token MTP growth rides the ordinary reserved-token charge; any
        // caller granularity or padding only rounds it up.
        config.auxiliaryBytesPerToken = rate
        // Existing scalar/projection fields are CALLER charges in this branch;
        // the inferred legacy drafter rate was never installed. Preserve them.
        return resolution
    }

    /// Installs a sequence-scaled declaration beside the fixed one. The
    /// projection slot is single-owner, so caller auxiliary charges refuse.
    static func applyTokenAllocation(
        specs: [CBv2AuxiliaryAllocationSpec]?, policy: AllocationFootprintPolicy?,
        resolution: CBv2MTPAdmissionResolution, to config: inout AdmissionV2.Config
    ) -> CBv2MTPAdmissionResolution {
        guard case .bounded(var value) = resolution else { return resolution }
        guard let specs, !specs.isEmpty, let policy,
            config.auxiliaryBytesPerToken == 0, config.auxiliaryAllocationProjection == nil,
            let projection = CBv2AuxiliaryAllocationProjection(policy: policy, buffers: specs),
            let rate = projection.linearBytesPerToken, rate > 0,
            let slack = projection.linearSlackBytes
        else {
            config.fixedBytesPerRequest = Int.max
            return .unavailable(.invalidDeclaration)
        }
        config.auxiliaryBytesPerToken = rate
        config.auxiliaryTokenGranularity = 1
        config.auxiliaryTokenAllocationPadding = 0
        config.auxiliaryAllocationProjection = projection
        value.auxiliaryBytesPerToken = rate
        value.auxiliaryLinearSlackBytes = slack
        return .bounded(value)
    }

    /// TargetAuxiliaryAdmission owns a single projection slot. Without target
    /// extras leave the caller's exact projection untouched. With extras, use
    /// its documented maximum one-token growth G as a conservative scalar
    /// floor: P(0)=0 and P(n)-P(n-1)<=G imply P(n)<=n*G for all representable
    /// nonnegative n. This includes padded first allocation and block crossings.
    /// Admission checks each product at query time; overflow refuses work.
    static func preserveCallerProjectionForTarget(
        model: any CBv2SteppableModel, config: inout AdmissionV2.Config
    ) {
        guard let target = model as? any CBv2TargetAuxiliaryAllocationProviding,
            let specs = target.cbv2TargetAuxiliaryAllocationSpecs, !specs.isEmpty,
            let projection = config.auxiliaryAllocationProjection
        else { return }
        // The current projection has zero base. Keep this explicit so a future
        // nonzero-base implementation is charged, not silently discarded.
        guard let base = projection.bytes(forTokens: 0), base >= 0,
            projection.maximumGrowthBytes >= 0,
            let fixed = add(config.fixedBytesPerRequest, base)
        else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        config.fixedBytesPerRequest = fixed
        config.auxiliaryBytesPerToken = max(
            config.auxiliaryBytesPerToken, projection.maximumGrowthBytes)
    }

    static func add(_ a: Int, _ b: Int) -> Int? {
        let (sum, overflow) = a.addingReportingOverflow(b)
        return a >= 0 && b >= 0 && !overflow ? sum : nil
    }
}
