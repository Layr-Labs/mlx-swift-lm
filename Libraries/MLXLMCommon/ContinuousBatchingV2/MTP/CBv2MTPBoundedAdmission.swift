import MLX

/// Immutable execution envelope, not a request or advertised context limit.
/// EngineV2 and SchedulerV2 copy their value-type configuration into let
/// properties. Raising either work limit therefore requires a new engine and
/// a new resolution before any work can use the larger envelope.
public struct CBv2MTPAllocationLimits: Sendable, Equatable {
    public let maximumPrefillTokens: Int
    public let maximumDraftTokens: Int

    public init(maximumPrefillTokens: Int, maximumDraftTokens: Int) {
        self.maximumPrefillTokens = maximumPrefillTokens
        self.maximumDraftTokens = maximumDraftTokens
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

    public init(
        resident: [CBv2MTPFixedBufferSpec], working: [CBv2MTPFixedBufferSpec], hostBytes: Int,
        hostBytesPerToken: Int = 0
    ) {
        self.resident = resident
        self.working = working
        self.hostBytes = hostBytes
        self.hostBytesPerToken = hostBytesPerToken
    }
}

/// Opt-in only. Nil refuses admission; it is not a fallback to an unpriced
/// state. The declaration covers request-owned roots and source-visible work,
/// not weights, target-forward activations or opaque kernel scratch. Existing
/// host activation/scratch safety reserves must remain in force.
public protocol CBv2MTPBoundedAllocationProviding: CBv2MTPDrafter {
    func boundedRequestAllocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec?
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
    /// Already included in the engine admission's auxiliary per-token rate.
    public let auxiliaryBytesPerToken: Int
    public var auxiliaryTokenGranularity: Int { 1 }
    public var auxiliaryTokenAllocationPadding: Int { 0 }

    init(
        limits: CBv2MTPAllocationLimits, residentBytes: Int, workingBytes: Int, hostBytes: Int,
        residentBytesPerToken: Int = 0, workingBytesPerToken: Int = 0, hostBytesPerToken: Int = 0,
        fixedBytesPerRequest: Int, auxiliaryBytesPerToken: Int = 0
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
    }

    /// The whole charge of a request holding `tokens` reserved tokens; nil on overflow.
    public func bytes(forTokens tokens: Int) -> Int? {
        let (growth, overflow) = auxiliaryBytesPerToken.multipliedReportingOverflow(by: tokens)
        guard !overflow, tokens >= 0 else { return nil }
        return CBv2MTPBoundedAdmission.add(fixedBytesPerRequest, growth)
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
        guard !spec.resident.isEmpty, !spec.working.isEmpty, spec.hostBytes >= 0,
            spec.hostBytesPerToken >= 0,
            (spec.resident + spec.working).allSatisfy({
                $0.logicalBytesPerToken >= 0 && $0.allocationCount > 0
                    && ($0.logicalBytes > 0 || $0.logicalBytesPerToken > 0)
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
        guard let resident = project(spec.resident), let working = project(spec.working) else {
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
                fixedBytesPerRequest: total, auxiliaryBytesPerToken: perToken))
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
