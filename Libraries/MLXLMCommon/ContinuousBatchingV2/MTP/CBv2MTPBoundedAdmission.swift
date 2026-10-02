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
public struct CBv2MTPFixedBufferSpec: Sendable, Equatable {
    public let logicalBytes: Int
    public let allocationCount: Int

    public init(logicalBytes: Int, allocationCount: Int = 1) {
        self.logicalBytes = logicalBytes
        self.allocationCount = allocationCount
    }
}

public struct CBv2MTPBoundedAllocationSpec: Sendable, Equatable {
    public let resident: [CBv2MTPFixedBufferSpec]
    public let working: [CBv2MTPFixedBufferSpec]
    public let hostBytes: Int

    public init(
        resident: [CBv2MTPFixedBufferSpec], working: [CBv2MTPFixedBufferSpec], hostBytes: Int
    ) {
        self.resident = resident
        self.working = working
        self.hostBytes = hostBytes
    }
}

/// Opt-in only. Nil refuses admission; it is not a fallback to an unpriced
/// state. The declaration covers request-owned roots and source-visible work,
/// not weights, target-forward activations or opaque kernel scratch. Existing
/// host activation/scratch safety reserves must remain in force.
public protocol CBv2MTPBoundedAllocationProviding: CBv2MTPDrafter {
    func boundedRequestAllocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec?
}

public struct CBv2ResolvedMTPAdmission: Sendable, Equatable {
    public let limits: CBv2MTPAllocationLimits
    public let residentBytes: Int
    public let workingBytes: Int
    public let hostBytes: Int
    /// This amount is already included in EngineV2.resolvedFixedBytesPerRequest.
    /// Consumers must replace the legacy MTP variable term, never add it twice.
    public let fixedBytesPerRequest: Int
    public var auxiliaryBytesPerToken: Int { 0 }
    public var auxiliaryTokenGranularity: Int { 1 }
    public var auxiliaryTokenAllocationPadding: Int { 0 }
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
        return resolve(spec: spec, limits: limits, upperBound: policy.upperBound(byteCount:))
    }

    // Scalar injection is internal and used only by arithmetic tests. Public
    // consumers receive the engine's captured real allocator resolution.
    static func resolve(
        spec: CBv2MTPBoundedAllocationSpec?, limits: CBv2MTPAllocationLimits,
        upperBound: (Int) -> Int?
    ) -> CBv2MTPAdmissionResolution {
        guard limits.maximumPrefillTokens > 0, limits.maximumDraftTokens > 0 else {
            return .unavailable(.invalidLimits)
        }
        guard let spec else { return .unavailable(.missingDeclaration) }
        guard !spec.resident.isEmpty, !spec.working.isEmpty, spec.hostBytes >= 0,
            (spec.resident + spec.working).allSatisfy({
                $0.logicalBytes > 0 && $0.allocationCount > 0
            })
        else {
            return .unavailable(.invalidDeclaration)
        }
        func project(_ buffers: [CBv2MTPFixedBufferSpec]) -> Int? {
            var total = 0
            for buffer in buffers {
                guard let bound = upperBound(buffer.logicalBytes), bound >= buffer.logicalBytes
                else { return nil }
                let (all, multiplyOverflow) = bound.multipliedReportingOverflow(
                    by: buffer.allocationCount)
                guard !multiplyOverflow, let sum = add(total, all) else { return nil }
                total = sum
            }
            return total
        }
        guard let resident = project(spec.resident), let working = project(spec.working) else {
            return .unavailable(.unrepresentableAllocation)
        }
        guard let device = add(resident, working), let total = add(device, spec.hostBytes) else {
            return .unavailable(.chargeOverflow)
        }
        return .bounded(
            .init(
                limits: limits, residentBytes: resident, workingBytes: working,
                hostBytes: spec.hostBytes, fixedBytesPerRequest: total))
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
        guard let total = add(config.fixedBytesPerRequest, value.fixedBytesPerRequest) else {
            config.fixedBytesPerRequest = Int.max
            return .unavailable(.chargeOverflow)
        }
        config.fixedBytesPerRequest = total
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
