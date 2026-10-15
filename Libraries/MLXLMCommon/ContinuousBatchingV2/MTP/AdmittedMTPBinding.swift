// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

public enum CBv2MTPBindingError: Error, Equatable {
    case missingAdmission, staleRequest, identityMismatch, changedLimit, capacityExceeded
}

/// Optional refinement. Existing drafters retain their original configuration path.
public protocol CBv2AdmittedMTPRequestStatefulDrafter: CBv2MTPRequestStatefulDrafter {
    func configureRequestState(
        _ requestState: any CBv2MTPRequestState, admission: CBv2AdmittedMTPContext
    ) throws
}

final class CBv2MTPAdmissionRecord {
    let id: CBv2RequestID
    let generation = UUID()
    let target: ObjectIdentifier
    let drafter: ObjectIdentifier
    let maximumInputs: Int
    let resolved: CBv2ResolvedMTPAdmission
    weak var context: CBv2AdmittedMTPContext?
    var checkpointImport: CBv2MTPCheckpointImportProof?
    /// Admission mutates this only with the matching real stage transfer or
    /// rollback; callers cannot use a phase flag in place of the proof owner.
    var checkpointImportCommitted = false
    init(
        id: CBv2RequestID, target: ObjectIdentifier, drafter: ObjectIdentifier,
        maximumInputs: Int, resolved: CBv2ResolvedMTPAdmission
    ) {
        self.id = id
        self.target = target
        self.drafter = drafter
        self.maximumInputs = maximumInputs
        self.resolved = resolved
    }
}

/// Issued only by Common from the active engine's existing fixed MTP charge.
/// Retaining this value alone grants no extra memory. Allocation leases retain
/// the whole prepaid charge after request retirement until their last owner dies.
public final class CBv2AdmittedMTPContext: @unchecked Sendable {
    public enum Pool: Sendable, Equatable { case resident, working, host }
    public let requestID: CBv2RequestID
    public let requestGeneration: UUID
    public let targetIdentity: ObjectIdentifier
    public let drafterIdentity: ObjectIdentifier
    public let maximumInputs: Int
    public let resolved: CBv2ResolvedMTPAdmission
    private let admission: AdmissionV2
    private let policy: AllocationFootprintPolicy
    private let lock = NSLock()
    private var active = true
    private var detached = false
    private var allocations = 0
    private var resident = 0
    private var working = 0
    private var host = 0

    init(admission: AdmissionV2, record: CBv2MTPAdmissionRecord, policy: AllocationFootprintPolicy)
    {
        self.admission = admission
        self.policy = policy
        requestID = record.id
        requestGeneration = record.generation
        targetIdentity = record.target
        drafterIdentity = record.drafter
        maximumInputs = record.maximumInputs
        resolved = record.resolved
    }

    public func validate(target: ObjectIdentifier, drafter: ObjectIdentifier, maximumInputs: Int)
        throws
    {
        guard target == targetIdentity, drafter == drafterIdentity else {
            throw CBv2MTPBindingError.identityMismatch
        }
        guard maximumInputs == self.maximumInputs else { throw CBv2MTPBindingError.changedLimit }
        try admission.validateMTPBinding(self)
    }

    /// Each independently allocated buffer receives its own captured allocator
    /// bound. Host declarations are bytes, not Metal allocator requests.
    public func reserve(_ pool: Pool, buffers: [CBv2MTPFixedBufferSpec]) throws
        -> CBv2MTPAllocationLease
    {
        try admission.validateMTPBinding(self)
        var bytes = 0
        for buffer in buffers {
            guard buffer.logicalBytes > 0, buffer.allocationCount > 0,
                let bound = pool == .host
                    ? buffer.logicalBytes : policy.upperBound(byteCount: buffer.logicalBytes)
            else { throw CBv2MTPBindingError.capacityExceeded }
            let product = bound.multipliedReportingOverflow(by: buffer.allocationCount)
            let sum = bytes.addingReportingOverflow(product.partialValue)
            guard !product.overflow && !sum.overflow else {
                throw CBv2MTPBindingError.capacityExceeded
            }
            bytes = sum.partialValue
        }
        lock.lock()
        defer { lock.unlock() }
        guard active else { throw CBv2MTPBindingError.staleRequest }
        let used: Int
        let limit: Int
        switch pool {
        case .resident:
            used = resident
            limit = resolved.residentBytes
        case .working:
            used = working
            limit = resolved.workingBytes
        case .host:
            used = host
            limit = resolved.hostBytes
        }
        guard bytes <= limit - used else { throw CBv2MTPBindingError.capacityExceeded }
        switch pool {
        case .resident: resident += bytes
        case .working: working += bytes
        case .host: host += bytes
        }
        allocations += 1
        return CBv2MTPAllocationLease(context: self, pool: pool, bytes: bytes)
    }

    public var allocatedBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return resident + working + host
    }

    /// Retired contexts permit draining their existing work, never new allocation
    /// or publication. The retained allocation leases still carry their charge.
    public var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    // Called with Admission's lock held. No callback into Admission here.
    func retire() -> Int {
        lock.lock()
        defer { lock.unlock() }
        active = false
        detached = allocations > 0
        return detached ? resolved.fixedBytesPerRequest : 0
    }

    fileprivate func release(_ pool: Pool, bytes: Int) {
        lock.lock()
        switch pool {
        case .resident: resident -= bytes
        case .working: working -= bytes
        case .host: host -= bytes
        }
        allocations -= 1
        let refund = detached && allocations == 0
        if refund { detached = false }
        lock.unlock()
        if refund { admission.releaseDetachedMTPBinding(bytes: resolved.fixedBytesPerRequest) }
    }
}

/// No public constructor or early-release operation. Arrays/checkpoints retain
/// this object until their last native reader and owner have retired.
public final class CBv2MTPAllocationLease: @unchecked Sendable {
    public let bytes: Int
    private let context: CBv2AdmittedMTPContext
    private let pool: CBv2AdmittedMTPContext.Pool
    fileprivate init(context: CBv2AdmittedMTPContext, pool: CBv2AdmittedMTPContext.Pool, bytes: Int)
    {
        self.context = context
        self.pool = pool
        self.bytes = bytes
    }
    deinit { context.release(pool, bytes: bytes) }
}

final class CBv2MTPAdmissionIssuer {
    let admission: AdmissionV2
    let target: ObjectIdentifier
    let drafter: ObjectIdentifier
    let resolved: CBv2ResolvedMTPAdmission
    let policy: AllocationFootprintPolicy
    init(
        admission: AdmissionV2, target: ObjectIdentifier, drafter: ObjectIdentifier,
        resolved: CBv2ResolvedMTPAdmission, policy: AllocationFootprintPolicy
    ) {
        self.admission = admission
        self.target = target
        self.drafter = drafter
        self.resolved = resolved
        self.policy = policy
    }
    func context(id: CBv2RequestID, maximumInputs: Int) throws -> CBv2AdmittedMTPContext {
        if let existing = try admission.existingImportedMTPBinding(
            id: id, maximumInputs: maximumInputs, target: target, drafter: drafter)
        {
            return existing
        }
        return try admission.issueMTPBinding(
            id: id, maximumInputs: maximumInputs, target: target,
            drafter: drafter, resolved: resolved, policy: policy)
    }

    func context(
        id: CBv2RequestID, maximumInputs: Int,
        importProof: CBv2MTPCheckpointImportProof
    ) throws -> CBv2AdmittedMTPContext {
        guard importProof.authorizes(self) else { throw CBv2MTPBindingError.identityMismatch }
        return try admission.issueImportedMTPBinding(
            id: id, maximumInputs: maximumInputs,
            target: target, drafter: drafter, resolved: resolved, policy: policy, proof: importProof
        )
    }
}
