import Foundation

/// Native blocks share the existing AdmissionV2 physical-floor and process
/// ledger. Nominal request pages overlap the pool's actual backing; canvas,
/// snapshot and transfer obligations remain additive. Only its owning native
/// engine queue may materialize or retire backend rows.
public final class CBv2NativeBlockPagedMemory: @unchecked Sendable {
    package let backend: PagedKVBackend
    package let admission: AdmissionV2
    public let usesProcessMemoryOwner: Bool
    private let lock = NSLock()
    private var sharedReservation: CBv2CheckpointReservation?
    private var sharedBytes = 0
    private var engineClaimed = false

    package init(
        layerKinds: [CBv2LayerKind], configuration: PagedKVPoolConfig,
        processMemoryOwner: (any CBv2ProcessMemoryOwner)?
    ) throws {
        guard configuration.segmentSizeBytes != nil, configuration.capacityBytes > 0,
            let dtypes = configuration.layerDTypes, dtypes.count == layerKinds.count
        else { throw CBv2NativeBlockError.invalidConfiguration }
        backend = try PagedKVBackend(layerKinds: layerKinds, config: configuration)
        admission = AdmissionV2(
            layerKinds: layerKinds, bytesCapacity: configuration.capacityBytes,
            config: .init(
                watermarkFraction: 0, elementBytes: configuration.dtype.size,
                layerElementBytes: dtypes.map(\.size)),
            residency: backend.kvResidency, processMemoryOwner: processMemoryOwner)
        backend.pool.bindAdmission(admission)
        usesProcessMemoryOwner = processMemoryOwner != nil
    }

    /// Immutable-geometry projection; safe on the submitting thread. Only the
    /// target page portion may overlap shared pool backing.
    package func nominalBytes(tokens: Int) -> Int { admission.allocatedBytes(forTokens: tokens) }

    func claimEngine(capacity: Int) throws {
        try lock.withLock {
            guard !engineClaimed, admission.bytesCapacity == capacity else {
                throw CBv2NativeBlockError.invalidConfiguration
            }
            engineClaimed = true
        }
    }

    package func reserve(id: CBv2RequestID, tokens: Int, totalBytes: Int) throws {
        let nominal = nominalBytes(tokens: tokens)
        guard tokens > 0, nominal >= 0, totalBytes >= nominal else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        guard
            let overhead = backend.pool.minimumSegmentedOverhead(
                tokens: tokens, layerKinds: backend.layerKinds)
        else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        let (extra, overflow) = (totalBytes - nominal).addingReportingOverflow(overhead)
        guard !overflow,
            admission.canEverFit(promptTokens: tokens, maxTokens: 0, additionalBackendBytes: extra)
        else {
            throw CBv2KVError.capacityExhausted(
                needed: overflow ? Int.max : totalBytes,
                available: admission.admissibleBytesCapacity)
        }
        try admission.reserve(
            id: id, additionalTokens: tokens, additionalBytes: totalBytes - nominal)
    }

    package func release(id: CBv2RequestID) { admission.releaseAll(id: id) }

    func reserveTransient(bytes: Int) throws -> CBv2CheckpointReservation {
        try admission.reserveTransient(bytes: bytes)
    }

    /// The cache partition switches between its configured maximum and zero.
    /// Shrinks happen only after queue-confined trim drops the actual owners.
    /// A failed growth preserves the old charge and must leave capture disabled.
    package func setSharedReservation(bytes: Int) throws {
        try lock.withLock {
            guard bytes >= 0 else { throw CBv2NativeBlockError.invalidConfiguration }
            if bytes == sharedBytes { return }
            if bytes == 0 {
                sharedReservation?.release()
                sharedReservation = nil
                sharedBytes = 0
                return
            }
            guard sharedBytes == 0 else { throw CBv2NativeBlockError.invalidConfiguration }
            sharedReservation = try admission.reserveTransient(bytes: bytes)
            sharedBytes = bytes
        }
    }

    package func updateCapacity(_ bytes: Int) {
        admission.updateBytesCapacity(max(0, bytes))
        backend.updateBytesCapacity(max(0, bytes))
    }

    public var reservedBytes: Int { admission.bytesReserved }

    // Do not explicitly retire Admission's process owner at engine shutdown:
    // an authenticated transfer may still own buffers/leases after the last
    // model request. The owner retires when its final real lease has drained.
}
