import Foundation

struct CBv2CheckpointStageEntry {
    var bytes: Int
    var transferred: Bool
    var settled = false
    var nativePreparationExtended = false
}

/// A native engine's typed staging charge. Its owner must drop all destination
/// aliases before close, or transfer them through Admission into its own pool.
/// After transfer, close releases only bounded scratch. Never provider credit.
final class CBv2CheckpointStageLease: @unchecked Sendable {
    struct Destination {
        let targetBytes: Int
        let auxiliaryBytes: Int
        var bytes: Int { targetBytes + auxiliaryBytes }
    }
    let admission: AdmissionV2
    let identity: UUID
    private let lock = NSLock()
    private var retainedDestination: Destination
    private var didSettle = false
    private var didExtendNativePreparation = false
    var destination: Destination { lock.withLock { retainedDestination } }
    var targetBytes: Int { destination.targetBytes }
    var auxiliaryBytes: Int { destination.auxiliaryBytes }
    let scratchBytes: Int
    var destinationBytes: Int { destination.bytes }
    var totalBytes: Int { destinationBytes + scratchBytes }

    init(admission: AdmissionV2, identity: UUID, targetBytes: Int,
         auxiliaryBytes: Int, scratchBytes: Int) {
        self.admission = admission
        self.identity = identity
        self.retainedDestination = .init(targetBytes: targetBytes, auxiliaryBytes: auxiliaryBytes)
        self.scratchBytes = scratchBytes
    }

    /// One private owner settles all evaluated destinations together before
    /// exposure. Scratch stays held through the later adoption transaction.
    func settleDestinationAfterEvaluation(targetBytes: Int, auxiliaryBytes: Int) throws {
        try lock.withLock {
            guard !didSettle, targetBytes >= 0, auxiliaryBytes >= 0,
                  targetBytes <= retainedDestination.targetBytes,
                  auxiliaryBytes <= retainedDestination.auxiliaryBytes else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let actual = Destination(targetBytes: targetBytes, auxiliaryBytes: auxiliaryBytes)
            try admission.settleCheckpointStage(
                identity: identity, expectedBytes: retainedDestination.bytes + scratchBytes,
                reduction: retainedDestination.bytes - actual.bytes)
            retainedDestination = actual
            didSettle = true
        }
    }

    /// One native paged preparation may add its EXACT missing-suffix and
    /// bounded host-map promise to this same stage, before constructing them.
    /// This only grows C. No pool/request transfer or native credit occurs.
    /// Initial imported destinations must already have settled; the enlarged
    /// phase must itself settle once before the final atomic adoption.
    func extendForNativePagedPreparation(targetBytes: Int, auxiliaryBytes: Int) throws {
        try lock.withLock {
            guard didSettle, !didExtendNativePreparation,
                  targetBytes >= 0, auxiliaryBytes >= 0,
                  targetBytes > 0 || auxiliaryBytes > 0 else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let target = try CBv2CheckpointAllocationFootprint.add(retainedDestination.targetBytes, targetBytes)
            let auxiliary = try CBv2CheckpointAllocationFootprint.add(retainedDestination.auxiliaryBytes, auxiliaryBytes)
            let additional = try CBv2CheckpointAllocationFootprint.add(targetBytes, auxiliaryBytes)
            let expected = try CBv2CheckpointAllocationFootprint.add(retainedDestination.bytes, scratchBytes)
            try admission.extendCheckpointStage(identity: identity,
                expectedBytes: expected, additionalBytes: additional)
            retainedDestination = .init(targetBytes: target, auxiliaryBytes: auxiliary)
            didSettle = false
            didExtendNativePreparation = true
        }
    }

    func closeAfterDroppingOwners() { admission.closeCheckpointStage(identity: identity) }
}

/// Short-lived engine-queue preparation ticket. It owns accounting, not arrays;
/// a failed preparation explicitly drops every private alias before rollback.
/// No automatic deinit refund can run ahead of stored-property destruction.
final class CBv2CheckpointAdoptionReservation {
    private var rollbackBody: (() -> Void)?
    private var commitBody: (() -> Void)?

    init(rollback: @escaping () -> Void, onCommit: (() -> Void)? = nil) {
        rollbackBody = rollback
        commitBody = onCommit
    }

    func commit() {
        let callback = commitBody
        rollbackBody = nil
        commitBody = nil
        callback?()
    }

    func rollbackAfterDroppingOwners() {
        let callback = rollbackBody
        rollbackBody = nil
        commitBody = nil
        callback?()
    }

    deinit { assert(rollbackBody == nil, "checkpoint transfer ticket was not resolved") }
}
