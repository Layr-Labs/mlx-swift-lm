// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// One genuine serial-MTP engine step. Composes existing Admission-backed
/// attention owners and the existing native operation loan; owns no new ledger,
/// success receipt, reservation discount, or alternative assistant cache.
final class CBv2NativePagedMTPWork {
    private final class Frame: @unchecked Sendable {
        let work: CBv2NativePagedMTPWork
        let thread = ObjectIdentifier(Thread.current)
        private let lock = NSLock()
        private var open = true
        init(_ work: CBv2NativePagedMTPWork) { self.work = work }
        func close() { lock.withLock { open = false } }
        var active: Bool { lock.withLock { open && thread == ObjectIdentifier(Thread.current) } }
    }
    @TaskLocal private static var frame: Frame?
    static var current: CBv2NativePagedMTPWork? {
        guard let frame, frame.active else { return nil }
        return frame.work
    }

    private let backend: PagedKVBackend
    private let binding: CBv2NativePagedModelBinding
    private var states: [CBv2RequestID: [CBv2SequenceKV?]]
    private var ordinary: [(id: CBv2RequestID, range: Range<Int>)]
    private var verification: [(id: CBv2RequestID, range: Range<Int>)]
    private var hostReservation: CBv2CheckpointReservation?
    private var operation: CBv2NativePagedOperation?
    private var owners: [CBv2PagedAttentionStepOwner] = []
    private var ordinaryOwner: CBv2PagedAttentionStepOwner?
    private var begunRows: Set<UInt64> = []
    private var expectedRows: [UInt64: (row: PagedSequenceKV, range: Range<Int>)] = [:]
    private var ordinaryClosed = false
    private var graphClosed = false
    private var verifiedWidth = 0
    private var evaluationFinished = false
    private var retirementQueued = false
    private(set) var completedColumns = 0

    private init(
        backend: PagedKVBackend, binding: CBv2NativePagedModelBinding,
        states: [CBv2RequestID: [CBv2SequenceKV?]], work: [CBv2MTPRowWork],
        hostReservation: CBv2CheckpointReservation
    ) {
        self.backend = backend
        self.binding = binding
        self.states = states
        self.hostReservation = hostReservation
        ordinary = work.filter { $0.carry == nil }.map {
            ($0.rec.id, $0.start ..< ($0.start + $0.count))
        }
        verification = work.filter { $0.carry != nil }.map {
            ($0.rec.id, $0.start ..< ($0.start + $0.count))
        }
    }

    static func make(
        backend: CBv2KVBackend, states: [CBv2RequestID: [CBv2SequenceKV?]],
        work: [CBv2MTPRowWork], driver: CBv2MTPRoundDriver
    ) throws -> CBv2NativePagedMTPWork? {
        guard let backend = backend as? PagedKVBackend, let binding = backend.nativeModelBinding
        else { return nil }
        guard binding.supportsSerialMTP, driver.config.verificationMode == .serialTarget,
            driver.config.maxSpeculativeBatch == 1,
            (1 ... 3).contains(driver.config.maxDraftTokens),
            work.lazy.filter({ $0.carry != nil }).count <= 1,
            !work.isEmpty, let limits = backend.pool.config.gatheredAttention,
            work.count <= limits.maximumBatchSize,
            work.allSatisfy({
                $0.start >= 0 && $0.count > 0
                    && CBv2KVGeometry.add($0.start, $0.count).map({
                        $0 <= binding.maximumContextTokens
                    }) == true
            }),
            work.allSatisfy({
                $0.rec.request.multimodal == nil && $0.rec.request.positionState == nil
            })
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        try binding.requireEngineQueue()
        try binding.validateAssistant(driver.drafter as AnyObject)
        guard let admission = backend.pool.memoryAdmission,
            let entries = CBv2KVGeometry.multiply(work.count, backend.layerKinds.count),
            let tables = CBv2KVGeometry.multiply(entries, 512),
            let hostBytes = CBv2KVGeometry.add(64 << 10, tables)
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        // Acquire the extra round metadata/table obligation BEFORE allocating
        // private maps or taking any native operation loan. No reserve discount.
        let host = try admission.reserveTransient(bytes: hostBytes)
        var result: CBv2NativePagedMTPWork?
        do {
            var selected: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
            for row in work {
                guard selected[row.rec.id] == nil, let state = states[row.rec.id],
                    state.count == backend.layerKinds.count
                else {
                    throw CBv2NativeShutdownError.unsupportedConsumer
                }
                selected[row.rec.id] = state
            }
            let created = CBv2NativePagedMTPWork(
                backend: backend, binding: binding,
                states: selected, work: work, hostReservation: host)
            result = created
            created.operation = try binding.beginWork(
                requests: Set(work.map { $0.rec.id }), nativeData: false)
            created.operation?.retain(owner: created)  // complete owner before any later veto
            if !created.ordinary.isEmpty {
                created.ordinaryOwner = try backend.prepareAttentionWork(
                    assignments: created.ordinary, states: selected)
                guard let owner = created.ordinaryOwner else {
                    throw CBv2NativeShutdownError.unsupportedConsumer
                }
                created.owners.append(owner)
            }
            return created
        } catch {
            if let result, result.operation != nil {
                result.discardAfterBuildFailure()
            } else {
                // Proven cold: no native operation/attention owner exists.
                result?.states.removeAll()
                result?.ordinary.removeAll()
                result?.verification.removeAll()
                result?.hostReservation = nil
                host.release()
            }
            throw error
        }
    }

    static func withConstruction<Result>(
        _ work: CBv2NativePagedMTPWork?,
        _ body: () throws -> Result
    ) throws -> Result {
        guard let work else { return try $frame.withValue(nil, operation: body) }
        try work.binding.requireEngineQueue()
        guard !work.graphClosed, Self.current == nil else {
            throw CBv2NativeShutdownError.operationClosed
        }
        let frame = Frame(work)
        defer {
            frame.close()
            work.graphClosed = true
        }
        return try $frame.withValue(frame, operation: body)
    }

    func finishOrdinaryConstruction() throws {
        try binding.requireEngineQueue()
        guard Self.current === self, !ordinaryClosed else {
            throw CBv2NativeShutdownError.operationClosed
        }
        if let ordinaryOwner {
            try ordinaryOwner.seal()
            backend.pool.endAttentionConstruction(ordinaryOwner)
        }
        ordinaryClosed = true
    }

    /// All rows/cohorts preflight before ANY row begins a transaction.
    func beginVerification(_ rows: [CBv2MTPRoundInFlight.VerifyRow], depth: Int) throws {
        try binding.requireEngineQueue()
        guard Self.current === self, ordinaryClosed, expectedRows.isEmpty,
            rows.count == verification.count, rows.count == 1, (1 ... 3).contains(depth)
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        var checked: [UInt64: (row: PagedSequenceKV, range: Range<Int>)] = [:]
        for (item, planned) in zip(rows, verification) {
            guard item.id == planned.id, planned.range.count == depth + 1,
                let original = states[item.id], original.count == item.storageRows.count
            else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            for (index, value) in item.storageRows.enumerated() {
                guard let row = value as? PagedSequenceKV, original[index] === row,
                    row.pool === backend.pool, !row.isReleased, row.speculativeBase == nil,
                    row.absoluteOffset == planned.range.lowerBound,
                    row.maxLength >= planned.range.upperBound,
                    row.speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan,
                    checked[row.serial] == nil
                else { throw CBv2NativeShutdownError.unsupportedConsumer }
                _ = try binding.metadata(row: row, layer: index)
                checked[row.serial] = (row, planned.range)
            }
        }
        expectedRows = checked
        verifiedWidth = depth + 1
    }

    func authorizeBegin(_ row: PagedSequenceKV) -> Bool {
        guard (try? binding.requireEngineQueue()) != nil else { return false }
        guard Self.current === self, !graphClosed, let expected = expectedRows[row.serial],
            expected.row === row, row.speculativeBase == nil,
            row.absoluteOffset == expected.range.lowerBound,
            begunRows.insert(row.serial).inserted
        else { return false }
        return true
    }

    /// Called by the existing actual-range planner, not by model/cache callers.
    func permitsPlannedColumn(_ row: PagedSequenceKV, range: Range<Int>) -> Bool {
        guard (try? binding.requireEngineQueue()) != nil else { return false }
        guard Self.current === self, !graphClosed, let expected = expectedRows[row.serial],
            expected.row === row, begunRows.contains(row.serial),
            row.speculativeBase == expected.range.lowerBound,
            range.count == 1, range.lowerBound == expected.range.lowerBound + completedColumns,
            range.upperBound <= expected.range.upperBound
        else { return false }
        return true
    }

    func prepareColumn(_ index: Int) throws -> CBv2PagedAttentionStepOwner {
        try binding.requireEngineQueue()
        guard Self.current === self, ordinaryClosed, index == completedColumns,
            index < verifiedWidth, begunRows.count == expectedRows.count,
            !expectedRows.isEmpty
        else { throw CBv2NativeShutdownError.unsupportedConsumer }
        let assignments = verification.map {
            (
                id: $0.id,
                range: ($0.range.lowerBound + index) ..< ($0.range.lowerBound + index + 1)
            )
        }
        guard let owner = try backend.prepareAttentionWork(assignments: assignments, states: states)
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        owners.append(owner)
        return owner
    }

    func completeColumn(_ owner: CBv2PagedAttentionStepOwner) throws {
        guard Self.current === self, owners.last === owner, !owner.completed else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        try owner.seal()
        // The canonical serial target already evaluated its scores/hidden/KV.
        // This additionally proves every owned transfer/output/sentinel root.
        try owner.finishEvaluation()
        owner.publish()
        backend.pool.endAttentionConstruction(owner)
        completedColumns += 1
    }

    var evaluationTargets: [MLXArray] { owners.flatMap(\.evaluationTargets) }
    func publish() { for owner in owners { owner.publish() } }

    func finishEvaluation() throws {
        guard graphClosed, ordinaryClosed,
            verification.isEmpty || completedColumns == verifiedWidth
        else {
            fail()
            throw CBv2NativeShutdownError.operationClosed
        }
        do {
            for owner in owners { try owner.finishEvaluation() }
            try operation?.requiredDrain()
            evaluationFinished = true
        } catch {
            fail()
            throw error
        }
    }

    /// Finalizer metadata only, AFTER the genuine required completion above.
    func permitsFinalization(_ row: PagedSequenceKV) -> Bool {
        guard evaluationFinished, let operation, operation.tracking.mayExecute,
            expectedRows[row.serial]?.row === row, begunRows.contains(row.serial)
        else { return false }
        do {
            try binding.requireEngineQueue()
            return true
        } catch { return false }
    }

    func fail() {
        operation?.fail()
        for owner in owners { owner.failCompletion() }
    }

    /// Failed graph locals must have unwound before the caller invokes this.
    /// No failed graph is evaluated. A failed REQUIRED fence remains sticky.
    func discardAfterBuildFailure() {
        guard let operation, operation.tracking.mayExecute else { return }
        backend.pool.endAttentionConstruction(backend.pool.activeAttentionWork)
        do {
            for owner in owners { try owner.discardAfterDrain() }
            try operation.requiredDrain()
            evaluationFinished = true
            closeAfterStep()
        } catch { fail() }
    }

    func closeAfterStep() {
        guard evaluationFinished, !retirementQueued, let operation else { return }
        retirementQueued = true
        for owner in owners { owner.closeGraph() }
        // Individual owners enqueue their true alias/charge retirement first.
        // No new native wait is performed under finalize's commit lock.
        operation.enqueueRetirement { [self, operation] in
            guard operation.tracking.mayExecute else { return }
            guard owners.allSatisfy(\.released) else {
                fail()
                return
            }
            operation.finish {
                owners.removeAll()
                ordinaryOwner = nil
                expectedRows.removeAll()
                begunRows.removeAll()
                states.removeAll()
                ordinary.removeAll()
                verification.removeAll()
                hostReservation?.release()
                hostReservation = nil
                self.operation = nil
            }
        }
    }
}
