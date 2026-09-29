// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// One checked prefill scope over the already installed MiMo block budget.
/// This is not an admission assertion supplied by a model/caller.
enum MiMoV26NAXKeyRangeNative {
    static let requested =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_MIMO_V26_NAX_KEY_RANGES"] == "1"
    @TaskLocal private static var current: MiMoV26NAXKeyRangeContext?

    static func withContext<Result>(
        _ value: MiMoV26NAXKeyRangeContext?,
        _ body: () throws -> Result
    ) rethrows -> Result {
        try $current.withValue(value, operation: body)
    }

    static func prepareLayer(
        budget: MiMoV26BlockBatchBudget, layerIndex: Int?,
        plan: MiMoV26BlockBatchAttention.Plan
    )
        -> MiMoV26NAXKeyRangeLayer?
    {
        guard requested, let current, current.budget === budget,
            MiMoV26BlockBatchAttention.matchesCurrentBudget(budget), let layerIndex
        else { return nil }
        return current.prepareLayer(index: layerIndex, plan: plan)
    }
}

/// Engine-queue confined, bounded by the native model's actual layer count.
/// A child task retaining the TaskLocal cannot use it after the synchronous
/// forward closes. Only the engine creates its real work owner.
final class MiMoV26NAXKeyRangeContext: @unchecked Sendable {
    let budget: MiMoV26BlockBatchBudget
    private let admission: AdmissionV2
    private let tracking: CBv2NativeShutdownState
    private let createWork: () throws -> MiMoV26NAXKeyRangeWork
    private let stream: MLX.Stream
    private let maximumBufferBytes: Int
    private var work: MiMoV26NAXKeyRangeWork?
    private var consumedLayers = Set<Int>()
    private var closed = false
    private var failure: Error?

    init?(
        budget: MiMoV26BlockBatchBudget, admission: AdmissionV2,
        tracking: CBv2NativeShutdownState,
        createWork: @escaping () throws -> MiMoV26NAXKeyRangeWork
    ) {
        guard MiMoV26NAXKeyRangeNative.requested, admission.hasProcessMemoryOwner,
            tracking.supported, tracking.engineID == budget.engineID,
            MiMoV26NAXGatherQMM.gpuStream(.default),
            StreamOrDevice.default.stream == MLX.Stream.gpu,
            MiMoV26NAXGatherQMM.naxAvailable
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let maximum = GPU.deviceInfo().maxBufferSize
            guard maximum > 0 else { return nil }
            maximumBufferBytes = maximum
        #else
            return nil
        #endif
        self.budget = budget
        self.admission = admission
        self.tracking = tracking
        self.createWork = createWork
        stream = StreamOrDevice.default.stream
    }

    func prepareLayer(index: Int, plan: MiMoV26BlockBatchAttention.Plan)
        -> MiMoV26NAXKeyRangeLayer?
    {
        guard !closed, failure == nil, (0 ..< budget.layerCount).contains(index),
            !consumedLayers.contains(index), tracking.mayExecute,
            StreamOrDevice.default.stream == stream,
            let bytes = MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
                plan: plan, maximumBufferBytes: maximumBufferBytes,
                upperBound: budget.allocationPolicy.upperBound(byteCount:)),
            bytes > 0
        else { return nil }
        do {
            let actual: MiMoV26NAXKeyRangeWork
            if let work {
                actual = work
            } else {
                actual = try createWork()
                work = actual  // native loan/root already installed before reserve
            }
            try tracking.requireWork()
            let reservation: CBv2CheckpointReservation
            do { reservation = try admission.reserveTransient(bytes: bytes) } catch {
                // Optional additional C was refused BEFORE encoding. The
                // unchanged grouped/scalar path was already admitted.
                try tracking.requireWork()
                return nil
            }
            actual.retain(reservation: reservation, bytes: bytes)
            // Capture a late returned actual reservation before this veto.
            try tracking.requireWork()
            guard let groups = MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: plan),
                !groups.isEmpty
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "key-range projection changed after admission")
            }
            consumedLayers.insert(index)
            actual.retainPlans(groups)
            return .init(owner: actual, groups: groups)
        } catch {
            failure = error
            work?.failCompletion()
            return nil
        }
    }

    func close() throws {
        closed = true
        if let failure { throw failure }
        try tracking.requireWork()
    }
}

struct MiMoV26NAXKeyRangeLayer {
    let owner: MiMoV26NAXKeyRangeWork
    let groups: [Int: MiMoV26NAXAttentionKeyRanges.GroupedSchedule]
    func retain(_ arrays: [MLXArray]) { owner.retain(arrays: arrays) }
}

/// One actual forward's extra state, using the existing native root+loan
/// machinery. All per-layer reservations are additive and explicitly released.
/// No destructor can authorize release: the native loan retains this owner
/// until the only successful post-fence retirement path ends that same loan.
final class MiMoV26NAXKeyRangeWork: @unchecked Sendable {
    private let tracking: CBv2NativeShutdownState
    private let queue: DispatchQueue
    private let didRetire: (MiMoV26NAXKeyRangeWork) -> Void
    private let failure: () -> Void
    private let streams: [MLX.Stream]
    private var loan: UUID?
    private(set) var rootID: UInt64 = 0
    private(set) var requestIDs = Set<CBv2RequestID>()
    private var reservations: [CBv2CheckpointReservation] = []
    private var arrays: [MLXArray] = []
    private var plans: [[Int: MiMoV26NAXAttentionKeyRanges.GroupedSchedule]] = []
    private(set) var reservedBytes = 0
    private(set) var completionFailed = false
    private(set) var completed = false
    private(set) var released = false
    private var retirementQueued = false
    // Failure/hold witness only, never a successful-completion replacement.
    var beforeRequiredDrainForTesting: (() throws -> Void)?

    init(
        tracking: CBv2NativeShutdownState, queue: DispatchQueue,
        didRetire: @escaping (MiMoV26NAXKeyRangeWork) -> Void,
        failure: @escaping () -> Void
    ) throws {
        self.tracking = tracking
        self.queue = queue
        self.didRetire = didRetire
        self.failure = failure
        let actual = StreamOrDevice.default.stream
        streams = actual == Stream.cpu ? [actual] : [actual, Stream.cpu]
        // Properties exist before self is retained by the actual native loan.
        loan = try tracking.beginLoan(owner: self, duringDrain: true)
    }

    func bindRoot(_ value: UInt64) {
        precondition(rootID == 0 && value > 0)
        rootID = value
    }
    func bindRequests(_ ids: Set<CBv2RequestID>) {
        precondition(requestIDs.isEmpty)
        requestIDs = ids
    }
    func retain(reservation: CBv2CheckpointReservation, bytes: Int) {
        reservations.append(reservation)
        let next = reservedBytes.addingReportingOverflow(bytes)
        precondition(bytes > 0 && !next.overflow)
        reservedBytes = next.partialValue
    }
    func retainPlans(_ value: [Int: MiMoV26NAXAttentionKeyRanges.GroupedSchedule]) {
        plans.append(value)
    }
    func retain(arrays incoming: [MLXArray]) { arrays.append(contentsOf: incoming) }
    var evaluationTargets: [MLXArray] { arrays }
    var retainedArrayCount: Int { arrays.count }

    /// Called at the existing step completion boundary, before beginCommit.
    /// Also covers custom-kernel outputs unused by the model's scalar readback.
    func finishEvaluation() throws {
        guard !completionFailed else { throw CBv2NativeShutdownError.operationClosed }
        guard !completed else { return }
        do {
            try tracking.requireWork()
            if arrays.isEmpty && reservations.isEmpty {
                // A capacity-only refusal created no native arrays/encoding.
                // Dispose only this genuine unstarted owner, not stock work.
                completed = true
                return
            }
            try withError { fault in
                if !arrays.isEmpty { eval(arrays) }
                try fault.check()
            }
            try requiredDrain()
            completed = true
        } catch {
            failCompletion()
            throw error
        }
    }

    /// Failed graph locals have unwound. Never evaluate a failed graph to
    /// fabricate a successful completion; drain the actually captured streams.
    func discardAfterDrain() throws {
        guard !completionFailed else { throw CBv2NativeShutdownError.operationClosed }
        guard !completed else { return }
        do {
            try requiredDrain()
            completed = true
        } catch {
            failCompletion()
            throw error
        }
    }

    private func requiredDrain() throws {
        try tracking.requireWork()
        try beforeRequiredDrainForTesting?()
        for stream in streams {
            try withError { fault in
                stream.synchronize()
                try fault.check()
            }
        }
        try tracking.requireWork()
    }
    func failCompletion() {
        completionFailed = true
        if tracking.fail(.nativeWorkFailed) { failure() }
    }

    /// The completed step has dropped its graph/native root holders. Queue on
    /// that SAME engine queue so the enclosing metadata commit fully unwinds.
    /// Actual array detach and reservation release occur outside the commit.
    func retireCompletedGraph() {
        guard completed, !completionFailed, !retirementQueued else { return }
        retirementQueued = true
        queue.async { [self] in
            var detached:
                (
                    [MLXArray], [[Int: MiMoV26NAXAttentionKeyRanges.GroupedSchedule]],
                    [CBv2CheckpointReservation]
                )?
            guard
                tracking.commitIfHealthy({
                    // Claim metadata retirement atomically before a later fault
                    // can win. Moving references retains every actual owner; no
                    // array destructor/accounting runs inside this commit.
                    precondition(completed && !completionFailed && !released)
                    detached = (arrays, plans, reservations)
                    arrays = []
                    plans = []
                    reservations = []
                    reservedBytes = 0
                    released = true
                }), var held = detached
            else { return }
            detached = nil  // sever the second tuple/Array COW owner before credit
            held.0.removeAll()
            held.1.removeAll()
            for reservation in held.2 { reservation.release() }
            held.2.removeAll()
            if let loan { tracking.endLoan(loan) }
            didRetire(self)
        }
    }
}
