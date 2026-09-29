import Foundation
import MLX

/// Request-local capture and retirement, with no resident index or idle arrays.
/// The engine queue owns staged; the retirement queue owns detached payloads.
final class CBv2CompleteCheckpointCapture: @unchecked Sendable {
    let codec: CBv2CompleteCheckpointCodec
    let store: any CBv2CompletePrefixCache
    let queue = DispatchQueue(label: "cbv2.complete-checkpoint-retirement", qos: .utility)
    var staged: [CBv2RequestID: [CBv2CapturedCompleteCheckpoint]] = [:]
    var makeContiguousCheckpoint: (CBv2CompleteCheckpointCodec, Int, Int, [CBv2SequenceKV?]) throws -> CBv2ContiguousHistoricalCheckpoint = {
        try .init(codec: $0, position: $1, chunkSize: $2, state: $3)
    }
    /// Every donor keeps at most the first boundary, the coordinator's fork
    /// target and the rolling latest (`CBv2CheckpointRetention`): a
    /// historical donor on the 1,024-token stride, a recurrent donor on
    /// whatever 256-token-aligned range ends land. Engine-queue owned, created with a request's
    /// first staged checkpoint and removed with its staged list.
    var retentions: [CBv2RequestID: CBv2CheckpointRetention] = [:]
    /// Staged window copies share the slot's ordinary admission ceiling
    /// (`AdmissionV2.reserveTransient`). Over 1/16 of that capacity a donor
    /// gives up its first, then its fork target, and always keeps its rolling
    /// latest: the same order as the slot-wide cap. Windows per checkpoint:
    /// gpt-oss-20b ~7 MB (12 owners x 128 tokens, float32 K/V), gemma-4-26b
    /// ~216 MB (25 owners x 1,024 tokens, 16-bit K/V).
    static let historicalStagedByteBudgetDivisor = 16
    /// Test seam; production reads the admission capacity at every commit so
    /// a resized slot budget is honored by requests already in flight.
    var historicalStagedByteBudgetOverride: Int?
    var historicalStagedByteBudget: Int {
        historicalStagedByteBudgetOverride
            ?? max(0, codec.admission.bytesCapacity) / Self.historicalStagedByteBudgetDivisor
    }
    /// Staged historical windows across ALL donors may hold at most 1/8 of
    /// the slot's admission capacity (`CBv2HistoricalStagingCap`), read at
    /// each capture because the slot can be re-sliced at runtime. Test seam.
    var historicalSlotStagedByteCapOverride: Int?
    var historicalSlotStagedByteCap: Int {
        historicalSlotStagedByteCapOverride
            ?? max(0, codec.admission.bytesCapacity) / CBv2HistoricalStagingCap.capacityDivisor
    }
    /// Window bytes a donor's checkpoints still hold while its files are
    /// written: `publish` moves them out of `staged`, but their reservations
    /// are released only when the batch closes after the last file. Written
    /// on the engine queue, cleared on the retirement queue, read by either.
    private let publishingLock = NSLock()
    private var publishingHistoricalBytes: [CBv2RequestID: Int] = [:]
    var publishingHistoricalBytesTotal: Int {
        publishingLock.withLock { publishingHistoricalBytes.values.reduce(0, +) }
    }
    /// Window bytes of every donor's historical checkpoints that still hold
    /// a reservation: staged, or in publication. Both count against the
    /// slot-wide cap, so donors finishing together cannot lift new donors'
    /// staging above it.
    var stagedHistoricalBytes: Int {
        staged.values.reduce(publishingHistoricalBytesTotal) { total, captures in
            captures.reduce(total) { $0 + $1.stagedHistoricalBytes }
        }
    }
    /// Net window bytes prepared by the step now launching and not yet
    /// committed: candidates minus the rolling latest each replaces. A step
    /// holding candidates never chains a successor
    /// (`CBv2InFlightStep.permitsChainedSuccessor`), so every candidate is
    /// committed or closed before the next step prepares and resets this.
    var inFlightHistoricalBytes = 0
    var historicalCheckpointStrideTokens = CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens
    /// A fork target within this many tokens of the final deepest boundary
    /// is dropped at publication, for every layout. Test seam.
    var targetAdjacencyTokens = CBv2CheckpointRetention.defaultTargetAdjacencyTokens
    // Deterministic native construction/evaluation fault seam, engine-queue
    // only. Production always uses the ordinary private historical owner.
    var makeHistoricalWindow: (PagedSequenceKV, Int, AdmissionV2) throws -> CBv2HistoricalWindow = {
        try .init(row: $0, position: $1, admission: $2)
    }
    private let handlerLock = NSLock()
    private var publicationHandler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    private var closed = false
    // Installed once by the exact issued native profile before the loop starts.
    var nativeWorkFactory: ((CBv2RequestID?) throws -> CBv2NativeCompletePrefixWork)?
    var nativeFailure: (() -> Void)?
    private var nativeFaultOwners: [AnyObject] = []
    var hasNativeTracking: Bool { nativeWorkFactory != nil }

    func retainAfterNativeFailure(_ owner: AnyObject) {
        handlerLock.withLock { nativeFaultOwners.append(owner) }
        nativeFailure?()
    }

    /// All rolling/rejected captures use the same tracked completion owner.
    /// Legacy/untracked cleanup remains byte-for-byte the old behavior.
    func retireCaptured(_ candidate: CBv2CapturedCompleteCheckpoint,
                        requestID: CBv2RequestID?) {
        guard let factory = nativeWorkFactory else {
            queue.async { candidate.finishEvaluationAndClose() }
            return
        }
        do {
            let work = try factory(requestID)
            try work.retain(arrays: candidate.evaluationRoots, owners: [candidate])
            queue.async {
                do {
                    try work.captureCurrentStreams()
                    try candidate.finishEvaluationForRetirement()
                    work.finishAfterDroppingConsumers { candidate.closeAfterCompletedEvaluation() }
                } catch { work.requiredCompletionFailed() }
            }
        } catch { retainAfterNativeFailure(candidate) }
    }

    init(codec: CBv2CompleteCheckpointCodec, store: any CBv2CompletePrefixCache) {
        self.codec = codec
        self.store = store
    }

    func setPublicationHandler(_ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?) {
        handlerLock.lock()
        publicationHandler = closed ? nil : handler
        handlerLock.unlock()
    }

    func close() {
        handlerLock.lock()
        closed = true
        publicationHandler = nil
        handlerLock.unlock()
    }

    var isClosed: Bool {
        handlerLock.lock()
        defer { handlerLock.unlock() }
        return closed
    }

    func reportPublication(receiptID: CBv2RequestID?, positions: [Int]) {
        guard let receiptID, !positions.isEmpty else { return }
        handlerLock.lock()
        let handler = publicationHandler
        handlerLock.unlock()
        handler?(receiptID, positions)
    }

    func hasCheckpoints(requestID: CBv2RequestID) -> Bool {
        staged[requestID]?.isEmpty == false
    }

    /// The retention verdicts for this request so far, or a fresh plan when
    /// nothing is staged yet. Read-only: state is stored at commit. `stride`
    /// is the historical stride, which names the fork target ahead of
    /// capture; nil for a recurrent donor, whose target is the deepest
    /// captured boundary at or below the hint.
    func retention(
        requestID: CBv2RequestID, stride: Int?, hintTokens: Int?, resumedAt: Int
    ) -> CBv2CheckpointRetention {
        retentions[requestID] ?? .init(stride: stride, hintTokens: hintTokens, resumedAt: resumedAt,
                                       targetAdjacencyTokens: targetAdjacencyTokens)
    }

    /// Reserve before any checkpoint copy graph is constructed. The extra
    /// allowance covers allocator padding, concatenate/copy intermediates and
    /// entire retained SSM backing; it survives until every alias retires.
    ///
    /// Recurrent boundaries are whatever aligned range ends land, so
    /// `CBv2CheckpointRetention` runs its dynamic rule (no stride): the first
    /// captured boundary, the deepest captured boundary at or below
    /// `hintTokens`, and the rolling latest stay staged; every other boundary
    /// retires at the next commit. `chunkSize` is the chunk that ended at
    /// this boundary, recorded in the manifest as provenance only.
    /// Retention changes only after the copy is staged, so a refused
    /// reservation leaves the previously retained boundaries as they were.
    /// The new copy is reserved before the old latest retires, so a donor
    /// rolling its latest transiently holds one copy more than it retains:
    /// up to four staged under the three-way rule, as it was three under
    /// the first/latest rule. What is unchanged is that +1, not the peak.
    ///
    /// Unlike historical windows there is no slot-wide cap here. Each
    /// staged checkpoint is one `reserveTransient` on the admission ledger,
    /// which fails closed but, once granted, takes room a running request's
    /// next chunk may need, exactly the exposure the historical cap bounds.
    /// The third slot is judged small enough not to need it: about 51 MB of
    /// conv/SSM state per checkpoint for Qwen3.5-9B (plus MTP history in
    /// production) against ~210 MB per Gemma 4 window, so N concurrent
    /// donors retain at most N x 3 x ~51 MB, one copy more each while a
    /// latest rolls. Revisit if capture geometry or state sizes grow.
    func capture(
        requestID: CBv2RequestID, position: Int, chunkSize: Int,
        layers: [Int: CBv2RecurrentLayerState], assistantState: (any CBv2MTPRequestState)?,
        rowStates: [CBv2SequenceKV?] = [],
        mediaIdentity: CBv2HybridPrefixIdentity? = nil, mediaTargetOnly: Bool = false,
        hintTokens: Int? = nil, resumedAt: Int = 0
    ) -> [MLXArray] {
        guard codec.contiguousLayout == nil, !codec.isNativePagedHistorical, !isClosed, position > 1, chunkSize > 1,
            CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(position, chunkSize: chunkSize),
            !mediaTargetOnly || mediaIdentity != nil,
            !(staged[requestID]?.contains { $0.checkpoint?.position == position } ?? false)
        else { return [] }
        do {
            let qwen4 = try codec.qwen4Snapshots(rows: rowStates, position: position)
            let logical = CBv2RecurrentCheckpoint(position: position, chunkSize: chunkSize,
                layers: [:], byteCount: 0, qwen4: qwen4)
            let descriptors = try codec.tensorDescriptors(position: position, qwen4: codec.qwen4Descriptors(logical),
                                                          mediaTargetOnly: mediaTargetOnly)
            var packedBytes = 0
            for descriptor in descriptors {
                let (next, overflow) = packedBytes.addingReportingOverflow(descriptor.byteCount)
                guard !overflow else { return [] }
                packedBytes = next
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: packedBytes) else { return [] }
            let stateDescriptors = descriptors.filter { $0.role != .keys && $0.role != .values }
            for spec in codec.recurrentSpec?.layers ?? [] {
                guard let state = layers[spec.modelLayerIndex],
                    let conv = state.conv, let ssm = state.ssm,
                    conv.shape == spec.convShape, conv.dtype == spec.convDType,
                    ssm.shape == spec.ssmShape, ssm.dtype == spec.ssmDType
                else { return [] }
            }
            let bytes = try CBv2CheckpointAllocationFootprint.captureBytes(stateDescriptors, layers: layers)
            let reservation = try codec.admission.reserveTransient(bytes: bytes)
            let checkpoint = try withError { error in
                var assistant: (any CBv2MTPPrefixCheckpoint)?
                if let drafter = codec.assistant, !mediaTargetOnly {
                    guard let assistantState,
                        let captured = drafter.capturePrefixCheckpoint(requestState: assistantState, targetInputCount: position)
                    else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                    assistant = captured
                }
                try error.check()
                var copies: [Int: CBv2RecurrentLayerState] = [:]
                for spec in codec.recurrentSpec?.layers ?? [] {
                    let layer = layers[spec.modelLayerIndex]!
                    copies[spec.modelLayerIndex] = .init(
                        conv: MLX.where(MLXArray(true), layer.conv!, layer.conv!), ssm: layer.ssm)
                }
                try error.check()
                return CBv2RecurrentCheckpoint(
                    position: position, chunkSize: chunkSize, layers: copies,
                    byteCount: stateDescriptors.reduce(0) { $0 + $1.byteCount }, assistant: assistant,
                    qwen4: try codec.compactQwen4(qwen4), mediaIdentity: mediaIdentity,
                    mediaTargetOnly: mediaTargetOnly)
            }
            let captured = CBv2CapturedCompleteCheckpoint(checkpoint: checkpoint, reservation: reservation)
            var verdicts = self.retention(
                requestID: requestID, stride: nil, hintTokens: hintTokens, resumedAt: resumedAt)
            let retired = Set(verdicts.commit(position))
            guard !retired.contains(position) else {
                retireCaptured(captured, requestID: requestID)
                return []
            }
            var checkpoints = staged[requestID] ?? []
            let retiring = checkpoints.filter { $0.position.map(retired.contains) ?? false }
            checkpoints.removeAll { $0.position.map(retired.contains) ?? false }
            checkpoints.append(captured)
            staged[requestID] = checkpoints
            retentions[requestID] = verdicts
            for previous in retiring { retireCaptured(previous, requestID: requestID) }
            return checkpoint.evaluationRoots
        } catch {
            return []
        }
    }

    /// A final queued drop follows that request's rolling retirement copies.
    /// The engine counts this callback in its existing shutdown drain barrier.
    func drop(requestID: CBv2RequestID, nativeWork: CBv2NativeCompletePrefixWork? = nil,
              completion: @escaping @Sendable () -> Void) -> Bool {
        retentions.removeValue(forKey: requestID)
        guard let captures = staged.removeValue(forKey: requestID) else { return false }
        if let nativeWork {
            do { try nativeWork.retain(arrays: captures.flatMap(\.evaluationRoots), owners: captures) }
            catch { return true } // loan already retains the late actual roots.
            queue.async {
                do {
                    try nativeWork.captureCurrentStreams()
                    for capture in captures { try capture.finishEvaluation() }
                    nativeWork.finishAfterDroppingConsumers {
                        captures.forEach { $0.closeAfterCompletedEvaluation() }
                        completion()
                    }
                } catch { nativeWork.requiredCompletionFailed() }
            }
            return true
        }
        queue.async {
            captures.forEach { $0.finishEvaluationAndClose() }
            completion()
        }
        return true
    }

    /// Legacy callers always complete. A tracked native failure retains its
    /// operation and poisons the outcome instead of invoking a success-shaped
    /// retirement callback. Donor backend/request C stays live until completion.
    func publish(
        intent: CBv2DonationIntent,
        state: [CBv2SequenceKV?],
        nativeWork: CBv2NativeCompletePrefixWork? = nil,
        completion: @escaping @Sendable ([Int]) -> Void
    ) {
        // Every donor publishes deepest first, then the fork target, then the
        // first: the store's queue, quota and demand gates see the most
        // valuable endpoint before a shallower one can consume them. A target
        // that turned out within `targetAdjacencyTokens` (1,024 in
        // production, every layout) of the final deepest boundary is retired
        // unwritten.
        var captures = staged.removeValue(forKey: intent.requestID) ?? []
        let retention = retentions.removeValue(forKey: intent.requestID)
        let dropped = Set(retention?.publication.drop ?? [])
        let retiring = captures.filter { $0.position.map(dropped.contains) ?? false }
        captures.removeAll { $0.position.map(dropped.contains) ?? false }
        captures.sort { ($0.position ?? 0) > ($1.position ?? 0) }
        for previous in retiring { retireCaptured(previous, requestID: intent.requestID) }
        if let nativeWork {
            do { try nativeWork.retain(arrays: captures.flatMap(\.evaluationRoots), owners: captures) }
            catch { return }
        }
        var exports: [CBv2CompleteCheckpointExport] = []
        for capture in captures where intent.allowsCompletePublication {
            if let nativeWork {
                do {
                    let source: CBv2CompleteCheckpointExport
                    if let checkpoint = capture.contiguous {
                        source = try checkpoint.export(codec: codec, state: state,
                            tokens: intent.tokens, cacheSalt: intent.cacheSalt)
                    } else if let checkpoint = capture.historical {
                        source = try codec.exportHistorical(checkpoint: checkpoint, state: state,
                            tokens: intent.tokens, cacheSalt: intent.cacheSalt)
                    } else if let checkpoint = capture.checkpoint {
                        source = try codec.export(checkpoint: checkpoint, state: state,
                            tokens: intent.tokens, cacheSalt: intent.cacheSalt)
                    } else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                    try nativeWork.retain(owners: [source])
                    try source.bindNativeCompletePrefixWork(nativeWork)
                    exports.append(source)
                } catch { nativeWork.requiredCompletionFailed(); return }
                continue
            }
            if let checkpoint = capture.contiguous {
                if let source = try? checkpoint.export(codec: codec, state: state, tokens: intent.tokens, cacheSalt: intent.cacheSalt) {
                    exports.append(source)
                }
            } else if let checkpoint = capture.historical {
                if let source = try? codec.exportHistorical(
                    checkpoint: checkpoint, state: state, tokens: intent.tokens, cacheSalt: intent.cacheSalt) {
                    exports.append(source)
                }
            } else if let checkpoint = capture.checkpoint {
                if let source = try? codec.export(
                    checkpoint: checkpoint, state: state, tokens: intent.tokens, cacheSalt: intent.cacheSalt) {
                    exports.append(source)
                }
            }
        }
        let publishingBytes = captures.reduce(0) { $0 + $1.stagedHistoricalBytes }
        var onRetired: (@Sendable () -> Void)?
        if publishingBytes > 0 {
            let requestID = intent.requestID
            publishingLock.withLock { publishingHistoricalBytes[requestID, default: 0] += publishingBytes }
            onRetired = { [self] in
                publishingLock.withLock {
                    let remaining = (publishingHistoricalBytes[requestID] ?? 0) - publishingBytes
                    if remaining > 0 {
                        publishingHistoricalBytes[requestID] = remaining
                    } else {
                        publishingHistoricalBytes.removeValue(forKey: requestID)
                    }
                }
            }
        }
        let batch = CBv2CompleteCheckpointPublication(
            captures: captures, exports: exports, receiptID: intent.receiptID,
            tokens: intent.tokens, cacheSalt: intent.cacheSalt,
            nativeWork: nativeWork, onRetired: onRetired, completion: completion)
        queue.async { [self] in
            if let nativeWork {
                do {
                    try nativeWork.retain(owners: [batch])
                    try nativeWork.captureCurrentStreams()
                } catch { return }
                // Even a closed store must complete/drop already-created
                // native captures truthfully; no swallowed required fence.
                let prepared = batch.prepare()
                guard !nativeWork.debugIsFailed else { return }
                guard !isClosed, prepared,
                      let scratch = try? codec.admission.reserveTransient(
                        bytes: codec.exportScratchBytes + (codec.admission.hasProcessMemoryOwner
                            ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes))
                else { batch.close(); return }
                batch.scratch = scratch
                do { try nativeWork.retain(owners: [scratch]) }
                catch { return }
                publishNext(batch)
                return
            }
            guard !isClosed, batch.prepare(),
                let scratch = try? codec.admission.reserveTransient(
                    bytes: codec.exportScratchBytes + (codec.admission.hasProcessMemoryOwner
                        ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes))
            else { batch.close(); return }
            batch.scratch = scratch
            publishNext(batch)
        }
    }

    private func publishNext(_ batch: CBv2CompleteCheckpointPublication) {
        guard !isClosed, let source = batch.nextSource else { batch.close(); return }
        store.donate(
            source, requestID: batch.receiptID, tokens: batch.tokens, cacheSalt: batch.cacheSalt
        ) { [self, batch, source] positions in
            source.close()
            queue.async { [self, batch, source] in
                if batch.didPublish(source: source, positions: positions) { publishNext(batch) }
            }
        }
    }
}

final class CBv2CapturedCompleteCheckpoint: @unchecked Sendable {
    private(set) var checkpoint: CBv2RecurrentCheckpoint?
    private(set) var historical: CBv2HistoricalCompleteCheckpoint?
    private(set) var contiguous: CBv2ContiguousHistoricalCheckpoint?
    var evaluationRoots: [MLXArray] { contiguous?.evaluationRoots ?? checkpoint?.evaluationRoots ?? historical?.evaluationRoots ?? [] }
    var position: Int? { contiguous?.position ?? checkpoint?.position ?? historical?.position }
    /// Transient admission bytes this staged historical capture holds.
    var stagedHistoricalBytes: Int { historical?.reservedBytes ?? 0 }
    private var reservation: CBv2CheckpointReservation?

    init(checkpoint: CBv2RecurrentCheckpoint, reservation: CBv2CheckpointReservation) {
        self.checkpoint = checkpoint
        self.reservation = reservation
    }

    init(historical: CBv2HistoricalCompleteCheckpoint) { self.historical = historical }
    init(contiguous: CBv2ContiguousHistoricalCheckpoint) { self.contiguous = contiguous }

    func finishEvaluation() throws {
        if let contiguous { try contiguous.finishEvaluation() }
        else if let historical { try historical.finishEvaluation() }
        else { try withError { eval(evaluationRoots) } }
    }

    func finishEvaluationAndClose() {
        try? finishEvaluation()
        closeAfterCompletedEvaluation()
    }

    func finishEvaluationForRetirement() throws {
        if let contiguous { try contiguous.finishEvaluationForRetirement() }
        else if let historical { try historical.finishEvaluationForRetirement() }
        else { try finishEvaluation() }
    }

    // Tracked callers invoke this only inside their actual work's post-fence
    // retirement callback; a caught failure must never arrive here.
    func closeAfterCompletedEvaluation() {
        contiguous?.close()
        contiguous = nil
        historical = nil
        checkpoint = nil
        reservation?.release()
        reservation = nil
    }
}

/// Serial-queue publication state; array aliases are cleared before completion.
private final class CBv2CompleteCheckpointPublication: @unchecked Sendable {
    private var captures: [CBv2CapturedCompleteCheckpoint]
    private var exports: [CBv2CompleteCheckpointExport]
    private var completedPositions: [Int] = []
    private var cursor = 0
    private var completion: (@Sendable ([Int]) -> Void)?
    /// Runs once the captures' reservations are released, before completion.
    private var onRetired: (@Sendable () -> Void)?
    let receiptID: CBv2RequestID?
    let tokens: [Int]
    let cacheSalt: String?
    var scratch: CBv2CheckpointReservation?
    private let nativeWork: CBv2NativeCompletePrefixWork?
    private var nativeFailure = false

    init(
        captures: [CBv2CapturedCompleteCheckpoint], exports: [CBv2CompleteCheckpointExport],
        receiptID: CBv2RequestID?, tokens: [Int], cacheSalt: String?,
        nativeWork: CBv2NativeCompletePrefixWork?,
        onRetired: (@Sendable () -> Void)? = nil,
        completion: @escaping @Sendable ([Int]) -> Void
    ) {
        self.captures = captures
        self.exports = exports
        self.receiptID = receiptID
        self.tokens = tokens
        self.cacheSalt = cacheSalt
        self.onRetired = onRetired
        self.completion = completion
        self.nativeWork = nativeWork
    }

    func prepare() -> Bool {
        do {
            for capture in captures {
                try capture.finishEvaluation()
            }
            return !exports.isEmpty
        } catch {
            if let nativeWork { nativeFailure = true; nativeWork.requiredCompletionFailed() }
            return false
        }
    }

    var nextSource: CBv2CompleteCheckpointExport? { cursor < exports.count ? exports[cursor] : nil }

    func didPublish(source: CBv2CompleteCheckpointExport, positions: [Int]) -> Bool {
        guard completion != nil, nextSource === source else { return false }
        // The provider cannot claim a boundary different from this file.
        if positions.contains(source.manifest.position) {
            completedPositions.append(source.manifest.position)
        }
        cursor += 1
        return true
    }

    func close() {
        if let nativeWork {
            guard !nativeFailure else { return }
            // The callback-held source aliases have unwound before this queued
            // work's real fence; keep captures and scratch until that fence.
            nativeWork.finishAfterDroppingConsumers { [self] in closeAfterNativeCompletion() }
            return
        }
        closeAfterNativeCompletion()
    }

    private func closeAfterNativeCompletion() {
        exports.forEach { $0.close() }
        exports.removeAll()
        if nativeWork != nil { captures.forEach { $0.closeAfterCompletedEvaluation() } }
        else { captures.forEach { $0.finishEvaluationAndClose() } }
        captures.removeAll()
        let retired = onRetired
        onRetired = nil
        retired?()
        scratch?.release()
        scratch = nil
        let callback = completion
        completion = nil
        callback?(completedPositions)
    }
}
