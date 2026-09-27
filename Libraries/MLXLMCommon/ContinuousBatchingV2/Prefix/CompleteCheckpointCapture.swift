import Foundation
import MLX

/// Request-local capture and retirement, with no resident index or idle arrays.
/// The engine queue owns staged; the retirement queue owns detached payloads.
final class CBv2CompleteCheckpointCapture: @unchecked Sendable {
    let codec: CBv2CompleteCheckpointCodec
    let store: any CBv2CompletePrefixCache
    let queue = DispatchQueue(label: "cbv2.complete-checkpoint-retirement", qos: .utility)
    var staged: [CBv2RequestID: [CBv2CapturedCompleteCheckpoint]] = [:]
    /// Historical donors keep at most the first boundary, the coordinator's
    /// fork target and the rolling latest (`CBv2HistoricalCheckpointRetention`).
    /// Engine-queue owned, created with a request's first staged checkpoint
    /// and removed with its staged list.
    var historicalRetention: [CBv2RequestID: CBv2HistoricalCheckpointRetention] = [:]
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
    // Deterministic native construction/evaluation fault seam, engine-queue
    // only. Production always uses the ordinary private historical owner.
    var makeHistoricalWindow: (PagedSequenceKV, Int, AdmissionV2) throws -> CBv2HistoricalWindow = {
        try .init(row: $0, position: $1, admission: $2)
    }
    private let handlerLock = NSLock()
    private var publicationHandler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    private var closed = false

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

    /// Reserve before any checkpoint copy graph is constructed. The extra
    /// allowance covers allocator padding, concatenate/copy intermediates and
    /// entire retained SSM backing; it survives until every alias retires.
    func capture(
        requestID: CBv2RequestID, position: Int, chunkSize: Int,
        layers: [Int: CBv2RecurrentLayerState], assistantState: (any CBv2MTPRequestState)?,
        rowStates: [CBv2SequenceKV?] = [],
        mediaIdentity: CBv2HybridPrefixIdentity? = nil, mediaTargetOnly: Bool = false
    ) -> [MLXArray] {
        guard !isClosed, position > 1, chunkSize > 1, position % chunkSize == 0,
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
            if staged[requestID, default: []].count == 2 {
                let previous = staged[requestID]!.removeLast()
                queue.async { previous.finishEvaluationAndClose() }
            }
            staged[requestID, default: []].append(captured)
            return checkpoint.evaluationRoots
        } catch {
            return []
        }
    }

    /// A final queued drop follows that request's rolling retirement copies.
    /// The engine counts this callback in its existing shutdown drain barrier.
    func drop(requestID: CBv2RequestID, completion: @escaping @Sendable () -> Void) -> Bool {
        historicalRetention.removeValue(forKey: requestID)
        guard let captures = staged.removeValue(forKey: requestID) else { return false }
        queue.async {
            captures.forEach { $0.finishEvaluationAndClose() }
            completion()
        }
        return true
    }

    /// Always completes. Call on the engine queue with retired KV owners;
    /// their backend/request reservation remains live until completion.
    func publish(
        intent: CBv2DonationIntent,
        state: [CBv2SequenceKV?],
        completion: @escaping @Sendable ([Int]) -> Void
    ) {
        // Historical donors publish deepest first, then the fork target, then
        // the first: the store's queue, quota and demand gates see the most
        // valuable endpoint before a shallower one can consume them. A target
        // that turned out adjacent to the final deepest boundary is retired
        // unwritten. Recurrent publication order is unchanged.
        var captures = staged.removeValue(forKey: intent.requestID) ?? []
        let retention = historicalRetention.removeValue(forKey: intent.requestID)
        if codec.historicalLayout != nil {
            let dropped = Set(retention?.publication.drop ?? [])
            let retiring = captures.filter { $0.position.map(dropped.contains) ?? false }
            captures.removeAll { $0.position.map(dropped.contains) ?? false }
            captures.sort { ($0.position ?? 0) > ($1.position ?? 0) }
            if !retiring.isEmpty {
                queue.async { retiring.forEach { $0.finishEvaluationAndClose() } }
            }
        }
        var exports: [CBv2CompleteCheckpointExport] = []
        for capture in captures where intent.allowsCompletePublication {
            if let checkpoint = capture.historical {
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
            onRetired: onRetired, completion: completion)
        queue.async { [self] in
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
    var evaluationRoots: [MLXArray] { checkpoint?.evaluationRoots ?? historical?.evaluationRoots ?? [] }
    var position: Int? { checkpoint?.position ?? historical?.position }
    /// Transient admission bytes this staged historical capture holds.
    var stagedHistoricalBytes: Int { historical?.reservedBytes ?? 0 }
    private var reservation: CBv2CheckpointReservation?

    init(checkpoint: CBv2RecurrentCheckpoint, reservation: CBv2CheckpointReservation) {
        self.checkpoint = checkpoint
        self.reservation = reservation
    }

    init(historical: CBv2HistoricalCompleteCheckpoint) { self.historical = historical }

    func finishEvaluation() throws {
        if let historical { try historical.finishEvaluation() }
        else { try withError { eval(evaluationRoots) } }
    }

    func finishEvaluationAndClose() {
        try? finishEvaluation()
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

    init(
        captures: [CBv2CapturedCompleteCheckpoint], exports: [CBv2CompleteCheckpointExport],
        receiptID: CBv2RequestID?, tokens: [Int], cacheSalt: String?,
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
    }

    func prepare() -> Bool {
        do {
            for capture in captures {
                try capture.finishEvaluation()
            }
            return !exports.isEmpty
        } catch { return false }
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
        exports.forEach { $0.close() }
        exports.removeAll()
        captures.forEach { $0.finishEvaluationAndClose() }
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
