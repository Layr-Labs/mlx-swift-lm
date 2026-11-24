import MLX

extension CBv2CompleteCheckpointCapture {
    /// Engine queue, before asyncEval and before constructing any successor.
    /// A candidate never enters the durable publication set before commit.
    /// Native `position` may sit below the row frontier: full pages are
    /// immutable below it and each window copy proves its own ring residency.
    /// Packed owners require the current frontier so a restored suffix sees
    /// the same original-precision prefill chunk as its donor.
    ///
    /// Refuses (nil, the ordinary graceful path) when every donor's staged
    /// windows plus this checkpoint's would exceed the slot-wide cap: a
    /// retained checkpoint must never take the ledger room a request being
    /// served needs for its next chunk. `allowance` names what the donor
    /// itself gives up first; without one the candidate only fits in free room.
    func prepareHistorical(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?],
        allowance: CBv2HistoricalStagingAllowance? = nil,
        nativeWork: CBv2NativeCompletePrefixWork? = nil,
        prefix: CBv2CheckpointManifestPrefix? = nil
    ) throws -> CBv2CapturedCompleteCheckpoint? {
        if codec.contiguousLayout != nil {
            return try prepareContiguous(
                position: position, chunkSize: chunkSize, state: state,
                allowance: allowance, prefix: prefix)
        }
        guard !isClosed, let layout = codec.historicalLayout,
            state.count == layout.layers.count
        else { return refuse(.layoutMismatch, position) }
        // The live packed row keeps every token of its current chunk native,
        // while a complete checkpoint retains only its exact recent band.
        // An interior cut would turn formerly current tokens into coded
        // history on replay. Refuse before quotes, copies or staging charges;
        // explicitly native owners retain their existing interior contract.
        for index in layout.owningIndices {
            if let row = state[index] as? PagedSequenceKV,
                row.groupKey.quantization != nil, row.absoluteOffset != position
            {
                return refuse(.packedInteriorCut, position)
            }
        }
        do {
            // Scalar policy projection first; no descriptor/page/token table
            // is built merely to ask the store whether this boundary fits.
            let descriptors = try codec.tensorDescriptors(position: position)
            let packedBytes = try descriptors.reduce(0) {
                try CBv2CheckpointAllocationFootprint.add($0, $1.byteCount)
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: packedBytes),
                manifestFits(
                    position: position, chunkSize: chunkSize, prefix: prefix,
                    descriptors: descriptors)
            else {
                return refuse(.storeRefused, position)
            }
            var owners: [(index: Int, row: PagedSequenceKV)] = []
            var windowBytes = 0
            if codec.isNativePagedHistorical {
                guard let nativeWork, nativeWork.codecIdentity == codec.identity else {
                    throw CBv2NativeShutdownError.unsupportedConsumer
                }
                windowBytes = try CBv2CheckpointAllocationFootprint.add(
                    64 << 10, layout.layers.count * 512)
                if codec.assistant != nil {
                    windowBytes = try CBv2CheckpointAllocationFootprint.add(
                        windowBytes,
                        CBv2HistoricalMTPCheckpointFootprint.captureBytes(
                            position: position,
                            descriptors: Array(descriptors.dropFirst(codec.targetTensorCount))))
                }
            }
            if codec.historicalRecurrent, codec.assistant != nil {
                windowBytes = try CBv2CheckpointAllocationFootprint.add(
                    windowBytes,
                    CBv2HistoricalMTPCheckpointFootprint.captureBytes(
                        position: position,
                        descriptors: try codec.assistantDescriptors(position: position)))
            }
            for (index, layer) in layout.layers.enumerated() {
                if layer.owner != index {
                    guard state[index] == nil else { return refuse(.rowMismatch, position) }
                    continue
                }
                guard let row = state[index] as? PagedSequenceKV,
                    row.absoluteOffset >= position, row.pool.layerKinds == codec.layerKinds,
                    row.groupKey.dtype == layer.dtype.mlxDType
                else { return refuse(.rowMismatch, position) }
                guard layer.window != nil else {
                    if row.groupKey.quantization != nil {
                        windowBytes = try CBv2CheckpointAllocationFootprint.add(
                            windowBytes,
                            CBv2QuantizedCheckpointRecent.reservationBytes(
                                row: row, position: position))
                    }
                    continue
                }
                // Allocation-free: the same figure the window reserves.
                let (next, overflow) = windowBytes.addingReportingOverflow(
                    try CBv2HistoricalWindow.reservationBytes(row: row, position: position))
                guard !overflow else { return refuse(.windowFootprintOverflow, position) }
                windowBytes = next
                owners.append((index, row))
            }
            let shed = historicalStagingSheddable(allowance)
            guard
                let displaced = CBv2HistoricalStagingCap.displaced(
                    candidateBytes: windowBytes,
                    slotBytes: stagedHistoricalBytes + max(0, inFlightHistoricalBytes),
                    replacingBytes: allowance?.replacingBytes ?? 0,
                    sheddable: shed.map(\.stagedHistoricalBytes), cap: historicalSlotStagedByteCap)
            else {
                events.record(
                    .slotStagingCap, position: position,
                    detail: "candidate \(windowBytes) slot \(stagedHistoricalBytes) "
                        + "inFlight \(inFlightHistoricalBytes) "
                        + "replacing \(allowance?.replacingBytes ?? 0) "
                        + "cap \(historicalSlotStagedByteCap) "
                        + "capacity \(codec.admission.bytesCapacity)")
                return nil
            }
            let candidate: CBv2CapturedCompleteCheckpoint
            if codec.isNativePagedHistorical, let nativeWork {
                let history = try CBv2HistoricalCompleteCheckpoint(
                    codec: codec,
                    position: position, chunkSize: chunkSize, state: state)
                candidate = .init(historical: history)
                // Partial window construction is already loan-owned. A later
                // required native failure cannot fall into the legacy deinit.
                try nativeWork.retain(owners: [candidate])
                for owner in owners {
                    let window = try makeHistoricalWindow(owner.row, position, codec.admission)
                    try history.installWindow(window, layer: owner.index)
                    try nativeWork.retain(arrays: window.evaluationRoots, owners: [window])
                }
            } else {
                var windows: [Int: CBv2HistoricalWindow] = [:]
                for owner in owners {
                    windows[owner.index] = try makeHistoricalWindow(
                        owner.row, position, codec.admission)
                }
                candidate = .init(
                    historical: .init(
                        position: position, chunkSize: chunkSize, windows: windows,
                        quantizedRecent: try codec.captureQuantizedRecent(
                            state: state,
                            position: position, includeWindows: false)))
                if codec.historicalRecurrent, codec.assistant != nil {
                    // The drafter's own history is copied once the step settles.
                    try candidate.historical?.armAssistant(codec: codec)
                }
            }
            inFlightHistoricalBytes +=
                candidate.stagedHistoricalBytes - (allowance?.replacingBytes ?? 0)
            if let allowance, displaced > 0 {
                releaseHistoricalStaging(
                    Array(shed.prefix(displaced)), requestID: allowance.requestID)
            }
            events.record(.prepared, position: position)
            return candidate
        } catch let error as MLXError { throw error } catch {
            events.record(.prepareFailed, position: position, error: error)
            return nil
        }
    }

    /// Graceful refusal of one boundary, counted.
    private func refuse(_ event: CBv2CheckpointCaptureEvent, _ position: Int)
        -> CBv2CapturedCompleteCheckpoint?
    {
        events.record(event, position: position)
        return nil
    }

    /// Copy only the boundaries of one computed range that retention will
    /// keep: the deepest capturable one (the rolling latest) and, below it,
    /// the fork target and an open first. Any other interior boundary would
    /// retire in this same step, so its windows are never copied. The
    /// deepest is tried first because it is only the latest if it lands; a
    /// refused boundary hands that place to the next one down. The order of
    /// the attempts is also the order of claims on the slot-wide cap: latest,
    /// then target, then first. Ascending.
    func prepareHistorical(
        positions: [Int], retention: CBv2CheckpointRetention, state: [CBv2SequenceKV?],
        requestID: CBv2RequestID? = nil, prefix: CBv2CheckpointManifestPrefix? = nil
    ) throws -> [CBv2CapturedCompleteCheckpoint] {
        func allowance(_ role: CBv2HistoricalStagingCap.Role) -> CBv2HistoricalStagingAllowance? {
            guard let requestID else { return nil }
            return historicalStagingAllowance(
                requestID: requestID, retention: retention, role: role)
        }
        let stride = retention.stride ?? historicalCheckpointStrideTokens
        var prepared: [CBv2CapturedCompleteCheckpoint] = []
        var latest: Int?
        for position in positions.reversed() {
            guard
                let candidate = try prepareHistorical(
                    position: position, chunkSize: stride, state: state,
                    allowance: allowance(.latest), prefix: prefix)
            else { continue }
            prepared.append(candidate)
            latest = position
            break
        }
        guard let latest else { return [] }
        let below = positions.filter { $0 < latest }
        if let target = retention.plannedTarget, below.contains(target),
            let candidate = try prepareHistorical(
                position: target, chunkSize: stride, state: state, allowance: allowance(.target),
                prefix: prefix)
        {
            prepared.append(candidate)
        }
        if retention.firstIsOpen {
            // The first is the lowest boundary that lands, the target included.
            let lowest = prepared.compactMap(\.position).min() ?? latest
            for position in below where position < lowest {
                guard
                    let candidate = try prepareHistorical(
                        position: position, chunkSize: stride, state: state,
                        allowance: allowance(.first), prefix: prefix)
                else { continue }
                prepared.append(candidate)
                break
            }
        }
        return prepared.sorted { ($0.position ?? 0) < ($1.position ?? 0) }
    }

    /// A prepared candidate that will not be staged: its step was discarded,
    /// its request preempted, or its copy failed. Closing it releases its
    /// windows; it is no longer in flight either way.
    func discardHistorical(_ candidate: CBv2CapturedCompleteCheckpoint) {
        inFlightHistoricalBytes = max(0, inFlightHistoricalBytes - candidate.stagedHistoricalBytes)
        candidate.finishEvaluationAndClose()
    }

    /// Stage one evaluated boundary under `CBv2CheckpointRetention`:
    /// at most the first, the fork target and the rolling latest stay staged.
    /// Over the donor's byte budget the claim order is the slot-wide cap's:
    /// the first goes, then the target, and the rolling latest is always
    /// kept. Without a target that is the first/latest pair, as before.
    func commitHistorical(
        _ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID,
        hintTokens: Int? = nil, resumedAt: Int = 0
    ) {
        if codec.contiguousLayout != nil {
            commitContiguousHistorical(
                candidate, requestID: requestID,
                hintTokens: hintTokens, resumedAt: resumedAt)
            return
        }
        // Staged or closed, the candidate is no longer in flight.
        inFlightHistoricalBytes = max(0, inFlightHistoricalBytes - candidate.stagedHistoricalBytes)
        guard !isClosed, let position = candidate.position,
            !(staged[requestID]?.contains { $0.position == position } ?? false)
        else {
            events.record(.duplicateBoundary, position: candidate.position)
            candidate.finishEvaluationAndClose()
            return
        }
        let retiring = stageHistorical(
            candidate, requestID: requestID,
            position: position,
            stride: codec.historicalRecurrent ? nil : historicalCheckpointStrideTokens,
            hintTokens: hintTokens, resumedAt: resumedAt)
        events.record(.staged, position: position)
        if !retiring.isEmpty {
            queue.async { retiring.forEach { $0.finishEvaluationAndClose() } }
        }
    }

    @discardableResult
    func commitContiguousHistorical(
        _ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID,
        nativeWork: CBv2NativeCompletePrefixWork? = nil,
        hintTokens: Int? = nil, resumedAt: Int = 0
    ) -> Bool {
        if codec.isNativePagedHistorical || codec.contiguousLayout != nil {
            inFlightHistoricalBytes = max(
                0, inFlightHistoricalBytes - candidate.stagedHistoricalBytes)
        }
        guard !isClosed, let position = candidate.position,
            position > (staged[requestID]?.compactMap(\.position).max() ?? 0),
            !(staged[requestID]?.contains { $0.position == candidate.position } ?? false)
        else {
            if let nativeWork {
                nativeWork.finishAfterDroppingConsumers {
                    candidate.closeAfterCompletedEvaluation()
                }
            } else if hasNativeTracking {
                retireCaptured(candidate, requestID: requestID)
            } else {
                candidate.finishEvaluationAndClose()
            }
            return false
        }
        let retiring = stageHistorical(
            candidate, requestID: requestID,
            position: position, stride: nil, hintTokens: hintTokens, resumedAt: resumedAt)
        for previous in retiring { retireCaptured(previous, requestID: requestID) }
        return true
    }
}

extension EngineLoopV2 {
    /// Paged historical donors follow upstream's independent stride and
    /// retention policy. Contiguous native checkpoints retain their actual
    /// committed-frontier contract; no older assistant state is fabricated.
    func prepareHistoricalCheckpoints(_ step: CBv2InFlightStep) throws -> [MLXArray] {
        guard let capture = completeCheckpointCapture,
            capture.codec.historicalLayout != nil || capture.codec.contiguousLayout != nil
        else { return [] }
        let stride = capture.historicalCheckpointStrideTokens
        capture.inFlightHistoricalBytes = 0
        for (id, range) in step.computedRanges {
            recurrentGeometryObserverForTesting?(
                id, range, step.recurrentCheckpointChunkSizes[id],
                step.packedPrefixRows.contains(id), "range", "historical")
            guard let rec = scheduler.record(for: id), rec.request.prefixCacheEnabled,
                rec.request.multimodal == nil, rec.request.positionState == nil,
                rec.preemptionCount == 0,
                let state = kvStates[id]
            else {
                if let rec = scheduler.record(for: id), rec.request.prefixCacheEnabled,
                    range.lowerBound < rec.request.promptTokens.count
                {
                    capture.events.record(.requestIneligible, position: range.upperBound)
                }
                continue
            }
            let prefix = CBv2CheckpointManifestPrefix(
                tokens: rec.request.promptTokens, cacheSalt: rec.request.checkpointCacheSalt)
            var geometry = recurrentCheckpointGeometry[id] ?? .init()
            if capture.codec.historicalRecurrent {
                // Recurrent state exists only at a range end, so the recurrent
                // boundary rule decides; windows are copied now, the state at commit.
                // Prompt ranges only are counted; decode ranges are never boundaries.
                let inPrompt = range.lowerBound < rec.request.promptTokens.count
                guard step.recurrentEvaluations[id] != nil else {
                    if inPrompt {
                        capture.events.record(.noRecurrentEvaluation, position: range.upperBound)
                    }
                    continue
                }
                guard rec.request.permitsDurableHybridCheckpoint(layerKinds: layerKinds) else {
                    if inPrompt {
                        capture.events.record(
                            .durableCheckpointNotPermitted, position: range.upperBound)
                    }
                    continue
                }
                guard let cap = step.recurrentCheckpointChunkSizes[id] else {
                    if inPrompt { capture.events.record(.noChunkCap, position: range.upperBound) }
                    continue
                }
                let wasArmed = geometry.isArmed
                let boundary = geometry.record(
                    range: range, cap: cap, promptLength: rec.request.promptTokens.count,
                    packed: step.packedPrefixRows.contains(id),
                    chunkGrid: recurrentCheckpointChunkGrid)
                recurrentCheckpointGeometry[id] = geometry
                if !geometry.isArmed { rec.shortCheckpointCaptureDisarmed = true }
                if wasArmed, !geometry.isArmed, inPrompt {
                    capture.events.record(.geometryDisarmed, position: range.lowerBound)
                }
                guard boundary else { continue }
                guard range.upperBound < rec.request.promptTokens.count else {
                    capture.events.record(.promptEnd, position: range.upperBound)
                    continue
                }
                let retention = capture.retention(
                    requestID: id, stride: nil,
                    hintTokens: rec.request.prefixCheckpointTargetTokens,
                    resumedAt: rec.prefixReusePlan?.matchedBoundary ?? 0)
                if let candidate = try capture.prepareHistorical(
                    position: range.upperBound, chunkSize: cap, state: state,
                    allowance: capture.historicalStagingAllowance(
                        requestID: id, retention: retention, role: .latest),
                    prefix: prefix)
                {
                    step.historicalCheckpoints[id] = [candidate]
                }
                continue
            }
            if capture.codec.contiguousLayout != nil || capture.codec.isNativePagedHistorical {
                guard let cap = step.recurrentCheckpointChunkSizes[id],
                    cap >= scheduler.config.prefillChunkSize,
                    CBv2AttentionV1.queryBlockSize <= 0 || cap % CBv2AttentionV1.queryBlockSize == 0
                else { continue }
                // Preserve the existing uniform complete-chunk rule for the
                // exact-frontier target/head capture. Upstream recurrent state
                // can use the newer chunk-agnostic rule independently.
                let eligible =
                    geometry.isArmed && !step.packedPrefixRows.contains(id)
                    && cap > 1 && range.lowerBound == geometry.position
                    && range.upperBound <= rec.request.promptTokens.count
                    && range.count == cap && range.lowerBound % cap == 0
                    && (geometry.chunkSize == nil || geometry.chunkSize == cap)
                if eligible {
                    geometry.position = range.upperBound
                    geometry.chunkSize = cap
                } else {
                    geometry.isArmed = false
                }
                recurrentCheckpointGeometry[id] = geometry
                if !geometry.isArmed { rec.shortCheckpointCaptureDisarmed = true }
                guard eligible, range.upperBound % cap == 0,
                    range.upperBound < rec.request.promptTokens.count
                else { continue }
                if capture.codec.assistant != nil {
                    guard
                        step.mtpRound?.committedObservationRows.contains(where: { $0.id == id })
                            == true
                    else { continue }
                }
                let work = try capture.nativeWorkFactory?(id)
                do {
                    try work?.captureCurrentStreams()
                    let retention = capture.retention(
                        requestID: id, stride: nil,
                        hintTokens: rec.request.prefixCheckpointTargetTokens,
                        resumedAt: rec.prefixReusePlan?.matchedBoundary ?? 0)
                    guard
                        let candidate = try capture.prepareHistorical(
                            position: range.upperBound, chunkSize: cap, state: state,
                            allowance: capture.historicalStagingAllowance(
                                requestID: id, retention: retention, role: .latest),
                            nativeWork: work, prefix: prefix)
                    else {
                        work?.finishAfterDroppingConsumers()
                        continue
                    }
                    step.historicalCheckpoints[id] = [candidate]
                    if let work {
                        step.nativeHistoricalCheckpointWork[id] = work
                        try work.retain(arrays: candidate.evaluationRoots, owners: [candidate])
                    }
                } catch {
                    work?.requiredCompletionFailed()
                    throw error
                }
                continue
            }
            let positions = geometry.recordHistorical(
                range: range,
                promptLength: rec.request.promptTokens.count,
                packed: step.packedPrefixRows.contains(id), stride: stride)
            recurrentCheckpointGeometry[id] = geometry
            if !geometry.isArmed { rec.shortCheckpointCaptureDisarmed = true }
            let capturable = positions.filter { $0 < rec.request.promptTokens.count }
            guard !capturable.isEmpty else { continue }
            let retention = capture.retention(
                requestID: id, stride: stride, hintTokens: rec.request.prefixCheckpointTargetTokens,
                resumedAt: rec.prefixReusePlan?.matchedBoundary ?? 0)
            let prepared = try capture.prepareHistorical(
                positions: capturable, retention: retention, state: state, requestID: id,
                prefix: prefix)
            if !prepared.isEmpty { step.historicalCheckpoints[id] = prepared }
        }
        let candidates = step.historicalCheckpoints.values.flatMap { $0 }
        let roots = candidates.flatMap(\.evaluationRoots)
        for candidate in candidates {
            candidate.historical?.markSubmitted()
            candidate.contiguous?.markSubmitted()
        }
        return roots
    }

    /// What `finishStep` does when `captureSettledHistoricalAssistants` throws.
    enum SettledAssistantCopyFailure: Equatable {
        /// Discard the steps of the rows that had checkpoints.
        case discardCheckpointRows
        /// Discard every participant and finish the cohort with the error.
        case failCohort
        /// Drop only the step's checkpoints; the donors' steps stand.
        case dropCheckpoints
    }

    /// The split rule (`failCohort` for an `MLXError`, else `dropCheckpoints`)
    /// applies only to an assistant that opts in
    /// (`CBv2HistoricalMTPPrefixCheckpointCoding.refusedAssistantCopyKeepsDonorStep`).
    static func settledAssistantCopyFailure(
        _ error: any Error, keepsDonorStep: Bool
    ) -> SettledAssistantCopyFailure {
        guard keepsDonorStep else { return .discardCheckpointRows }
        return error is MLXError ? .failCohort : .dropCheckpoints
    }

    /// See `CBv2HistoricalMTPPrefixCheckpointCoding.refusedAssistantCopyKeepsDonorStep`.
    var refusedAssistantCopyKeepsDonorStep: Bool {
        (completeCheckpointCapture?.codec.assistant
            as? any CBv2HistoricalMTPPrefixCheckpointCoding)?
            .refusedAssistantCopyKeepsDonorStep == true
    }

    /// Runs after the actual observation fence and before the native commit.
    ///
    /// For an assistant that opts in (`refusedAssistantCopyKeepsDonorStep`),
    /// a paged historical (non-native) candidate whose assistant copy is
    /// refused is only a lost checkpoint: it is already marked failed (or
    /// carries no assistant), so the commit closes it, and the donor's own
    /// step stays committed. Device errors, native work and contiguous
    /// candidates still throw, as does every refusal of other assistants.
    func captureSettledHistoricalAssistants(_ step: CBv2InFlightStep) throws {
        let keepsDonorStep = refusedAssistantCopyKeepsDonorStep
        for (id, candidates) in step.historicalCheckpoints {
            for candidate in candidates {
                guard
                    candidate.contiguous?.requiresAssistant == true
                        || candidate.historical?.requiresAssistant == true
                else { continue }
                guard !step.discard.contains(id), scheduler.record(for: id)?.preemptionCount == 0
                else { continue }
                let work = step.nativeHistoricalCheckpointWork[id]
                let refusable =
                    keepsDonorStep && work == nil && nativeShutdownState == nil
                    && candidate.historical != nil
                guard
                    let observation = step.mtpRound?.committedObservationRows.first(where: {
                        $0.id == id
                    })
                else {
                    if refusable {
                        completeCheckpointCapture?.events.record(
                            .assistantCaptureRefused, position: candidate.position)
                        continue
                    }
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                do {
                    try work?.captureCurrentStreams()
                    try work?.retain(owners: [observation.assistantState])
                    if let contiguous = candidate.contiguous {
                        try contiguous.captureSettledAssistant(
                            requestState: observation.assistantState)
                    } else if let historical = candidate.historical {
                        try historical.captureSettledAssistant(
                            requestState: observation.assistantState)
                    } else {
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                    try work?.retain(arrays: candidate.evaluationRoots, owners: [candidate])
                } catch let error as MLXError {
                    work?.requiredCompletionFailed()
                    throw error
                } catch {
                    if refusable {
                        completeCheckpointCapture?.events.record(
                            .assistantCaptureRefused, position: candidate.position, error: error)
                        continue
                    }
                    work?.requiredCompletionFailed()
                    throw error
                }
            }
        }
    }

    /// All required native evaluation/fences remain outside beginCommit.
    func finishHistoricalCheckpointCopiesForNative(_ step: CBv2InFlightStep) -> Bool {
        guard let capture = completeCheckpointCapture,
            let factory = capture.nativeWorkFactory
        else { return true }
        for (id, candidates) in step.historicalCheckpoints {
            var work = step.nativeHistoricalCheckpointWork[id]
            do {
                if work == nil {
                    work = try factory(id)
                    step.nativeHistoricalCheckpointWork[id] = work
                }
                guard let work else { throw CBv2NativeShutdownError.unsupportedConsumer }
                try work.retain(arrays: candidates.flatMap(\.evaluationRoots), owners: candidates)
                try work.captureCurrentStreams()
                for candidate in candidates {
                    if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0
                    {
                        try candidate.finishEvaluationForRetirement()
                    } else {
                        try candidate.finishEvaluation()
                    }
                }
                try work.fenceForProtectedPromotion()
            } catch {
                if let work {
                    work.requiredCompletionFailed()
                } else {
                    candidates.forEach { capture.retainAfterNativeFailure($0) }
                }
                return false
            }
        }
        return true
    }

    func commitHistoricalCheckpoints(_ step: CBv2InFlightStep) -> MLXError? {
        guard let capture = completeCheckpointCapture else { return nil }
        var nativeFailure: MLXError?
        for (id, candidates) in step.historicalCheckpoints {
            if capture.hasNativeTracking {
                guard let active = step.nativeHistoricalCheckpointWork[id],
                    active.hasProtectedPromotionCompletion,
                    capture.codec.contiguousLayout != nil || capture.codec.isNativePagedHistorical,
                    candidates.count == 1, let candidate = candidates.first
                else {
                    candidates.forEach { capture.retainAfterNativeFailure($0) }
                    return nil
                }
                if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0 {
                    active.finishAfterDroppingConsumers {
                        candidate.closeAfterCompletedEvaluation()
                    }
                } else if capture.commitContiguousHistorical(
                    candidate, requestID: id, nativeWork: active,
                    hintTokens: scheduler.record(for: id)?.request.prefixCheckpointTargetTokens,
                    resumedAt: scheduler.record(for: id)?.prefixReusePlan?.matchedBoundary ?? 0)
                {
                    active.finishAfterDroppingConsumers()
                }
                continue
            }
            for candidate in candidates {
                do {
                    try candidate.finishEvaluation()
                    if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0
                    {
                        capture.events.record(.stepDiscarded, position: candidate.position)
                        capture.discardHistorical(candidate)
                    } else {
                        if capture.codec.historicalRecurrent {
                            // The recurrent commit for this step already ran,
                            // so the confirmed state sits at the range end.
                            guard let historical = candidate.historical,
                                let layers = recurrentStates[id]?.confirmedStateSnapshot()
                            else {
                                capture.events.record(
                                    .recurrentStateMissing, position: candidate.position)
                                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                            }
                            try historical.attachRecurrentState(layers, codec: capture.codec)
                        }
                        let rec = scheduler.record(for: id)
                        capture.commitHistorical(
                            candidate, requestID: id,
                            hintTokens: rec?.request.prefixCheckpointTargetTokens,
                            resumedAt: rec?.prefixReusePlan?.matchedBoundary ?? 0)
                    }
                } catch {
                    capture.events.record(.commitFailed, position: candidate.position, error: error)
                    if let error = error as? MLXError { nativeFailure = nativeFailure ?? error }
                    capture.discardHistorical(candidate)
                }
            }
        }
        step.historicalCheckpoints.removeAll()
        step.nativeHistoricalCheckpointWork.removeAll()
        if nativeFailure != nil { step.discard.formUnion(step.participants) }
        return nativeFailure
    }
}
