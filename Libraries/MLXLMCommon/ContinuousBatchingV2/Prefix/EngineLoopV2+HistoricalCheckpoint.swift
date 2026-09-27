import MLX

/// What one donor can give up so a candidate fits under the slot-wide cap.
struct CBv2HistoricalStagingAllowance {
    let requestID: CBv2RequestID
    /// Window bytes of the donor's role-less rolling latest, which the
    /// candidate's commit retires anyway.
    var replacingBytes = 0
    /// The donor's own staged boundaries the candidate may displace, lowest
    /// priority first.
    var sheddable: [Int] = []
}

extension CBv2CompleteCheckpointCapture {
    /// Engine queue, before asyncEval and before constructing any successor.
    /// A candidate never enters the durable publication set before commit.
    /// `position` may sit below the row frontier: full pages are immutable
    /// below the frontier and each window copy proves its own ring residency.
    ///
    /// Refuses (nil, the ordinary graceful path) when every donor's staged
    /// windows plus this checkpoint's would exceed the slot-wide cap: a
    /// retained checkpoint must never take the ledger room a request being
    /// served needs for its next chunk. `allowance` names what the donor
    /// itself gives up first; without one the candidate only fits in free room.
    func prepareHistorical(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?],
        allowance: CBv2HistoricalStagingAllowance? = nil
    ) throws -> CBv2CapturedCompleteCheckpoint? {
        guard !isClosed, let layout = codec.historicalLayout,
              state.count == layout.layers.count else { return nil }
        do {
            // Scalar policy projection first; no descriptor/page/token table
            // is built merely to ask the store whether this boundary fits.
            var packedBytes = 0
            for layer in layout.layers.enumerated() where layer.element.owner == layer.offset {
                let item = layer.element
                let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                    shape: [2, item.kvHeads, position - item.tokenStart(at: position), item.headDim],
                    dtype: item.dtype.mlxDType)
                let (next, overflow) = packedBytes.addingReportingOverflow(bytes)
                guard !overflow else { return nil }
                packedBytes = next
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: packedBytes) else { return nil }
            var owners: [(index: Int, row: PagedSequenceKV)] = []
            var windowBytes = 0
            for (index, layer) in layout.layers.enumerated() {
                if layer.owner != index {
                    guard state[index] == nil else { return nil }
                    continue
                }
                guard let row = state[index] as? PagedSequenceKV,
                      row.absoluteOffset >= position, row.pool.layerKinds == codec.layerKinds,
                      row.groupKey.dtype == layer.dtype.mlxDType
                else { return nil }
                guard layer.window != nil else { continue }
                // Allocation-free: the same figure the window reserves.
                let (next, overflow) = windowBytes.addingReportingOverflow(
                    try CBv2HistoricalWindow.reservationBytes(row: row, position: position))
                guard !overflow else { return nil }
                windowBytes = next
                owners.append((index, row))
            }
            let shed = allowance.flatMap { staged[$0.requestID] }.map { captures in
                (allowance?.sheddable ?? []).compactMap { position in
                    captures.first { $0.position == position }
                }
            } ?? []
            guard let displaced = CBv2HistoricalStagingCap.displaced(
                candidateBytes: windowBytes,
                slotBytes: stagedHistoricalBytes + max(0, inFlightHistoricalBytes),
                replacingBytes: allowance?.replacingBytes ?? 0,
                sheddable: shed.map(\.stagedHistoricalBytes), cap: historicalSlotStagedByteCap)
            else { return nil }
            var windows: [Int: CBv2HistoricalWindow] = [:]
            for owner in owners {
                windows[owner.index] = try makeHistoricalWindow(owner.row, position, codec.admission)
            }
            let candidate = CBv2CapturedCompleteCheckpoint(
                historical: .init(position: position, chunkSize: chunkSize, windows: windows))
            inFlightHistoricalBytes += candidate.stagedHistoricalBytes - (allowance?.replacingBytes ?? 0)
            if let allowance, displaced > 0 {
                release(Array(shed.prefix(displaced)), requestID: allowance.requestID)
            }
            return candidate
        } catch let error as MLXError { throw error }
        catch { return nil }
    }

    /// Give up staged boundaries of one donor. Whenever the candidate that
    /// displaced them then fails to commit, its step was discarded and the
    /// donor's whole staged set is dropped with it.
    private func release(_ captures: [CBv2CapturedCompleteCheckpoint], requestID: CBv2RequestID) {
        let positions = Set(captures.compactMap(\.position))
        guard !positions.isEmpty else { return }
        staged[requestID]?.removeAll { $0.position.map(positions.contains) ?? false }
        for position in positions { historicalRetention[requestID]?.shed(position) }
        if staged[requestID]?.isEmpty == true { staged.removeValue(forKey: requestID) }
        queue.async { captures.forEach { $0.finishEvaluationAndClose() } }
    }

    /// The retention verdicts for this request so far, or a fresh plan when
    /// nothing is staged yet. Read-only: state is stored at commit.
    func historicalRetention(
        requestID: CBv2RequestID, hintTokens: Int?, resumedAt: Int
    ) -> CBv2HistoricalCheckpointRetention {
        historicalRetention[requestID] ?? .init(
            stride: historicalCheckpointStrideTokens, hintTokens: hintTokens, resumedAt: resumedAt)
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
        positions: [Int], retention: CBv2HistoricalCheckpointRetention, state: [CBv2SequenceKV?],
        requestID: CBv2RequestID? = nil
    ) throws -> [CBv2CapturedCompleteCheckpoint] {
        func allowance(_ role: CBv2HistoricalStagingCap.Role) -> CBv2HistoricalStagingAllowance? {
            guard let requestID else { return nil }
            // The stored verdicts, which earlier attempts of this range may
            // have changed by displacing a boundary.
            let current = historicalRetention[requestID] ?? retention
            var result = CBv2HistoricalStagingAllowance(requestID: requestID, sheddable: current.sheddable(for: role))
            if role == .latest, let replaced = current.replaceableLatest {
                result.replacingBytes = staged[requestID]?
                    .first { $0.position == replaced }?.stagedHistoricalBytes ?? 0
            }
            return result
        }
        var prepared: [CBv2CapturedCompleteCheckpoint] = []
        var latest: Int?
        for position in positions.reversed() {
            guard let candidate = try prepareHistorical(
                position: position, chunkSize: retention.stride, state: state,
                allowance: allowance(.latest)) else { continue }
            prepared.append(candidate)
            latest = position
            break
        }
        guard let latest else { return [] }
        let below = positions.filter { $0 < latest }
        if let target = retention.target, below.contains(target),
           let candidate = try prepareHistorical(
               position: target, chunkSize: retention.stride, state: state, allowance: allowance(.target))
        {
            prepared.append(candidate)
        }
        if retention.firstIsOpen {
            // The first is the lowest boundary that lands, the target included.
            let lowest = prepared.compactMap(\.position).min() ?? latest
            for position in below where position < lowest {
                guard let candidate = try prepareHistorical(
                    position: position, chunkSize: retention.stride, state: state,
                    allowance: allowance(.first)) else { continue }
                prepared.append(candidate)
                break
            }
        }
        return prepared.sorted { ($0.position ?? 0) < ($1.position ?? 0) }
    }

    /// Stage one evaluated boundary under `CBv2HistoricalCheckpointRetention`:
    /// at most the first, the fork target and the rolling latest stay staged.
    /// Over the byte budget the target goes first; the first/latest pair is
    /// never split, so a large-window model keeps the older rule's coverage.
    func commitHistorical(
        _ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID,
        hintTokens: Int? = nil, resumedAt: Int = 0
    ) {
        // Staged or closed, the candidate is no longer in flight.
        inFlightHistoricalBytes = max(0, inFlightHistoricalBytes - candidate.stagedHistoricalBytes)
        guard !isClosed, let position = candidate.position,
              !(staged[requestID]?.contains { $0.position == position } ?? false)
        else { candidate.finishEvaluationAndClose(); return }
        var retention = historicalRetention(
            requestID: requestID, hintTokens: hintTokens, resumedAt: resumedAt)
        var retired = Set(retention.commit(position))
        var checkpoints = (staged[requestID] ?? []) + [candidate]
        let bytes = checkpoints.reduce(0) {
            $0 + (retired.contains($1.position ?? 0) ? 0 : $1.stagedHistoricalBytes)
        }
        if bytes > historicalStagedByteBudget, let dropped = retention.dropInterior() {
            retired.insert(dropped)
        }
        let retiring = checkpoints.filter { retired.contains($0.position ?? 0) }
        checkpoints.removeAll { retired.contains($0.position ?? 0) }
        if checkpoints.isEmpty {
            staged.removeValue(forKey: requestID)
            historicalRetention.removeValue(forKey: requestID)
        } else {
            staged[requestID] = checkpoints
            historicalRetention[requestID] = retention
        }
        if !retiring.isEmpty {
            queue.async { retiring.forEach { $0.finishEvaluationAndClose() } }
        }
    }
}

extension EngineLoopV2 {
    /// Exact scalar geometry advances on launch. It belongs to that immutable
    /// step; scheduler cursors and mutable windows may already be ahead when
    /// finalize runs. Preemption/packed history disarms this generation; the
    /// chunk cap is not part of the historical rule (see
    /// `CBv2RecurrentCheckpointGeometry.recordHistorical`).
    func prepareHistoricalCheckpoints(_ step: CBv2InFlightStep) throws -> [MLXArray] {
        guard let capture = completeCheckpointCapture, capture.codec.historicalLayout != nil else { return [] }
        let stride = capture.historicalCheckpointStrideTokens
        // Every candidate of the previous capturing step was committed or
        // closed before this one could launch.
        capture.inFlightHistoricalBytes = 0
        for (id, range) in step.computedRanges {
            guard let rec = scheduler.record(for: id), rec.request.prefixCacheEnabled,
                  rec.request.multimodal == nil, rec.request.positionState == nil, rec.preemptionCount == 0,
                  let state = kvStates[id]
            else { continue }
            var geometry = recurrentCheckpointGeometry[id] ?? .init()
            let positions = geometry.recordHistorical(range: range,
                promptLength: rec.request.promptTokens.count,
                packed: step.packedPrefixRows.contains(id), stride: stride)
            recurrentCheckpointGeometry[id] = geometry
            // Interior positions are exact while every sliding ring still holds
            // `[p - W, p)`; `CBv2HistoricalWindow` refuses the rest, and the
            // gathers land in this step's own asyncEval before any successor
            // may write the ring (`permitsChainedSuccessor`).
            let capturable = positions.filter { $0 < rec.request.promptTokens.count }
            guard !capturable.isEmpty else { continue }
            let retention = capture.historicalRetention(
                requestID: id, hintTokens: rec.request.prefixCheckpointTargetTokens,
                resumedAt: rec.prefixReusePlan?.matchedBoundary ?? 0)
            let prepared = try capture.prepareHistorical(
                positions: capturable, retention: retention, state: state, requestID: id)
            if !prepared.isEmpty { step.historicalCheckpoints[id] = prepared }
        }
        let candidates = step.historicalCheckpoints.values.flatMap { $0 }
        let roots = candidates.flatMap(\.evaluationRoots)
        for candidate in candidates { candidate.historical?.markSubmitted() }
        return roots
    }

    func commitHistoricalCheckpoints(_ step: CBv2InFlightStep) -> MLXError? {
        guard let capture = completeCheckpointCapture else { return nil }
        var nativeFailure: MLXError?
        for (id, candidates) in step.historicalCheckpoints {
            for candidate in candidates {
                // This step's sample can finish before its final gather witness.
                // Wait for the already submitted roots before promotion/retirement.
                do {
                    try candidate.finishEvaluation()
                    if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0 {
                        candidate.finishEvaluationAndClose()
                    } else {
                        let rec = scheduler.record(for: id)
                        capture.commitHistorical(
                            candidate, requestID: id,
                            hintTokens: rec?.request.prefixCheckpointTargetTokens,
                            resumedAt: rec?.prefixReusePlan?.matchedBoundary ?? 0)
                    }
                } catch {
                    if let error = error as? MLXError { nativeFailure = nativeFailure ?? error }
                    candidate.finishEvaluationAndClose()
                }
            }
        }
        step.historicalCheckpoints.removeAll()
        if nativeFailure != nil { step.discard.formUnion(step.participants) }
        return nativeFailure
    }
}
