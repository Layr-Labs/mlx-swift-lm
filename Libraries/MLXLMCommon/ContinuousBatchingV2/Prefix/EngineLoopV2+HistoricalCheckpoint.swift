import MLX

extension CBv2CompleteCheckpointCapture {
    /// Engine queue, before asyncEval and before constructing any successor.
    /// A candidate never enters the durable publication set before commit.
    /// `position` may sit below the row frontier: full pages are immutable
    /// below the frontier and each window copy proves its own ring residency.
    func prepareHistorical(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?]
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
            var windows: [Int: CBv2HistoricalWindow] = [:]
            for (index, layer) in layout.layers.enumerated() {
                if layer.owner != index {
                    guard state[index] == nil else { return nil }
                    continue
                }
                guard let row = state[index] as? PagedSequenceKV,
                      row.absoluteOffset >= position, row.pool.layerKinds == codec.layerKinds,
                      row.groupKey.dtype == layer.dtype.mlxDType
                else { return nil }
                if layer.window != nil {
                    windows[index] = try makeHistoricalWindow(row, position, codec.admission)
                }
            }
            return .init(historical: .init(position: position, chunkSize: chunkSize, windows: windows))
        } catch let error as MLXError { throw error }
        catch { return nil }
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
    /// an open first and the fork target. Any other interior boundary would
    /// retire in this same step, so its windows are never copied. The
    /// deepest is tried first because it is only the latest if it lands; a
    /// refused boundary hands that place to the next one down. Ascending.
    func prepareHistorical(
        positions: [Int], retention: CBv2HistoricalCheckpointRetention, state: [CBv2SequenceKV?]
    ) throws -> [CBv2CapturedCompleteCheckpoint] {
        var prepared: [CBv2CapturedCompleteCheckpoint] = []
        var latest: Int?
        for position in positions.reversed() {
            guard let candidate = try prepareHistorical(
                position: position, chunkSize: retention.stride, state: state) else { continue }
            prepared.append(candidate)
            latest = position
            break
        }
        guard let latest else { return [] }
        var firstIsOpen = retention.firstIsOpen
        for position in positions where position < latest {
            guard firstIsOpen || position == retention.target,
                  let candidate = try prepareHistorical(
                      position: position, chunkSize: retention.stride, state: state)
            else { continue }
            firstIsOpen = false
            prepared.append(candidate)
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
                positions: capturable, retention: retention, state: state)
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
