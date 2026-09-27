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

    /// Keep the first checkpoint and the deepest of the rest. Over the count
    /// or byte cap, the shallowest interior checkpoint retires first; the
    /// byte cap never drops the set below two so a large-window model keeps
    /// at least the older first/latest pair.
    func commitHistorical(_ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID) {
        guard !isClosed,
              !(staged[requestID]?.contains { $0.position == candidate.position } ?? false)
        else { candidate.finishEvaluationAndClose(); return }
        var checkpoints = staged[requestID] ?? []
        checkpoints.append(candidate)
        var bytes = checkpoints.reduce(0) { $0 + $1.stagedHistoricalBytes }
        var retiring: [CBv2CapturedCompleteCheckpoint] = []
        while checkpoints.count > max(2, maximumStagedHistoricalCheckpoints)
            || (checkpoints.count > 2 && bytes > historicalStagedByteBudget)
        {
            let previous = checkpoints.remove(at: 1)
            bytes -= previous.stagedHistoricalBytes
            retiring.append(previous)
        }
        staged[requestID] = checkpoints
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
            for position in positions {
                guard position < rec.request.promptTokens.count,
                      let candidate = try capture.prepareHistorical(position: position, chunkSize: stride, state: state)
                else { continue }
                step.historicalCheckpoints[id, default: []].append(candidate)
            }
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
                        capture.commitHistorical(candidate, requestID: id)
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
