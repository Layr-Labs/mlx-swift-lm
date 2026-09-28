import MLX

extension CBv2CompleteCheckpointCapture {
    /// Engine queue, before asyncEval and before constructing any successor.
    /// A candidate never enters the durable publication set before commit.
    func prepareHistorical(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?]
    ) throws -> CBv2CapturedCompleteCheckpoint? {
        if codec.contiguousLayout != nil { return try prepareContiguous(position: position, chunkSize: chunkSize, state: state) }
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
                      row.absoluteOffset == position, row.pool.layerKinds == codec.layerKinds,
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

    @discardableResult
    func commitHistorical(_ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID,
                          nativeWork: CBv2NativeCompletePrefixWork? = nil) -> Bool {
        guard !isClosed,
              !(staged[requestID]?.contains { $0.position == candidate.position } ?? false)
        else {
            if let nativeWork {
                nativeWork.finishAfterDroppingConsumers { candidate.closeAfterCompletedEvaluation() }
            } else if hasNativeTracking {
                retireCaptured(candidate, requestID: requestID)
            } else { candidate.finishEvaluationAndClose() }
            return false
        }
        if staged[requestID, default: []].count == 2 {
            let previous = staged[requestID]!.removeLast()
            retireCaptured(previous, requestID: requestID)
        }
        staged[requestID, default: []].append(candidate)
        return true
    }
}

extension EngineLoopV2 {
    /// Exact scalar geometry advances on launch. It belongs to that immutable
    /// step; scheduler cursors and mutable windows may already be ahead when
    /// finalize runs. Preemption/ragged/packed history disarms this generation.
    func prepareHistoricalCheckpoints(_ step: CBv2InFlightStep) throws -> [MLXArray] {
        guard let capture = completeCheckpointCapture,
              capture.codec.historicalLayout != nil || capture.codec.contiguousLayout != nil else { return [] }
        for (id, range) in step.computedRanges {
            guard let rec = scheduler.record(for: id), rec.request.prefixCacheEnabled,
                  rec.request.multimodal == nil, rec.request.positionState == nil, rec.preemptionCount == 0,
                  let cap = step.recurrentCheckpointChunkSizes[id], cap >= scheduler.config.prefillChunkSize,
                  CBv2AttentionV1.queryBlockSize <= 0 || cap % CBv2AttentionV1.queryBlockSize == 0,
                  let state = kvStates[id]
            else { continue }
            var geometry = recurrentCheckpointGeometry[id] ?? .init()
            let eligible = geometry.record(range: range, cap: cap,
                promptLength: rec.request.promptTokens.count, packed: step.packedPrefixRows.contains(id))
            recurrentCheckpointGeometry[id] = geometry
            guard eligible, range.upperBound < rec.request.promptTokens.count else { continue }
            if capture.codec.assistant != nil {
                // Only an actual committed prompt observation can supply the
                // assistant at this interior boundary. A target-only fallback
                // must not publish a partial MTP checkpoint.
                guard step.mtpRound?.committedObservationRows.contains(where: { $0.id == id }) == true
                else { continue }
            }
            let nativeWork = try capture.nativeWorkFactory?(id)
            do {
                try nativeWork?.captureCurrentStreams()
                guard let candidate = try capture.prepareHistorical(
                    position: range.upperBound, chunkSize: cap, state: state) else {
                    nativeWork?.finishAfterDroppingConsumers()
                    continue
                }
                step.historicalCheckpoints[id] = candidate
                if let nativeWork {
                    step.nativeHistoricalCheckpointWork[id] = nativeWork
                    try nativeWork.retain(arrays: candidate.evaluationRoots, owners: [candidate])
                }
            } catch {
                nativeWork?.requiredCompletionFailed()
                throw error
            }
        }
        let roots = step.historicalCheckpoints.values.flatMap(\.evaluationRoots)
        for candidate in step.historicalCheckpoints.values {
            candidate.historical?.markSubmitted()
            candidate.contiguous?.markSubmitted()
        }
        return roots
    }

    /// Called after finishStatefulMTPEvaluation and before the native commit
    /// lock. The observation still owns the genuine request-local head state.
    func captureSettledHistoricalAssistants(_ step: CBv2InFlightStep) throws {
        for (id, candidate) in step.historicalCheckpoints {
            guard let contiguous = candidate.contiguous, contiguous.requiresAssistant,
                  !step.discard.contains(id), scheduler.record(for: id)?.preemptionCount == 0 else { continue }
            guard let observation = step.mtpRound?.committedObservationRows.first(where: { $0.id == id }) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let work = step.nativeHistoricalCheckpointWork[id]
            do {
                try work?.captureCurrentStreams()
                try work?.retain(owners: [observation.assistantState])
                try contiguous.captureSettledAssistant(requestState: observation.assistantState)
                try work?.retain(arrays: candidate.evaluationRoots, owners: [candidate])
            } catch {
                work?.requiredCompletionFailed()
                throw error
            }
        }
    }


    /// Target-window prepare remains before the successor can overwrite it.
    /// Root may attach settled assistant copies in the pre-commit hook. This
    /// phase performs ALL required evaluation/fences outside beginCommit.
    func finishHistoricalCheckpointCopiesForNative(_ step: CBv2InFlightStep) -> Bool {
        guard let capture = completeCheckpointCapture,
              let factory = capture.nativeWorkFactory else { return true }
        for (id, candidate) in step.historicalCheckpoints {
            var work = step.nativeHistoricalCheckpointWork[id]
            do {
                if work == nil {
                    work = try factory(id)
                    step.nativeHistoricalCheckpointWork[id] = work
                }
                guard let work else { throw CBv2NativeShutdownError.unsupportedConsumer }
                try work.retain(arrays: candidate.evaluationRoots, owners: [candidate])
                try work.captureCurrentStreams()
                if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0 {
                    try candidate.finishEvaluationForRetirement()
                } else {
                    try candidate.finishEvaluation()
                }
                try work.fenceForProtectedPromotion()
            } catch {
                if let work { work.requiredCompletionFailed() }
                else { capture.retainAfterNativeFailure(candidate) }
                return false // step/loan retains actual candidates and credit.
            }
        }
        return true
    }

    func commitHistoricalCheckpoints(_ step: CBv2InFlightStep) -> MLXError? {
        guard let capture = completeCheckpointCapture else { return nil }
        var nativeFailure: MLXError?
        for (id, candidate) in step.historicalCheckpoints {
            if capture.hasNativeTracking {
                guard let active = step.nativeHistoricalCheckpointWork[id],
                      active.hasProtectedPromotionCompletion else {
                    capture.retainAfterNativeFailure(candidate)
                    return nil
                }
                // Metadata-only promotion under the healthy commit. Any
                // actual close/fence is queued to the owned operation.
                if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0 {
                    active.finishAfterDroppingConsumers { candidate.closeAfterCompletedEvaluation() }
                } else if capture.commitHistorical(candidate, requestID: id, nativeWork: active) {
                    active.finishAfterDroppingConsumers()
                }
                continue
            }
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
        step.historicalCheckpoints.removeAll()
        step.nativeHistoricalCheckpointWork.removeAll()
        if nativeFailure != nil { step.discard.formUnion(step.participants) }
        return nativeFailure
    }
}
