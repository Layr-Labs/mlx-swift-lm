import MLX

extension CBv2CompleteCheckpointCodec {
    /// Validate every slot before reserving metadata, constructing copy graphs
    /// or registering a restored row. Borrowers never own a second allocation.
    func validateContiguousRows(_ state: [CBv2SequenceKV?], position: Int, exactWindow: Bool) throws {
        guard let layout = contiguousLayout, state.count == layout.layers.count, position > 1 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var owners = Set<ObjectIdentifier>()
        for (i, layer) in layout.layers.enumerated() {
            if layer.owner != i {
                guard state[i] == nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                continue
            }
            guard let row = state[i], owners.insert(ObjectIdentifier(row)).inserted,
                  row.absoluteOffset >= position else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            if let window = layer.window {
                guard let row = row as? CBv2WindowedSequenceKV, row.window == window,
                      row.kvHeads == layer.kvHeads, row.headDim == layer.headDim, row.valueHeadDim == layer.valueHeadDim,
                      !exactWindow || row.absoluteOffset == position,
                      row.retainedCount == min(row.absoluteOffset, window) else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            } else {
                guard row is CBv2FullSequenceKV || row is CBv2FrozenReplayFullSequenceKV else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            }
            let snapshot = row.snapshot()
            let count = layer.window.map { min(snapshot.offset, $0) } ?? snapshot.offset
            guard snapshot.keys.shape == [1, layer.kvHeads, count, layer.headDim],
                  snapshot.values.shape == [1, layer.kvHeads, count, layer.valueHeadDim],
                  snapshot.keys.dtype == layer.dtype.mlxDType, snapshot.values.dtype == layer.dtype.mlxDType else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
    }

    func exportContiguousV2(checkpoint: CBv2RecurrentCheckpoint,
        kv: [(keys: MLXArray, values: MLXArray, offset: Int)?], tokens: [Int], cacheSalt: String?,
        retainedOwners: [AnyObject] = []) throws
        -> CBv2CompleteCheckpointExport {
        guard let layout = contiguousLayout, checkpoint.layers.isEmpty,
              (assistant == nil) == (checkpoint.assistant == nil),
              checkpoint.qwen4.isEmpty, checkpoint.mediaIdentity == nil, !checkpoint.mediaTargetOnly,
              checkpoint.position < tokens.count, kv.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let descriptors = try tensorDescriptors(position: checkpoint.position)
        let assistantArrays: [MLXArray]
        if let assistant {
            guard assistant is any CBv2HistoricalMTPPrefixCheckpointCoding,
                  let state = checkpoint.assistant, state.targetInputCount == checkpoint.position,
                  let encoded = assistant.encodePrefixCheckpoint(state) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let expected = Array(descriptors.dropFirst(targetTensorCount))
            guard encoded.count == expected.count,
                  zip(encoded, expected).allSatisfy({ $0.0.shape == $0.1.shape && $0.0.dtype == $0.1.dtype.mlxDType }) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            assistantArrays = encoded
        } else { assistantArrays = [] }
        // Complete metadata preflight before permit and before slicing any row.
        for (i, layer) in layout.layers.enumerated() {
            if layer.owner != i {
                guard kv[i] == nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                continue
            }
            guard let entry = kv[i], entry.offset >= checkpoint.position,
                  layer.window == nil || entry.offset == checkpoint.position else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let count = layer.window.map { min(entry.offset, $0) } ?? entry.offset
            guard entry.keys.shape == [1, layer.kvHeads, count, layer.headDim],
                  entry.values.shape == [1, layer.kvHeads, count, layer.valueHeadDim],
                  entry.keys.dtype == layer.dtype.mlxDType, entry.values.dtype == layer.dtype.mlxDType else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
        let validation = CBv2CompleteCheckpointManifest(identity: identity, position: checkpoint.position,
            chunkSize: checkpoint.chunkSize, prefixTokens: Array(tokens.prefix(checkpoint.position)),
            cacheSalt: cacheSalt, assistantCodecID: assistant?.prefixCheckpointCodecID,
            tensors: descriptors, backendLayout: backendLayout,
            attentionLayers: layout.layers)
        _ = try validation.validateStructure()
        let manifest = try validation.owningMetadata(admission: admission)
        var arrays: [MLXArray] = []
        for i in layout.owningIndices {
            let entry = kv[i]!, count = checkpoint.position - layout.layers[i].tokenStart(at: checkpoint.position)
            arrays.append(entry.keys[.ellipsis, ..<count, 0...])
            arrays.append(entry.values[.ellipsis, ..<count, 0...])
        }
        arrays.append(contentsOf: assistantArrays)
        var exportOwners = retainedOwners
        if let checkpoint = checkpoint.assistant { exportOwners.append(checkpoint as AnyObject) }
        return .init(manifest: manifest, arrays: arrays, usesProcessMemoryOwner: admission.hasProcessMemoryOwner,
                     retainedOwners: exportOwners)
    }

    func preparedContiguousV2(manifest: CBv2CompleteCheckpointManifest, arrays: [MLXArray],
                              maximumSequenceLength: Int, backing: CBv2ContiguousCheckpointBacking) throws -> CBv2PreparedCompleteCheckpoint {
        _ = try manifest.validateStructure()
        guard let layout = contiguousLayout, manifest.backendLayout == backendLayout, manifest.identity == identity,
              manifest.attentionLayers == layout.layers, maximumSequenceLength > manifest.position,
              manifest.assistantCodecID == assistant?.prefixCheckpointCodecID,
              manifest.tensors == (try tensorDescriptors(position: manifest.position)), arrays.count == manifest.tensors.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        for (cursor, i) in layout.owningIndices.enumerated() {
            let layer = layout.layers[i], capacity = layer.window ?? maximumSequenceLength
            guard arrays[2 * cursor].shape == [1, layer.kvHeads, capacity, layer.headDim],
                  arrays[2 * cursor + 1].shape == [1, layer.kvHeads, capacity, layer.valueHeadDim],
                  arrays[2 * cursor].dtype == layer.dtype.mlxDType,
                  arrays[2 * cursor + 1].dtype == layer.dtype.mlxDType else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
        var rows = Array<CBv2SequenceKV?>(repeating: nil, count: layerKinds.count)
        for (cursor, i) in layout.owningIndices.enumerated() {
            let l = layout.layers[i], k = arrays[cursor * 2], v = arrays[cursor * 2 + 1]
            if let window = l.window {
                rows[i] = try CBv2WindowedSequenceKV(restoredKeys: k, restoredValues: v, offset: manifest.position,
                    window: window, kvHeads: l.kvHeads, headDim: l.headDim, valueHeadDim: l.valueHeadDim, checkpointBacking: backing)
            } else {
                rows[i] = try CBv2FullSequenceKV(restoredKeys: k, restoredValues: v, offset: manifest.position,
                    maxLength: maximumSequenceLength, kvHeads: l.kvHeads, headDim: l.headDim, valueHeadDim: l.valueHeadDim, checkpointBacking: backing)
            }
        }
        let checkpoint: CBv2RecurrentCheckpoint
        if let assistant {
            guard let historical = assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            checkpoint = try recurrentCheckpoint(manifest: manifest, auxiliary: Array(arrays.dropFirst(targetTensorCount)))
            guard let value = checkpoint.assistant else { throw CBv2CompleteCheckpointError.incompleteTransfer }
            // Import backing contains no checkpoint/arrays, so this creates no
            // owner cycle. Target AND assistant aliases keep its full debt.
            try historical.bindImportedPrefixCheckpointOwner(value, owner: backing)
        } else {
            guard arrays.count == targetTensorCount else { throw CBv2CompleteCheckpointError.incompleteTransfer }
            checkpoint = .init(position: manifest.position, chunkSize: manifest.chunkSize, layers: [:], byteCount: 0)
        }
        return .init(state: rows, checkpoint: checkpoint)
    }

    func contiguousReusePlan(position: Int, maximumSequenceLength: Int) throws -> CBv2PrefixReusePlan {
        guard let layout = contiguousLayout, maximumSequenceLength > position else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var rate = 0
        for i in layout.owningIndices where layout.layers[i].window == nil {
            let l = layout.layers[i]
            let k = try CBv2CheckpointTensorDescriptor.checkedByteCount(shape: [l.kvHeads, l.headDim], dtype: l.dtype.mlxDType)
            let v = try CBv2CheckpointTensorDescriptor.checkedByteCount(shape: [l.kvHeads, l.valueHeadDim], dtype: l.dtype.mlxDType)
            rate = try CBv2CheckpointAllocationFootprint.add(rate, CBv2CheckpointAllocationFootprint.add(k, v))
        }
        let (bytes, overflow) = rate.multipliedReportingOverflow(by: position)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return .init(backend: .contiguousUnquantized, strategy: .direct, matchedBoundary: position,
            replayStart: position, replayTokens: 0, prefillTokensSaved: position, restoredFullTokens: position,
            capacityReservationTokens: maximumSequenceLength, nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
            fullKVBytesPerToken: rate, additionalFullKVBytesPerToken: max(0, rate - admission.fullKVBytesPerToken),
            initialAdditionalCapacityBytes: 0, fullCapacityTokensReserved: maximumSequenceLength,
            stagedFullKVBytes: bytes, residentFullKVBytes: bytes)
    }
}

/// A bounded native allocation lease, shared only by this imported row set and
/// its authorized exports. No native array/row is retained by the lease itself.
/// Mutable contiguous storage does NOT receive evaluated materialization credit:
/// later same-shape updates may replace backing. C retains the allocator bound.
final class CBv2ContiguousCheckpointBacking: @unchecked Sendable {
    let lease: CBv2CheckpointStageLease
    let allocationBoundsByLayer: [Int?]
    let measuredBytesByLayer: [Int?]
    let auxiliaryNativeBound: Int
    let retainedHostBytes: Int
    private(set) var requestID: CBv2RequestID?
    private var rollback: CBv2CheckpointAdoptionReservation?

    init(lease: CBv2CheckpointStageLease, codec: CBv2CompleteCheckpointCodec,
         arrays: [MLXArray], expectedBound: Int, auxiliaryArrays: [MLXArray] = [],
         hostBytes: Int = 0, position: Int? = nil) throws {
        guard let layout = codec.contiguousLayout, arrays.count == layout.owningIndices.count * 2 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var bounds = Array<Int?>(repeating: nil, count: layout.layers.count)
        var actual = bounds, total = 0
        for (cursor, layer) in layout.owningIndices.enumerated() {
            let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(Array(arrays[(cursor*2)..<(cursor*2+2)]))
            bounds[layer] = footprint.bound; actual[layer] = footprint.actual
            total = try CBv2CheckpointAllocationFootprint.add(total, footprint.bound)
        }
        let auxiliary: Int
        if let assistant = codec.assistant {
            guard assistant is any CBv2HistoricalMTPPrefixCheckpointCoding, let position,
                  hostBytes == (try CBv2HistoricalMTPCheckpointFootprint.retainedHostBytes(position: position)) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let expected = Array(try codec.tensorDescriptors(position: position).dropFirst(codec.targetTensorCount))
            guard auxiliaryArrays.count == expected.count,
                  zip(auxiliaryArrays, expected).allSatisfy({ $0.0.shape == $0.1.shape && $0.0.dtype == $0.1.dtype.mlxDType }) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(auxiliaryArrays)
            auxiliary = try CBv2HistoricalMTPCheckpointFootprint.nativeDestinationBound(expected)
            guard footprint.bound == auxiliary, footprint.actual <= auxiliary else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
        } else {
            guard auxiliaryArrays.isEmpty, hostBytes == 0 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            auxiliary = 0
        }
        let retainedAuxiliary = try CBv2CheckpointAllocationFootprint.add(auxiliary, hostBytes)
        guard total == expectedBound, lease.targetBytes == expectedBound,
              lease.auxiliaryBytes == retainedAuxiliary else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        self.lease = lease; allocationBoundsByLayer = bounds; measuredBytesByLayer = actual
        auxiliaryNativeBound = auxiliary; retainedHostBytes = hostBytes
        // Seal the verified bound rather than refunding padding that a later
        // same-shape contiguous allocation is still permitted to need.
        try lease.settleDestinationAfterEvaluation(targetBytes: total, auxiliaryBytes: retainedAuxiliary)
    }

    func arm(requestID: CBv2RequestID) {
        precondition(self.requestID == nil)
        self.requestID = requestID
    }
    func finishAdoption() { lease.closeAfterDroppingOwners() } // transferred => scratch only
    func abandonAdoption(_ ticket: CBv2CheckpointAdoptionReservation) {
        rollback = ticket
        // Unpublish the request ID without refunding any still-borrowed array.
        // The deferred ticket resolves only when this backing owner dies.
        if let requestID { lease.admission.releaseAll(id: requestID) }
    }
    deinit {
        lease.closeAfterDroppingOwners()
        rollback?.rollbackAfterDroppingOwners()
        if let requestID { lease.admission.retireContiguousCheckpointRows(id: requestID, owner: lease.identity) }
    }
}
