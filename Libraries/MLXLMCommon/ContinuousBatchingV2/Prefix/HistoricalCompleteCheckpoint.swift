import Foundation
import MLX

private final class CBv2HistoricalPagedWeakRow {
    weak var value: (any CBv2SequenceKV)?
    init(_ value: any CBv2SequenceKV) { self.value = value }
}

/// Export and capture share this actual payload owner, not a copied byte claim.
/// A tracked capture/work keeps it until its required native completion.
private final class CBv2HistoricalPagedAssistantBacking {
    var checkpoint: (any CBv2MTPPrefixCheckpoint)?
    let reservation: CBv2CheckpointReservation
    let reservedBytes: Int
    let destinationBound: Int
    init(reservation: CBv2CheckpointReservation, bytes: Int, destinationBound: Int) {
        self.reservation = reservation
        reservedBytes = bytes
        self.destinationBound = destinationBound
    }
    deinit {
        checkpoint = nil
        reservation.release()
    }
}

/// Request-local exact boundary. Window owners were copied at launch; full
/// pages remain with the donor until its existing retirement/donation barrier.
final class CBv2HistoricalCompleteCheckpoint {
    let position: Int
    let chunkSize: Int
    private(set) var windows: [Int: CBv2HistoricalWindow]
    private(set) var quantizedRecent: [Int: CBv2QuantizedCheckpointRecent] = [:]
    let requiresAssistant: Bool
    private weak var codec: CBv2CompleteCheckpointCodec?
    private var sourceRows: [Int: CBv2HistoricalPagedWeakRow] = [:]
    private var assistantBacking: CBv2HistoricalPagedAssistantBacking?
    private var ready = false
    private var failed = false
    private let stream: StreamOrDevice
    var evaluationRoots: [MLXArray] {
        windows.values.flatMap(\.evaluationRoots)
            + quantizedRecent.values.flatMap(\.evaluationRoots)
            + (assistantBacking?.checkpoint?.evaluationTargets ?? [])
    }
    /// Copies and their host witnesses; full pages stay with the actual donor.
    var reservedBytes: Int {
        windows.values.reduce(
            quantizedRecent.values.reduce(assistantBacking?.reservedBytes ?? 0) {
                $0 + $1.reservedBytes
            }
        ) { $0 + $1.reservedBytes }
    }
    var assistantCheckpoint: (any CBv2MTPPrefixCheckpoint)? { assistantBacking?.checkpoint }
    var assistantRetainedOwners: [AnyObject] { assistantBacking.map { [$0] } ?? [] }

    // Legacy target-only callers keep their existing construction contract.
    init(
        position: Int, chunkSize: Int, windows: [Int: CBv2HistoricalWindow],
        quantizedRecent: [Int: CBv2QuantizedCheckpointRecent] = [:]
    ) {
        self.position = position
        self.chunkSize = chunkSize
        self.windows = windows
        self.quantizedRecent = quantizedRecent
        requiresAssistant = false
        stream = .default
    }

    /// Native preparation creates/retains this owner BEFORE constructing any
    /// window copy. Actual row-ledger identities bind all captured full pages.
    init(
        codec: CBv2CompleteCheckpointCodec, position: Int, chunkSize: Int,
        state: [CBv2SequenceKV?]
    ) throws {
        guard codec.isNativePagedHistorical, let binding = codec.nativePagedBinding,
            let layout = codec.historicalLayout, position > 1, chunkSize > 1,
            position % chunkSize == 0, state.count == layout.layers.count
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        self.position = position
        self.chunkSize = chunkSize
        windows = [:]
        requiresAssistant = codec.assistant != nil
        self.codec = codec
        stream = .default
        var cohort: UUID?
        for index in layout.owningIndices {
            guard let row = state[index] else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let actual = try binding.metadata(row: row, layer: index)
            guard actual.offset == position, cohort == nil || actual.request == cohort else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            cohort = actual.request
        }
        let descriptors = try codec.tensorDescriptors(position: position)
        let auxiliary = Array(descriptors.dropFirst(codec.targetTensorCount))
        let host = try CBv2CheckpointAllocationFootprint.add(64 << 10, layout.layers.count * 512)
        let bytes =
            requiresAssistant
            ? try CBv2CheckpointAllocationFootprint.add(
                host,
                CBv2HistoricalMTPCheckpointFootprint.captureBytes(
                    position: position, descriptors: auxiliary))
            : host
        let destination =
            requiresAssistant
            ? try CBv2HistoricalMTPCheckpointFootprint.nativeDestinationBound(auxiliary) : 0
        assistantBacking = .init(
            reservation: try codec.admission.reserveTransient(bytes: bytes),
            bytes: bytes, destinationBound: destination)
        for index in layout.owningIndices { sourceRows[index] = .init(state[index]!) }
    }

    func installWindow(_ window: CBv2HistoricalWindow, layer: Int) throws {
        guard codec != nil, !ready, !failed, windows[layer] == nil, window.position == position
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        windows[layer] = window
    }

    func validatesSourceRows(
        _ state: [CBv2SequenceKV?], codec expected: CBv2CompleteCheckpointCodec
    ) -> Bool {
        guard expected.isNativePagedHistorical else { return codec == nil }
        guard codec === expected, let layout = expected.historicalLayout, ready, !failed,
            state.count == layout.layers.count
        else { return false }
        return layout.owningIndices.allSatisfy { sourceRows[$0]?.value === state[$0] }
    }

    func captureSettledAssistant(requestState: any CBv2MTPRequestState) throws {
        guard !ready, !failed, requiresAssistant, let backing = assistantBacking,
            backing.checkpoint == nil, let codec,
            let assistant = codec.assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        do {
            try withError { fault in
                guard
                    let value = assistant.capturePrefixCheckpoint(
                        requestState: requestState,
                        targetInputCount: position), value.targetInputCount == position
                else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                backing.checkpoint = value  // keep the actual partial copy before inspecting errors
                try fault.check()
                let expected = Array(
                    try codec.tensorDescriptors(position: position).dropFirst(
                        codec.targetTensorCount))
                guard let arrays = assistant.encodePrefixCheckpoint(value),
                    arrays.count == expected.count,
                    zip(arrays, expected).allSatisfy({
                        $0.0.shape == $0.1.shape && $0.0.dtype == $0.1.dtype.mlxDType
                    })
                else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            }
        } catch {
            failed = true
            throw error
        }
    }

    func markSubmitted() { for window in windows.values { window.markSubmitted() } }
    func finishEvaluation() throws {
        guard !failed else { throw CBv2CompleteCheckpointError.allocationFailed }
        if ready { return }
        guard !requiresAssistant || assistantBacking?.checkpoint != nil else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        do {
            try finishEvaluationForRetirement()
            if let backing = assistantBacking, let checkpoint = backing.checkpoint {
                let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(
                    checkpoint.evaluationTargets)
                guard footprint.bound == backing.destinationBound else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
            }
            ready = true
        } catch {
            failed = true
            throw error
        }
    }

    /// Cancelled pre-observation captures may have no head copy. Their actual
    /// submitted target copies must still finish; no assistant is synthesized.
    func finishEvaluationForRetirement() throws {
        guard !failed else { throw CBv2CompleteCheckpointError.allocationFailed }
        do {
            for window in windows.values { try window.finishEvaluation() }
            for recent in quantizedRecent.values { try recent.finishEvaluation() }
            if let checkpoint = assistantBacking?.checkpoint {
                try withError { fault in
                    eval(checkpoint.evaluationTargets)
                    try fault.check()
                    stream.stream.synchronize()
                    try fault.check()
                }
            }
        } catch {
            failed = true
            throw error
        }
    }
}

extension CBv2CompleteCheckpointCodec {
    /// The stage transfers complete target state, including exact windows.
    /// This is a direct resume contract; attention-only replay capability is
    /// intentionally not used to infer a historical window that it never owns.
    func historicalReusePlan(position: Int, maximumSequenceLength: Int) throws
        -> CBv2PrefixReusePlan
    {
        guard let layout = historicalLayout, position > 1, maximumSequenceLength > position else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var bytesPerToken = 0
        for (index, layer) in layout.layers.enumerated()
        where layer.owner == index && layer.window == nil {
            let bytes = try checkpointGroupKey(layer: index).bytesPerToken()
            let (next, overflow) = bytesPerToken.addingReportingOverflow(bytes)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            bytesPerToken = next
        }
        let (fullBytes, overflow) = bytesPerToken.multipliedReportingOverflow(by: position)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return .init(
            backend: .pagedFP16, strategy: .direct, matchedBoundary: position,
            replayStart: position, replayTokens: 0, prefillTokensSaved: position,
            restoredFullTokens: position, capacityReservationTokens: maximumSequenceLength,
            nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
            fullKVBytesPerToken: bytesPerToken,
            additionalFullKVBytesPerToken: max(0, bytesPerToken - admission.fullKVBytesPerToken),
            initialAdditionalCapacityBytes: 0, fullCapacityTokensReserved: maximumSequenceLength,
            stagedFullKVBytes: fullBytes, residentFullKVBytes: fullBytes)
    }

    func exportHistorical(
        checkpoint: CBv2HistoricalCompleteCheckpoint, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard let layout = historicalLayout, checkpoint.position < tokens.count,
            state.count == layerKinds.count, checkpoint.validatesSourceRows(state, codec: self)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let permit = try CBv2CheckpointManifestMemory.Permit(
            admission: admission, position: checkpoint.position)
        return try withExtendedLifetime(permit) {
            try makeHistoricalExport(
                checkpoint: checkpoint, state: state, layout: layout,
                tokens: tokens, cacheSalt: cacheSalt, permit: permit)
        }
    }

    private func makeHistoricalExport(
        checkpoint: CBv2HistoricalCompleteCheckpoint, state: [CBv2SequenceKV?],
        layout: CBv2HistoricalAttentionLayout, tokens: [Int], cacheSalt: String?,
        permit: CBv2CheckpointManifestMemory.Permit
    ) throws -> CBv2CompleteCheckpointExport {
        let descriptors = try tensorDescriptors(position: checkpoint.position)
        var sources: [CBv2CompleteCheckpointTensorSource] = []
        for index in layout.owningIndices {
            guard let row = state[index] as? PagedSequenceKV,
                row.pool.layerKinds == layerKinds, row.groupKey.dtype == kvDTypes[index],
                row.groupKey == row.pool.groupKey(forLayer: index)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            if layout.layers[index].window != nil {
                guard let window = checkpoint.windows[index],
                    window.position == checkpoint.position,
                    window.start == layout.layers[index].tokenStart(at: checkpoint.position)
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                sources.append(try checkpointSource(
                    .historicalWindow(.init(window: window, values: false)), layer: index,
                    position: checkpoint.position, values: false))
                sources.append(try checkpointSource(
                    .historicalWindow(.init(window: window, values: true)), layer: index,
                    position: checkpoint.position, values: true))
            } else {
                let map = try CBv2PagedCheckpointPageMap(
                    row: row, position: checkpoint.position, admission: admission)
                sources.append(
                    try checkpointSource(.paged(
                        try .init(
                            pageMap: map, values: false,
                            recent: checkpoint.quantizedRecent[index])), layer: index,
                        position: checkpoint.position, values: false))
                sources.append(
                    try checkpointSource(.paged(
                        try .init(
                            pageMap: map, values: true,
                            recent: checkpoint.quantizedRecent[index])), layer: index,
                        position: checkpoint.position, values: true))
            }
        }
        for index in layout.layers.indices where layout.layers[index].owner != index {
            guard state[index] == nil else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
        if let assistant {
            guard isNativePagedHistorical, let checkpoint = checkpoint.assistantCheckpoint,
                let arrays = assistant.encodePrefixCheckpoint(checkpoint)
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            sources.append(contentsOf: arrays.map { .array($0) })
        } else if checkpoint.assistantCheckpoint != nil {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard sources.count == descriptors.count,
            zip(sources, descriptors).allSatisfy({ $0.0.matches($0.1) })
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let manifest = CBv2CompleteCheckpointManifest(
            schemaVersion: CBv2CompleteCheckpointManifest.currentSchemaVersion, identity: identity,
            backendLayout: backendLayout, position: checkpoint.position,
            chunkSize: checkpoint.chunkSize,
            cacheSalt: cacheSalt, assistantCodecID: assistant?.prefixCheckpointCodecID,
            checkpointQuantization: checkpointQuantization,
            checkpointNativeDTypes: checkpointNativeDTypes,
            metadata: .init(
                tokens: Array(tokens.prefix(checkpoint.position)), tensors: descriptors,
                attentionLayers: layout.layers, permit: permit))
        _ = try manifest.validateStructure()
        return .init(
            manifest: manifest, sources: sources,
            usesProcessMemoryOwner: admission.hasProcessMemoryOwner,
            retainedOwners: checkpoint.assistantRetainedOwners)
    }
}
