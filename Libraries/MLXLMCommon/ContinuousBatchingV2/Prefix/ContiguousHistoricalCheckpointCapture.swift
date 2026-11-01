import Foundation
import MLX

private final class CBv2WeakCheckpointRow {
    weak var row: (any CBv2SequenceKV)?
    init(_ row: any CBv2SequenceKV) { self.row = row }
}

/// Shared immutable copy owner. Explicit capture.close only drops one owner;
/// exported tensor sources retain this same lease until their aliases retire.
private final class CBv2ContiguousWindowBacking {
    var windows: [Int: (MLXArray, MLXArray)] = [:]
    var assistant: (any CBv2MTPPrefixCheckpoint)?
    let reservation: CBv2CheckpointReservation
    let reservedBytes: Int
    let allocationBound: Int
    var measuredBytes: Int?
    init(reservation: CBv2CheckpointReservation, reservedBytes: Int, allocationBound: Int) {
        self.reservation = reservation
        self.reservedBytes = reservedBytes
        self.allocationBound = allocationBound
    }
    deinit {
        windows.removeAll()
        assistant = nil
        reservation.release()
    }
}

/// A failed required completion cannot refund its buffers through ARC. This
/// rare process-lifetime quarantine retains the actual owner, not a byte claim.
private final class CBv2ContiguousCaptureQuarantine: @unchecked Sendable {
    static let shared = CBv2ContiguousCaptureQuarantine()
    private let lock = NSLock()
    private var owners: [CBv2ContiguousWindowBacking] = []
    func retain(_ owner: CBv2ContiguousWindowBacking) { lock.withLock { owners.append(owner) } }
}

/// Private historical SWA copies, never an alias to a mutable donor ring.
/// Full rows remain with the donor's existing retirement owner.
final class CBv2ContiguousHistoricalCheckpoint {
    let position: Int
    let chunkSize: Int
    let requiresAssistant: Bool
    private weak var capturingCodec: CBv2CompleteCheckpointCodec?
    private let stream: StreamOrDevice
    private var backing: CBv2ContiguousWindowBacking?
    private var owners: [Int: CBv2WeakCheckpointRow] = [:]
    private enum Evaluation { case pending, ready, failed }
    private var evaluation = Evaluation.pending
    private var submitted = false
    private var drained = false
    private(set) var completionFailed = false
    private var closed = false
    // Failure-only observer; cannot supply a successful native completion.
    var beforeRequiredDrainForTesting: (() throws -> Void)?
    var evaluate: ([MLXArray]) throws -> Void = { arrays in try withError { eval(arrays) } }
    var evaluationRoots: [MLXArray] {
        guard let backing else { return [] }
        return backing.windows.values.flatMap { [$0.0, $0.1] }
            + (backing.assistant?.evaluationTargets ?? [])
    }
    var compactAllocationEvidence: (bound: Int, actual: Int)? {
        guard let backing, let actual = backing.measuredBytes else { return nil }
        return (backing.allocationBound, actual)
    }
    var reservedBytes: Int { backing?.reservedBytes ?? 0 }

    init(
        codec: CBv2CompleteCheckpointCodec, position: Int, chunkSize: Int, state: [CBv2SequenceKV?]
    ) throws {
        guard let layout = codec.contiguousLayout, chunkSize > 1, position % chunkSize == 0 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try codec.validateContiguousRows(state, position: position, exactWindow: true)
        self.position = position
        self.chunkSize = chunkSize
        requiresAssistant = codec.assistant != nil
        capturingCodec = codec
        let footprint = try Self.reservationFootprint(codec: codec, position: position)
        backing = .init(
            reservation: try codec.admission.reserveTransient(bytes: footprint.reservedBytes),
            reservedBytes: footprint.reservedBytes, allocationBound: footprint.allocationBound)
        stream = .default
        do {
            try withError { fault in
                for i in layout.owningIndices {
                    owners[i] = CBv2WeakCheckpointRow(state[i]!)
                    guard layout.layers[i].window != nil else { continue }
                    let s = state[i]!.snapshot()
                    // Selection copies preserve NaN payloads and signed zero;
                    // arithmetic identity operations would not do so.
                    backing!.windows[i] = (
                        MLX.where(MLXArray(true), s.keys, s.keys, stream: stream),
                        MLX.where(MLXArray(true), s.values, s.values, stream: stream)
                    )
                }
                try fault.check()
            }
        } catch {
            close()
            throw error
        }
    }

    /// The same scalar reservation used by construction, so the slot cap can
    /// refuse a copy before constructing any native array graph.
    static func reservationFootprint(codec: CBv2CompleteCheckpointCodec, position: Int) throws
        -> (reservedBytes: Int, allocationBound: Int)
    {
        guard let layout = codec.contiguousLayout else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let wireDescriptors = try codec.tensorDescriptors(position: position)
        let descriptors =
            try codec.nativeTargetDescriptors(position: position)
            + Array(wireDescriptors.dropFirst(codec.targetTensorCount))
        // Wrap concatenation and compact copies can coexist. Include the bool
        // scalar for each Where and a conservative host owner/table envelope.
        // Two bounded dictionaries, weak owner boxes and array/control entries;
        // this is a conservative host envelope, not measured Swift heap bytes.
        var bytes = (64 << 10) + layout.layers.count * 512
        var allocationBound = 0
        for descriptor in descriptors
        where (descriptor.role == .keys || descriptor.role == .values)
            && layout.layers.contains(where: {
                $0.modelLayer == descriptor.layer && $0.window != nil
            })
        {
            let copy = try CBv2CheckpointAllocationFootprint.bound(descriptor.byteCount)
            allocationBound = try CBv2CheckpointAllocationFootprint.add(allocationBound, copy)
            bytes = try CBv2CheckpointAllocationFootprint.add(
                bytes,
                CBv2CheckpointAllocationFootprint.add(
                    CBv2CheckpointAllocationFootprint.add(copy, copy),
                    CBv2CheckpointAllocationFootprint.bound(1)))
        }
        if codec.assistant != nil {
            guard codec.assistant is any CBv2HistoricalMTPPrefixCheckpointCoding else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let auxiliary = Array(descriptors.dropFirst(codec.targetTensorCount))
            allocationBound = try CBv2CheckpointAllocationFootprint.add(
                allocationBound,
                CBv2HistoricalMTPCheckpointFootprint.nativeDestinationBound(auxiliary))
            bytes = try CBv2CheckpointAllocationFootprint.add(
                bytes,
                CBv2HistoricalMTPCheckpointFootprint.captureBytes(
                    position: position, descriptors: auxiliary))
        }
        return (bytes, allocationBound)
    }

    func markSubmitted() { submitted = true }

    /// Target windows were copied before a successor could overwrite them.
    /// The assistant is attached only after the actual committed observation
    /// fence, outside native commit locks and under this pre-reserved owner.
    func captureSettledAssistant(requestState: any CBv2MTPRequestState) throws {
        guard !closed, !completionFailed, case .pending = evaluation,
            requiresAssistant, let backing, backing.assistant == nil,
            let codec = capturingCodec,
            let assistant = codec.assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        submitted = true
        drained = false
        try withError { fault in
            guard
                let captured = assistant.capturePrefixCheckpoint(
                    requestState: requestState, targetInputCount: position),
                captured.targetInputCount == position
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            // Retain before checking a native error signalled during copy graph
            // construction; cleanup must not lose the actual partial owner.
            backing.assistant = captured
            try fault.check()
            let expected = Array(
                try codec.tensorDescriptors(position: position).dropFirst(codec.targetTensorCount))
            guard let arrays = assistant.encodePrefixCheckpoint(captured),
                arrays.count == expected.count,
                zip(arrays, expected).allSatisfy({
                    $0.0.shape == $0.1.shape && $0.0.dtype == $0.1.dtype.mlxDType
                })
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
    }

    func finishEvaluation() throws {
        guard !closed else { throw CBv2CompleteCheckpointError.closed }
        switch evaluation {
        case .ready: return
        case .failed: throw CBv2CompleteCheckpointError.allocationFailed
        case .pending: break
        }
        guard !requiresAssistant || backing?.assistant != nil else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        submitted = true
        do {
            try evaluate(evaluationRoots)
            try requiredDrain()
            let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(evaluationRoots)
            guard let backing, footprint.bound == backing.allocationBound,
                footprint.actual <= backing.allocationBound
            else { throw CBv2CompleteCheckpointError.allocationFailed }
            // Unique, row-contiguous, offset-zero, exact-extent native backing
            // proved by freshBytes. A view/oversized-parent result fails closed.
            backing.measuredBytes = footprint.actual
            evaluation = .ready
        } catch {
            // The owning step may have partially submitted these roots.
            // Drain issued work, never evaluate/retry a failed graph on close.
            do { if !drained { try requiredDrain() } } catch {
                evaluation = .failed
                throw error
            }
            evaluation = .failed
            throw error
        }
    }

    /// Discard is not publication. A cancelled interior capture can lack its
    /// assistant payload, but its already-submitted target copies still need
    /// real completion before any reservation is released.
    func finishEvaluationForRetirement() throws {
        guard !closed, !completionFailed else { throw CBv2CompleteCheckpointError.allocationFailed }
        if case .pending = evaluation {
            submitted = true
            do { try evaluate(evaluationRoots) } catch {
                evaluation = .failed
                throw error
            }
        }
        if submitted && !drained { try requiredDrain() }
    }

    private func requiredDrain() throws {
        guard !completionFailed else { throw CBv2CompleteCheckpointError.allocationFailed }
        do {
            try beforeRequiredDrainForTesting?()
            try withError { fault in
                stream.stream.synchronize()
                try fault.check()
            }
            drained = true
        } catch {
            completionFailed = true
            throw error
        }
    }

    func export(
        codec: CBv2CompleteCheckpointCodec, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard !closed, case .ready = evaluation, let backing,
            capturingCodec === codec, backing.measuredBytes != nil,
            let layout = codec.contiguousLayout, state.count == layout.layers.count
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var kv = [(keys: MLXArray, values: MLXArray, offset: Int)?](
            repeating: nil, count: state.count)
        var seen = Set<ObjectIdentifier>()
        for (i, layer) in layout.layers.enumerated() {
            if layer.owner != i {
                guard state[i] == nil else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                continue
            }
            guard let row = state[i], owners[i]?.row === row,
                seen.insert(ObjectIdentifier(row)).inserted, row.absoluteOffset >= position
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            if let window = layer.window {
                guard let ring = row as? CBv2WindowedSequenceKV, ring.window == window,
                    ring.kvHeads == layer.kvHeads, ring.headDim == layer.headDim,
                    ring.valueHeadDim == layer.valueHeadDim,
                    ring.retainedCount >= 0, ring.retainedCount <= min(ring.absoluteOffset, window)
                else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                // Plain deferred rollback can leave W-1 (or zero) CURRENT rows
                // after overwrite. Export the exact earlier immutable copy,
                // never revive discarded history or inspect empty placeholders.
                guard let pair = backing.windows[i] else {
                    throw CBv2CompleteCheckpointError.closed
                }
                kv[i] = (pair.0, pair.1, position)
            } else {
                guard row is CBv2FullSequenceKV || row is CBv2FrozenReplayFullSequenceKV else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                kv[i] = row.snapshot()
            }
        }
        // Unchanged export validation checks the ACTUAL captured SWA shape,
        // dtype and boundary plus every still-available full prefix, before
        // reserving manifest metadata or slicing. Capture's native compact/
        // independent-backing proof and shared lifetime owner remain in force.
        return try codec.exportContiguousV2(
            checkpoint: .init(
                position: position, chunkSize: chunkSize,
                layers: [:], byteCount: backing.assistant?.materializedBytes ?? 0,
                assistant: backing.assistant), kv: kv, tokens: tokens, cacheSalt: cacheSalt,
            retainedOwners: [backing] + state.compactMap { $0.map { $0 as AnyObject } })
    }

    func close() {
        guard !closed else { return }
        closed = true
        if submitted && !drained {
            if !completionFailed { try? requiredDrain() }
            if !drained, let backing { CBv2ContiguousCaptureQuarantine.shared.retain(backing) }
        }
        owners.removeAll()
        backing = nil
    }
    deinit { close() }
}

extension CBv2CompleteCheckpointCapture {
    func prepareContiguous(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?],
        allowance: CBv2HistoricalStagingAllowance? = nil
    ) throws
        -> CBv2CapturedCompleteCheckpoint?
    {
        guard !isClosed, codec.contiguousLayout != nil else { return nil }
        do {
            let descriptors = try codec.tensorDescriptors(position: position)
            let bytes = try descriptors.reduce(0) {
                try CBv2CheckpointAllocationFootprint.add($0, $1.byteCount)
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: bytes) else {
                return nil
            }
            let footprint = try CBv2ContiguousHistoricalCheckpoint.reservationFootprint(
                codec: codec, position: position)
            let shed = historicalStagingSheddable(allowance)
            guard
                let displaced = CBv2HistoricalStagingCap.displaced(
                    candidateBytes: footprint.reservedBytes,
                    slotBytes: stagedHistoricalBytes + max(0, inFlightHistoricalBytes),
                    replacingBytes: allowance?.replacingBytes ?? 0,
                    sheddable: shed.map(\.stagedHistoricalBytes), cap: historicalSlotStagedByteCap)
            else { return nil }
            let candidate = CBv2CapturedCompleteCheckpoint(
                contiguous: try makeContiguousCheckpoint(codec, position, chunkSize, state))
            inFlightHistoricalBytes +=
                candidate.stagedHistoricalBytes - (allowance?.replacingBytes ?? 0)
            if let allowance, displaced > 0 {
                releaseHistoricalStaging(
                    Array(shed.prefix(displaced)), requestID: allowance.requestID)
            }
            return candidate
        } catch let error as MLXError { throw error } catch { return nil }
    }
}
