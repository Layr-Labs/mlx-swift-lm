import Foundation
import MLX

/// A generation of original-precision rows. Consumers of its arrays must also
/// retain this owner: its admission charge ends only after the last consumer.
final class PagedKVNativeRecentOwner {
    let start: Int
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private let reservation: CBv2CheckpointReservation?
    private let admission: AdmissionV2?
    private var coverage: CBv2MemoryCoverage?
    private let stream: StreamOrDevice
    let reservedBytes: Int

    init(
        start: Int, keys: MLXArray, values: MLXArray,
        reservation: CBv2CheckpointReservation?, reservedBytes: Int, admission: AdmissionV2?
    ) {
        self.start = start
        self.keys = keys
        self.values = values
        self.reservation = reservation
        self.admission = admission
        self.reservedBytes = reservedBytes
        stream = .default
    }
    var count: Int { keys?.dim(1) ?? 0 }
    var evaluationRoots: [MLXArray] { [keys, values].compactMap { $0 } }
    func coverEvaluatedStorage() throws {
        guard coverage == nil, let admission, admission.hasProcessMemoryOwner else { return }
        var actual = 0
        for array in evaluationRoots {
            guard let info = try array.evaluatedBufferInfo(), info.isRowContiguous,
                info.dataOffset == 0, info.dataElements == array.size,
                info.allocatedBytes >= array.nbytes
            else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            actual = try CBv2CheckpointAllocationFootprint.add(actual, info.allocatedBytes)
        }
        guard actual <= reservedBytes else { throw CBv2CompleteCheckpointError.allocationFailed }
        coverage = try admission.coverEvaluatedAllocation(bytes: actual)
    }
    deinit {
        // Evaluation may have failed after submitting a partial copy. Keep its
        // real arrays and C/M coverage until the allocation stream has drained.
        stream.stream.synchronize()
        keys = nil
        values = nil
        coverage?.invalidate()
        reservation?.release()
    }
}

final class PagedKVRecentWeakRow {
    weak var row: PagedSequenceKV?
    init(_ row: PagedSequenceKV) { self.row = row }
}

extension PagedSequenceKV {
    /// Native block/canvas engines explicitly distinguish confirmed history
    /// from mutable pending rows. Ordinary causal CBv2 leaves this nil.
    package func setQuantizedConfirmedFrontier(_ position: Int?) {
        precondition(
            position == nil || (position! >= baseOffset && position! <= absoluteOffset),
            "quantized confirmed frontier outside the written row")
        quantizedConfirmedFrontier = position
    }

    /// Absolute beginning and compact [KV heads, tokens, channels] arrays.
    var nativeRecentStart: Int { nativeRecentOwner?.start ?? absoluteOffset }
    var nativeRecentKeys: MLXArray? { nativeRecentOwner?.keys }
    var nativeRecentValues: MLXArray? { nativeRecentOwner?.values }
    var nativeRecentStorageOwner: AnyObject? { nativeRecentOwner }
    var nativeRecentEvaluationRoots: [MLXArray] {
        nativeRecentOwner?.evaluationRoots ?? []
    }

    func nativeRecentRange(start: Int, count: Int) -> (keys: MLXArray, values: MLXArray)? {
        guard count >= 0, let owner = nativeRecentOwner,
            start >= owner.start, start + count <= absoluteOffset,
            start + count <= owner.start + owner.count,
            let keys = owner.keys, let values = owner.values
        else { return nil }
        let range = (start - owner.start) ..< (start - owner.start + count)
        return (keys[0..., range, 0...], values[0..., range, 0...])
    }

    /// Before the first storage mutation, reserve the fresh generation. During
    /// a prompt chunk all its original rows remain visible; during MTP the old
    /// confirmed tail survives every pending column until rollback/commit.
    func appendNativeRecent(keys: MLXArray, values: MLXArray) throws {
        guard let quantization = groupKey.quantization else { return }
        let old = nativeRecentOwner
        let oldCount = old.map { max(0, absoluteOffset - $0.start) } ?? 0
        let count = try CBv2CheckpointAllocationFootprint.add(oldCount, keys.dim(1))
        let maximum = try CBv2CheckpointAllocationFootprint.add(
            quantization.recentTokenCount,
            max(pool.config.maxPrefillChunk, CBv2PagedSpeculation.maxSpeculativeSpan))
        guard count <= maximum else {
            throw CBv2KVError.backendIneligible(
                reason: "native recent rows did not retire after the completed step")
        }
        let group = pool.group(groupKey)
        let firstPage = absoluteOffset / group.pageSize
        let lastPage = (absoluteOffset + keys.dim(1) - 1) / group.pageSize
        let transferSegments = Set(
            (firstPage ... lastPage).map { logical in
                group.segmentLayout!.segmentIndex(
                    page: table[ringPages.map { logical % $0 } ?? logical])
            })
        let transferOffsets = try PagedKVQuantizationConfig.multiply(
            transferSegments.count,
            CBv2CheckpointAllocationFootprint.bound(MemoryLayout<Int64>.stride))
        let owner = try makeNativeRecentOwner(
            start: old?.start ?? absoluteOffset,
            count: count,
            keyParts: (old?.keys.map { [$0[0..., ..<oldCount, 0...]] } ?? []) + [keys],
            valueParts: (old?.values.map { [$0[0..., ..<oldCount, 0...]] } ?? []) + [values],
            additionalReservedBytes: transferOffsets)
        if let old { retiredNativeRecentOwners.append(old) }
        nativeRecentOwner = owner
        pool.trackQuantizedRecent(self)
    }

    /// Caller supplies the actual imported native-tail DTO, never a dtype cast
    /// of packed bytes. Preparation occurs before the adopted row is published.
    func installNativeRecent(
        keys: MLXArray, values: MLXArray, start: Int, storedThrough: Int? = nil
    ) throws {
        guard let quantization = groupKey.quantization,
            keys.ndim == 3, values.ndim == 3,
            keys.dtype == groupKey.dtype, values.dtype == groupKey.dtype,
            keys.dim(0) == groupKey.kvHeads, values.dim(0) == groupKey.kvHeads,
            keys.dim(1) == values.dim(1), keys.dim(2) == groupKey.headDim,
            values.dim(2) == groupKey.valueHeadDim,
            start >= baseOffset, start + keys.dim(1) == (storedThrough ?? absoluteOffset),
            (storedThrough ?? absoluteOffset) <= maxLength,
            keys.dim(1) <= quantization.recentTokenCount
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let owner = try makeNativeRecentOwner(
            start: start, count: keys.dim(1),
            keyParts: [keys], valueParts: [values])
        if let old = nativeRecentOwner { retiredNativeRecentOwners.append(old) }
        nativeRecentOwner = owner
        pool.trackQuantizedRecent(self)
    }

    private func makeNativeRecentOwner(
        start: Int, count: Int, keyParts: [MLXArray], valueParts: [MLXArray],
        additionalReservedBytes: Int = 0
    ) throws -> PagedKVNativeRecentOwner {
        let keyBytes = try PagedKVQuantizationConfig.multiply(
            try PagedKVQuantizationConfig.multiply(groupKey.kvHeads, count),
            try PagedKVQuantizationConfig.multiply(groupKey.headDim, groupKey.dtype.size))
        let valueBytes = try PagedKVQuantizationConfig.multiply(
            try PagedKVQuantizationConfig.multiply(groupKey.kvHeads, count),
            try PagedKVQuantizationConfig.multiply(groupKey.valueHeadDim, groupKey.dtype.size))
        let bytes = try CBv2CheckpointAllocationFootprint.add(
            CBv2CheckpointAllocationFootprint.bound(keyBytes),
            CBv2CheckpointAllocationFootprint.bound(valueBytes))
        let booleanBytes =
            keyParts.count == 1
            ? try PagedKVQuantizationConfig.multiply(CBv2CheckpointAllocationFootprint.bound(1), 2)
            : 0
        let reserved = try CBv2CheckpointAllocationFootprint.add(
            CBv2CheckpointAllocationFootprint.add(bytes, booleanBytes), additionalReservedBytes)
        let reservation = try pool.memoryAdmission?.reserveTransient(bytes: reserved)
        // A single input may be a view of an entire prompt allocation. Force a
        // compact copy; concatenation already creates a fresh complete owner.
        func copied(_ parts: [MLXArray]) -> MLXArray {
            if parts.count > 1 { return concatenated(parts, axis: 1) }
            let input = parts[0]
            return MLX.where(MLXArray(true), input, input)
        }
        return PagedKVNativeRecentOwner(
            start: start, keys: copied(keyParts), values: copied(valueParts),
            reservation: reservation, reservedBytes: reserved, admission: pool.memoryAdmission)
    }

    fileprivate func prepareRecentRetirement() throws -> PagedKVNativeRecentOwner? {
        guard speculativeBase == nil, let config = groupKey.quantization,
            let old = nativeRecentOwner, let keys = old.keys, let values = old.values
        else { return nil }
        let confirmed = quantizedConfirmedFrontier ?? absoluteOffset
        let start = max(old.start, confirmed - config.recentTokenCount)
        let count = absoluteOffset - start
        if start == old.start, count == old.count { return nil }
        let offset = start - old.start
        return try makeNativeRecentOwner(
            start: start, count: count,
            keyParts: [keys[0..., offset ..< offset + count, 0...]],
            valueParts: [values[0..., offset ..< offset + count, 0...]])
    }

    fileprivate func installRecentRetirement(_ owner: PagedKVNativeRecentOwner?) {
        if let owner { nativeRecentOwner = owner }
        retiredNativeRecentOwners.removeAll()
    }
}

extension PagedKVPool {
    public var nativeRecentBytesInUse: Int {
        var owners = Set<ObjectIdentifier>()
        return quantizedRecentRows.compactMap(\.row).reduce(0) { total, row in
            ([row.nativeRecentOwner].compactMap { $0 } + row.retiredNativeRecentOwners).reduce(
                total
            ) { value, owner in
                owners.insert(ObjectIdentifier(owner)).inserted
                    ? value + owner.reservedBytes : value
            }
        }
    }

    func trackQuantizedRecent(_ row: PagedSequenceKV) {
        quantizedRecentRows.removeAll { $0.row == nil || $0.row!.isReleased }
        guard !quantizedRecentRows.contains(where: { $0.row === row }) else { return }
        quantizedRecentRows.append(PagedKVRecentWeakRow(row))
    }

    /// AFTER the actual step completion, and with no chained consumer in flight.
    /// Prepare every compact destination under its own charge, evaluate/fence
    /// once, then drop source owners and refund their charges. Never refund on
    /// a failed evaluation; the old generations remain owned by their rows.
    func finishQuantizedRecentStep() throws {
        quantizedRecentRows.removeAll { $0.row == nil || $0.row!.isReleased }
        let rows = quantizedRecentRows.compactMap(\.row)
        var prepared: [(PagedSequenceKV, PagedKVNativeRecentOwner?)] = []
        for row in rows where row.speculativeBase == nil {
            prepared.append((row, try row.prepareRecentRetirement()))
        }
        // Publish ownership before the first native evaluation: on a partial
        // failure the engine may retire rows later, but no local deinit can
        // refund destinations while Metal still holds an executing copy.
        for (row, owner) in prepared {
            if let owner { row.retiredNativeRecentOwners.append(owner) }
        }
        let roots = prepared.flatMap { $0.1?.evaluationRoots ?? [] }
        if !roots.isEmpty {
            try quantizedRecentEvaluate(roots)
        }
        for (row, owner) in prepared {
            try (owner ?? row.nativeRecentOwner)?.coverEvaluatedStorage()
            row.installRecentRetirement(owner)
        }
    }
}
