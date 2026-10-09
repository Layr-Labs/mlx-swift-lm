import Cmlx
import Foundation
import MLX

/// Exact native band at checkpoint M, copied while M is still available in
/// the current original-precision generation. Later tail compaction cannot
/// change this checkpoint's precision or retain a whole prompt allocation.
final class CBv2QuantizedCheckpointRecent: @unchecked Sendable {
    let start: Int
    let position: Int
    let reservedBytes: Int
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private var sourceOwner: AnyObject?
    private let reservation: CBv2CheckpointReservation?
    private let admission: AdmissionV2?
    private var coverage: CBv2MemoryCoverage?
    private let stream: StreamOrDevice
    private let lock = NSLock()
    private var ready = false

    var evaluationRoots: [MLXArray] { [keys, values].compactMap { $0 } }

    static func reservationBytes(row: PagedSequenceKV, position: Int) throws -> Int {
        guard let config = row.groupKey.quantization else { return 0 }
        let count = min(
            config.recentTokenCount, position - max(0, row.windowSize.map { position - $0 } ?? 0))
        guard count > 0 else { return 0 }
        return try [row.groupKey.headDim, row.groupKey.valueHeadDim].reduce(0) { total, width in
            let bytes = try PagedKVQuantizationConfig.multiply(
                PagedKVQuantizationConfig.multiply(row.groupKey.kvHeads, count),
                PagedKVQuantizationConfig.multiply(width, row.groupKey.dtype.size))
            return try CBv2CheckpointAllocationFootprint.add(
                total,
                CBv2CheckpointAllocationFootprint.add(
                    CBv2CheckpointAllocationFootprint.bound(bytes),
                    CBv2CheckpointAllocationFootprint.bound(1)))
        }
    }

    init(row: PagedSequenceKV, position: Int, admission: AdmissionV2) throws {
        guard let config = row.groupKey.quantization,
            let source = row.nativeRecentStorageOwner
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let tokenStart = row.windowSize.map { max(0, position - $0) } ?? 0
        let count = min(config.recentTokenCount, position - tokenStart)
        guard count > 0, let native = row.nativeRecentRange(start: position - count, count: count)
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        start = position - count
        self.position = position
        reservedBytes = try Self.reservationBytes(row: row, position: position)
        reservation = try admission.reserveTransient(bytes: reservedBytes)
        self.admission = admission
        stream = .default
        sourceOwner = source
        keys = MLX.where(MLXArray(true), native.keys, native.keys)
        values = MLX.where(MLXArray(true), native.values, native.values)
    }

    func finishEvaluation() throws {
        try lock.withLock {
            guard !ready else { return }
            try withError { fault in
                eval(evaluationRoots)
                try fault.check()
                stream.stream.synchronize()
                try fault.check()
            }
            var actual = 0
            for array in evaluationRoots {
                guard let info = try array.evaluatedBufferInfo(), info.isRowContiguous,
                    info.dataOffset == 0, info.dataElements == array.size,
                    mlx_array_data_uint8(array.ctx) != nil
                else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                actual = try CBv2CheckpointAllocationFootprint.add(actual, info.allocatedBytes)
            }
            guard actual <= reservedBytes else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            if let admission, admission.hasProcessMemoryOwner {
                coverage = try admission.coverEvaluatedAllocation(bytes: actual)
            }
            sourceOwner = nil
            ready = true
        }
    }

    func copy(
        values: Bool, head: Int, byteOffset: Int, count: Int,
        destination: UnsafeMutableRawPointer
    ) throws {
        try finishEvaluation()
        guard let array = values ? self.values : keys,
            head >= 0, head < array.dim(0), byteOffset >= 0,
            count > 0, byteOffset + count <= array.dim(1) * array.dim(2) * array.dtype.size,
            let pointer = mlx_array_data_uint8(array.ctx)
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let stride = array.dim(1) * array.dim(2) * array.dtype.size
        destination.copyMemory(
            from: UnsafeRawPointer(pointer).advanced(by: head * stride + byteOffset),
            byteCount: count)
    }

    deinit {
        // A discarded asynchronous capture still retains its source charge
        // until submitted device readers have drained and every alias drops.
        try? stream.stream.synchronize()
        keys = nil
        values = nil
        sourceOwner = nil
        coverage?.invalidate()
        reservation?.release()
    }
}

extension CBv2CompleteCheckpointCapture {
    /// Newly captured recurrent tails rode the current step's sole evaluator.
    /// Detach their old prompt generations after completion, before the live
    /// rows' compaction refunds those generations.
    func finishQuantizedRecentCaptures() throws {
        for captures in staged.values {
            for capture in captures {
                if let checkpoint = capture.checkpoint {
                    for recent in checkpoint.quantizedRecent.values {
                        try recent.finishEvaluation()
                    }
                }
            }
        }
    }
}

extension CBv2CompleteCheckpointCodec {
    func captureQuantizedRecent(
        state: [CBv2SequenceKV?], position: Int, includeWindows: Bool = true
    ) throws
        -> [Int: CBv2QuantizedCheckpointRecent]
    {
        guard pagedConfig?.quantization != nil else { return [:] }
        guard state.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var captured: [Int: CBv2QuantizedCheckpointRecent] = [:]
        for (index, entry) in state.enumerated()
        where checkpointGroupKey(layer: index).quantization != nil
            && layerKinds[index].sharesKVWithLayer == nil
        {
            if !includeWindows, case .slidingWindow = layerKinds[index].attention { continue }
            guard let row = entry as? PagedSequenceKV,
                row.groupKey == checkpointGroupKey(layer: index)
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            if row.groupKey.quantization!.recentTokenCount > 0 {
                captured[index] = try .init(row: row, position: position, admission: admission)
            }
        }
        return captured
    }
}
