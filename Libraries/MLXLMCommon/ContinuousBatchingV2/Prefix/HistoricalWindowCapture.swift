import Cmlx
import Foundation
import MLX

/// Immutable native-byte copy before a successor writes its donor ring.
/// Equal widths keep the original [2,1,H,W,D] allocation; asymmetric roles own
/// separate [1,H,W,Dk/Dv] buffers. No materialization credit comes from aliases.
/// The conservative charge survives the last exported K/V source.
final class CBv2HistoricalWindow: @unchecked Sendable {
    let start: Int
    let position: Int
    let heads: Int
    let headDim: Int
    let valueHeadDim: Int
    let dtype: DType
    /// Transient admission bytes charged for this window until retirement.
    let reservedBytes: Int
    private var combined: MLXArray?
    private var asymmetricKeys: MLXArray?
    private var asymmetricValues: MLXArray?
    private var reservation: CBv2CheckpointReservation?
    private enum Evaluation { case pending, ready, failed }
    private var evaluation = Evaluation.pending
    private var submitted = false
    private var drained = false
    // Snapshot the task-local default once. Retirement may run outside its
    // construction scope; neither a later default nor global GPU/CPU is ours.
    let copyStream: StreamOrDevice
    private let synchronize: (StreamOrDevice) throws -> Void
    private let evaluate: (MLXArray) throws -> Void
    // Preserve the legacy one-root observation for equal-width captures.
    var evaluationRoot: MLXArray? { combined }
    // Each Depends result must itself be observed as evaluated, even though
    // both roots reach the same final copy fence. No whole-output concat/cast.
    var evaluationRoots: [MLXArray] {
        if let combined { return [combined] }
        return [asymmetricKeys, asymmetricValues].compactMap { $0 }
    }

    static func reservationBytes(row: PagedSequenceKV, position: Int) throws -> Int {
        guard let window = row.windowSize, row.pool.segmentGrant != nil,
            position <= row.absoluteOffset, position > 1, position <= Int(Int32.max)
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        // An interior position (behind the frontier) is exact only while the
        // ring still holds its whole window: the ring keeps
        // `max(window + speculative span, maxPrefillChunk)` tokens behind the
        // highest written position, so a 1,024-aligned point inside a 2,048
        // stripe qualifies for gpt-oss (W = 128) and, at the stripe's own
        // alignment, for gemma-4 (W = 1,024). Anything older is refused here.
        let count = min(window, position)
        guard position - count >= row.oldestValidPosition else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let key = row.groupKey
        func add(_ a: Int, _ b: Int) throws -> Int {
            try CBv2CheckpointAllocationFootprint.add(a, b)
        }
        func multiply(_ a: Int, _ b: Int) throws -> Int {
            guard let value = CBv2KVGeometry.multiply(a, b) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            return value
        }
        func bound(_ bytes: Int) throws -> Int {
            try Memory.allocationFootprintUpperBound(byteCount: bytes)
        }
        let output: Int
        let scalar: Int
        if key.isAsymmetric {
            let keys = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: [1, key.kvHeads, count, key.headDim], dtype: key.dtype)
            let values = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: [1, key.kvHeads, count, key.valueHeadDim], dtype: key.dtype)
            // Distinct output/zero-scalar allocations: never bound the sum as
            // if it were a single buffer or discount V to the K width.
            output = try add(bound(keys), bound(values))
            scalar = try multiply(2, bound(key.dtype.size))
        } else {
            let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: [2, 1, key.kvHeads, count, key.headDim], dtype: key.dtype)
            output = try bound(bytes)
            scalar = try bound(key.dtype.size)
        }
        let numerator = try add(count - 1, row.pool.config.pageSize - 1)
        let pageSpan = numerator / row.pool.config.pageSize + 1
        let segments = min(pageSpan, row.pool.group(row.groupKey).segments.count)
        // Each bucket has <=count records. Bound separately because cached
        // buffers may be larger than logical sizes; a total-byte bound is wrong.
        let records = try bound(multiply(max(24, multiply(count, 3)), MemoryLayout<Int32>.stride))
        let fence = try bound(MemoryLayout<Int32>.stride)
        let hostRecords = try multiply(multiply(count, 4), MemoryLayout<Int32>.stride)
        let hostPages = try multiply(pageSpan, MemoryLayout<Int32>.stride)
        let host = try add(64 << 10, multiply(4, add(hostRecords, hostPages)))
        let transfer = try multiply(segments, add(records, fence))
        return try add(transfer, add(output, add(scalar, add(fence, host))))
    }

    init(
        row: PagedSequenceKV, position: Int, admission: AdmissionV2,
        stream: StreamOrDevice = .default,
        beforeAllocation: () throws -> Void = {},
        afterConstruction: (MLXArray) throws -> Void = { _ in },
        evaluate: @escaping (MLXArray) throws -> Void = { array in try withError { eval(array) } },
        synchronize: @escaping (StreamOrDevice) throws -> Void = { stream in
            try withError { stream.stream.synchronize() }
        }
    ) throws {
        let bytes = try Self.reservationBytes(row: row, position: position)
        let permit = try admission.reserveTransient(bytes: bytes)
        self.position = position
        reservedBytes = bytes
        start = max(0, position - row.windowSize!)
        heads = row.groupKey.kvHeads
        headDim = row.groupKey.headDim
        valueHeadDim = row.groupKey.valueHeadDim
        dtype = row.groupKey.dtype
        reservation = permit
        self.evaluate = evaluate
        copyStream = stream
        self.synchronize = synchronize
        do {
            // No per-capture page list, host records or lazy GPU graph exists
            // before the permit. Production never calls the fault seam.
            try beforeAllocation()
            try withError {
                let pageSize = row.pool.config.pageSize
                let pages = (start / pageSize ... (position - 1) / pageSize).map {
                    row.table[$0 % row.ringPages!]
                }
                let group = row.pool.group(row.groupKey)
                if row.groupKey.isAsymmetric {
                    let copied = PagedAsymmetricTransfers.gatherSegmented(
                        group: group, pages: pages, firstSlot: start % pageSize,
                        count: position - start, publishReadFence: false, stream: copyStream)
                    asymmetricKeys = copied.keys
                    asymmetricValues = copied.values
                } else {
                    combined = PagedSegmentTransfers.gatherCombined(
                        group: group, pages: pages,
                        firstSlot: start % pageSize, count: position - start,
                        publishReadFence: false,
                        stream: copyStream)
                }
                for root in evaluationRoots { try afterConstruction(root) }
            }
        } catch {
            // Construction submits no copy work and never publishes a group
            // fence. Private graph locals have unwound; no serving alias can
            // retain this destination after the output is dropped.
            evaluation = .failed
            combined = nil
            asymmetricKeys = nil
            asymmetricValues = nil
            reservation = nil
            throw error
        }
    }

    /// Called by the owning step before any successor may build cache writes.
    /// A failed private copy cannot poison the target's serving write fence.
    func markSubmitted() { submitted = true }

    func finishEvaluation() throws {
        switch evaluation {
        case .ready: return
        case .failed: throw CBv2CompleteCheckpointError.allocationFailed
        case .pending: break
        }
        let roots = evaluationRoots
        guard roots.count == (headDim == valueHeadDim ? 1 : 2) else {
            throw CBv2CompleteCheckpointError.closed
        }
        do {
            submitted = true
            for root in roots { try evaluate(root) }
            evaluation = .ready
        } catch {
            // asyncEval may already have submitted part of the private copy.
            // Drain issued work; never retry the failed graph during retirement.
            var failure = error
            do {
                try synchronize(copyStream)
                drained = true
            } catch { failure = error }
            evaluation = .failed
            throw failure
        }
    }

    func read(values: Bool, byteOffset: Int, maximumBytes: Int) throws -> Data {
        guard let array = combined ?? (values ? asymmetricValues : asymmetricKeys) else {
            throw CBv2CompleteCheckpointError.closed
        }
        let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
            shape: [1, heads, position - start, values ? valueHeadDim : headDim], dtype: dtype)
        guard byteOffset >= 0, byteOffset < bytes, byteOffset % dtype.size == 0,
            maximumBytes >= dtype.size,
            maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        // Launch submitted the completion edge on the engine's sole evaluator.
        // Readback here only waits for that already detached immutable output.
        try finishEvaluation()
        guard let info = try array.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataOffset == 0, info.dataElements == array.size,
            let pointer = mlx_array_data_uint8(array.ctx)
        else { throw CBv2CompleteCheckpointError.allocationFailed }
        let count = min(maximumBytes - maximumBytes % dtype.size, bytes - byteOffset)
        let roleOffset = values && combined != nil ? bytes : 0
        return Data(
            bytes: UnsafeRawPointer(pointer).advanced(by: roleOffset + byteOffset), count: count)
    }

    deinit {
        // eval() can return before Metal completion handlers drop their data
        // references. Keep the output and permit through that final retirement
        // even for a successful .ready capture; reads need no additional wait.
        if submitted && !drained {
            // Drain only submitted work, never evaluate/retry the graph. An
            // evaluation failure already drained successfully is not waited twice.
            try? synchronize(copyStream)
        }
        combined = nil
        asymmetricKeys = nil
        asymmetricValues = nil
        reservation?.release()
        reservation = nil
    }
}

final class CBv2HistoricalWindowTensorSource {
    private var window: CBv2HistoricalWindow?
    private let values: Bool
    init(window: CBv2HistoricalWindow, values: Bool) {
        self.window = window
        self.values = values
    }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        guard let window else { return false }
        return descriptor.role == (values ? .values : .keys)
            && descriptor.dtype.mlxDType == window.dtype
            && descriptor.shape == [
                1, window.heads, window.position - window.start,
                values ? window.valueHeadDim : window.headDim,
            ]
    }

    func retainForNativeExport(_ work: CBv2NativeCompletePrefixWork) throws {
        try work.requireNativePagedSources()
        guard let window else { throw CBv2CompleteCheckpointError.closed }
        try work.retain(arrays: window.evaluationRoots, owners: [self, window])
    }

    func readSegment(byteOffset: Int, maximumBytes: Int) throws -> Data {
        try readSegment(byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nil)
    }

    func readSegment(
        byteOffset: Int, maximumBytes: Int,
        nativeWork: CBv2NativeCompletePrefixWork?
    ) throws -> Data {
        guard let window else { throw CBv2CompleteCheckpointError.closed }
        if let nativeWork {
            try retainForNativeExport(nativeWork)
            try nativeWork.captureCurrentStreams()
        }
        do {
            return try window.read(
                values: values, byteOffset: byteOffset, maximumBytes: maximumBytes)
        } catch {
            if error is MLXError || (error as? CBv2CompleteCheckpointError) == .allocationFailed {
                nativeWork?.requiredCompletionFailed()
            }
            throw error
        }
    }

    func close() { window = nil }
}
