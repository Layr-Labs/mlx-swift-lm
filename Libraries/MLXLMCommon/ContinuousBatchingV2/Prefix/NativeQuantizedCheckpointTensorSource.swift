import Foundation
import MLX

/// Decorates an immutable native source. Existing live-packed sources never
/// reach this encoder and keep their byte-exact packed mirror semantics.
final class CBv2NativeQuantizedCheckpointTensorSource {
    private var source: CBv2CompleteCheckpointTensorSource?
    private let nativeDescriptor: CBv2CheckpointTensorDescriptor
    private let role: CBv2CheckpointPagedRoleLayout
    private let admission: AdmissionV2

    init(
        source: CBv2CompleteCheckpointTensorSource,
        nativeDescriptor: CBv2CheckpointTensorDescriptor,
        role: CBv2CheckpointPagedRoleLayout, admission: AdmissionV2
    ) throws {
        guard role.isQuantized, source.matches(nativeDescriptor),
            nativeDescriptor.dtype.isFloatingPoint
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.source = source
        self.nativeDescriptor = nativeDescriptor
        self.role = role
        self.admission = admission
    }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        (try? role.descriptor(layer: nativeDescriptor.layer ?? 0)) == descriptor
    }

    func retainForNativeExport(_ work: CBv2NativeCompletePrefixWork) throws {
        guard let source else { throw CBv2CompleteCheckpointError.closed }
        try source.retainForNativeExport(work)
    }

    func readSegment(
        byteOffset: Int, maximumBytes: Int, nativeWork: CBv2NativeCompletePrefixWork?
    ) throws -> Data {
        guard let source else { throw CBv2CompleteCheckpointError.closed }
        let bytes = try PagedKVQuantizationConfig.multiply(role.key.kvHeads, role.bytesPerHead)
        guard byteOffset >= 0, byteOffset < bytes, maximumBytes > 0,
            maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let batchRows = max(
            1, CBv2NativeCheckpointRowCodec.exportBatchBytes / max(1, role.nativeRowBytes))
        // Reserve before the first CPU row read/allocation: one batch of native
        // rows and its coded rows beside the encoders' row buffers. The donor's
        // own native completion owner remains responsible for its immutable roots.
        let scratch = try admission.reserveTransient(
            bytes: CBv2NativeCheckpointRowCodec.scratchBytes
                + batchRows * (role.nativeRowBytes + role.packedRowBytes))
        defer { scratch.release() }
        let count = min(maximumBytes, bytes - byteOffset)
        let packedBytes = role.packedCount * role.packedRowBytes
        let coding = try CBv2NativeCheckpointBatchCoding(
            role: role, workers: CBv2NativeCheckpointRowCodec.exportCodingWorkers)
        var coded = Data(count: batchRows * role.packedRowBytes)
        let tokenCount = role.position - role.tokenStart
        var result = Data(count: count)
        try result.withUnsafeMutableBytes { destination in
            var copied = 0
            while copied < count {
                let offset = byteOffset + copied
                let head = offset / role.bytesPerHead
                let inHead = offset % role.bytesPerHead
                let packed = inHead < packedBytes
                let inBand = packed ? inHead : inHead - packedBytes
                let rowBytes = packed ? role.packedRowBytes : role.nativeRowBytes
                let bandRows = packed ? role.packedCount : role.nativeCount
                let firstRow = inBand / rowBytes
                let inRow = inBand % rowBytes
                // One source read covers consecutive rows of this head's band:
                // the read's fixed cost (native binding checks, evaluation, a
                // buffer) is paid per batch, not per row.
                let neededEnd = min(bandRows * rowBytes, inBand + count - copied)
                let rows = min(batchRows, (neededEnd + rowBytes - 1) / rowBytes - firstRow)
                let token = firstRow + (packed ? 0 : role.packedCount)
                let native = try source.readSegment(
                    descriptor: nativeDescriptor,
                    byteOffset: (head * tokenCount + token) * role.nativeRowBytes,
                    maximumBytes: rows * role.nativeRowBytes, nativeWork: nativeWork)
                guard native.count == rows * role.nativeRowBytes else {
                    throw CBv2CompleteCheckpointError.incompleteTransfer
                }
                if packed { try coding.code(native, rows: rows, into: &coded) }
                let length = min(count - copied, rows * rowBytes - inRow)
                (packed ? coded : native).withUnsafeBytes { band in
                    destination.baseAddress!.advanced(by: copied).copyMemory(
                        from: band.baseAddress!.advanced(by: inRow), byteCount: length)
                }
                copied += length
            }
        }
        return result
    }

    func close() {
        source?.close()
        source = nil
    }
}

/// Codes one batch of native rows on up to `workers` cores. Each worker owns
/// one prepared encoder and writes only its own rows, so the bytes and the
/// first refusal are those of coding the rows one by one. One instance per
/// concurrent caller.
final class CBv2NativeCheckpointBatchCoding: @unchecked Sendable {
    private let role: CBv2CheckpointPagedRoleLayout
    private let encoders: [CBv2PreparedNativeCheckpointRowEncoder]
    /// Fewer rows per worker cost more to dispatch than they save.
    private static let minimumRowsPerWorker = 16

    init(role: CBv2CheckpointPagedRoleLayout, workers: Int) throws {
        self.role = role
        encoders = try (0 ..< max(1, workers)).map { _ in
            try CBv2PreparedNativeCheckpointRowEncoder(role: role)
        }
    }

    func code(_ native: Data, rows: Int, into coded: inout Data) throws {
        guard rows > 0, native.count == rows * role.nativeRowBytes,
            coded.count >= rows * role.packedRowBytes
        else { throw CBv2CompleteCheckpointError.incompleteTransfer }
        let workers = min(encoders.count, max(1, rows / Self.minimumRowsPerWorker))
        try native.withUnsafeBytes { native in
            try coded.withUnsafeMutableBytes { coded in
                let batch = Batch(
                    native: native.baseAddress!, coded: coded.baseAddress!, rows: rows,
                    workers: workers)
                guard workers > 1 else { return try code(batch, worker: 0) }
                let failures = Failures()
                DispatchQueue.concurrentPerform(iterations: workers) { worker in
                    do { try self.code(batch, worker: worker) } catch {
                        failures.record(worker, error)
                    }
                }
                if let failure = failures.first { throw failure }
            }
        }
    }

    private struct Batch: @unchecked Sendable {
        let native: UnsafeRawPointer
        let coded: UnsafeMutableRawPointer
        let rows: Int
        let workers: Int
    }

    private func code(_ batch: Batch, worker: Int) throws {
        let encoder = encoders[worker]
        let start = batch.rows * worker / batch.workers
        let end = batch.rows * (worker + 1) / batch.workers
        for row in start ..< end {
            try encoder.encode(
                UnsafeRawBufferPointer(
                    start: batch.native + row * role.nativeRowBytes, count: role.nativeRowBytes),
                into: UnsafeMutableRawBufferPointer(
                    start: batch.coded + row * role.packedRowBytes, count: role.packedRowBytes))
        }
    }

    /// The lowest worker's refusal: the one a row-by-row pass would meet first.
    private final class Failures: @unchecked Sendable {
        private let lock = NSLock()
        private var failures: [Int: Error] = [:]
        func record(_ worker: Int, _ error: Error) { lock.withLock { failures[worker] = error } }
        var first: Error? { lock.withLock { failures.min { $0.key < $1.key }?.value } }
    }
}

extension CBv2CompleteCheckpointCodec {
    func checkpointSource(
        _ source: CBv2CompleteCheckpointTensorSource, layer index: Int,
        position: Int, values: Bool
    ) throws -> CBv2CompleteCheckpointTensorSource {
        guard checkpointQuantization != nil else { return source }
        let role = try checkpointRole(layer: index, position: position, values: values)
        guard role.isQuantized else { return source }
        // The rows the encoder reads: the role's own dtype, which is the
        // pool's, or GLM's narrower latent wire dtype.
        let key = role.key
        let native = try CBv2CheckpointPagedRoleLayout(
            key: PagedKVGroupKey(
                kvHeads: key.kvHeads, headDim: key.headDim, dtype: key.dtype,
                windowSize: key.windowSize, valueHeadDim: key.valueHeadDim),
            position: position, tokenStart: role.tokenStart, values: values
        ).descriptor(layer: layerKinds[index].modelLayerIndex ?? index)
        return .nativeQuantized(
            try .init(
                source: source, nativeDescriptor: native, role: role, admission: admission))
    }

    func checkpointSources(arrays: [MLXArray], position: Int) throws
        -> [CBv2CompleteCheckpointTensorSource]
    {
        let owners =
            (contiguousLayout ?? historicalLayout)?.owningIndices ?? Array(layerKinds.indices)
        guard arrays.count >= owners.count * 2 else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        return try arrays.enumerated().map { cursor, array in
            guard cursor < owners.count * 2 else { return .array(array) }
            return try checkpointSource(
                .array(array), layer: owners[cursor / 2], position: position,
                values: cursor % 2 == 1)
        }
    }
}
