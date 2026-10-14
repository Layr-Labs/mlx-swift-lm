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
        // Reserve before the first CPU row read/allocation. The donor's own
        // native completion owner remains responsible for its immutable roots.
        let scratch = try admission.reserveTransient(
            bytes: CBv2NativeCheckpointRowCodec.scratchBytes)
        defer { scratch.release() }
        let count = min(maximumBytes, bytes - byteOffset)
        let packedBytes = role.packedCount * role.packedRowBytes
        let tokenCount = role.position - role.tokenStart
        // Prepared once per segment (inside the reserved scratch); a packed
        // row is encoded straight into the result unless the segment splits it.
        var encoder: CBv2PreparedNativeCheckpointRowEncoder?
        var partial = Data()
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
                let token = inBand / rowBytes + (packed ? 0 : role.packedCount)
                let inRow = inBand % rowBytes
                let native = try source.readSegment(
                    descriptor: nativeDescriptor,
                    byteOffset: (head * tokenCount + token) * role.nativeRowBytes,
                    maximumBytes: role.nativeRowBytes, nativeWork: nativeWork)
                let length = min(count - copied, rowBytes - inRow)
                guard packed else {
                    guard native.count == rowBytes else {
                        throw CBv2CompleteCheckpointError.incompleteTransfer
                    }
                    native.withUnsafeBytes { source in
                        destination.baseAddress!.advanced(by: copied).copyMemory(
                            from: source.baseAddress!.advanced(by: inRow), byteCount: length)
                    }
                    copied += length
                    continue
                }
                if encoder == nil {
                    encoder = try CBv2PreparedNativeCheckpointRowEncoder(role: role)
                }
                guard let encoder, encoder.encodedRowBytes == rowBytes,
                    native.count == role.nativeRowBytes
                else {
                    throw CBv2CompleteCheckpointError.incompleteTransfer
                }
                try native.withUnsafeBytes { source in
                    if inRow == 0 && length == rowBytes {
                        try encoder.encode(
                            source,
                            into: UnsafeMutableRawBufferPointer(
                                rebasing: destination[copied ..< copied + rowBytes]))
                    } else {
                        if partial.count != rowBytes { partial = Data(count: rowBytes) }
                        try partial.withUnsafeMutableBytes { try encoder.encode(source, into: $0) }
                        partial.withUnsafeBytes { row in
                            destination.baseAddress!.advanced(by: copied).copyMemory(
                                from: row.baseAddress!.advanced(by: inRow), byteCount: length)
                        }
                    }
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

extension CBv2CompleteCheckpointCodec {
    func checkpointSource(
        _ source: CBv2CompleteCheckpointTensorSource, layer index: Int,
        position: Int, values: Bool
    ) throws -> CBv2CompleteCheckpointTensorSource {
        guard checkpointQuantization != nil else { return source }
        let role = try checkpointRole(layer: index, position: position, values: values)
        guard role.isQuantized else { return source }
        let native = try CBv2CheckpointPagedRoleLayout(
            key: checkpointGroupKey(layer: index), position: position,
            tokenStart: role.tokenStart, values: values
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
