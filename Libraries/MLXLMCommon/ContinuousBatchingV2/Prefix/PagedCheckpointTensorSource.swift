import Cmlx
import Foundation
import MLX

/// Immutable page-map capture, built on the engine queue while the donor row
/// remains charged and pinned. Export never reads mutable pool dictionaries on
/// the I/O queue. Full-attention prefix pages remain immutable during donation;
/// a rolling window requires a different boundary capture and is refused here.
final class CBv2PagedCheckpointTensorSource {
    private let key: PagedKVGroupKey
    private let pageSize: Int
    private let position: Int
    private let values: Bool
    private var roleWidth: Int { values ? key.valueHeadDim : key.headDim }
    private var pageMap: CBv2PagedCheckpointPageMap?
    private var recent: CBv2QuantizedCheckpointRecent?
    private let role: CBv2CheckpointPagedRoleLayout
    let byteCount: Int

    convenience init(row: PagedSequenceKV, position: Int, values: Bool, admission: AdmissionV2)
        throws
    {
        try self.init(
            pageMap: .init(row: row, position: position, admission: admission), values: values,
            recent: row.groupKey.quantization?.recentTokenCount ?? 0 > 0
                ? .init(row: row, position: position, admission: admission) : nil)
    }

    convenience init(
        storage: CBv2PagedCheckpointStorage, layerIndex: Int, values: Bool,
        admission: AdmissionV2
    ) throws {
        guard storage.plan.layers.indices.contains(layerIndex) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let layer = storage.plan.layers[layerIndex]
        guard layer.ringPages == nil, layer.tokenStart == 0, layer.key.quantization == nil else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard let group = storage.groups[layer.key] else {
            throw CBv2CompleteCheckpointError.closed
        }
        try self.init(
            pageMap: .init(
                key: layer.key, pageSize: storage.plan.pageSize, position: storage.plan.position,
                table: group.pages[layer.firstPage ..< layer.firstPage + layer.pageCount],
                layout: group.layout, segments: group.segments,
                previous: MLXArray.zeros([1], dtype: .int32),
                admission: admission), values: values)
    }

    init(
        pageMap: CBv2PagedCheckpointPageMap, values: Bool,
        recent: CBv2QuantizedCheckpointRecent? = nil
    ) throws {
        self.key = pageMap.key
        self.pageSize = pageMap.pageSize
        self.position = pageMap.position
        self.values = values
        self.pageMap = pageMap
        role = try .init(key: pageMap.key, position: pageMap.position, values: values)
        if role.isQuantized, role.nativeCount > 0 {
            guard let recent, recent.position == pageMap.position,
                recent.start == role.nativeStart
            else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
        }
        self.recent = recent
        byteCount = try PagedKVQuantizationConfig.multiply(key.kvHeads, role.bytesPerHead)
    }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        guard let expected = try? role.descriptor(layer: descriptor.layer ?? 0) else {
            return false
        }
        return descriptor.byteCount == expected.byteCount && descriptor.shape == expected.shape
            && descriptor.dtype == expected.dtype && descriptor.role == expected.role
    }

    /// The enclosing export serializes reads and close. The supported Metal
    /// allocator exposes shared CPU-readable storage, so only the returned
    /// provider-owned Data is allocated. No full backing view or GPU gather is
    /// constructed; the page map pins every source throughout the copy.
    func retainForNativeExport(_ work: CBv2NativeCompletePrefixWork) throws {
        try work.requireNativePagedSources()
        guard let pageMap else { throw CBv2CompleteCheckpointError.closed }
        try work.retain(arrays: [pageMap.previous], owners: [self, pageMap])
    }

    func readSegment(byteOffset: Int, maximumBytes: Int) throws -> Data {
        try readSegment(byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nil)
    }

    func readSegment(
        byteOffset: Int, maximumBytes: Int,
        nativeWork: CBv2NativeCompletePrefixWork?
    ) throws -> Data {
        guard let pageMap else { throw CBv2CompleteCheckpointError.closed }
        let width = role.isQuantized ? 1 : key.dtype.size
        guard byteOffset >= 0, byteOffset < byteCount, byteOffset % width == 0,
            maximumBytes > 0, maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let count = min(byteCount - byteOffset, maximumBytes - maximumBytes % width)
        guard count > 0 else { throw CBv2CompleteCheckpointError.invalidSegment }
        if let nativeWork {
            try retainForNativeExport(nativeWork)
            try nativeWork.captureCurrentStreams()
            try pageMap.prepareForReading { array in
                do {
                    try withError { fault in
                        eval(array)
                        try fault.check()
                    }
                } catch {
                    nativeWork.requiredCompletionFailed()
                    throw error
                }
            }
        } else {
            try pageMap.prepareForReading()
        }
        if role.isQuantized {
            return try readQuantized(pageMap: pageMap, byteOffset: byteOffset, count: count)
        }
        var result = Data(count: count)
        try result.withUnsafeMutableBytes { destination in
            try CBv2PagedCheckpointByteLayout.runs(
                headDim: roleWidth, position: position, pageSize: pageSize,
                itemSize: width, byteOffset: byteOffset, count: count
            ) { logicalPage, head, slot, feature, packedOffset, length in
                let page = pageMap[logicalPage]
                let segment = page.segment
                let source =
                    ((page.localPage * key.kvHeads + head) * pageSize + slot)
                    * roleWidth + feature + (values ? segment.valueOffset : 0)
                guard let pointer = mlx_array_data_uint8(segment.storage.ctx) else {
                    nativeWork?.requiredCompletionFailed()
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                destination.baseAddress!.advanced(by: packedOffset).copyMemory(
                    from: UnsafeRawPointer(pointer).advanced(by: source * width), byteCount: length)
            }
        }
        return result
    }

    private func readQuantized(pageMap: CBv2PagedCheckpointPageMap, byteOffset: Int, count: Int)
        throws -> Data
    {
        var result = Data(count: count)
        try result.withUnsafeMutableBytes { destination in
            var offset = byteOffset
            var copied = 0
            let packedBytes = role.packedCount * role.packedRowBytes
            while copied < count {
                let head = offset / role.bytesPerHead
                let inHead = offset % role.bytesPerHead
                let length: Int
                if inHead < packedBytes {
                    let token = inHead / role.packedRowBytes
                    let feature = inHead % role.packedRowBytes
                    let slot = token % pageSize
                    let page = pageMap[token / pageSize]
                    let segment = page.segment
                    let address =
                        ((page.localPage * key.kvHeads + head) * pageSize + slot)
                        * role.packedRowBytes + feature + (values ? segment.valueOffset : 0)
                    guard let pointer = mlx_array_data_uint8(segment.storage.ctx) else {
                        throw CBv2CompleteCheckpointError.allocationFailed
                    }
                    length = min(
                        count - copied, packedBytes - inHead,
                        (pageSize - slot) * role.packedRowBytes - feature)
                    destination.baseAddress!.advanced(by: copied).copyMemory(
                        from: UnsafeRawPointer(pointer).advanced(by: address), byteCount: length)
                } else {
                    guard let recent else { throw CBv2CompleteCheckpointError.incompleteTransfer }
                    length = min(count - copied, role.bytesPerHead - inHead)
                    try recent.copy(
                        values: values, head: head, byteOffset: inHead - packedBytes,
                        count: length, destination: destination.baseAddress!.advanced(by: copied))
                }
                copied += length
                offset += length
            }
        }
        return result
    }

    func close() {
        pageMap = nil
        recent = nil
    }
}
