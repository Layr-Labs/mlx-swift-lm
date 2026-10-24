import Foundation
import MLX

extension CBv2CompleteCheckpointCodec {
    func glm5NextReusePlan(position: Int, maximumSequenceLength: Int) throws -> CBv2PrefixReusePlan
    {
        guard let glm5NextLayout else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let descriptors = try glm5NextLayout.attention.tensorDescriptors(position: position)
        let bytes = try descriptors.reduce(0) {
            try CBv2CheckpointAllocationFootprint.add($0, $1.byteCount)
        }
        // `.pagedFP16` names the unquantized paged adoption contract, not the
        // element width: GLM rows are FP32. Both per-token byte inputs below
        // are exact (staged descriptors and FP32 admission), so the
        // capability's FP16 nominal width never prices this plan.
        guard
            var plan = CBv2PrefixReuseCapability.derive(layerKinds: layerKinds, backend: .pagedFP16)
                .plan(
                    matchedBoundary: position, exactStagedFullKVBytes: bytes,
                    maximumSequenceLength: maximumSequenceLength,
                    nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
                    reserveFullSequenceTokens: true)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        plan.excludesPackedPrefill = true
        return plan
    }

    /// GLM eligibility checks precede the ordinary paged exporter; packing,
    /// assistant coding and manifest ownership have only one implementation.
    func validateGLM5NextDonor(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?]
    ) throws {
        guard let layout = glm5NextLayout, let pagedConfig,
            state.count == layerKinds.count,
            checkpoint.qwen4.isEmpty, checkpoint.quantizedRecent.isEmpty,
            Set(checkpoint.layers.keys) == Set(layout.recurrentLayers.map(\.modelLayer))
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV, !row.isReleased,
                row.pool === layout.backend.pool, row.pool.memoryAdmission === admission,
                row.pool.layerKinds == layerKinds,
                row.pool.glm5NextPayloadBinding === layout.binding,
                row.pool.config.pageSize == pagedConfig.pageSize,
                row.pool.config.segmentSizeBytes == pagedConfig.segmentSizeBytes,
                row.pool.activeAttentionWork == nil, row.speculativeBase == nil,
                row.groupKey == row.pool.groupKey(forLayer: index),
                row.groupKey.dtype == .float32, row.groupKey.quantization == nil
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
    }
}

extension CBv2CompleteCheckpointCodec {
    /// Import destinations: the manifest's own descriptors, except GLM roles
    /// written narrower than their FP32 pages and state, which import widens.
    func nativeDescriptors(for manifest: CBv2CompleteCheckpointManifest) throws
        -> [CBv2CheckpointTensorDescriptor]
    {
        guard let layout = glm5NextLayout, layout.narrowsPayload else { return manifest.tensors }
        let native = try layout.nativeDescriptors(position: manifest.position)
        guard manifest.tensors.count >= native.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return native + manifest.tensors.dropFirst(native.count)
    }
}

/// Wire contents that shape checks alone cannot establish. Validate at the
/// authenticated segment boundary, before a byte reaches native destinations.
enum CBv2GLM5NextCheckpointPayload {
    static func validate(
        manifest: CBv2CompleteCheckpointManifest, tensorIndex: Int,
        byteOffset: Int, data: Data
    ) throws {
        guard manifest.backendLayout == CBv2CompleteCheckpointManifest.glm5NextPagedLayout,
            let state = manifest.glm5NextState
        else { return }
        let descriptor = manifest.tensors[tensorIndex]
        guard descriptor.role == .values else { return }
        let width = 2 * state.indexWidth + 1
        let size = descriptor.dtype.mlxDType.size
        // The validity column is located by element index; a segment that is
        // not whole, aligned elements cannot be classified.
        guard descriptor.dtype == .float32 || CBv2CheckpointHalfPrecision.isHalf(descriptor.dtype),
            byteOffset >= 0, byteOffset % size == 0, data.count % size == 0
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        // Visit only the validity column: the first element at or after the
        // segment start whose row position is `width - 1`, then every row.
        let first = byteOffset / size
        try data.withUnsafeBytes { bytes in
            for element in stride(
                from: first + (width - 1 - first % width), to: first + data.count / size,
                by: width)
            {
                let offset = (element - first) * size
                let value: Float
                switch descriptor.dtype {
                case .bfloat16:
                    value = Float(
                        bitPattern: UInt32(
                            UInt16(
                                littleEndian: bytes.loadUnaligned(
                                    fromByteOffset: offset, as: UInt16.self))) << 16)
                case .float16:
                    value = Float(
                        Float16(
                            bitPattern: UInt16(
                                littleEndian: bytes.loadUnaligned(
                                    fromByteOffset: offset, as: UInt16.self))))
                default:
                    value = Float(
                        bitPattern: UInt32(
                            littleEndian:
                                bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                }
                guard value == 0 || value == 1 else {
                    throw CBv2CompleteCheckpointError.invalidSegment
                }
            }
        }
    }
}
