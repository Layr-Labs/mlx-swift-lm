import Cmlx
import Foundation
import MLX

extension CBv2CompleteCheckpointImportPlan {
    /// Decode directly to private native destinations, before publication. All
    /// buffers and bounded CPU scratch were admitted by this plan's stage lease.
    func appendQuantizedTarget(
        codec: CBv2CompleteCheckpointCodec, decoder: CBv2NativeCheckpointRowDecoder?,
        tensorIndex: Int, byteOffset: Int, data: Data,
        storage: CBv2PagedCheckpointStorage?, arrays: [MLXArray],
        nativeWork: CBv2NativeCompletePrefixWork? = nil
    ) throws -> Bool {
        guard let decoder, tensorIndex < codec.targetTensorCount,
            manifest.tensors[tensorIndex].dtype == .uint8
        else { return false }
        let owners =
            (codec.contiguousLayout ?? codec.historicalLayout)?.owningIndices
            ?? Array(codec.layerKinds.indices)
        let layer = owners[tensorIndex / 2]
        let role = try codec.checkpointRole(
            layer: layer, position: manifest.position, values: tensorIndex % 2 == 1)
        let descriptor = destinationDescriptors[tensorIndex]
        try decoder.append(role: role, byteOffset: byteOffset, data: data) { offset, row in
            if let storage {
                try storage.append(
                    layerIndex: tensorIndex / 2, values: tensorIndex % 2 == 1,
                    byteOffset: offset, data: row)
                return
            }
            guard arrays.indices.contains(tensorIndex),
                let pointer = mlx_array_data_uint8(arrays[tensorIndex].ctx)
            else {
                nativeWork?.requiredCompletionFailed()
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            let destination = UnsafeMutableRawPointer(mutating: pointer)
            let window = codec.contiguousLayout?.layers[layer].window
            row.withUnsafeBytes { source in
                if let window {
                    CBv2CheckpointByteLayout.copyRing(
                        shape: descriptor.shape, window: window, firstPosition: role.tokenStart,
                        itemSize: role.key.dtype.size, byteOffset: offset, count: row.count
                    ) { physical, packed, count in
                        destination.advanced(by: physical).copyMemory(
                            from: source.baseAddress!.advanced(by: packed), byteCount: count)
                    }
                } else {
                    CBv2CheckpointByteLayout.copy(
                        shape: descriptor.shape,
                        strides: CBv2CheckpointByteLayout.contiguousStrides(
                            destinationShapes[tensorIndex]),
                        itemSize: role.key.dtype.size, byteOffset: offset, count: row.count
                    ) { physical, packed, count in
                        destination.advanced(by: physical).copyMemory(
                            from: source.baseAddress!.advanced(by: packed), byteCount: count)
                    }
                }
            }
        }
        return true
    }
}
