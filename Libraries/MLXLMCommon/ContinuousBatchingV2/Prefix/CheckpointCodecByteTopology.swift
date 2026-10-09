import Foundation

extension CBv2CompleteCheckpointCodec {
    /// The loaded codec has target attention records to describe. Keep the
    /// producer and its pre-construction host reservation on the same boundary.
    var emitsTokenByteTopologies: Bool {
        nativePagedBinding == nil && contiguousLayout == nil
            && layerKinds.contains { $0.sharesKVWithLayer == nil }
    }

    /// Existing package-issued asymmetric/MiMo producers retain their exact
    /// legacy serialization. No model adapter or eligibility changes here.
    func checkpointTokenByteTopologies(
        descriptors: [CBv2CheckpointTensorDescriptor], position: Int
    ) throws -> [CBv2CheckpointTokenByteTopology]? {
        guard emitsTokenByteTopologies else { return nil }
        let records = try descriptors.enumerated().compactMap {
            tensorIndex, descriptor
                -> CBv2CheckpointTokenByteTopology? in
            guard descriptor.role == .keys || descriptor.role == .values else { return nil }
            guard
                let index = layerKinds.indices.first(where: {
                    (layerKinds[$0].modelLayerIndex ?? $0) == descriptor.layer
                        && layerKinds[$0].sharesKVWithLayer == nil
                }), let dtype = CBv2CheckpointDType(kvDTypes[index])
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let kind = layerKinds[index]
            let start: Int
            let window: Int?
            switch kind.attention {
            case .full:
                start = 0
                window = nil
            case .slidingWindow(let size):
                start = max(0, position - size)
                window = size
            }
            let quantization = checkpointGroupKey(layer: index).quantization
            return try .init(
                tensorIndex: tensorIndex, descriptor: descriptor, nativeDType: dtype,
                roleWidth: descriptor.role == .values ? kind.valueHeadDim : kind.headDim,
                absoluteTokenStart: start, position: position, attentionWindow: window,
                quantization: quantization,
                nativeExempt: usesQuantizedCheckpoint && quantization == nil)
        }
        // Complete auxiliary-only checkpoints have no owning attention bytes.
        // Keep their legacy omission; a present empty table is invalid.
        return records.isEmpty ? nil : records
    }
}
