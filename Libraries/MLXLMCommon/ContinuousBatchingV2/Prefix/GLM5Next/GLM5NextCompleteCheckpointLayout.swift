import Foundation
import MLX

/// Encrypted, lossless GLM target geometry. Physical positions are the dense
/// NoPE row ordinals: no external position tensor is accepted by this format.
public struct CBv2GLM5NextCheckpointState: Codable, Sendable, Equatable {
    public struct RecurrentLayer: Codable, Sendable, Equatable {
        public let modelLayer: Int
        public let convolutionShape: [Int]
        public let recurrentShape: [Int]
    }

    public let physicalPosition: Int
    public let acceptedFrontier: Int
    public let latentWidth: Int
    public let indexWidth: Int
    public let recurrentLayers: [RecurrentLayer]

    func targetDescriptors(attention: [CBv2CheckpointAttentionLayer], position: Int) throws
        -> [CBv2CheckpointTensorDescriptor]
    {
        guard physicalPosition == position, acceptedFrontier == position,
            (64 ... 512).contains(latentWidth), (1 ... 512).contains(indexWidth),
            !attention.isEmpty, attention.count <= 64,
            // The bounded geometry fits the existing manifest permit's 64KiB
            // control-metadata allowance; tensor payloads are priced separately.
            !recurrentLayers.isEmpty, recurrentLayers.count <= 64
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var descriptors: [CBv2CheckpointTensorDescriptor] = []
        var modelLayers = Set<Int>()
        for (index, layer) in attention.enumerated() {
            guard layer.owner == index, layer.window == nil, layer.kvHeads == 1,
                layer.headDim == latentWidth, layer.valueHeadDim == 2 * indexWidth + 1,
                layer.headDim != layer.valueHeadDim, layer.queryHeads > 0,
                !layer.hasSinks, layer.dtype == .float32, layer.modelLayer >= 0,
                modelLayers.insert(layer.modelLayer).inserted
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            descriptors.append(
                try .init(
                    role: .keys, layer: layer.modelLayer,
                    shape: [1, 1, position, latentWidth], dtype: .float32))
            // Raw index keys, raw learned pool gates, then exact 0/1 validity.
            descriptors.append(
                try .init(
                    role: .values, layer: layer.modelLayer,
                    shape: [1, 1, position, 2 * indexWidth + 1], dtype: .float32))
        }
        guard recurrentLayers.map(\.modelLayer) == recurrentLayers.map(\.modelLayer).sorted()
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        for layer in recurrentLayers {
            guard layer.modelLayer >= 0, modelLayers.insert(layer.modelLayer).inserted,
                layer.convolutionShape.count == 3, layer.convolutionShape.first == 1,
                layer.recurrentShape.count == 4, layer.recurrentShape.first == 1,
                layer.recurrentShape[2] == layer.recurrentShape[3],
                let channels = CBv2KVGeometry.multiply(
                    layer.recurrentShape[1], layer.recurrentShape[2]),
                CBv2KVGeometry.multiply(3, channels) == layer.convolutionShape[1]
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            descriptors.append(
                try .init(
                    role: .convolution, layer: layer.modelLayer,
                    shape: layer.convolutionShape, dtype: .float32))
            descriptors.append(
                try .init(
                    role: .recurrent, layer: layer.modelLayer,
                    shape: layer.recurrentShape, dtype: .float32))
        }
        guard modelLayers == Set(0 ..< modelLayers.count) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return descriptors
    }
}

/// Constructed only from a live package-issued GLM paged binding. This does not
/// confer native-shutdown/profile permission or register provider persistence.
struct CBv2GLM5NextCompleteCheckpointLayout {
    let backend: PagedKVBackend
    let binding: GLM5NextPagedModelBinding
    let attention: CBv2HistoricalAttentionLayout
    let latentWidth: Int
    let indexWidth: Int
    let recurrentLayers: [CBv2GLM5NextCheckpointState.RecurrentLayer]

    init(
        backend: PagedKVBackend, admission: AdmissionV2, layerKinds: [CBv2LayerKind],
        recurrentSpec: CBv2RecurrentStateSpec, dtypes: [DType],
        config: PagedKVPoolConfig
    ) throws {
        guard let binding = backend.glm5NextModelBinding,
            backend.nativeModelBinding == nil || backend.nativeModelBinding?.isGLM5Next == true,
            backend.pool.nativeModelBinding === backend.nativeModelBinding,
            binding.matches(binding.targetIdentity), binding.kinds == layerKinds,
            backend.pool.memoryAdmission === admission, backend.layerKinds == layerKinds,
            backend.pool.usesStepOwnedAttention,
            backend.pool.config.pageSize == config.pageSize,
            backend.pool.config.segmentSizeBytes == config.segmentSizeBytes,
            backend.pool.config.maxBufferLength == config.maxBufferLength,
            backend.pool.layerDTypes == dtypes,
            config.quantization == nil, config.segmentSizeBytes != nil,
            config.gatheredAttention == binding.limits,
            dtypes == Array(repeating: .float32, count: layerKinds.count),
            config.layerDTypes == dtypes,
            recurrentSpec.layers.allSatisfy({ $0.convDType == .float32 && $0.ssmDType == .float32 })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.backend = backend
        self.binding = binding
        attention = try .init(layerKinds: layerKinds, dtypes: dtypes, allowAsymmetric: true)
        guard let first = layerKinds.first, first.valueHeadDim % 2 == 1 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        latentWidth = first.headDim
        indexWidth = (first.valueHeadDim - 1) / 2
        recurrentLayers = recurrentSpec.layers.map {
            .init(
                modelLayer: $0.modelLayerIndex, convolutionShape: $0.convShape,
                recurrentShape: $0.ssmShape)
        }
        _ = try state(position: 2).targetDescriptors(attention: attention.layers, position: 2)
    }

    func state(position: Int) -> CBv2GLM5NextCheckpointState {
        .init(
            physicalPosition: position, acceptedFrontier: position,
            latentWidth: latentWidth, indexWidth: indexWidth, recurrentLayers: recurrentLayers)
    }

    func descriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try state(position: position).targetDescriptors(
            attention: attention.layers, position: position)
    }
}

extension CBv2CompleteCheckpointManifest {
    func validateGLM5NextStructure() throws {
        guard let glm5NextState, let attentionLayers,
            checkpointQuantization == nil, checkpointNativeDTypes == nil,
            nativeBlockState == nil, !mediaTargetOnly || assistantCodecID == nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let target = try glm5NextState.targetDescriptors(
            attention: attentionLayers, position: position)
        guard tensors.prefix(target.count).elementsEqual(target) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        if assistantCodecID == nil {
            guard tensors == target else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        } else {
            let auxiliary = tensors.dropFirst(target.count)
            let roles: Set<CBv2CheckpointTensorRole> = [
                .assistantKeys, .assistantValues, .assistantCacheMetadata,
                .assistantHidden, .assistantTokens, .assistantFrontier,
            ]
            guard assistantCodecID?.isEmpty == false, !auxiliary.isEmpty,
                auxiliary.allSatisfy({ roles.contains($0.role) }),
                auxiliary.contains(where: { $0.role == .assistantKeys }),
                auxiliary.contains(where: { $0.role == .assistantValues }),
                auxiliary.contains(where: {
                    $0.role == .assistantHidden && $0.dtype.isFloatingPoint
                }),
                auxiliary.contains(where: {
                    $0.role == .assistantFrontier || $0.role == .assistantCacheMetadata
                })
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
    }
}
