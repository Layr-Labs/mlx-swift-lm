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

    /// Wire dtypes of the roles whose FP32 pool and state copies only widen a
    /// narrower activation dtype: latent keys, the [index keys, pool gates,
    /// 0/1 validity] values row and the KDA convolution history. They are
    /// stored at that dtype and widened back exactly on import; the export
    /// refuses any value the wire dtype cannot represent. The accumulated KDA
    /// recurrent state is always FP32.
    public struct Storage: Codable, Sendable, Equatable {
        public let latent: CBv2CheckpointDType
        public let index: CBv2CheckpointDType
        public let convolution: CBv2CheckpointDType

        public static let float32 = Storage(
            latent: .float32, index: .float32, convolution: .float32)

        public init(
            latent: CBv2CheckpointDType, index: CBv2CheckpointDType,
            convolution: CBv2CheckpointDType
        ) {
            self.latent = latent
            self.index = index
            self.convolution = convolution
        }

        var isValid: Bool {
            [latent, index, convolution].allSatisfy {
                $0 == .float32 || CBv2CheckpointHalfPrecision.isHalf($0)
            }
        }
    }

    public let physicalPosition: Int
    public let acceptedFrontier: Int
    public let latentWidth: Int
    public let indexWidth: Int
    public let recurrentLayers: [RecurrentLayer]
    public let storage: Storage

    /// Wire descriptors: the narrowed roles use `storage`. With a
    /// `latentQuantization` profile each latent (keys) tensor is the lossy
    /// `uint8` stream of `latentRole`.
    func targetDescriptors(
        attention: [CBv2CheckpointAttentionLayer], position: Int,
        latentQuantization: PagedKVQuantizationConfig? = nil
    ) throws -> [CBv2CheckpointTensorDescriptor] {
        try targetDescriptors(
            attention: attention, position: position, storage: storage,
            latentQuantization: latentQuantization)
    }

    /// The FP32 destinations that import widens the wire roles into.
    func nativeTargetDescriptors(attention: [CBv2CheckpointAttentionLayer], position: Int) throws
        -> [CBv2CheckpointTensorDescriptor]
    {
        try targetDescriptors(attention: attention, position: position, storage: .float32)
    }

    /// One latent row stream under the checkpoint profile: every head's older
    /// rows coded, then its exact last `recentTokenCount` rows at the latent
    /// wire dtype, which restore widens back into the FP32 pages.
    func latentRole(position: Int, quantization: PagedKVQuantizationConfig) throws
        -> CBv2CheckpointPagedRoleLayout
    {
        guard (try? quantization.validate(headDim: latentWidth)) != nil else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try .init(
            key: PagedKVGroupKey(
                kvHeads: 1, headDim: latentWidth, dtype: storage.latent.mlxDType,
                valueHeadDim: 2 * indexWidth + 1, quantization: quantization),
            position: position, values: false, preservePackedRecent: false)
    }

    private func targetDescriptors(
        attention: [CBv2CheckpointAttentionLayer], position: Int, storage: Storage,
        latentQuantization: PagedKVQuantizationConfig? = nil
    ) throws -> [CBv2CheckpointTensorDescriptor] {
        guard physicalPosition == position, acceptedFrontier == position,
            (64 ... 512).contains(latentWidth), (1 ... 512).contains(indexWidth),
            !attention.isEmpty, attention.count <= 64,
            // The bounded geometry fits the existing manifest permit's 64KiB
            // control-metadata allowance; tensor payloads are priced separately.
            !recurrentLayers.isEmpty, recurrentLayers.count <= 64,
            self.storage.isValid, storage.isValid
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
            if let latentQuantization {
                descriptors.append(
                    try latentRole(position: position, quantization: latentQuantization)
                        .descriptor(layer: layer.modelLayer))
            } else {
                descriptors.append(
                    try .init(
                        role: .keys, layer: layer.modelLayer,
                        shape: [1, 1, position, latentWidth], dtype: storage.latent))
            }
            // Raw index keys, raw learned pool gates, then exact 0/1 validity.
            descriptors.append(
                try .init(
                    role: .values, layer: layer.modelLayer,
                    shape: [1, 1, position, 2 * indexWidth + 1], dtype: storage.index))
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
                    shape: layer.convolutionShape, dtype: storage.convolution))
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
    /// The loaded target's declared payload dtypes; FP32 when undeclared.
    let storage: CBv2GLM5NextCheckpointState.Storage
    /// The resolved checkpoint profile of the older latent rows; nil is exact.
    let latentQuantization: PagedKVQuantizationConfig?

    init(
        backend: PagedKVBackend, admission: AdmissionV2, layerKinds: [CBv2LayerKind],
        recurrentSpec: CBv2RecurrentStateSpec, dtypes: [DType],
        config: PagedKVPoolConfig, latentQuantization: PagedKVQuantizationConfig? = nil
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
        storage = binding.payloadStorage ?? .float32
        self.latentQuantization = latentQuantization
        _ = try state(position: 2).targetDescriptors(
            attention: attention.layers, position: 2, latentQuantization: latentQuantization)
    }

    func state(position: Int) -> CBv2GLM5NextCheckpointState {
        .init(
            physicalPosition: position, acceptedFrontier: position,
            latentWidth: latentWidth, indexWidth: indexWidth, recurrentLayers: recurrentLayers,
            storage: storage)
    }

    /// Wire descriptors, at the declared storage dtypes.
    func descriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try state(position: position).targetDescriptors(
            attention: attention.layers, position: position,
            latentQuantization: latentQuantization)
    }

    /// The export/import row layout of one target role. Only the latent
    /// (keys) role is ever coded; the values role stays exact at its wire dtype.
    func role(position: Int, values: Bool) throws -> CBv2CheckpointPagedRoleLayout {
        if !values, let latentQuantization {
            return try state(position: position).latentRole(
                position: position, quantization: latentQuantization)
        }
        return try .init(
            key: PagedKVGroupKey(
                kvHeads: 1, headDim: latentWidth,
                dtype: (values ? storage.index : storage.latent).mlxDType,
                valueHeadDim: 2 * indexWidth + 1),
            position: position, values: values)
    }

    /// FP32 import destinations for the same tensors.
    func nativeDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try state(position: position).nativeTargetDescriptors(
            attention: attention.layers, position: position)
    }

    /// Whether any wire role is narrower than its FP32 destination.
    var narrowsPayload: Bool { storage != .float32 }
}

extension CBv2CompleteCheckpointManifest {
    func validateGLM5NextStructure() throws {
        // Only the lossy layout carries a profile; its common checks already
        // bound the profile and native dtypes to the attention map.
        let lossy = backendLayout == Self.glm5NextQuantizedPagedLayout
        guard let glm5NextState, let attentionLayers,
            lossy == (checkpointQuantization != nil), lossy == (checkpointNativeDTypes != nil),
            nativeBlockState == nil, !mediaTargetOnly || assistantCodecID == nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let target = try glm5NextState.targetDescriptors(
            attention: attentionLayers, position: position,
            latentQuantization: checkpointQuantization)
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
