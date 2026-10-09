import Foundation

extension CBv2CompleteCheckpointManifest {
    static let maximumTokenByteTopologyRecordCount = 4096

    /// Positive host envelope for decoded and independently derived records,
    /// plus a bounded transient index and one component iteration. Consumers
    /// retaining additional component collections must charge those separately.
    public static var maximumTokenByteTopologyHostBytes: Int {
        let capacity = 2 * maximumTokenByteTopologyRecordCount
        let recordBytes = MemoryLayout<CBv2CheckpointTokenByteTopology>.stride
        let arrays = 2 * capacity * recordBytes
        let index = capacity * (recordBytes + MemoryLayout<Int>.stride + 64)
        return arrays + index + (64 << 10)
    }

    /// Validate once for a whole transfer. Present records borrow manifest
    /// ownership. Derived legacy records are bounded to 4096 fixed-size values
    /// and belong to the caller's existing provider I/O scratch envelope.
    public func validatedTokenByteTopologies() throws -> [CBv2CheckpointTokenByteTopology] {
        _ = try validateStructure()
        if let tokenByteTopologies { return tokenByteTopologies }
        return try tensors.indices.compactMap { index in
            let role = tensors[index].role
            guard role == .keys || role == .values else { return nil }
            return try legacyNativeTokenByteTopology(tensorIndex: index)
        }
    }

    /// Nil means this tensor remains endpoint-owned. Opaque legacy affine
    /// streams never acquire a guessed topology; auxiliary state is never KV.
    public func validatedTokenByteTopology(tensorIndex: Int) throws
        -> CBv2CheckpointTokenByteTopology?
    {
        _ = try validateStructure()
        guard tensors.indices.contains(tensorIndex) else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let descriptor = tensors[tensorIndex]
        guard descriptor.role == .keys || descriptor.role == .values else { return nil }
        if let tokenByteTopologies {
            return tokenByteTopologies.first { $0.tensorIndex == tensorIndex }
        }
        return try legacyNativeTokenByteTopology(tensorIndex: tensorIndex)
    }

    func validateTokenByteTopologyRecords() throws {
        try validateAttentionOwnerMap()
        guard let tokenByteTopologies else { return }
        guard !tokenByteTopologies.isEmpty,
            tokenByteTopologies.count <= Self.maximumTokenByteTopologyRecordCount,
            tokenByteTopologies.count
                == tensors.reduce(
                    0,
                    { count, tensor in
                        count + (tensor.role == .keys || tensor.role == .values ? 1 : 0)
                    }),
            backendLayout != Self.diffusionBlockLayout
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        var lastIndex = -1
        for topology in tokenByteTopologies {
            guard topology.tensorIndex > lastIndex,
                tensors.indices.contains(topology.tensorIndex)
            else { throw CBv2CompleteCheckpointError.invalidManifest }
            lastIndex = topology.tensorIndex
            let descriptor = tensors[topology.tensorIndex]
            try topology.validate(descriptor: descriptor, position: position)
            let affine =
                backendLayout == Self.quantizedPagedLayout
                || backendLayout == Self.quantizedHistoricalLayout
            guard affine || (topology.quantization == nil && !topology.nativeExempt),
                !affine || topology.quantization != nil || topology.nativeExempt
            else { throw CBv2CompleteCheckpointError.invalidManifest }
            if affine {
                if backendLayout == Self.quantizedPagedLayout {
                    guard topology.absoluteTokenStart == 0, topology.attentionWindow == nil else {
                        throw CBv2CompleteCheckpointError.invalidManifest
                    }
                } else {
                    try validateAttentionTopology(topology)
                }
            } else {
                guard
                    let expected = try legacyNativeTokenByteTopology(
                        tensorIndex: topology.tensorIndex),
                    expected == topology
                else { throw CBv2CompleteCheckpointError.invalidManifest }
            }
        }
    }

    private func validateAttentionTopology(_ topology: CBv2CheckpointTokenByteTopology) throws {
        guard let attentionLayers,
            let index = attentionLayers.firstIndex(where: { $0.modelLayer == topology.layer }),
            attentionLayers[index].owner == index
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let owner = attentionLayers[index]
        try validateAttentionOwner(owner)
        guard topology.attentionWindow == owner.window, topology.headCount == owner.kvHeads,
            topology.roleWidth == (topology.role == .values ? owner.valueHeadDim : owner.headDim),
            topology.nativeDType == owner.dtype,
            topology.absoluteTokenStart == owner.tokenStart(at: position)
        else { throw CBv2CompleteCheckpointError.invalidManifest }
    }

    private func validateAttentionOwner(_ owner: CBv2CheckpointAttentionLayer) throws {
        guard owner.modelLayer >= 0, owner.headDim > 0, owner.headDim <= Int(Int32.max),
            owner.valueHeadDim > 0, owner.valueHeadDim <= Int(Int32.max),
            owner.kvHeads > 0, owner.kvHeads <= Int(Int32.max),
            owner.queryHeads > 0, owner.queryHeads <= Int(Int32.max),
            owner.queryHeads.isMultiple(of: owner.kvHeads), owner.dtype.isFloatingPoint,
            owner.window == nil || (owner.window! > 0 && owner.window! <= Int(Int32.max))
        else { throw CBv2CompleteCheckpointError.invalidManifest }
    }

    /// Validate the whole table before selecting an owner or computing token
    /// offsets. Slices borrow the bounded table; no additional index is built.
    private func validateAttentionOwnerMap() throws {
        guard let attentionLayers else { return }
        for (index, owner) in attentionLayers.enumerated() {
            try validateAttentionOwner(owner)
            guard owner.owner >= 0, owner.owner <= index,
                !attentionLayers[..<index].contains(where: { $0.modelLayer == owner.modelLayer })
            else { throw CBv2CompleteCheckpointError.invalidManifest }
            if owner.owner != index {
                let source = attentionLayers[owner.owner]
                guard source.owner == owner.owner, source.window == owner.window,
                    source.kvHeads == owner.kvHeads, source.headDim == owner.headDim,
                    source.valueHeadDim == owner.valueHeadDim, source.dtype == owner.dtype
                else { throw CBv2CompleteCheckpointError.invalidManifest }
            }
        }
    }

    private func legacyNativeTokenByteTopology(tensorIndex: Int) throws
        -> CBv2CheckpointTokenByteTopology?
    {
        let descriptor = tensors[tensorIndex]
        guard descriptor.dtype.isFloatingPoint, descriptor.shape.count == 4,
            descriptor.shape[0] == 1, let layer = descriptor.layer
        else { return nil }
        let start: Int
        let window: Int?
        switch backendLayout {
        case Self.layout, Self.pagedLayout:
            guard attentionLayers == nil, descriptor.shape[2] == position else { return nil }
            start = 0
            window = nil
        case Self.historicalAttentionLayout, Self.contiguousAsymmetricLayout,
            Self.contiguousAsymmetricMTPLayout, Self.pagedAsymmetricLayout,
            Self.pagedAsymmetricMTPLayout:
            guard let attentionLayers,
                let index = attentionLayers.firstIndex(where: { $0.modelLayer == layer }),
                attentionLayers[index].owner == index
            else { return nil }
            let owner = attentionLayers[index]
            try validateAttentionOwner(owner)
            window = owner.window
            start = owner.tokenStart(at: position)
            guard descriptor.dtype == owner.dtype, descriptor.shape[1] == owner.kvHeads,
                descriptor.shape[2] == position - start,
                descriptor.shape[3]
                    == (descriptor.role == .values ? owner.valueHeadDim : owner.headDim)
            else { return nil }
        default:
            // Includes affine layouts: native exceptions cannot be inferred
            // from an old mixed-layout descriptor without the typed owner set.
            return nil
        }
        return try .init(
            tensorIndex: tensorIndex, descriptor: descriptor, nativeDType: descriptor.dtype,
            roleWidth: descriptor.shape[3], absoluteTokenStart: start,
            position: position, attentionWindow: window, quantization: nil)
    }
}
