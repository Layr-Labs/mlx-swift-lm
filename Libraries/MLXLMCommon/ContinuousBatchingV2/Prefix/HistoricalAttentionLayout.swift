import MLX

/// A loaded causal target whose complete state is attention KV. A per-round
/// assistant may rebuild from target rows; persistent assistants need their own
/// checkpoint codec and are deliberately excluded from this contract.
public protocol CBv2HistoricalAttentionCheckpointProviding {
    var cbv2SupportsHistoricalAttentionCheckpoint: Bool { get }
}

/// Canonical compact-layer mapping. Borrowers name an earlier owning row, never
/// another borrower. The same table drives disk identity, tensors and adoption.
public struct CBv2CheckpointAttentionLayer: Codable, Sendable, Equatable {
    public let modelLayer: Int
    public let owner: Int
    public let window: Int?
    public let kvHeads: Int
    public let headDim: Int
    private let explicitValueHeadDim: Int?
    public var valueHeadDim: Int { explicitValueHeadDim ?? headDim }
    public let queryHeads: Int
    public let hasSinks: Bool
    public let dtype: CBv2CheckpointDType

    init(modelLayer: Int, owner: Int, window: Int?, kvHeads: Int, headDim: Int,
         valueHeadDim: Int? = nil, queryHeads: Int, hasSinks: Bool, dtype: CBv2CheckpointDType) {
        self.modelLayer = modelLayer; self.owner = owner; self.window = window
        self.kvHeads = kvHeads; self.headDim = headDim
        explicitValueHeadDim = valueHeadDim == headDim ? nil : valueHeadDim
        self.queryHeads = queryHeads; self.hasSinks = hasSinks; self.dtype = dtype
    }

    private enum CodingKeys: String, CodingKey {
        case modelLayer, owner, window, kvHeads, headDim, valueHeadDim, queryHeads, hasSinks, dtype
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(modelLayer: try c.decode(Int.self, forKey: .modelLayer), owner: try c.decode(Int.self, forKey: .owner),
            window: try c.decodeIfPresent(Int.self, forKey: .window), kvHeads: try c.decode(Int.self, forKey: .kvHeads),
            headDim: try c.decode(Int.self, forKey: .headDim), valueHeadDim: try c.decodeIfPresent(Int.self, forKey: .valueHeadDim),
            queryHeads: try c.decode(Int.self, forKey: .queryHeads), hasSinks: try c.decode(Bool.self, forKey: .hasSinks),
            dtype: try c.decode(CBv2CheckpointDType.self, forKey: .dtype))
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelLayer, forKey: .modelLayer); try c.encode(owner, forKey: .owner)
        try c.encodeIfPresent(window, forKey: .window); try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim); try c.encodeIfPresent(explicitValueHeadDim, forKey: .valueHeadDim)
        try c.encode(queryHeads, forKey: .queryHeads); try c.encode(hasSinks, forKey: .hasSinks)
        try c.encode(dtype, forKey: .dtype)
    }

    /// Provider identity must use the same owner map as native staging.
    public static func resolve(layerKinds: [CBv2LayerKind], dtypes: [DType]) throws -> [Self] {
        try CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: dtypes).layers
    }

    /// The distinct contiguous-asymmetric v2 format uses this exact map.
    /// Do not broaden the legacy equal-width historical resolver.
    public static func resolveContiguousAsymmetric(layerKinds: [CBv2LayerKind], dtypes: [DType]) throws -> [Self] {
        guard layerKinds.contains(where: { $0.headDim != $0.valueHeadDim }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: dtypes,
                                                 allowAsymmetric: true).layers
    }

    /// Scalar identity only for the distinct page-native asymmetric format.
    /// Native permission still requires the package-issued loaded/page/store
    /// tuple. Never broadens the legacy equal-width resolver.
    public static func resolvePagedAsymmetric(layerKinds: [CBv2LayerKind], dtypes: [DType]) throws -> [Self] {
        guard layerKinds.contains(where: { $0.headDim != $0.valueHeadDim }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: dtypes,
                                                 allowAsymmetric: true).layers
    }

    func tokenStart(at position: Int) -> Int { window.map { max(0, position - $0) } ?? 0 }
}

struct CBv2HistoricalAttentionLayout: Sendable {
    let layers: [CBv2CheckpointAttentionLayer]
    var owningIndices: [Int] { layers.indices.filter { layers[$0].owner == $0 } }

    init(layerKinds: [CBv2LayerKind], dtypes: [DType], allowAsymmetric: Bool = false) throws {
        guard !layerKinds.isEmpty, layerKinds.count <= 2048, layerKinds.count == dtypes.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        // The new contiguous table covers target attention KV only. Do not
        // silently discard learned indexer or any other non-KV side state.
        if allowAsymmetric {
            guard layerKinds.allSatisfy({ $0.qwen4IndexerCompressRatio == nil && $0.extraStorageBytesPerToken == 0 }) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
        var result: [CBv2CheckpointAttentionLayer] = []
        var modelIndices = Set<Int>()
        for (index, kind) in layerKinds.enumerated() {
            guard !kind.isBidirectional, kind.kvHeads > 0, kind.headDim >= 64,
                  kind.valueHeadDim > 0, kind.kvGeometry != nil,
                  allowAsymmetric || kind.valueHeadDim == kind.headDim,
                  kind.queryHeads > 0, kind.queryHeads % kind.kvHeads == 0,
                  (kind.modelLayerIndex ?? index) >= 0, let dtype = CBv2CheckpointDType(dtypes[index]), dtype.isFloatingPoint,
                  modelIndices.insert(kind.modelLayerIndex ?? index).inserted
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let window: Int?
            switch kind.attention {
            case .full: window = nil
            case .slidingWindow(let size):
                guard size > 0 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                window = size
            }
            let owner = kind.sharesKVWithLayer ?? index
            if owner != index {
                guard owner >= 0, owner < index, result[owner].owner == owner,
                      result[owner].window == window, result[owner].kvHeads == kind.kvHeads,
                      result[owner].headDim == kind.headDim, result[owner].valueHeadDim == kind.valueHeadDim,
                      result[owner].dtype == dtype
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            } else if kind.sharesKVWithLayer != nil {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            result.append(.init(modelLayer: kind.modelLayerIndex ?? index, owner: owner,
                                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                                valueHeadDim: kind.valueHeadDim, queryHeads: kind.queryHeads, hasSinks: kind.hasSinks, dtype: dtype))
        }
        layers = result
    }

    func tensorDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try owningIndices.flatMap { index in
            let layer = layers[index]
            return try [CBv2CheckpointTensorRole.keys, .values].map {
                try CBv2CheckpointTensorDescriptor(role: $0, layer: layer.modelLayer,
                          shape: [1, layer.kvHeads, position - layer.tokenStart(at: position), $0 == .keys ? layer.headDim : layer.valueHeadDim],
                          dtype: layer.dtype)
            }
        }
    }
}
