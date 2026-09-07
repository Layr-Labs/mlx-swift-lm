import MLX

/// Native compute and packed-format identity. Segmented roots separate full and windowed
/// ownership, so a rolling window never shares an allocator root with a full
/// history. Uniform fixed-reference pools retain their historical grouping.
public struct PagedKVGroupKey: Hashable, Sendable, CustomStringConvertible {
    public let kvHeads: Int
    public let headDim: Int
    public let dtype: DType
    public let windowSize: Int?
    public let quantization: PagedKVQuantizationConfig?

    public init(kvHeads: Int, headDim: Int, dtype: DType = .float16, windowSize: Int? = nil,
                quantization: PagedKVQuantizationConfig? = nil) {
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.dtype = dtype
        self.windowSize = windowSize
        self.quantization = quantization
    }

    public init(_ kind: CBv2LayerKind, dtype: DType = .float16, separateWindow: Bool = false,
                quantization: PagedKVQuantizationConfig? = nil) {
        let window: Int?
        if separateWindow, case .slidingWindow(let size) = kind.attention {
            window = size
        } else {
            window = nil
        }
        self.init(kvHeads: kind.kvHeads, headDim: kind.headDim, dtype: dtype, windowSize: window,
                  quantization: kind.attention == .full ? quantization : nil)
    }

    var sortKey: (Int, Int, String, Int, String) {
        (headDim, kvHeads, String(describing: dtype), windowSize ?? 0, quantization?.identity ?? "native")
    }
    public var description: String {
        "kv\(kvHeads)xd\(headDim)-\(dtype)-\(windowSize.map { "w\($0)" } ?? "full")" + (quantization.map { "-\($0.identity)" } ?? "")
    }
}

enum PagedKVStorageLayout {
    static func resolve(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig) throws -> [DType] {
        let types = config.layerDTypes ?? Array(repeating: config.dtype, count: layerKinds.count)
        guard types.count == layerKinds.count,
            types.allSatisfy({ [.float16, .bfloat16, .float32].contains($0) })
        else {
            throw CBv2KVError.backendIneligible(reason: "paged native dtype table is missing or unsupported")
        }
        for (index, kind) in layerKinds.enumerated() {
            guard let source = kind.sharesKVWithLayer else { continue }
            guard source >= 0, source < layerKinds.count, source != index,
                layerKinds[source].sharesKVWithLayer == nil,
                layerKinds[source].kvHeads == kind.kvHeads,
                layerKinds[source].headDim == kind.headDim,
                layerKinds[source].attention == kind.attention,
                types[source] == types[index]
            else {
                throw CBv2KVError.backendIneligible(reason: "paged borrowed layer \(index) has a different native layout")
            }
        }
        return types
    }
}
