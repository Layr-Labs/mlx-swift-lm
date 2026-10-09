import MLX

/// A loaded-model geometry and one concrete query/native-cache dtype pair.
/// Window topology and bidirectional bounds run through their actual adapters.
public struct PagedQuantizedKernelSmokeShape: Hashable, Sendable {
    public let headDim: Int
    public let kvHeads: Int
    public let queryHeads: Int
    public let hasSinks: Bool
    public let windowSize: Int?
    public let isBidirectional: Bool
    public let dtype: DType
    public let queryDType: DType

    public init(
        headDim: Int, kvHeads: Int, queryHeads: Int, hasSinks: Bool,
        windowSize: Int? = nil, isBidirectional: Bool = false,
        dtype: DType, queryDType: DType? = nil
    ) {
        self.headDim = headDim
        self.kvHeads = kvHeads
        self.queryHeads = queryHeads
        self.hasSinks = hasSinks
        self.windowSize = windowSize
        self.isBidirectional = isBidirectional
        self.dtype = dtype
        self.queryDType = queryDType ?? dtype
    }

    public var argumentValue: String {
        "\(headDim):\(kvHeads):\(queryHeads):\(hasSinks ? 1 : 0):\(windowSize ?? 0):\(isBidirectional ? 1 : 0):\(Self.dtypeName(dtype)):\(Self.dtypeName(queryDType))"
    }

    public init(argumentValue: String) throws {
        let fields = argumentValue.split(separator: ":", omittingEmptySubsequences: false)
        guard fields.count == 8,
            let dimension = Int(fields[0]), let kvHeads = Int(fields[1]),
            let queryHeads = Int(fields[2]), let sinks = Int(fields[3]),
            let window = Int(fields[4]), let bidirectional = Int(fields[5]),
            [0, 1].contains(sinks), [0, 1].contains(bidirectional),
            window >= 0, let dtype = Self.dtype(fields[6]),
            let queryDType = Self.dtype(fields[7])
        else {
            throw PagedAttentionKernelSmokeError.invalidShape(argumentValue)
        }
        self.init(
            headDim: dimension, kvHeads: kvHeads, queryHeads: queryHeads,
            hasSinks: sinks == 1, windowSize: window == 0 ? nil : window,
            isBidirectional: bidirectional == 1, dtype: dtype, queryDType: queryDType)
    }

    private static func dtypeName(_ value: DType) -> String {
        switch value {
        case .float16: return "f16"
        case .bfloat16: return "bf16"
        case .float32: return "f32"
        default: return "unsupported"
        }
    }

    private static func dtype(_ value: Substring) -> DType? {
        switch value {
        case "f16": return .float16
        case "bf16": return .bfloat16
        case "f32": return .float32
        default: return nil
        }
    }
}

/// Evaluated operation variants, keyed by the actual geometry/dtype probe.
public typealias PagedQuantizedKernelSmokeCoverage = [PagedQuantizedKernelSmokeShape: Set<String>]

public enum PagedQuantizedKernelSmoke {
    public static func smokeShapes(
        layerKinds: [CBv2LayerKind],
        quantization: PagedKVQuantizationConfig,
        nativeLayerIndices: Set<Int> = []
    ) throws -> [PagedQuantizedKernelSmokeShape] {
        try quantization.validateParameters()
        guard
            nativeLayerIndices.allSatisfy({
                layerKinds.indices.contains($0) && layerKinds[$0].sharesKVWithLayer == nil
            })
        else {
            throw PagedAttentionKernelSmokeError.invalidShape("invalid native owner indices")
        }
        let dtypes: [DType] = [.float16, .bfloat16, .float32]
        var shapes = Set<PagedQuantizedKernelSmokeShape>()
        for (index, kind) in layerKinds.enumerated() {
            let owner = kind.sharesKVWithLayer ?? index
            guard layerKinds.indices.contains(owner), owner <= index,
                layerKinds[owner].sharesKVWithLayer == nil
            else {
                throw PagedAttentionKernelSmokeError.invalidShape("invalid shared KV owner")
            }
            guard !nativeLayerIndices.contains(owner),
                PagedKVGroupKey.quantization(for: kind, config: quantization) != nil
            else { continue }
            guard kind.headDim == kind.valueHeadDim else {
                throw PagedAttentionKernelSmokeError.ineligibleShape(
                    "packed reader requires equal K/V widths")
            }
            try quantization.validate(headDim: kind.headDim)
            let window: Int?
            switch kind.attention {
            case .full: window = nil
            case .slidingWindow(let size): window = size
            }
            for dtype in dtypes {
                for queryDType in dtypes {
                    shapes.insert(
                        .init(
                            headDim: kind.headDim, kvHeads: kind.kvHeads,
                            queryHeads: kind.queryHeads, hasSinks: kind.hasSinks,
                            windowSize: window, isBidirectional: kind.isBidirectional,
                            dtype: dtype, queryDType: queryDType))
                }
            }
        }
        return shapes.sorted { $0.argumentValue < $1.argumentValue }
    }
}
