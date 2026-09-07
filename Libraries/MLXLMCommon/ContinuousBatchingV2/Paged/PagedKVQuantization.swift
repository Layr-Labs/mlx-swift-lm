import MLX

/// Versioned token-local affine storage. Scales and offsets are always FP32:
/// BF16/FP32 cache ranges must not be narrowed to FP16 merely to store metadata.
/// Rotation is normalized Walsh-Hadamard, with a fixed deterministic sign mask.
public struct PagedKVQuantizationConfig: Codable, Hashable, Sendable {
    public let keyBits: Int
    public let valueBits: Int
    public let groupSize: Int
    public let rotationBlockSize: Int
    public let version: Int

    public init(keyBits: Int = 4, valueBits: Int = 4, groupSize: Int = 64,
                rotationBlockSize: Int = 128) {
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.rotationBlockSize = rotationBlockSize
        self.version = 1
    }

    public var identity: String {
        "affine-v\(version)-k\(keyBits)v\(valueBits)-g\(groupSize)-f32-h\(rotationBlockSize)-s1"
    }

    public func resolvedRotationBlockSize(headDim: Int) -> Int {
        rotationBlockSize == 0 ? 0 : min(rotationBlockSize, headDim)
    }

    public func validateParameters() throws {
        guard version == 1, [4, 8].contains(keyBits), [4, 8].contains(valueBits),
              groupSize >= 32, groupSize <= 128, groupSize.nonzeroBitCount == 1,
              rotationBlockSize == 0 || (rotationBlockSize >= 32 && rotationBlockSize <= 512
                  && rotationBlockSize.nonzeroBitCount == 1)
        else {
            throw CBv2KVError.backendIneligible(reason: "unsupported paged KV quantization format")
        }
    }

    public func validate(headDim: Int) throws {
        try validateParameters()
        let rotation = resolvedRotationBlockSize(headDim: headDim)
        guard [64, 128, 256, 512].contains(headDim), headDim % groupSize == 0,
              rotation == 0 || headDim % rotation == 0
        else {
            throw CBv2KVError.backendIneligible(reason: "unsupported paged KV quantization geometry")
        }
    }

    public func rowLayout(headDim: Int) throws -> PagedKVQuantizedRowLayout {
        try validate(headDim: headDim)
        return try PagedKVQuantizedRowLayout(config: self, headDim: headDim)
    }

    /// Both K and V, including FP32 scale/offset metadata, for one token.
    public func bytesPerToken(kvHeads: Int, headDim: Int) throws -> Int {
        guard kvHeads > 0 else { throw CBv2KVError.backendIneligible(reason: "invalid KV heads") }
        let row = try rowLayout(headDim: headDim)
        return try Self.multiply(kvHeads, row.keyRowBytes + row.valueRowBytes)
    }

    static func multiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow, value >= 0 else {
            throw CBv2KVError.backendIneligible(reason: "paged quantization byte overflow")
        }
        return value
    }
}

/// Each K or V row is [packed codes][FP32 scales][FP32 offsets]. All offsets
/// and strides are bytes. A segment contains all K rows, then all V rows;
/// each region orders rows [physical page, KV head, token within page].
public struct PagedKVQuantizedRowLayout: Hashable, Sendable {
    public let keyDataBytes: Int
    public let valueDataBytes: Int
    public let keyScaleOffset: Int
    public let valueScaleOffset: Int
    public let keyOffsetOffset: Int
    public let valueOffsetOffset: Int
    public let keyRowBytes: Int
    public let valueRowBytes: Int

    init(config: PagedKVQuantizationConfig, headDim: Int) throws {
        keyDataBytes = try PagedKVQuantizationConfig.multiply(headDim / 8, config.keyBits)
        valueDataBytes = try PagedKVQuantizationConfig.multiply(headDim / 8, config.valueBits)
        let scaleBytes = try PagedKVQuantizationConfig.multiply(headDim / config.groupSize, 4)
        keyScaleOffset = keyDataBytes
        valueScaleOffset = valueDataBytes
        keyOffsetOffset = keyDataBytes + scaleBytes
        valueOffsetOffset = valueDataBytes + scaleBytes
        keyRowBytes = keyDataBytes + 2 * scaleBytes
        valueRowBytes = valueDataBytes + 2 * scaleBytes
    }
}

extension PagedKVGroupKey {
    /// Physical storage bytes, independently of the model's native compute dtype.
    public func bytesPerToken() throws -> Int {
        if let quantization {
            return try quantization.bytesPerToken(kvHeads: kvHeads, headDim: headDim)
        }
        var bytes = 2
        for value in [kvHeads, headDim, dtype.size] {
            bytes = try PagedKVQuantizationConfig.multiply(bytes, value)
        }
        return bytes
    }
}
