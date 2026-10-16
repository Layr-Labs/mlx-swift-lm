import MLX

/// Request-owned unit-scale E4M3 attention history for the Super MTP head.
/// Each appended row is encoded once with `NemotronSuperFP8KVCodec`. A decoded
/// mirror is kept beside the codes: every E4M3 value (and NaN) is exactly
/// representable in F16/BF16/F32, and decoding is elementwise, so returned K/V
/// are bit-identical to `NemotronSuperFP8KVCache` without re-decoding history.
/// Storage grows in whole blocks up to a fixed capacity. `rewind` only moves
/// the offset; the next append overwrites abandoned rows, which are never read.
/// The caller serializes access, owns admission and fences evaluation before
/// retiring state. Not a paged, persistent, shared or sliding-window cache.
public final class NemotronSuperFP8KVHistory: KVCache {
    public enum Failure: Error, Equatable {
        case configuration, shape, dtype, capacity, rewind, state
    }
    public static let blockTokens = 256
    /// Bound on resident elements of one K or V buffer.
    public static let maximumElements = 1 << 28
    public let heads: Int
    public let headDimension: Int
    public let capacityTokens: Int
    public let dtype: DType
    public private(set) var offset = 0
    /// Not a window: masks must never treat capacity as a sliding size.
    public var maxSize: Int? { nil }
    private var codes: [MLXArray] = []
    private var decoded: [MLXArray] = []

    public init(
        heads: Int, headDimension: Int, capacityTokens: Int, dtype: DType = .bfloat16
    ) throws {
        guard heads > 0, headDimension > 0, capacityTokens > 0,
            [.float16, .bfloat16, .float32].contains(dtype)
        else { throw Failure.configuration }
        let (width, o1) = heads.multipliedReportingOverflow(by: headDimension)
        let blocks = (capacityTokens + Self.blockTokens - 1) / Self.blockTokens
        let (elements, o2) = width.multipliedReportingOverflow(by: blocks * Self.blockTokens)
        guard !o1, !o2, elements <= Self.maximumElements,
            width <= NemotronSuperFP8KVCodec.maximumElements
        else { throw Failure.capacity }
        self.heads = heads
        self.headDimension = headDimension
        self.capacityTokens = capacityTokens
        self.dtype = dtype
    }

    /// K plus V, FP8 codes plus decoded mirror, for one token row.
    public static func bytesPerToken(heads: Int, headDimension: Int, dtype: DType) throws -> Int {
        let (width, o1) = heads.multipliedReportingOverflow(by: headDimension)
        let (perBuffer, o2) = width.multipliedReportingOverflow(by: 1 + dtype.size)
        let (both, o3) = perBuffer.multipliedReportingOverflow(by: 2)
        guard heads > 0, headDimension > 0, !o1, !o2, !o3 else { throw Failure.configuration }
        return both
    }

    /// Whole-block resident bytes at full capacity.
    public var maximumResidentBytes: Int {
        let blocks = (capacityTokens + Self.blockTokens - 1) / Self.blockTokens
        let perToken =
            (try? Self.bytesPerToken(
                heads: heads, headDimension: headDimension, dtype: dtype)) ?? Int.max
        return perToken * blocks * Self.blockTokens
    }

    /// Largest append the codec bounds admit in one call.
    public var maximumAppendRows: Int {
        NemotronSuperFP8KVCodec.maximumElements / (heads * headDimension)
    }

    public var allocatedTokens: Int { codes.first?.dim(2) ?? 0 }

    /// Explicit arrays of one append: codec temporaries for K and V at 32 bytes
    /// per element plus one growth copy of every buffer. Excludes retained
    /// storage, backend scratch and concurrent lazy evaluations.
    public func appendAllowanceBytes(rows: Int) throws -> Int {
        guard rows > 0, rows <= maximumAppendRows else { throw Failure.capacity }
        let width = heads * headDimension
        let codec = rows * width * 2 * 32
        let (growth, overflow) = maximumResidentBytes.addingReportingOverflow(codec)
        guard !overflow else { throw Failure.capacity }
        return growth
    }

    private func validate(_ tensor: MLXArray, encoded: Bool) throws -> Int {
        guard tensor.ndim == 4, tensor.dim(0) == 1, tensor.dim(1) == heads,
            tensor.dim(3) == headDimension, tensor.dim(2) > 0
        else { throw Failure.shape }
        guard tensor.dtype == (encoded ? .uint8 : dtype) else { throw Failure.dtype }
        return tensor.dim(2)
    }

    private func reserve(rows: Int) throws {
        let needed = offset + rows
        guard rows > 0, needed <= capacityTokens else { throw Failure.capacity }
        guard needed > allocatedTokens else { return }
        let rounded = (needed + Self.blockTokens - 1) / Self.blockTokens * Self.blockTokens
        let extra = min(rounded, capacityTokens) - allocatedTokens
        let shape = [1, heads, extra, headDimension]
        // Distinct wrappers: slice assignment mutates an MLXArray in place.
        func blocks(_ dtype: DType) -> [MLXArray] {
            (0 ..< 2).map { _ in MLXArray.zeros(shape, dtype: dtype) }
        }
        if codes.isEmpty {
            codes = blocks(.uint8)
            decoded = blocks(dtype)
        } else {
            codes = zip(codes, blocks(.uint8)).map { concatenated([$0, $1], axis: 2) }
            decoded = zip(decoded, blocks(dtype)).map { concatenated([$0, $1], axis: 2) }
        }
    }

    private func write(codes keyCodes: MLXArray, _ valueCodes: MLXArray) throws {
        let count = keyCodes.dim(2)
        guard count <= maximumAppendRows else { throw Failure.capacity }
        let keys = try NemotronSuperFP8KVCodec.decode(keyCodes, dtype: dtype)
        let values = try NemotronSuperFP8KVCodec.decode(valueCodes, dtype: dtype)
        try reserve(rows: count)
        let range = offset ..< (offset + count)
        codes[0][.ellipsis, range, 0...] = keyCodes
        codes[1][.ellipsis, range, 0...] = valueCodes
        decoded[0][.ellipsis, range, 0...] = keys
        decoded[1][.ellipsis, range, 0...] = values
        offset += count
    }

    /// Validation, capacity and codec errors leave the previous state unchanged.
    /// Later evaluation errors require owner invalidation, as for native KV.
    public func checkedUpdate(keys newKeys: MLXArray, values newValues: MLXArray) throws -> (
        MLXArray, MLXArray
    ) {
        let count = try validate(newKeys, encoded: false)
        guard try validate(newValues, encoded: false) == count else { throw Failure.shape }
        guard count <= capacityTokens - offset, count <= maximumAppendRows else {
            throw Failure.capacity
        }
        let keyCodes = try NemotronSuperFP8KVCodec.encode(newKeys)
        let valueCodes = try NemotronSuperFP8KVCodec.encode(newValues)
        try write(codes: keyCodes, valueCodes)
        return (decoded[0][.ellipsis, ..<offset, 0...], decoded[1][.ellipsis, ..<offset, 0...])
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        do { return try checkedUpdate(keys: keys, values: values) } catch {
            preconditionFailure("Invalid Super FP8 KV history update: \(error)")
        }
    }

    /// Abandon every row at or after `newOffset`.
    public func rewind(to newOffset: Int) throws {
        guard newOffset >= 0, newOffset <= offset else { throw Failure.rewind }
        offset = newOffset
    }

    @discardableResult public func trim(_ n: Int) -> Int {
        let count = min(offset, max(0, n))
        offset -= count
        return count
    }

    /// Lazy views of the committed FP8 codes `[K, V]`, `[1, heads, offset, d]`.
    public func committedCodes() -> [MLXArray] {
        guard offset > 0 else { return [] }
        return codes.map { $0[.ellipsis, ..<offset, 0...] }
    }

    public func innerState() -> [MLXArray] { codes + decoded }

    public var state: [MLXArray] {
        get { committedCodes().map { $0 + MLXArray(UInt8(0)) } }
        set {
            do { try restore(newValue) } catch {
                preconditionFailure("Invalid Super FP8 KV history state: \(error)")
            }
        }
    }

    /// Replace the committed rows with FP8 codes `[K, V]`. Rows are staged in
    /// new buffers, so rejected input keeps the previous state.
    public func restore(_ state: [MLXArray]) throws {
        if state.isEmpty {
            offset = 0
            return
        }
        guard state.count == 2 else { throw Failure.state }
        let count = try validate(state[0], encoded: true)
        guard try validate(state[1], encoded: true) == count, count <= capacityTokens else {
            throw Failure.state
        }
        let staged = try NemotronSuperFP8KVHistory(
            heads: heads, headDimension: headDimension, capacityTokens: capacityTokens,
            dtype: dtype)
        var start = 0
        while start < count {
            let end = min(count, start + maximumAppendRows)
            try staged.write(
                codes: state[0][.ellipsis, start ..< end, 0...],
                state[1][.ellipsis, start ..< end, 0...])
            start = end
        }
        codes = staged.codes
        decoded = staged.decoded
        offset = staged.offset
    }

    public var metaState: [String] {
        get {
            [
                "super-unit-fp8-e4m3-history-v1", String(heads), String(headDimension),
                String(capacityTokens), String(describing: dtype),
            ]
        }
        set { precondition(newValue == metaState, "Super FP8 KV history metadata mismatch") }
    }

    public var isTrimmable: Bool { true }

    public func copy() -> any KVCache {
        // Configuration was validated by this instance. The copy consumes
        // another caller-owned reservation.
        let result = try! NemotronSuperFP8KVHistory(
            heads: heads, headDimension: headDimension,
            capacityTokens: capacityTokens, dtype: dtype)
        try! result.restore(committedCodes())
        return result
    }

    public func checkedMask(n: Int, windowSize: Int?, returnArray: Bool) throws
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        guard n > 0, n <= capacityTokens - offset, windowSize == nil else {
            throw Failure.capacity
        }
        if n == 1 { return .none }
        if !returnArray { return .causal }
        let (elements, overflow) = n.multipliedReportingOverflow(by: offset + n)
        guard !overflow, elements <= NemotronSuperFP8KVCodec.maximumElements else {
            throw Failure.capacity
        }
        return .array(createCausalMask(n: n, offset: offset, windowSize: nil))
    }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        do { return try checkedMask(n: n, windowSize: windowSize, returnArray: returnArray) } catch
        { preconditionFailure("Invalid Super FP8 KV history mask: \(error)") }
    }
}
