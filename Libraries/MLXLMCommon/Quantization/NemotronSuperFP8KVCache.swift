import MLX

/// Experimental request-local unit-scale FP8 cache. Caller serializes all access,
/// owns memory admission and must evaluate/fence before retiring state. No paged
/// attention, automatic growth, persistent-prefix registration or MTP activation.
/// Returns decoded KV; fused CUDA attention arithmetic is not reproduced here.
public final class NemotronSuperFP8KVCache: KVCache {
    public enum Failure: Error { case configuration, shape, dtype, capacity, state }
    public let heads: Int
    public let headDimension: Int
    public let capacityTokens: Int
    public let dtype: DType
    public let maximumResidentBytes: Int
    /// Conservative explicit-array envelope per operation, excluding backend
    /// scratch, retained copies/returned arrays and concurrent lazy evaluations.
    public let operationAllowanceBytes: Int
    public private(set) var offset = 0
    public var maxSize: Int? { capacityTokens }
    private var keys: MLXArray?
    private var values: MLXArray?

    public init(
        heads: Int, headDimension: Int, capacityTokens: Int,
        dtype: DType = .bfloat16
    ) throws {
        guard heads > 0, headDimension > 0, capacityTokens > 0,
            [.float16, .bfloat16, .float32].contains(dtype)
        else { throw Failure.configuration }
        let (width, o1) = heads.multipliedReportingOverflow(by: headDimension)
        let (elements, o2) = width.multipliedReportingOverflow(by: capacityTokens)
        guard !o1, !o2, elements <= NemotronSuperFP8KVCodec.maximumElements else {
            throw Failure.capacity
        }
        self.heads = heads
        self.headDimension = headDimension
        self.capacityTokens = capacityTokens
        self.dtype = dtype
        maximumResidentBytes = elements * 2
        operationAllowanceBytes = elements * 72
    }

    private func validate(_ tensor: MLXArray, encoded: Bool) throws -> Int {
        guard tensor.ndim == 4, tensor.dim(0) == 1, tensor.dim(1) == heads,
            tensor.dim(3) == headDimension, tensor.dim(2) > 0
        else { throw Failure.shape }
        guard tensor.dtype == (encoded ? .uint8 : dtype) else { throw Failure.dtype }
        guard tensor.dim(2) <= capacityTokens else { throw Failure.capacity }
        return tensor.dim(2)
    }

    /// Validation and graph-construction errors leave previous state unchanged.
    /// As with native KVCache, later evaluation errors require owner invalidation.
    public func checkedUpdate(keys newKeys: MLXArray, values newValues: MLXArray) throws -> (
        MLXArray, MLXArray
    ) {
        let count = try validate(newKeys, encoded: false)
        guard try validate(newValues, encoded: false) == count else { throw Failure.shape }
        guard count <= capacityTokens - offset else { throw Failure.capacity }
        let encodedKeys = try NemotronSuperFP8KVCodec.encode(newKeys)
        let encodedValues = try NemotronSuperFP8KVCodec.encode(newValues)
        let nextKeys = keys.map { concatenated([$0, encodedKeys], axis: 2) } ?? encodedKeys
        let nextValues = values.map { concatenated([$0, encodedValues], axis: 2) } ?? encodedValues
        let resultKeys = try NemotronSuperFP8KVCodec.decode(nextKeys, dtype: dtype)
        let resultValues = try NemotronSuperFP8KVCodec.decode(nextValues, dtype: dtype)
        keys = nextKeys
        values = nextValues
        offset += count
        return (resultKeys, resultValues)
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        do { return try checkedUpdate(keys: keys, values: values) } catch {
            preconditionFailure("Invalid Super FP8 KV update: \(error)")
        }
    }

    public func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }
    public var state: [MLXArray] {
        get { innerState().map { $0 + MLXArray(UInt8(0)) } }
        set {
            do { try restore(newValue) } catch {
                preconditionFailure("Invalid Super FP8 KV state: \(error)")
            }
        }
    }
    public func restore(_ state: [MLXArray]) throws {
        if state.isEmpty {
            keys = nil
            values = nil
            offset = 0
            return
        }
        guard state.count == 2 else { throw Failure.state }
        let count = try validate(state[0], encoded: true)
        guard try validate(state[1], encoded: true) == count else { throw Failure.shape }
        let k = state[0] + MLXArray(UInt8(0))
        let v = state[1] + MLXArray(UInt8(0))
        keys = k
        values = v
        offset = count
    }
    public var metaState: [String] {
        get {
            [
                "super-unit-fp8-e4m3-v1", String(heads), String(headDimension),
                String(capacityTokens), String(describing: dtype),
            ]
        }
        set { precondition(newValue == metaState, "Super FP8 KV metadata mismatch") }
    }
    public var isTrimmable: Bool { true }
    @discardableResult public func trim(_ n: Int) -> Int {
        let count = min(offset, max(0, n))
        guard count > 0 else { return 0 }
        let remaining = offset - count
        if remaining == 0 {
            keys = nil
            values = nil
        } else {
            keys = keys![.ellipsis, ..<remaining, 0...]
            values = values![.ellipsis, ..<remaining, 0...]
        }
        offset = remaining
        return count
    }
    public func copy() -> any KVCache {
        //Configuration already validated; copy consumes another caller-owned
        //reservation and may retain source storage until lazy copy evaluation.
        let result = try! NemotronSuperFP8KVCache(
            heads: heads, headDimension: headDimension,
            capacityTokens: capacityTokens, dtype: dtype)
        try! result.restore(state)
        return result
    }
    public func checkedMask(n: Int, windowSize: Int?, returnArray: Bool) throws
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        guard n > 0, n <= capacityTokens - offset,
            windowSize == nil || windowSize! > 0
        else { throw Failure.capacity }
        if n == 1 && windowSize == nil { return .none }
        if !returnArray && windowSize == nil { return .causal }
        let (elements, overflow) = n.multipliedReportingOverflow(by: offset + n)
        guard !overflow, elements <= NemotronSuperFP8KVCodec.maximumElements else {
            throw Failure.capacity
        }
        return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        do { return try checkedMask(n: n, windowSize: windowSize, returnArray: returnArray) } catch
        { preconditionFailure("Invalid Super FP8 KV mask: \(error)") }
    }
}
