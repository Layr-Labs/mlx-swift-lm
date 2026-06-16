import Foundation
import MLX

/// Quantized, continuously-batchable KV cache.
///
/// Mirrors ``BatchKVCache`` in layout (right-justified per-row storage with
/// `leftPadding` / `batchOffset`) but stores keys and values as quantized
/// tuples `(wq, scales, biases?)`. New tokens are quantized incrementally
/// (`O(step)`) and appended along the time axis; the full logical view can be
/// returned either dequantized (``update(keys:values:)``) or as quantized
/// tuples (``updateQuantized(keys:values:)``) for the native quantized
/// attention kernel.
public final class QuantizedBatchKVCache: QuantizedBatchKVCacheBase,
    QuantizedKVCacheProtocol
{
}

/// Quantized-storage, continuously-batchable KV cache that does **not**
/// conform to ``QuantizedKVCacheProtocol``.
///
/// Use this for models whose quantized attention kernel is not sink-safe
/// (e.g. GPT-OSS). The cache still stores K/V in quantized form for the
/// capacity win, but ``update(keys:values:)`` returns dequantized fp16 so the
/// model falls back to the regular sink-aware attention path.
public final class DequantBatchKVCache: QuantizedBatchKVCacheBase {
}

/// Base implementation for quantized batched KV caches.
///
/// Holds all the storage, batched-cache logic, and the dequantizing
/// ``update(keys:values:)`` path. The two final subclasses differ only in
/// whether they also conform to ``QuantizedKVCacheProtocol`` (the native
/// quantized-kernel path) or stay off that path (the dequant-only path).
public class QuantizedBatchKVCacheBase: BaseKVCache, BatchPositionedKVCache,
    BatchedCache
{

    // MARK: - Quantization parameters

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    // MARK: - BatchedCache / BatchPositionedKVCache

    /// Allocation chunk size along the time axis.
    public static let allocationStep = 256

    /// Per-row position counter `[B]`. Starts at `-leftPadding[b]`.
    public private(set) var batchOffset: MLXArray

    /// Per-row left padding `[B]`.
    public private(set) var leftPadding: MLXArray

    // MARK: - Storage

    /// Quantized key tuple: `(wq, scales, biases?)`. Time axis is `-2`.
    private var keys: (MLXArray, MLXArray, MLXArray?)?

    /// Quantized value tuple: `(wq, scales, biases?)`. Time axis is `-2`.
    private var values: (MLXArray, MLXArray, MLXArray?)?

    /// Rightmost-valid slot along the time axis.
    private var _idx: Int = 0

    /// Pending right-padding applied by ``finalizeBatched()``.
    private var _rightPadding: MLXArray?

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    public override var maxSize: Int? { nil }
    public override var isTrimmable: Bool { true }

    // MARK: - Init

    public required init(
        leftPadding: [Int],
        groupSize: Int = 64,
        bits: Int = 8,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { Int32(-$0) })
        super.init()
    }

    fileprivate init(
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        keys: (MLXArray, MLXArray, MLXArray?)?,
        values: (MLXArray, MLXArray, MLXArray?)?,
        offset: MLXArray,
        leftPadding: MLXArray,
        idx: Int,
        rightPadding: MLXArray?
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.keys = keys
        self.values = values
        self.batchOffset = offset
        self.leftPadding = leftPadding
        self._idx = idx
        self._rightPadding = rightPadding
        super.init()
    }

    // MARK: - Update

    /// Append `[B, kvHeads, T, D]` keys/values, quantize only the incoming
    /// slice, and return the **dequantized** full logical view. This is the
    /// fallback path for models without a quantized attention kernel.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (quantizedKeys, quantizedValues) = updateQuantized(keys: keys, values: values)

        let dequantizedKeys = dequantized(
            quantizedKeys.0,
            scales: quantizedKeys.1,
            biases: quantizedKeys.2,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let dequantizedValues = dequantized(
            quantizedValues.0,
            scales: quantizedValues.1,
            biases: quantizedValues.2,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

        return (dequantizedKeys, dequantizedValues)
    }

    /// Append `[B, kvHeads, T, D]` keys/values, quantize only the incoming
    /// slice, and return the **quantized** full logical view.
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let prev = _idx
        let stepCount = keys.dim(2)
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)

        precondition(
            kHeadDim % groupSize == 0 && vHeadDim % groupSize == 0,
            "QuantizedBatchKVCacheBase requires groupSize to divide head_dim"
        )

        var currentKeys = self.keys
        var currentValues = self.values

        let needGrow: Bool = {
            guard let storedKeys = currentKeys else { return true }
            return (prev + stepCount) > storedKeys.0.dim(-2)
        }()

        if needGrow {
            let nSteps = (Self.allocationStep + stepCount - 1) / Self.allocationStep
            let newShape = [B, nKVHeads, nSteps * Self.allocationStep]

            if var existingKeys = currentKeys, var existingValues = currentValues {
                if prev % Self.allocationStep != 0 {
                    existingKeys = trimmedTo(existingKeys, limit: prev)
                    existingValues = trimmedTo(existingValues, limit: prev)
                }
                currentKeys = expandQuant(existingKeys, newShape: newShape)
                currentValues = expandQuant(existingValues, newShape: newShape)
            } else {
                currentKeys = initQuant(dim: kHeadDim, shape: newShape, dtype: keys.dtype)
                currentValues = initQuant(dim: vHeadDim, shape: newShape, dtype: keys.dtype)
            }

            self.keys = currentKeys
            self.values = currentValues
        }

        guard var ck = currentKeys, var cv = currentValues else {
            fatalError("QuantizedBatchKVCacheBase storage not initialized")
        }

        let qk = quantized(keys, groupSize: groupSize, bits: bits)
        let qv = quantized(values, groupSize: groupSize, bits: bits)
        let newKeys = (qk.wq, qk.scales, qk.biases)
        let newValues = (qv.wq, qv.scales, qv.biases)

        _idx += stepCount
        batchOffset = batchOffset + Int32(stepCount)

        assign(into: &ck, from: newKeys, range: prev..<_idx)
        assign(into: &cv, from: newValues, range: prev..<_idx)

        self.keys = ck
        self.values = cv

        return (trimmedTo(ck, limit: _idx), trimmedTo(cv, limit: _idx))
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }
        return (trimmedTo(keys, limit: _idx), trimmedTo(values, limit: _idx))
    }

    // MARK: - Mask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let mask = createCausalMask(
            n: n,
            offset: _idx,
            windowSize: windowSize,
            leftPadding: leftPadding
        )
        return .array(mask)
    }

    // MARK: - BatchedCache

    public func prepareBatched(
        leftPadding additionalLeftPadding: [Int]?,
        lengths _: [Int]?,
        rightPadding: [Int]?
    ) {
        if let additionalLeftPadding {
            precondition(
                keys == nil,
                "prepareBatched() with leftPadding can only be called on an empty QuantizedBatchKVCacheBase"
            )
            let additional = MLXArray(additionalLeftPadding.map { Int32($0) })
            leftPadding = leftPadding + additional
            batchOffset = batchOffset - additional
        }

        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            _rightPadding = MLXArray(rightPadding.map { Int32($0) })
        }
    }

    public func finalizeBatched() {
        guard let pending = _rightPadding else { return }
        guard let storedK = keys, let storedV = values else {
            _rightPadding = nil
            return
        }

        let shifts = pending[0..., .newAxis]
        keys = treeMap({ dynamicRoll($0, shifts: shifts, axis: 2) }, storedK)
        values = treeMap({ dynamicRoll($0, shifts: shifts, axis: 2) }, storedV)
        batchOffset = batchOffset - pending
        leftPadding = leftPadding + pending
        _rightPadding = nil
    }

    public func filterBatched(batchIndices: MLXArray) {
        if keys != nil {
            keys = treeMap({ take($0, batchIndices, axis: 0) }, keys!)
            values = treeMap({ take($0, batchIndices, axis: 0) }, values!)
        }
        batchOffset = take(batchOffset, batchIndices, axis: 0)
        leftPadding = take(leftPadding, batchIndices, axis: 0)

        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0 {
            let minPadInt = Int(minLeftPad)
            if let storedK = keys, let storedV = values {
                keys = treeMap({ $0[.ellipsis, minPadInt..., 0...] }, storedK)
                values = treeMap({ $0[.ellipsis, minPadInt..., 0...] }, storedV)
            }
            _idx -= minPadInt
            leftPadding = leftPadding - Int32(minPadInt)
        }
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? QuantizedBatchKVCacheBase else {
            preconditionFailure(
                "QuantizedBatchKVCacheBase.extendBatched requires another QuantizedBatchKVCacheBase"
            )
        }
        extend(other)
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        let cache = KVCacheSimple()
        guard let storedK = keys, let storedV = values else {
            return cache
        }

        let leftPad = max(0, Int(leftPadding[idx].item(Int32.self)))
        let kTuple = treeMap(
            { $0[idx..<(idx + 1), 0..., leftPad..<_idx, 0...] }, storedK)
        let vTuple = treeMap(
            { $0[idx..<(idx + 1), 0..., leftPad..<_idx, 0...] }, storedV)

        let kDQ = dequantized(
            kTuple.0, scales: kTuple.1, biases: kTuple.2,
            groupSize: groupSize, bits: bits, mode: mode)
        let vDQ = dequantized(
            vTuple.0, scales: vTuple.1, biases: vTuple.2,
            groupSize: groupSize, bits: bits, mode: mode)

        eval(kDQ, vDQ)
        cache.state = [kDQ, vDQ]
        return cache
    }

    public func advanceBatched(_: Int) {}

    // MARK: - Extend (in-place admission)

    /// In-place concatenation of another batched cache's rows onto this one.
    /// Both caches are padded to the same right-justified time-axis size, then
    /// concatenated along the batch axis.
    public func extend(_ other: QuantizedBatchKVCacheBase) {
        precondition(
            groupSize == other.groupSize && bits == other.bits && mode == other.mode,
            "QuantizedBatchKVCacheBase.extend requires matching quantization params"
        )

        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            batchOffset = concatenated([batchOffset, other.batchOffset], axis: 0)
            return
        }

        let maxIdx = max(_idx, other._idx)
        var L1 = 0
        var L2 = 0
        var H = 0
        var Dk = 0
        var Dv = 0

        if let storedK = keys {
            L1 = storedK.0.dim(-2)
            H = storedK.0.dim(-3)
            // The unpacked head dim is derived from the scales, not the packed
            // weights, so the empty-cache branch creates correctly-shaped zeros.
            Dk = storedK.1.dim(-1) * groupSize
            Dv = values!.1.dim(-1) * groupSize
        }
        if let storedK = other.keys {
            L2 = storedK.0.dim(-2)
            H = storedK.0.dim(-3)
            Dk = storedK.1.dim(-1) * groupSize
            Dv = other.values!.1.dim(-1) * groupSize
        }
        let maxSize = max(L1, L2)

        func pad(
            _ cacheKeys: (MLXArray, MLXArray, MLXArray?)?,
            _ cacheValues: (MLXArray, MLXArray, MLXArray?)?,
            cacheIdx: Int,
            cacheOffset: MLXArray,
            cacheLeftPadding: MLXArray
        ) -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?), MLXArray, MLXArray)
        {
            var k: (MLXArray, MLXArray, MLXArray?)
            var v: (MLXArray, MLXArray, MLXArray?)
            if let cacheKeys = cacheKeys, let cacheValues = cacheValues {
                k = cacheKeys
                v = cacheValues
            } else {
                let Bc = cacheOffset.dim(0)
                let keyZeros = initQuant(dim: Dk, shape: [Bc, H, 0], dtype: .float32)
                let valueZeros = initQuant(dim: Dv, shape: [Bc, H, 0], dtype: .float32)
                k = keyZeros
                v = valueZeros
            }

            let leftPad = maxIdx - cacheIdx
            var rightPad = maxSize - k.0.dim(-2) - leftPad
            if rightPad < 0 {
                k = trimmedTo(k, limit: k.0.dim(-2) + rightPad)
                v = trimmedTo(v, limit: v.0.dim(-2) + rightPad)
                rightPad = 0
            }
            if leftPad != 0 || rightPad != 0 {
                let widths: [IntOrPair] = [
                    .init(0), .init(0), .init((leftPad, rightPad)), .init(0),
                ]
                k = treeMap({ padded($0, widths: widths) }, k)
                v = treeMap({ padded($0, widths: widths) }, v)
            }
            return (k, v, cacheOffset, cacheLeftPadding + Int32(leftPad))
        }

        let (k1, v1, o1, lp1) = pad(
            keys, values, cacheIdx: _idx,
            cacheOffset: batchOffset, cacheLeftPadding: leftPadding)
        let (k2, v2, o2, lp2) = pad(
            other.keys, other.values, cacheIdx: other._idx,
            cacheOffset: other.batchOffset, cacheLeftPadding: other.leftPadding)

        keys = (
            concatenated([k1.0, k2.0], axis: 0),
            concatenated([k1.1, k2.1], axis: 0),
            zipOptional(k1.2, k2.2) { concatenated([$0, $1], axis: 0) }
        )
        values = (
            concatenated([v1.0, v2.0], axis: 0),
            concatenated([v1.1, v2.1], axis: 0),
            zipOptional(v1.2, v2.2) { concatenated([$0, $1], axis: 0) }
        )
        batchOffset = concatenated([o1, o2], axis: 0)
        leftPadding = concatenated([lp1, lp2], axis: 0)
        _idx = maxIdx
    }

    // MARK: - Trim / state / copy

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    public override var state: [MLXArray] {
        get {
            guard let storedK = keys, let storedV = values else {
                return [batchOffset, leftPadding]
            }
            let tk = trimmedTo(storedK, limit: _idx)
            let tv = trimmedTo(storedV, limit: _idx)
            return [
                tk.0, tk.1, tk.2,
                tv.0, tv.1, tv.2,
                batchOffset, leftPadding,
            ].compactMap { $0 }
        }
        set {
            switch newValue.count {
            case 2:
                batchOffset = newValue[0]
                leftPadding = newValue[1]
                keys = nil
                values = nil
                _idx = 0
            case 6:
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
                batchOffset = newValue[4]
                leftPadding = newValue[5]
                _idx = newValue[0].dim(-2)
            case 8:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
                batchOffset = newValue[6]
                leftPadding = newValue[7]
                _idx = newValue[0].dim(-2)
            default:
                fatalError(
                    "QuantizedBatchKVCacheBase.state setter expects 2, 6, or 8 arrays"
                )
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override var metaState: [String] {
        get {
            let modeName = mode == .affine ? "affine" : "symmetric"
            return [
                String(Self.allocationStep),
                String(_idx),
                String(groupSize),
                String(bits),
                modeName,
            ]
        }
        set {
            // TODO(DAR-319): metaState currently restores only _idx; groupSize,
            // bits, and mode are not serialized here. Acceptable for v1 because
            // prefix-cache/checkpoint is disabled when quantization is on.
            guard newValue.count == 5 else {
                fatalError("QuantizedBatchKVCacheBase.metaState expects 5 values")
            }
            if let off = Int(newValue[1]) {
                _idx = off
            }
        }
    }

    public override func copy() -> any KVCache {
        let new = Self(
            leftPadding: Array(repeating: 0, count: leftPadding.dim(0)),
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    // MARK: - Quantized tuple helpers

    private func initQuant(dim: Int, shape: [Int], dtype: DType)
        -> (MLXArray, MLXArray, MLXArray?)
    {
        let temp = MLXArray.zeros(shape + [dim], dtype: dtype)
        let q = quantized(temp, groupSize: groupSize, bits: bits)
        return (q.wq, q.scales, q.biases)
    }

    private func expandQuant(
        _ tuple: (MLXArray, MLXArray, MLXArray?),
        newShape: [Int]
    ) -> (MLXArray, MLXArray, MLXArray?) {
            return treeMap({ array in
            let zeros = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
            return concatenated([array, zeros], axis: 2)
        }, tuple)
    }

    private func trimmedTo(
        _ tuple: (MLXArray, MLXArray, MLXArray?),
        limit: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        return treeMap({ $0[.ellipsis, ..<limit, 0...] }, tuple)
    }

    private func assign(
        into target: inout (MLXArray, MLXArray, MLXArray?),
        from source: (MLXArray, MLXArray, MLXArray?),
        range: Range<Int>
    ) {
        target.0[.ellipsis, range, 0...] = source.0
        target.1[.ellipsis, range, 0...] = source.1
        switch (target.2, source.2) {
        case (.some(let tb), .some(let sb)):
            tb[.ellipsis, range, 0...] = sb
        case (.none, .none):
            break
        default:
            fatalError("QuantizedBatchKVCacheBase bias mismatch during assignment")
        }
    }

    private func treeMap<T>(
        _ transform: (MLXArray) -> T,
        _ tuple: (MLXArray, MLXArray, MLXArray?)
    ) -> (T, T, T?) {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))
        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    private func zipOptional<T>(
        _ a: T?, _ b: T?,
        combine: (T, T) -> T
    ) -> T? {
        switch (a, b) {
        case (.some(let x), .some(let y)):
            return combine(x, y)
        case (.none, .none):
            return nil
        default:
            fatalError("QuantizedBatchKVCacheBase optional mismatch")
        }
    }
}
