import MLX
import MLXLMCommon

/// Full-attention history owned by one inline assistant request. Pages grow
/// only after a forward reaches them; ordinary attention receives the same
/// chronological K/V rows as KVCacheSimple. Gathered arrays are step roots,
/// never persistent storage. The caller owns admission and checked retirement.
final class Qwen35MTPPagedKVCache: KVCache {
    static let pageTokens = 64
    static let maximumTokens = 1 << 20
    private struct Page {
        let keys: MLXArray
        let values: MLXArray
    }
    private var pages: [Page] = []
    private var geometry: (heads: Int, width: Int, dtype: DType)?
    private(set) var offset = 0
    let capacityTokens: Int
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }
    var pageCount: Int { pages.count }

    init(capacityTokens: Int) {
        precondition((1 ... Self.maximumTokens).contains(capacityTokens))
        self.capacityTokens = capacityTokens
    }

    func innerState() -> [MLXArray] { pages.flatMap { [$0.keys, $0.values] } }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(
            keys.ndim == 4 && keys.shape == values.shape && keys.dim(0) == 1
                && keys.dim(1) > 0 && keys.dim(2) > 0 && keys.dim(3) > 0
                && keys.dtype == values.dtype
                && [.float16, .bfloat16, .float32].contains(keys.dtype)
                && keys.dim(2) <= capacityTokens - offset)
        if let geometry {
            precondition(
                keys.dim(1) == geometry.heads && keys.dim(3) == geometry.width
                    && keys.dtype == geometry.dtype)
        } else {
            geometry = (keys.dim(1), keys.dim(3), keys.dtype)
        }
        var source = 0
        while source < keys.dim(2) {
            let destination = offset + source
            let index = destination / Self.pageTokens
            let row = destination % Self.pageTokens
            if index == pages.count {
                let shape = [1, keys.dim(1), Self.pageTokens, keys.dim(3)]
                pages.append(
                    .init(
                        keys: .zeros(shape, dtype: keys.dtype),
                        values: .zeros(shape, dtype: keys.dtype)))
            }
            let count = min(Self.pageTokens - row, keys.dim(2) - source)
            // Separate handles preserve snapshots/copies across later writes.
            let keyPage = pages[index].keys[.ellipsis]
            let valuePage = pages[index].values[.ellipsis]
            keyPage[.ellipsis, row ..< (row + count), 0...] =
                keys[.ellipsis, source ..< (source + count), 0...]
            valuePage[.ellipsis, row ..< (row + count), 0...] =
                values[.ellipsis, source ..< (source + count), 0...]
            pages[index] = .init(keys: keyPage, values: valuePage)
            source += count
        }
        offset += keys.dim(2)
        let result = state
        return (result[0], result[1])
    }

    var state: [MLXArray] {
        get {
            guard offset > 0 else { return [] }
            let rows = pages.enumerated().map { index, page in
                let count = min(Self.pageTokens, offset - index * Self.pageTokens)
                return Page(
                    keys: page.keys[.ellipsis, 0 ..< count, 0...],
                    values: page.values[.ellipsis, 0 ..< count, 0...])
            }
            if rows.count == 1 { return [rows[0].keys, rows[0].values] }
            return [
                concatenated(rows.map(\.keys), axis: 2), concatenated(rows.map(\.values), axis: 2),
            ]
        }
        set {
            if newValue.isEmpty {
                pages = []
                geometry = nil
                offset = 0
                return
            }
            precondition(
                newValue.count == 2 && newValue[0].ndim == 4
                    && newValue[0].shape == newValue[1].shape
                    && newValue[0].dim(2) <= capacityTokens)
            pages = []
            geometry = nil
            offset = 0
            _ = update(keys: newValue[0], values: newValue[1])
        }
    }

    var metaState: [String] {
        get { [""] }
        set { precondition(newValue == [""]) }
    }

    @discardableResult func trim(_ n: Int) -> Int {
        let removed = min(offset, max(0, n))
        offset -= removed
        let count = (offset + Self.pageTokens - 1) / Self.pageTokens
        pages = Array(pages.prefix(count))
        return removed
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> any KVCache {
        let result = Qwen35MTPPagedKVCache(capacityTokens: capacityTokens)
        result.offset = offset
        result.geometry = geometry
        result.pages = pages
        return result
    }
}
