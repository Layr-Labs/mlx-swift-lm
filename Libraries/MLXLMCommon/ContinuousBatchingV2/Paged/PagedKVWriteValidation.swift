import MLX

/// A runtime projection changed the native KV storage contract established at
/// engine construction. These are metadata only; no prompt or tensors escape.
public struct CBv2PagedKVWriteError: Error, Sendable, CustomStringConvertible {
    public let layerIndex: Int?
    public let expected: DType
    public let keys: DType
    public let values: DType
    public let reason: String?

    public init(layerIndex: Int?, expected: DType, keys: DType, values: DType, reason: String? = nil) {
        self.layerIndex = layerIndex; self.expected = expected; self.keys = keys; self.values = values
        self.reason = reason
    }

    public var description: String {
        if let reason { return "paged KV contract mismatch at layer \(layerIndex.map(String.init) ?? "unknown"): \(reason)" }
        return "paged KV dtype mismatch at layer \(layerIndex.map(String.init) ?? "unknown"): expected \(expected), keys \(keys), values \(values)"
    }
}

/// Shared by all groups/caches of one pool, confined to its engine queue.
/// A nonthrowing model forward can finish constructing a shape-valid graph,
/// but no later attention operation may mutate/read cache after the first fault.
/// The engine checks this latch before sampling or evaluating that graph.
final class CBv2PagedKVWriteValidation {
    private(set) var fault: CBv2PagedKVWriteError?
    var isFaulted: Bool { fault != nil }

    @discardableResult
    func validate(keys: MLXArray, values: MLXArray, expected: DType, layerIndex: Int? = nil) -> Bool {
        guard fault == nil else { return false }
        guard keys.dtype == expected, values.dtype == expected else {
            fault = CBv2PagedKVWriteError(
                layerIndex: layerIndex, expected: expected, keys: keys.dtype, values: values.dtype)
            return false
        }
        return true
    }

    func record(_ error: CBv2PagedKVWriteError) { if fault == nil { fault = error } }
    @discardableResult
    func refuse(_ reason: String, expected: DType, layerIndex: Int? = nil) -> Bool {
        record(.init(layerIndex: layerIndex, expected: expected, keys: expected, values: expected, reason: reason))
        return false
    }

    @discardableResult
    func validateShape(keys: MLXArray, values: MLXArray, group: PagedKVGroupKey,
                       rank: Int, batch: Int?, tokens: Int? = nil, layerIndex: Int? = nil) -> Bool {
        guard fault == nil else { return false }
        guard (rank == 3 || rank == 4), keys.ndim == rank, values.ndim == rank else {
            return refuse("invalid projected K/V rank", expected: group.dtype, layerIndex: layerIndex)
        }
        let count = tokens ?? keys.dim(rank - 2)
        let prefix = rank == 4 ? [batch ?? keys.dim(0), group.kvHeads, count] : [group.kvHeads, count]
        guard count >= 0, keys.shape == prefix + [group.headDim], values.shape == prefix + [group.valueHeadDim] else {
            return refuse("projected K/V shape differs from native role geometry", expected: group.dtype, layerIndex: layerIndex)
        }
        return true
    }
    func check() throws { if let fault { throw fault } }
    func clearAfterRetirement() { fault = nil }
}

/// Retain the last valid write-fence graph before building a step. On failure
/// no newly built fence may survive into another request's cache operation.
/// The affected rows are retired completely; this does not claim cursor-only
/// rollback can restore a window overwritten by an already evaluated MTP column.
final class CBv2PagedWriteBoundary {
    private var fences: [(PagedKVGroup, MLXArray)]

    init(pool: PagedKVPool) {
        fences = pool.groupKeys.map {
            let group = pool.group($0)
            return (group, group.writeFence)
        }
    }

    func discardFailedGraphAfterSynchronization() {
        for (group, fence) in fences { group.writeFence = fence }
        fences.removeAll()
    }
}
