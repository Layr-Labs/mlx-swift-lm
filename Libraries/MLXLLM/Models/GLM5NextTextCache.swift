// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon

/// Request-owned physical history. Pool keys are rebuilt from the retained raw
/// index/gate history, so a trim cannot expose a stale compressed pool.
///
/// Storage keeps geometric spare capacity along the token axis and appends
/// with slice updates, so a request's appends copy amortized O(1) history per
/// token instead of re-concatenating the whole history every round. MLX keeps
/// value semantics: an update writes in place only when no other array (a
/// copy, checkpoint or pending graph) still references the buffer. Readers
/// only ever see the logical `0 ..< offset` prefix.
public final class GLM5NextDSACache: KVCache {
    /// Smallest token capacity allocated by an append that must grow.
    static let minimumCapacity = 256

    /// `.exact` keeps every append a single concatenation of the logical
    /// prefix and the new columns into a fresh, exactly sized buffer. Owners
    /// that publish each round's history as a compact measured snapshot use it
    /// so no spare columns are allocated or charged.
    enum Growth { case geometric, exact }
    let growth: Growth

    private(set) var revision: UInt64 = 0
    /// `[latents, indexKeys, poolGates, validity]` with `capacity >= offset`
    /// columns; columns past `offset` are unspecified and never read.
    private var storage: [MLXArray] = []
    public private(set) var offset = 0
    public var latents: MLXArray? { logical(0) }
    public var indexKeys: MLXArray? { logical(1) }
    public var poolGates: MLXArray? { logical(2) }
    public var validity: MLXArray? { logical(3) }
    var capacity: Int { storage.first?.dim(1) ?? 0 }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    public convenience init() { self.init(growth: .geometric) }
    init(growth: Growth) { self.growth = growth }
    public func innerState() -> [MLXArray] { state }
    /// Exactly the logical history; serialization never sees spare capacity.
    public var state: [MLXArray] {
        get { (0 ..< storage.count).compactMap(logical) }
        set {
            precondition(
                newValue.isEmpty
                    || (newValue.count == 4 && newValue[0].ndim == 3
                        && newValue[1].ndim == 3 && newValue[2].shape == newValue[1].shape
                        && newValue[3].shape == Array(newValue[0].shape.prefix(2))
                        && newValue[3].dtype == .bool && newValue[1].dim(0) == newValue[0].dim(0)
                        && newValue[1].dim(1) == newValue[0].dim(1)),
                "invalid GLM DSA serialized state")
            replace(storage: newValue, offset: newValue.first?.dim(1) ?? 0)
        }
    }
    public var metaState: [String] {
        get { ["glm5_dsa_v1", String(offset)] }
        set { precondition(newValue == metaState, "GLM DSA metadata must match serialized arrays") }
    }
    func append(latents: MLXArray, keys: MLXArray, gates: MLXArray, validity: MLXArray) throws {
        guard latents.ndim == 3, keys.ndim == 3, gates.shape == keys.shape,
            validity.shape == Array(latents.shape.prefix(2)), validity.dtype == .bool,
            keys.dim(0) == latents.dim(0), keys.dim(1) == latents.dim(1)
        else { throw GLM5NextLoadError.invalid("DSA append geometry") }
        let incoming = [latents, keys, gates, validity]
        if storage.isEmpty {
            state = incoming
            return
        }
        for (old, next) in zip(storage, incoming) {
            guard old.dtype == next.dtype, old.dim(0) == next.dim(0),
                Array(old.shape.dropFirst(2)) == Array(next.shape.dropFirst(2))
            else { throw GLM5NextLoadError.invalid("DSA history changed batch, dtype or width") }
        }
        let count = latents.dim(1)
        let end = offset + count
        guard end > offset else { return }
        if growth == .exact || end > capacity {
            // Copy the logical prefix once; geometric growth adds spare
            // columns that this and later appends fill in place.
            let grown = growth == .exact ? end : max(end, 2 * capacity, Self.minimumCapacity)
            let next = zip(storage, incoming).map { old, new -> MLXArray in
                let prefix = offset == old.dim(1) ? old : old[0..., ..<offset]
                guard grown > end else { return concatenated([prefix, new], axis: 1) }
                var shape = old.shape
                shape[1] = grown - end
                return concatenated(
                    [prefix, new, MLXArray.zeros(shape, dtype: old.dtype)], axis: 1)
            }
            replace(storage: next, offset: end)
            return
        }
        let next = zip(storage, incoming).map { old, new -> MLXArray in
            // Never mutate `old`: copies and snapshots share its object.
            let updated = old[.ellipsis]
            updated[0..., offset ..< end] = new
            return updated
        }
        replace(storage: next, offset: end)
    }
    public func truncate(to count: Int) throws {
        guard count >= 0 && count <= offset else {
            throw GLM5NextLoadError.invalid("DSA rewind position")
        }
        if count == offset { return }
        if count == 0 {
            state = []
            return
        }
        // Rewound columns become spare capacity; the next append overwrites them.
        replace(storage: storage, offset: count)
    }
    @discardableResult public func trim(_ n: Int) -> Int {
        guard n >= 0 else { return 0 }
        let removed = min(n, offset)
        // Bounds are established above; this does not suppress a failed rewind.
        do { try truncate(to: offset - removed) } catch { preconditionFailure("\(error)") }
        return removed
    }
    public func copy() -> any KVCache {
        let result = GLM5NextDSACache(growth: growth)
        result.adopt(self)
        return result
    }
    /// Shares the storage arrays: no path mutates a stored array object (an
    /// in-place append writes through a fresh wrapper), so copies stay
    /// independent while MLX may still donate an unshared buffer.
    func adopt(_ other: GLM5NextDSACache) {
        replace(storage: other.storage, offset: other.offset)
    }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("GLM DSA requires latent, index, gate and validity together")
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        precondition(windowSize == nil, "GLM DSA is not a sliding-window cache")
        return .causal
    }
    private func logical(_ index: Int) -> MLXArray? {
        guard storage.indices.contains(index) else { return nil }
        let array = storage[index]
        guard offset < array.dim(1) else { return array }
        return array.ndim == 2 ? array[0..., ..<offset] : array[0..., ..<offset, 0...]
    }
    private func replace(storage next: [MLXArray], offset count: Int) {
        storage = next
        offset = next.isEmpty ? 0 : count
        precondition(revision < UInt64.max, "DSA cache revision exhausted")
        revision += 1
    }
}

/// Fixed-size KDA state plus one forward-start rollback checkpoint. Arbitrary
/// prefix truncation cannot be implemented by slicing this recurrent state.
/// A forward that captured its window's per-position states (an MTP
/// verification window) can also be truncated to any token of that window.
public final class GLM5NextKDACache: KVCache {
    private(set) var revision: UInt64 = 0
    var current: GLM5NextLinearAttention.State?
    private var rollback: GLM5NextLinearAttention.State?
    private var rollbackOffset = 0
    /// The last forward's per-position states when it captured them: entry t
    /// is the state after token `windowStart + t`. Dropped by the next
    /// forward, any truncation and any state replacement.
    private var window: GLM5NextKDASublayer.CapturedStates?
    private var windowStart = 0
    public private(set) var offset = 0
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }
    public init() {}
    public func innerState() -> [MLXArray] {
        state + (rollback.map { [$0.convolution, $0.recurrent] } ?? [])
            + (window.map { [$0.convolution, $0.recurrent] } ?? [])
    }
    public var state: [MLXArray] {
        get { current.map { [$0.convolution, $0.recurrent] } ?? [] }
        set {
            precondition(
                newValue.isEmpty
                    || (newValue.count == 2 && newValue[0].ndim == 3
                        && newValue[1].ndim == 4 && newValue[1].dtype == .float32
                        && newValue[0].dim(0) == newValue[1].dim(0)),
                "invalid GLM KDA serialized state")
            current =
                newValue.isEmpty
                ? nil : Self.compact(.init(convolution: newValue[0], recurrent: newValue[1]))
            rollback = nil
            rollbackOffset = 0
            window = nil
            if newValue.isEmpty { offset = 0 }
            changed()
        }
    }
    public var metaState: [String] {
        get { ["glm5_kda_v1", String(offset)] }
        set {
            precondition(
                newValue.count == 2 && newValue[0] == "glm5_kda_v1", "invalid KDA metadata")
            guard let count = Int(newValue[1]), count >= 0, count == 0 || current != nil else {
                preconditionFailure("invalid KDA offset")
            }
            offset = count
            changed()
        }
    }
    /// `window`: the forward's captured per-position states (`[B,count,...]`),
    /// kept until the next forward or truncation so any of its tokens can end
    /// the committed prefix.
    func advance(
        _ next: GLM5NextLinearAttention.State, count: Int,
        window captured: GLM5NextKDASublayer.CapturedStates? = nil
    ) {
        rollback = current.map { Self.clone($0) }
        rollbackOffset = offset
        current = Self.compact(next)
        window = captured
        windowStart = offset
        offset += count
        changed()
    }
    func canTruncate(to count: Int) -> Bool {
        count >= 0 && count <= offset
            && (count == 0 || count == offset || count == rollbackOffset
                || (window != nil && count > windowStart))
    }
    public func truncate(to count: Int) throws {
        guard canTruncate(to: count) else {
            throw GLM5NextLoadError.invalid(
                "KDA rewind requires a saved boundary plus accepted-token replay")
        }
        if count == offset {
            // The logical state is unchanged; only the window is released.
            window = nil
            return
        }
        if let window, count > windowStart {
            // Entry t is the state after window token t, from the same
            // recurrence that produced the window's final state.
            let index = count - windowStart - 1
            current = Self.compact(
                .init(
                    convolution: window.convolution[0..., index],
                    recurrent: window.recurrent[0..., index]))
        } else {
            current = count == 0 ? nil : rollback.map { Self.clone($0) }
        }
        offset = count
        rollback = nil
        rollbackOffset = count
        window = nil
        changed()
    }
    @discardableResult public func trim(_ n: Int) -> Int { 0 }
    public func copy() -> any KVCache {
        let result = GLM5NextKDACache()
        result.adopt(self)
        return result
    }
    func adopt(_ other: GLM5NextKDACache) {
        current = other.current.map { Self.clone($0) }
        rollback = other.rollback.map { Self.clone($0) }
        offset = other.offset
        rollbackOffset = other.rollbackOffset
        window = other.window.map {
            .init(convolution: $0.convolution[.ellipsis], recurrent: $0.recurrent[.ellipsis])
        }
        windowStart = other.windowStart
        changed()
    }
    private func changed() {
        precondition(revision < UInt64.max, "KDA cache revision exhausted")
        revision += 1
    }
    /// Lazy ownership boundary: the owner must evaluate and join innerState()
    /// before treating this cache as compact. Pending graphs still retain inputs.
    /// In the pinned CPU/Metal core, contiguous retains at most logical bytes
    /// plus 16 KiB of backing storage (before allocator rounding).
    private static func compact(_ value: GLM5NextLinearAttention.State)
        -> GLM5NextLinearAttention.State
    {
        .init(
            convolution: value.convolution.contiguous(),
            recurrent: value.recurrent.contiguous())
    }
    private static func clone(_ value: GLM5NextLinearAttention.State)
        -> GLM5NextLinearAttention.State
    {
        .init(convolution: value.convolution[.ellipsis], recurrent: value.recurrent[.ellipsis])
    }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("KDA must stage convolution and FP32 recurrent state together")
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        precondition(windowSize == nil, "KDA is not a sliding-window cache")
        return .none
    }
}

public final class GLM5NextTextCache {
    struct Revision: Equatable {
        let identities: [ObjectIdentifier]
        let versions: [UInt64]
        let offsets: [Int]
    }
    func revision() -> Revision {
        Revision(
            identities: layers.map { ObjectIdentifier($0 as AnyObject) },
            versions: layers.map {
                if let layer = $0 as? GLM5NextKDACache { return layer.revision }
                if let layer = $0 as? GLM5NextDSACache { return layer.revision }
                return 0
            }, offsets: layers.map(\.offset))
    }
    public let layers: [any KVCache]
    public var offset: Int { layers.first?.offset ?? 0 }
    public init(configuration: GLM5NextTextConfiguration) {
        layers = configuration.layerTypes.map { kind -> any KVCache in
            kind == .linearAttention ? GLM5NextKDACache() : GLM5NextDSACache()
        }
    }
    public init(layers: [any KVCache]) { self.layers = layers }
    public func copy() -> GLM5NextTextCache { .init(layers: layers.map { $0.copy() }) }
    func validate(_ configuration: GLM5NextTextConfiguration) throws {
        guard layers.count == configuration.hiddenLayers else {
            throw GLM5NextLoadError.invalid("cache layer count")
        }
        for (index, layer) in layers.enumerated() {
            guard layer.offset == offset,
                configuration.layerTypes[index] == .linearAttention
                    ? layer is GLM5NextKDACache : layer is GLM5NextDSACache
            else { throw GLM5NextLoadError.invalid("cache layer kind/frontier \(index)") }
        }
    }
    public func truncate(to count: Int) throws {
        for layer in layers {
            if let kda = layer as? GLM5NextKDACache {
                guard kda.canTruncate(to: count) else {
                    throw GLM5NextLoadError.invalid("KDA rewind requires replay")
                }
            } else if layer is GLM5NextDSACache {
                guard count >= 0 && count <= layer.offset else {
                    throw GLM5NextLoadError.invalid("DSA rewind position")
                }
            } else {
                throw GLM5NextLoadError.invalid("foreign GLM cache")
            }
        }
        for layer in layers {
            if let kda = layer as? GLM5NextKDACache { try kda.truncate(to: count) }
            if let dsa = layer as? GLM5NextDSACache { try dsa.truncate(to: count) }
        }
    }
    func adopt(_ other: GLM5NextTextCache) {
        precondition(layers.count == other.layers.count)
        for (destination, source) in zip(layers, other.layers) {
            if let destination = destination as? GLM5NextKDACache,
                let source = source as? GLM5NextKDACache
            {
                destination.adopt(source)
            } else if let destination = destination as? GLM5NextDSACache,
                let source = source as? GLM5NextDSACache
            {
                destination.adopt(source)
            } else {
                preconditionFailure("foreign cache in GLM transaction")
            }
        }
    }
}
