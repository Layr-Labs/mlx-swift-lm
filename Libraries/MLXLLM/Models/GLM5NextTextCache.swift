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
    /// Fitting appends on the default GPU stream run `GLM5NextDSAAppendKernel`
    /// (same bits as the slice updates); false is the slice-update oracle.
    var usesAppendKernel = true
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
        if usesAppendKernel,
            let next = GLM5NextDSAAppendKernel.append(
                storage: storage, incoming: incoming, offset: offset)
        {
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
        result.usesAppendKernel = usesAppendKernel
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
public final class GLM5NextKDACache: KVCache {
    private(set) var revision: UInt64 = 0
    var current: GLM5NextLinearAttention.State?
    private var rollback: GLM5NextLinearAttention.State?
    private var rollbackOffset = 0
    public private(set) var offset = 0
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }
    public init() {}
    public func innerState() -> [MLXArray] {
        state + (rollback.map { [$0.convolution, $0.recurrent] } ?? [])
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
    func advance(_ next: GLM5NextLinearAttention.State, count: Int) {
        rollback = current.map { Self.clone($0) }
        rollbackOffset = offset
        current = Self.compact(next)
        offset += count
        changed()
    }
    func canTruncate(to count: Int) -> Bool {
        count >= 0 && count <= offset && (count == 0 || count == offset || count == rollbackOffset)
    }
    public func truncate(to count: Int) throws {
        guard canTruncate(to: count) else {
            throw GLM5NextLoadError.invalid(
                "KDA rewind requires a saved boundary plus accepted-token replay")
        }
        if count == offset { return }
        current = count == 0 ? nil : rollback.map { Self.clone($0) }
        offset = count
        rollback = nil
        rollbackOffset = count
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

/// A fitting DSA append in two launches instead of four slice updates (eight
/// dispatches in five dependent steps): each launch writes FRESH buffers of the
/// old capacity holding the logical prefix `0 ..< offset` copied from the old
/// planes and the new columns `offset ..< end`. No input is mutated, so copies,
/// snapshots and in-flight readers of the old buffers are untouched, exactly as
/// with `SliceUpdate`'s copy. Columns past `end` are left unwritten: the cache
/// never reads them (`logical`), and every later append or growth overwrites or
/// drops them. The index planes (keys, gates, validity) the selection reads
/// next are one launch; the latents, read only by the attention, are the other.
/// Bitwise: pure copies. Default GPU stream only; nil means the slice updates.
enum GLM5NextDSAAppendKernel {
    static func append(storage: [MLXArray], incoming: [MLXArray], offset: Int) -> [MLXArray]? {
        guard storage.count == 4, incoming.count == 4, glm5NextOnDefaultGPUStream() else {
            return nil
        }
        let (batch, capacity, count) = (storage[0].dim(0), storage[0].dim(1), incoming[0].dim(1))
        let end = offset + count
        guard count > 0, end <= capacity, storage[3].ndim == 2, storage[3].dtype == .bool,
            incoming[3].dtype == .bool, storage.prefix(3).allSatisfy({ $0.ndim == 3 }),
            storage[2].dim(2) == storage[1].dim(2),
            zip(storage, incoming).allSatisfy({
                $0.dtype == $1.dtype && $0.dim(0) == batch && $1.dim(0) == batch
            }),
            zip(storage.prefix(3), incoming.prefix(3)).allSatisfy({ $0.dim(2) == $1.dim(2) }),
            batch * capacity * max(storage[0].dim(2), 2 * storage[1].dim(2) + 1) <= Int(Int32.max)
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            // Rows of whole 32-bit words move as words when every row plane is
            // a device buffer (MLX passes inputs under 8 elements as constant).
            let words =
                storage[1].dtype.size == storage[2].dtype.size
                && zip(storage.prefix(3), incoming.prefix(3)).allSatisfy {
                    ($0.dim(2) * $0.dtype.size) % 4 == 0 && $0.size >= 8 && $1.size >= 8
                }
            // Elements per row, or 32-bit words per row.
            func width(_ plane: MLXArray) -> Int {
                words ? plane.dim(2) * plane.dtype.size / 4 : plane.dim(2)
            }
            let (latentWidth, keyWidth) = (width(storage[0]), width(storage[1]))
            let params = GLM5NextKernelConstants.shared.int32s([Int32(offset), Int32(count)])
            // One thread per (unit, column, batch row) of the copied prefix and
            // the new columns; a typed load and store moves the bits as is.
            let index = (words ? indexWordKernel : indexKernel)(
                [storage[1], incoming[1], storage[2], incoming[2], storage[3], incoming[3], params],
                template: [("KD", keyWidth)],
                grid: (2 * keyWidth + 1, end, batch),
                threadGroup: (min(256, 2 * keyWidth + 1), 1, 1),
                outputShapes: [storage[1].shape, storage[2].shape, storage[3].shape],
                outputDTypes: [storage[1].dtype, storage[2].dtype, .bool])
            let latents = (words ? latentWordKernel : latentKernel)(
                [storage[0], incoming[0], params],
                template: [("LD", latentWidth)],
                grid: (latentWidth, end, batch), threadGroup: (min(256, latentWidth), 1, 1),
                outputShapes: [storage[0].shape], outputDTypes: [storage[0].dtype])
            return [latents[0], index[0], index[1], index[2]]
        #else
            return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let indexKernel = MLXFast.metalKernel(
            name: "glm5_next_dsa_append_index",
            inputNames: ["keys", "new_keys", "gates", "new_gates", "valid", "new_valid", "params"],
            outputNames: ["keys_out", "gates_out", "valid_out"],
            source: """
                const uint e = thread_position_in_grid.x;
                const int n = int(thread_position_in_grid.y);
                const int b = int(thread_position_in_grid.z);
                const int capacity = keys_shape[1];
                const int offset = params[0];
                const int count = params[1];
                const bool fresh = n >= offset;
                const size_t old_row = size_t(b) * capacity + n;
                const size_t new_row = size_t(b) * count + (n - offset);
                if (e < uint(KD)) {
                    keys_out[old_row * KD + e] =
                        fresh ? new_keys[new_row * KD + e] : keys[old_row * KD + e];
                } else if (e < uint(2 * KD)) {
                    const uint v = e - uint(KD);
                    gates_out[old_row * KD + v] =
                        fresh ? new_gates[new_row * KD + v] : gates[old_row * KD + v];
                } else if (e == uint(2 * KD)) {
                    valid_out[old_row] = fresh ? new_valid[new_row] : valid[old_row];
                }
                """)

        /// `indexKernel` moving 32-bit words (KD words per key/gate row).
        private static let indexWordKernel = MLXFast.metalKernel(
            name: "glm5_next_dsa_append_index_words",
            inputNames: ["keys", "new_keys", "gates", "new_gates", "valid", "new_valid", "params"],
            outputNames: ["keys_out", "gates_out", "valid_out"],
            source: """
                const uint e = thread_position_in_grid.x;
                const int n = int(thread_position_in_grid.y);
                const int b = int(thread_position_in_grid.z);
                const int capacity = keys_shape[1];
                const int offset = params[0];
                const int count = params[1];
                const bool fresh = n >= offset;
                const size_t old_row = size_t(b) * capacity + n;
                const size_t new_row = size_t(b) * count + (n - offset);
                if (e < uint(KD)) {
                    const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_keys : keys);
                    reinterpret_cast<device uint*>(keys_out)[old_row * KD + e] =
                        src[(fresh ? new_row : old_row) * KD + e];
                } else if (e < uint(2 * KD)) {
                    const uint v = e - uint(KD);
                    const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_gates : gates);
                    reinterpret_cast<device uint*>(gates_out)[old_row * KD + v] =
                        src[(fresh ? new_row : old_row) * KD + v];
                } else if (e == uint(2 * KD)) {
                    valid_out[old_row] = fresh ? new_valid[new_row] : valid[old_row];
                }
                """)

        /// `latentKernel` moving 32-bit words (LD words per latent row).
        private static let latentWordKernel = MLXFast.metalKernel(
            name: "glm5_next_dsa_append_latents_words",
            inputNames: ["latents", "new_latents", "params"],
            outputNames: ["latents_out"],
            source: """
                const uint e = thread_position_in_grid.x;
                const int n = int(thread_position_in_grid.y);
                const int b = int(thread_position_in_grid.z);
                const int capacity = latents_shape[1];
                const int offset = params[0];
                const int count = params[1];
                const bool fresh = n >= offset;
                const size_t old_row = size_t(b) * capacity + n;
                const size_t new_row = size_t(b) * count + (n - offset);
                const device uint* src = reinterpret_cast<const device uint*>(fresh ? new_latents : latents);
                reinterpret_cast<device uint*>(latents_out)[old_row * LD + e] =
                    src[(fresh ? new_row : old_row) * LD + e];
                """)

        private static let latentKernel = MLXFast.metalKernel(
            name: "glm5_next_dsa_append_latents",
            inputNames: ["latents", "new_latents", "params"],
            outputNames: ["latents_out"],
            source: """
                const uint e = thread_position_in_grid.x;
                const int n = int(thread_position_in_grid.y);
                const int b = int(thread_position_in_grid.z);
                const int capacity = latents_shape[1];
                const int offset = params[0];
                const int count = params[1];
                const size_t old_row = size_t(b) * capacity + n;
                latents_out[old_row * LD + e] = n >= offset
                    ? new_latents[(size_t(b) * count + (n - offset)) * LD + e]
                    : latents[old_row * LD + e];
                """)
    #endif
}
