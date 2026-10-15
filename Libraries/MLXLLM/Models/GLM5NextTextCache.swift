// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon

/// Request-owned physical history. Pool keys are rebuilt from the retained raw
/// index/gate history, so a trim cannot expose a stale compressed pool.
public final class GLM5NextDSACache: KVCache {
    private(set) var revision: UInt64 = 0
    public private(set) var latents: MLXArray?
    public private(set) var indexKeys: MLXArray?
    public private(set) var poolGates: MLXArray?
    public private(set) var validity: MLXArray?
    public var offset: Int { latents?.dim(1) ?? 0 }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    public init() {}
    public func innerState() -> [MLXArray] { state }
    public var state: [MLXArray] {
        get { [latents, indexKeys, poolGates, validity].compactMap { $0 } }
        set {
            precondition(
                newValue.isEmpty
                    || (newValue.count == 4 && newValue[0].ndim == 3
                        && newValue[1].ndim == 3 && newValue[2].shape == newValue[1].shape
                        && newValue[3].shape == Array(newValue[0].shape.prefix(2))
                        && newValue[3].dtype == .bool && newValue[1].dim(0) == newValue[0].dim(0)
                        && newValue[1].dim(1) == newValue[0].dim(1)),
                "invalid GLM DSA serialized state")
            latents = newValue.first
            indexKeys = newValue.isEmpty ? nil : newValue[1]
            poolGates = newValue.isEmpty ? nil : newValue[2]
            validity = newValue.isEmpty ? nil : newValue[3]
            precondition(revision < UInt64.max, "DSA cache revision exhausted")
            revision += 1
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
        if state.isEmpty {
            state = incoming
            return
        }
        for (old, next) in zip(state, incoming) {
            guard old.dtype == next.dtype, old.dim(0) == next.dim(0),
                Array(old.shape.dropFirst(2)) == Array(next.shape.dropFirst(2))
            else { throw GLM5NextLoadError.invalid("DSA history changed batch, dtype or width") }
        }
        state = zip(state, incoming).map { concatenated([$0.0, $0.1], axis: 1) }
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
        state = state.map { $0.ndim == 2 ? $0[0..., ..<count] : $0[0..., ..<count, 0...] }
    }
    @discardableResult public func trim(_ n: Int) -> Int {
        guard n >= 0 else { return 0 }
        let removed = min(n, offset)
        // Bounds are established above; this does not suppress a failed rewind.
        do { try truncate(to: offset - removed) } catch { preconditionFailure("\(error)") }
        return removed
    }
    public func copy() -> any KVCache {
        let result = GLM5NextDSACache()
        result.state = state.map { $0[.ellipsis] }
        return result
    }
    func adopt(_ other: GLM5NextDSACache) { state = other.state.map { $0[.ellipsis] } }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("GLM DSA requires latent, index, gate and validity together")
    }
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        precondition(windowSize == nil, "GLM DSA is not a sliding-window cache")
        return .causal
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
    public func innerState() -> [MLXArray] { state }
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
                newValue.isEmpty ? nil : .init(convolution: newValue[0], recurrent: newValue[1])
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
        current = next
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
