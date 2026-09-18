// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX

/// Shares an integer expression, not mutable Swift wrappers or K/V storage.
/// The bank owns caches; this coordinator keeps only weak cache references.
final class Gemma4UnifiedPositions: CBv2PositionBindingCoordinator {
    private final class Member {
        weak var cache: CBv2LayerCache?
        init(_ cache: CBv2LayerCache) { self.cache = cache }
    }
    private let members: [Member]
    private let cycle: Gemma4PositionCycle
    private let stream: StreamOrDevice
    private var value: MLXArray
    private var nextValue: MLXArray?
    private var identity: UInt
    private var rowCount = 0
    private var bindingVersion: UInt64 = 0
    private var updateVersion: UInt64 = 0
    private var completedWidth: Int?
    private(set) var isActive = true

    struct EvaluationStamp {
        let binding: UInt64
        let updates: UInt64
        let rows: Int
    }

    private init(caches: [CBv2LayerCache], value: MLXArray, identity: UInt) {
        members = caches.map(Member.init)
        cycle = Gemma4PositionCycle(layers: caches.count)
        stream = StreamOrDevice.default
        self.value = value
        self.identity = identity
    }

    static func configure(_ caches: [any CBv2AttendingLayerCache]) {
        let concrete = caches.compactMap { $0 as? CBv2LayerCache }
        guard !concrete.isEmpty, concrete.count == caches.count,
            concrete.enumerated().allSatisfy({ index, cache in
                cache.layerIndex == index && cache.kind.sharesKVWithLayer == nil && cache.rows.isEmpty
                    && cache.gemmaUnifiedPositions == nil
            }), let value = snapshot(concrete[0].positionOffsets), let identity = constantIdentity(value) else { return }
        let group = Gemma4UnifiedPositions(caches: concrete, value: value, identity: identity)
        for (index, cache) in concrete.enumerated() {
            guard cache.adoptUnifiedPosition(value) else { group.detach(); return }
            cache.gemmaUnifiedPositions = group
            cache.gemmaUnifiedPositionIndex = index
        }
    }

    func accepts(_ caches: [any CBv2AttendingLayerCache]) -> Bool {
        isActive && caches.count == members.count && zip(caches, members).allSatisfy { live, member in
            guard let live = live as? CBv2LayerCache, let bound = member.cache else { return false }
            return live === bound && live.gemmaUnifiedPositions === self
        }
    }

    func finishBinding() {
        guard isActive else { return }
        let caches = members.compactMap(\.cache)
        guard caches.count == members.count, let canonical = caches.last else { rebuildSeparately(caches); return }
        let rows = canonical.rows
        guard caches.allSatisfy({ cache in
            cache.rows.count == rows.count && cache.rows.indices.allSatisfy {
                cache.rows[$0].absoluteOffset == rows[$0].absoluteOffset
            }
        }), StreamOrDevice.default == stream else { rebuildSeparately(caches); return }
        let base = canonical.rebuildUnifiedPosition()
        guard let held = Self.snapshot(base), let id = Self.constantIdentity(held) else { rebuildSeparately(caches); return }
        for cache in caches {
            guard cache.adoptUnifiedPosition(held) else { rebuildSeparately(caches); return }
        }
        value = held
        identity = id
        rowCount = rows.count
        bindingVersion &+= 1
        completedWidth = nil
        nextValue = nil
        cycle.reset()
    }

    func prepare(_ cache: CBv2LayerCache, count: Int) {
        guard isActive else { return }
        let matches = StreamOrDevice.default == stream && rowCount > 0 && cache.rows.count == rowCount
            && Self.constantIdentity(cache.positionOffsets) == identity
        guard cycle.begin(layer: cache.gemmaUnifiedPositionIndex, count: count, inputMatches: matches) else {
            detach(); return
        }
        if nextValue == nil { nextValue = value + Int32(count) }
    }

    /// False leaves the caller on its original per-cache increment.
    func complete(_ cache: CBv2LayerCache, count: Int) -> Bool {
        guard isActive, Self.constantIdentity(cache.positionOffsets) == identity, let nextValue else {
            detach(); return false
        }
        let completion = cycle.finish(layer: cache.gemmaUnifiedPositionIndex, count: count)
        guard completion != .declined, cache.adoptUnifiedPosition(nextValue) else { detach(); return false }
        if completion == .advanced {
            guard let id = Self.constantIdentity(nextValue) else { detach(); return true }
            value = nextValue
            identity = id
            self.nextValue = nil
            updateVersion &+= 1
            completedWidth = count
        }
        return true
    }

    func evaluationStamp() -> EvaluationStamp? {
        guard isActive, rowCount > 0, cycle.isIdle, StreamOrDevice.default == stream else { return nil }
        return EvaluationStamp(binding: bindingVersion, updates: updateVersion, rows: rowCount)
    }

    func evaluationRoot(after stamp: EvaluationStamp, expectedUpdates: Int, expectedWidth: Int) -> MLXArray? {
        guard isActive, rowCount == stamp.rows, StreamOrDevice.default == stream,
            Gemma4CacheRootPolicy.validates(binding: bindingVersion, updates: updateVersion,
                previousBinding: stamp.binding, previousUpdates: stamp.updates,
                expectedUpdates: expectedUpdates, expectedWidth: expectedWidth,
                completedWidth: completedWidth, idle: cycle.isIdle),
            members.allSatisfy({ member in
                guard let cache = member.cache else { return false }
                return cache.rows.count == rowCount && Self.constantIdentity(cache.positionOffsets) == identity
            }) else { return nil }
        return Self.snapshot(value)
    }

    func detach() {
        guard isActive else { return }
        isActive = false
        nextValue = nil
        cycle.reset()
        for member in members {
            if member.cache?.gemmaUnifiedPositions === self { member.cache?.gemmaUnifiedPositions = nil }
        }
    }

    private func rebuildSeparately(_ caches: [CBv2LayerCache]) {
        detach()
        for cache in caches { _ = cache.rebuildUnifiedPosition() }
    }

    static func snapshot(_ array: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, array.ctx) == 0 else { mlx_array_free(context); return nil }
        return MLXArray(context)
    }

    private static func constantIdentity(_ array: MLXArray) -> UInt? {
        var id: UInt = 0
        var allowed = false
        guard _mlx_array_constant_cache_identity(&id, &allowed, array.ctx) == 0, allowed else { return nil }
        return id
    }
}

extension CBv2LayerCache: CBv2CoordinatedPositionBinding {
    var positionBindingCoordinator: (any CBv2PositionBindingCoordinator)? { gemmaUnifiedPositions }

    /// Explicit Gemma-only construction hook. Paged/mixed/shared banks decline.
    public static func configureGemmaUnifiedPositions(_ caches: [any CBv2AttendingLayerCache]) {
        Gemma4UnifiedPositions.configure(caches)
    }
}
