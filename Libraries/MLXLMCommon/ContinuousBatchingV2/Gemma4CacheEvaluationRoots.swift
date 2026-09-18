// Copyright © 2026 Eigen Labs.
import MLX

/// The standard decode logits / full verify hidden tensor consumes every
/// cache-returned attention result. Custom output paths must decline.
public protocol CBv2CacheOutputCoverageModel: AnyObject {
    func cbv2CacheOutputCoversAttention(_ scope: Gemma4CacheEvaluationScope) -> Bool
}

public protocol CBv2CacheOutputCoverageProviding: AnyObject {
    func cacheOutputCoversAttention(_ scope: Gemma4CacheEvaluationScope) -> Bool
}

/// A row may retain storage roots not covered by its last returned views.
/// Unknown row implementations do not qualify for omission.
protocol CBv2CacheRootCoverageRow: CBv2SequenceKV {
    func uncoveredCacheEvaluationRoots() -> [MLXArray]
}

extension CBv2FullSequenceKV: CBv2CacheRootCoverageRow {
    // Keep the degenerate single-key context explicit as well.
    func uncoveredCacheEvaluationRoots() -> [MLXArray] { retainedCount > 1 ? [] : cbv2InnerState() }
}

extension CBv2WindowedSequenceKV: CBv2CacheRootCoverageRow {
    func uncoveredCacheEvaluationRoots() -> [MLXArray] {
        cacheOutputCoversStorage && retainedCount > 1 ? [] : cbv2InnerState()
    }
}

/// One forward/verify scope only. Rebinding, partial cycles and unexpected
/// update counts decline; no tensor value is read back to make that decision.
struct Gemma4CacheEvaluationRequest {
    private let caches: [CBv2LayerCache]
    private let group: Gemma4UnifiedPositions
    private let stamp: Gemma4UnifiedPositions.EvaluationStamp
    private let updates: Int
    private let width: Int

    static func prepare(model: any CBv2SteppableModel, caches: [any CBv2AttendingLayerCache],
                        scope: Gemma4CacheEvaluationScope, expectedUpdates: Int, expectedWidth: Int,
                        policy: Gemma4CacheRootPolicy = .process) -> Self? {
        guard policy.enabled(scope), expectedUpdates > 0, expectedWidth > 0,
            scope != .decode || (expectedUpdates == 1 && expectedWidth == 1),
            (model as? any CBv2CacheOutputCoverageProviding)?.cacheOutputCoversAttention(scope) == true else { return nil }
        let owners = caches.compactMap { $0 as? CBv2LayerCache }
        guard owners.count == caches.count, let group = owners.first?.gemmaUnifiedPositions,
            group.accepts(caches), let stamp = group.evaluationStamp(),
            owners.allSatisfy({ cache in
                cache.kind.sharesKVWithLayer == nil && cache.rows.count == stamp.rows
                    && cache.boundSpanContexts == nil
                    && cache.rows.allSatisfy { $0 is any CBv2CacheRootCoverageRow }
            }) else { return nil }
        return Self(caches: owners, group: group, stamp: stamp, updates: expectedUpdates, width: expectedWidth)
    }

    func roots(forwardOutput: MLXArray) -> [MLXArray]? {
        guard forwardOutput.ndim > 0, forwardOutput.dim(0) == stamp.rows, forwardOutput.size > 0,
            group.accepts(caches), let positions = group.evaluationRoot(after: stamp,
                expectedUpdates: updates, expectedWidth: width) else { return nil }
        var roots = [forwardOutput, positions]
        for cache in caches {
            guard cache.rows.count == stamp.rows else { return nil }
            for row in cache.rows {
                guard let covered = row as? any CBv2CacheRootCoverageRow else { return nil }
                roots.append(contentsOf: covered.uncoveredCacheEvaluationRoots())
            }
        }
        return roots
    }
}
