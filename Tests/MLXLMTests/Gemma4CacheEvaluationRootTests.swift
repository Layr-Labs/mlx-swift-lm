// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX
import Testing

@testable import MLXLMCommon

/// Native materialization tests. Authored only while model/GPU execution is held.
@Suite("Gemma4 scoped cache evaluation roots", .serialized)
struct Gemma4CacheEvaluationRootTests {
    private final class Fixture: CBv2SteppableModel, CBv2CacheOutputCoverageProviding {
        var affirms = true
        func cacheOutputCoversAttention(_ scope: Gemma4CacheEvaluationScope) -> Bool { affirms }
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let n = tokens.dim(1)
            // Actual Gemma full/sliding geometry/dtype. The original toy
            // FP32/head4 fixture has its own ungated-substrate repro below.
            var output = MLXArray.zeros([1], dtype: .float32)
            for cache in caches {
                let dimension = cache.kind.headDim
                let q = (MLXArray(0..<(16 * n * dimension)).asType(.float32) % 31 / 16 + 0.5)
                    .asType(.bfloat16).reshaped(1, 16, n, dimension)
                let kv = q[0..., 0..<cache.kind.kvHeads, 0..., 0...]
                output = output + sum(cache.updateAndAttend(queries: q, keys: kv, values: kv,
                    scale: 1 / Float(dimension).squareRoot(), sinks: nil), axes: [1, 2, 3])
            }
            return output
        }
    }

    private func fixture(window: Int = 8) -> (Fixture, [CBv2LayerCache], CBv2LayerCacheBank, [[CBv2SequenceKV?]]) {
        let caches = [
            CBv2LayerCache(layerIndex: 0, kind: .init(attention: .full, headDim: 512, kvHeads: 2, queryHeads: 16)),
            CBv2LayerCache(layerIndex: 1, kind: .init(attention: .slidingWindow(window), headDim: 256, kvHeads: 8, queryHeads: 16))
        ]
        CBv2LayerCache.configureGemmaUnifiedPositions(caches)
        let bank = CBv2LayerCacheBank(caches: caches)
        let rows: [[CBv2SequenceKV?]] = [[
            CBv2FullSequenceKV(promptLength: 0, maxLength: 64, kvHeads: 2, headDim: 512),
            CBv2WindowedSequenceKV(window: window, kvHeads: 8, headDim: 256)
        ]]
        _ = bank.layerCaches(rowStates: rows)
        return (Fixture(), caches, bank, rows)
    }

    private let enabled = Gemma4CacheRootPolicy(environment: [
        "DARKBLOOM_GEMMA4_COMPACT_DECODE_ROOTS": "1", "DARKBLOOM_GEMMA4_COMPACT_MTP_ROOTS": "1"])

    private func available(_ array: MLXArray) -> Bool {
        var result = false
        return _mlx_array_is_available(&result, array.ctx) == 0 && result
    }

    @Test func decodeAndDisconnectedChunkWritesMaterialize() throws {
        let (model, caches, _, _) = fixture()
        for count in [1, 1, 3, 1] {
            let scope: Gemma4CacheEvaluationScope = count == 1 ? .decode : .mtpVerify
            let request = try #require(Gemma4CacheEvaluationRequest.prepare(model: model, caches: caches,
                scope: scope, expectedUpdates: 1, expectedWidth: count, policy: enabled))
            let output = model.forward(tokens: MLXArray.zeros([1, count], dtype: .int32), caches: caches)
            let roots = try #require(request.roots(forwardOutput: output))
            if count > 1 { #expect(roots.count == 4) } // output, positions, disconnected ring K/V
            eval(roots)
            // Do not call asArray on storage: that could conceal an omitted root.
            #expect(caches.flatMap { $0.innerState() }.allSatisfy(available))
        }
    }

    @Test func stagedRingRootsRemainExplicitEvenWithoutHistory() throws {
        let (model, caches, _, rows) = fixture(window: 1)
        _ = model.forward(tokens: MLXArray.zeros([1, 1], dtype: .int32), caches: caches)
        let window = try #require(rows[0][1] as? CBv2WindowedSequenceKV)
        window.beginSpeculativeWrite()
        let request = try #require(Gemma4CacheEvaluationRequest.prepare(model: model, caches: caches,
            scope: .mtpVerify, expectedUpdates: 1, expectedWidth: 2, policy: enabled))
        let output = model.forward(tokens: MLXArray.zeros([1, 2], dtype: .int32), caches: caches)
        let roots = try #require(request.roots(forwardOutput: output))
        #expect(roots.count == 4)
        eval(roots)
        #expect(caches.flatMap { $0.innerState() }.allSatisfy(available))
        window.rollback(1)
        window.commitSpeculativeWrite()
        #expect(!window.cacheOutputCoversStorage)
    }

    @Test func staleReboundAndUnprovenScopesDecline() throws {
        let (model, caches, bank, rows) = fixture()
        model.affirms = false
        #expect(Gemma4CacheEvaluationRequest.prepare(model: model, caches: caches,
            scope: .decode, expectedUpdates: 1, expectedWidth: 1, policy: enabled) == nil)
        model.affirms = true
        let stale = try #require(Gemma4CacheEvaluationRequest.prepare(model: model, caches: caches,
            scope: .decode, expectedUpdates: 1, expectedWidth: 1, policy: enabled))
        #expect(stale.roots(forwardOutput: MLXArray.zeros([1])) == nil)
        bank.invalidateBoundComposition()
        _ = bank.layerCaches(rowStates: rows)
        let output = model.forward(tokens: MLXArray.zeros([1, 1], dtype: .int32), caches: caches)
        #expect(stale.roots(forwardOutput: output) == nil)
    }
}

/// Independent baseline-substrate diagnostic: no unified positions or compact
/// roots. Preserve this case rather than silently treating its missing stock
/// dot-product symbol as a root-optimization failure or a passed test.
@Suite("Gemma4 original small Float32 fixture substrate", .serialized)
struct Gemma4CacheRootSmallFloatFixtureTests {
    @Test func ordinaryAttentionRequiresItsStockKernel() {
        let cache = CBv2LayerCache(layerIndex: 0,
            kind: .init(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 1))
        cache.setRows([CBv2FullSequenceKV(promptLength: 0, maxLength: 64, kvHeads: 1, headDim: 4)])
        let q = (MLXArray(0..<4).asType(.float32) / 16 + 0.5).reshaped(1, 1, 1, 4)
        let output = cache.updateAndAttend(queries: q, keys: q, values: q, scale: 0.5, sinks: nil)
        eval([output] + cache.innerState())
    }
}
