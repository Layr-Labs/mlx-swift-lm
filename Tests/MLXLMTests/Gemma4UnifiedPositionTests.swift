// Copyright © 2026 Eigen Labs.
import MLX
import Testing

@testable import MLXLMCommon

/// Native cache tests authored only; not run in the source-only phase.
@Suite("Gemma4 coordinated contiguous positions", .serialized)
struct Gemma4UnifiedPositionTests {
    private func caches() -> [CBv2LayerCache] {
        (0..<2).map {
            CBv2LayerCache(layerIndex: $0,
                kind: CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 1))
        }
    }

    private func row() -> CBv2FullSequenceKV {
        CBv2FullSequenceKV(promptLength: 0, maxLength: 32, kvHeads: 1, headDim: 4)
    }

    @Test func partialAndCompleteFramesPreserveIndependentWrappers() throws {
        let caches = caches()
        CBv2LayerCache.configureGemmaUnifiedPositions(caches)
        let group = try #require(caches[0].gemmaUnifiedPositions)
        let bank = CBv2LayerCacheBank(caches: caches)
        let rows = [row(), row()]
        _ = bank.layerCaches(rowStates: [[rows[0], rows[1]]])
        let old = caches[0].positionOffsets
        group.prepare(caches[0], count: 3)
        #expect(group.complete(caches[0], count: 3))
        #expect(caches[0].positionOffsets.asArray(Int32.self) == [3])
        #expect(caches[1].positionOffsets.asArray(Int32.self) == [0])
        old._updateInternal(MLXArray([Int32(99)]))
        group.prepare(caches[1], count: 3)
        #expect(group.complete(caches[1], count: 3))
        #expect(caches[0].positionOffsets !== caches[1].positionOffsets)
        #expect(caches[1].positionOffsets.asArray(Int32.self) == [3])
        caches[0].positionOffsets._updateInternal(MLXArray([Int32(7)]))
        group.prepare(caches[0], count: 1)
        #expect(!group.isActive)
        #expect(caches[0].positionOffsets.asArray(Int32.self) == [7])
        #expect(caches[1].positionOffsets.asArray(Int32.self) == [3])
        #expect(caches.allSatisfy { $0.gemmaUnifiedPositions == nil })
    }

    @Test func realAttentionKeepsAllEvaluationRootsAndPositions() throws {
        let unified = caches(), ordinary = caches()
        CBv2LayerCache.configureGemmaUnifiedPositions(unified)
        let unifiedBank = CBv2LayerCacheBank(caches: unified)
        let ordinaryBank = CBv2LayerCacheBank(caches: ordinary)
        let unifiedRows = [[row(), row()], [row(), row()]]
        let ordinaryRows = [[row(), row()], [row(), row()]]
        _ = unifiedBank.layerCaches(rowStates: unifiedRows)
        _ = ordinaryBank.layerCaches(rowStates: ordinaryRows)
        var total: Int32 = 0
        for count in [3, 1, 2, 1] {
            let q = MLXArray.ones([2, 1, count, 4], dtype: .bfloat16)
            var actual: [MLXArray] = [], expected: [MLXArray] = []
            for layer in 0..<2 {
                actual.append(unified[layer].updateAndAttend(queries: q, keys: q, values: q, scale: 0.5, sinks: nil))
                expected.append(ordinary[layer].updateAndAttend(queries: q, keys: q, values: q, scale: 0.5, sinks: nil))
            }
            eval(actual + expected + unified.flatMap { $0.innerState() } + ordinary.flatMap { $0.innerState() })
            total += Int32(count)
            for layer in 0..<2 {
                #expect(actual[layer].asData(access: .copy).data == expected[layer].asData(access: .copy).data)
                #expect(unified[layer].positionOffsets.asArray(Int32.self) == [total, total])
                #expect(unified[layer].innerState().count == ordinary[layer].innerState().count)
            }
            #expect(unified[0].gemmaUnifiedPositions?.isActive == true)
        }
        for request in unifiedRows { for row in request { row.rollback(1) } }
        unifiedBank.invalidateBoundComposition()
        _ = unifiedBank.layerCaches(rowStates: unifiedRows)
        #expect(unified[0].positionOffsets.asArray(Int32.self) == [total - 1, total - 1])
        unifiedBank.releaseBoundRows()
        #expect(unified.allSatisfy { $0.rows.isEmpty && $0.positionOffsets.size == 0 })
    }

    @Test func incompatibleBindingAndPublicMutationDecline() throws {
        let caches = caches()
        CBv2LayerCache.configureGemmaUnifiedPositions(caches)
        let bank = CBv2LayerCacheBank(caches: caches)
        let first = row(), second = row()
        let k = MLXArray.zeros([1, 1, 1, 4], dtype: .bfloat16)
        _ = second.update(keys: k, values: k)
        _ = bank.layerCaches(rowStates: [[first, second]])
        #expect(caches.allSatisfy { $0.gemmaUnifiedPositions == nil })
        #expect(caches[0].positionOffsets.asArray(Int32.self) == [0])
        #expect(caches[1].positionOffsets.asArray(Int32.self) == [1])
        let fresh = self.caches()
        CBv2LayerCache.configureGemmaUnifiedPositions(fresh)
        fresh[0].setRows([row()])
        #expect(fresh.allSatisfy { $0.gemmaUnifiedPositions == nil })
    }
}
