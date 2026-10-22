// CBv2PagedSharedPositionOffsetsTests.swift
//
// The paged layer caches of one pool share their RoPE offset arrays: one
// host build per membership change and one device add per step, instead of
// one of each per layer. Values must match the per-layer arithmetic exactly.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedSharedPositionOffsets", .serialized)
struct CBv2PagedSharedPositionOffsetsTests {

    private func qkv(_ kind: CBv2LayerKind, batch: Int, tokens: Int)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        (
            MLXRandom.normal([batch, kind.queryHeads, tokens, kind.headDim], dtype: .float16),
            MLXRandom.normal([batch, kind.kvHeads, tokens, kind.headDim], dtype: .float16),
            MLXRandom.normal([batch, kind.kvHeads, tokens, kind.headDim], dtype: .float16)
        )
    }

    @Test func cacheSharesByValueAndAdvancesByIdentity() {
        let cache = PagedPositionOffsetsCache()
        let a = cache.offsets([3, 7])
        #expect(cache.offsets([3, 7]) === a)
        let b = cache.offsets([3, 8])
        #expect(b !== a)
        #expect(cache.hostBuilds == 2)

        let a1 = cache.advance(a, by: 1)
        #expect(cache.advance(a, by: 1) === a1)
        #expect(cache.advance(a, by: 2) !== a1)
        #expect(a1.asArray(Int32.self) == [4, 8])
        // An equal-valued but distinct source is advanced on its own.
        let c = MLXArray([Int32(3), 7])
        #expect(cache.advance(c, by: 1) !== a1)
        #expect(cache.deviceAdvances == 3)

        // Bounded: the oldest entries fall out, results stay exact.
        for step in 0 ..< 8 { _ = cache.advance(a, by: Int32(10 + step)) }
        #expect(cache.advance(a, by: 1).asArray(Int32.self) == [4, 8])
    }

    /// Four layers (full and sliding) of one pool decode a 2-row batch: every
    /// layer holds the same offsets array, the pool issues one add per step,
    /// and the values are the rows' absolute positions.
    @Test func layersShareOneOffsetsArrayPerStep() throws {
        let full = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let sliding = CBv2LayerKind(
            attention: .slidingWindow(32), headDim: 64, kvHeads: 2, queryHeads: 4)
        let kinds = [full, sliding, sliding, full]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 16 << 20, maxPrefillChunk: 64, nominalMaxSequenceLength: 512))
        let caches = backend.makeLayerCaches()
        let pool = backend.pool
        let scale = 1 / Float(64).squareRoot()
        let a = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 256)
        let b = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 256)
        defer {
            eval(caches.flatMap { $0.innerState() })
            for cache in caches { cache.setRows([]) }
            backend.release(a)
            backend.release(b)
        }

        // Prefill each row alone (different lengths), then decode the batch.
        for (state, length) in [(a, 5), (b, 9)] {
            for (index, cache) in caches.enumerated() { cache.setRows([state[index]!]) }
            for (index, cache) in caches.enumerated() {
                let x = qkv(kinds[index], batch: 1, tokens: length)
                _ = cache.updateAndAttend(
                    queries: x.q, keys: x.k, values: x.v, scale: scale, sinks: nil)
            }
            eval(caches.flatMap { $0.innerState() })
        }
        let buildsBefore = pool.positionOffsets.hostBuilds
        for (index, cache) in caches.enumerated() { cache.setRows([a[index]!, b[index]!]) }
        #expect(pool.positionOffsets.hostBuilds == buildsBefore + 1, "one build for all layers")
        #expect(caches.allSatisfy { $0.positionOffsets === caches[0].positionOffsets })
        #expect(caches.allSatisfy { $0.positionOffsetsHostRebuilds >= 1 })

        for step in 0 ..< 6 {
            let advancesBefore = pool.positionOffsets.deviceAdvances
            for (index, cache) in caches.enumerated() {
                let x = qkv(kinds[index], batch: 2, tokens: 1)
                _ = cache.updateAndAttend(
                    queries: x.q, keys: x.k, values: x.v, scale: scale, sinks: nil)
            }
            eval(caches.flatMap { $0.innerState() })
            #expect(pool.positionOffsets.deviceAdvances == advancesBefore + 1)
            #expect(caches.allSatisfy { $0.positionOffsets === caches[0].positionOffsets })
            #expect(
                caches[0].positionOffsets.asArray(Int32.self) == [
                    Int32(6 + step), Int32(10 + step),
                ])
        }
    }
}
