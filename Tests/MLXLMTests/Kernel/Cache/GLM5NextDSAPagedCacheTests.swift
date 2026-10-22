// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// A paged DSA history (`GLM5NextDSACache(pagePool:account:)`) reads
    /// exactly what the contiguous one reads, bit for bit, through appends of
    /// any size, rollbacks inside and across pages, copies that diverge (copy
    /// on write at a shared partial page), adoption, serialization and the
    /// pool's growth; and every page comes back when the histories go.
    @Suite(.serialized)
    struct GLM5NextDSAPagedCacheTests {
        /// Row values that encode (position, column, plane, history).
        private struct Chunk {
            let latents, keys, gates, validity: MLXArray
            init(start: Int, count: Int, widths: (Int, Int), dtype: DType, marker: Int) {
                func rows(width: Int, plane: Int) -> MLXArray {
                    var values: [Float] = []
                    values.reserveCapacity(count * width)
                    for position in start ..< start + count {
                        for column in 0 ..< width {
                            let code: Int = (marker * 7 + plane * 3 + position * 11 + column * 13) % 251
                            values.append(Float(code) / 8 - 15)
                        }
                    }
                    return MLXArray(values).reshaped(1, count, width).asType(dtype)
                }
                latents = rows(width: widths.0, plane: 1)
                keys = rows(width: widths.1, plane: 2)
                gates = rows(width: widths.1, plane: 3)
                validity = MLXArray((start ..< start + count).map { ($0 + marker) % 5 != 2 })
                    .reshaped(1, count)
            }
        }

        private static func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        private func append(_ cache: GLM5NextDSACache, _ chunk: Chunk) throws {
            try cache.append(
                latents: chunk.latents, keys: chunk.keys, gates: chunk.gates,
                validity: chunk.validity)
        }

        /// The paged history's readers against the contiguous one's.
        private func expectSame(
            _ paged: GLM5NextDSACache, _ contiguous: GLM5NextDSACache, _ label: String
        ) {
            #expect(paged.offset == contiguous.offset, "\(label) offset")
            #expect(paged.metaState == contiguous.metaState, "\(label) meta")
            let left = [paged.latents, paged.indexKeys, paged.poolGates, paged.validity]
            let right = [
                contiguous.latents, contiguous.indexKeys, contiguous.poolGates, contiguous.validity,
            ]
            for (index, (a, b)) in zip(left, right).enumerated() {
                #expect((a == nil) == (b == nil), "\(label) plane \(index) presence")
                guard let a, let b else { continue }
                eval(a, b)
                #expect(
                    a.shape == b.shape && a.dtype == b.dtype && Self.bytes(a) == Self.bytes(b),
                    "\(label) plane \(index)")
            }
        }

        @Test(arguments: [(3, 2, DType.float32, 4), (3, 2, DType.bfloat16, 4), (512, 128, DType.bfloat16, 64), (512, 128, DType.bfloat16, 16)])
        func pagedHistoryReadsTheContiguousHistory(latent: Int, key: Int, dtype: DType, page: Int)
            throws
        {
            try Device.withDefaultDevice(.gpu) {
                let pool = try GLM5NextDSAPagePool(
                    pageSize: page, latentWidth: latent, keyWidth: key, latentType: dtype,
                    keyType: dtype, gateType: dtype, minimumGrowthPages: 4)
                let account = try pool.reserve(pages: 400)
                let paged = GLM5NextDSACache(pagePool: pool, account: account)
                let contiguous = GLM5NextDSACache()
                var position = 0
                func step(_ count: Int, marker: Int = 0) throws {
                    let chunk = Chunk(
                        start: position, count: count, widths: (latent, key), dtype: dtype,
                        marker: marker)
                    try append(paged, chunk)
                    try append(contiguous, chunk)
                    position += count
                }
                // Prefill-sized, decode-sized and window-sized appends.
                for (index, count) in [5, 1, page * 3 + 2, 3, 1, 1, 2 * page, 7].enumerated() {
                    try step(count, marker: index)
                    expectSame(paged, contiguous, "append \(index)")
                }
                // Rollback inside the tail page, across a page, to a boundary.
                for back in [{ position - 2 }, { position - page - 1 }, { (position / page) * page }] {
                    let keep = back()
                    try paged.truncate(to: keep)
                    try contiguous.truncate(to: keep)
                    position = keep
                    expectSame(paged, contiguous, "rollback to \(keep)")
                    try step(3, marker: keep)
                    expectSame(paged, contiguous, "append after rollback to \(keep)")
                }
                // A transaction: a working copy appends; the original is intact;
                // the original adopts it.
                let working = paged.copy() as! GLM5NextDSACache
                let workingReference = contiguous.copy() as! GLM5NextDSACache
                let window = Chunk(
                    start: position, count: 3, widths: (latent, key), dtype: dtype, marker: 99)
                try append(working, window)
                try append(workingReference, window)
                expectSame(working, workingReference, "working copy")
                expectSame(paged, contiguous, "original under a working copy")
                try working.truncate(to: position + 1)
                try workingReference.truncate(to: position + 1)
                paged.adopt(working)
                contiguous.adopt(workingReference)
                position += 1
                expectSame(paged, contiguous, "adopted")
                // Two copies diverge from one shared partial page (a prefix
                // entry restored twice) and the original goes on too.
                let first = paged.copy() as! GLM5NextDSACache
                let second = paged.copy() as! GLM5NextDSACache
                let firstReference = contiguous.copy() as! GLM5NextDSACache
                let secondReference = contiguous.copy() as! GLM5NextDSACache
                for (marker, pair) in [
                    (1, (first, firstReference)), (2, (second, secondReference)),
                    (3, (paged, contiguous)),
                ] {
                    let chunk = Chunk(
                        start: position, count: page + 3, widths: (latent, key), dtype: dtype,
                        marker: 1000 + marker)
                    try append(pair.0, chunk)
                    try append(pair.1, chunk)
                }
                expectSame(first, firstReference, "first copy")
                expectSame(second, secondReference, "second copy")
                expectSame(paged, contiguous, "original after copies")
                // Serialized state round trip into fresh pages.
                let restored = GLM5NextDSACache(pagePool: pool, account: account)
                restored.state = first.state
                expectSame(restored, firstReference, "restored state")
                // Every page comes back once nothing holds it.
                eval(paged.innerState() + first.innerState() + second.innerState())
                #expect(pool.allocatedPages > 0)
            }
        }

        @Test func pagesComeBackAndThePoolShrinks() throws {
            try Device.withDefaultDevice(.gpu) {
                let pool = try GLM5NextDSAPagePool(
                    pageSize: 4, latentWidth: 3, keyWidth: 2, latentType: .float32,
                    keyType: .float32, gateType: .float32, minimumGrowthPages: 4)
                let account = try pool.reserve(pages: 40)
                #expect(pool.capacityPages >= 40)
                var histories: [GLM5NextDSACache] = (0 ..< 3).map { _ in
                    GLM5NextDSACache(pagePool: pool, account: account)
                }
                for (index, history) in histories.enumerated() {
                    try append(
                        history,
                        Chunk(
                            start: 0, count: 9 + index, widths: (3, 2), dtype: .float32,
                            marker: index))
                }
                eval(histories.flatMap { $0.innerState() })
                #expect(pool.allocatedPages == 9)
                histories.removeAll()
                pool.recycle()
                #expect(pool.allocatedPages == 0)
                pool.release(account)
                #expect(try pool.trim() > 0)
                #expect(pool.capacityPages == 0)
            }
        }
    }
}
