// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The capacity-backed DSA history must be indistinguishable from the
    /// historical whole-history concatenation for every reader and every
    /// rewind/copy/adopt sequence.
    @Suite(.serialized)
    struct GLM5NextDSACacheTests {
        /// A chunk whose every element encodes (position, column, plane).
        private struct Chunk {
            let latents, keys, gates, validity: MLXArray
            init(start: Int, count: Int, marker: Float = 0) {
                func rows(width: Int, plane: Float) -> MLXArray {
                    let values = (start ..< start + count).flatMap { position in
                        (0 ..< width).map { marker + plane * 1000 + Float(position * 10 + $0) }
                    }
                    return MLXArray(values).reshaped(1, count, width)
                }
                latents = rows(width: 3, plane: 1)
                keys = rows(width: 2, plane: 2)
                gates = rows(width: 2, plane: 3)
                validity = MLXArray((start ..< start + count).map { $0 % 3 != 1 })
                    .reshaped(1, count)
            }
            var arrays: [MLXArray] { [latents, keys, gates, validity] }
        }

        /// Reference: the previous implementation's concatenated history.
        private final class Reference {
            var arrays: [MLXArray] = []
            var offset: Int { arrays.first?.dim(1) ?? 0 }
            func append(_ chunk: Chunk) {
                arrays =
                    arrays.isEmpty
                    ? chunk.arrays
                    : zip(arrays, chunk.arrays).map { concatenated([$0, $1], axis: 1) }
            }
            func truncate(to count: Int) {
                arrays = count == 0 ? [] : arrays.map { $0[0..., ..<count] }
            }
        }

        private func append(_ cache: GLM5NextDSACache, _ chunk: Chunk) throws {
            try cache.append(
                latents: chunk.latents, keys: chunk.keys, gates: chunk.gates,
                validity: chunk.validity)
        }

        private func expectEqual(_ cache: GLM5NextDSACache, _ reference: Reference) {
            #expect(cache.offset == reference.offset)
            #expect(cache.metaState == ["glm5_dsa_v1", String(reference.offset)])
            let views = [cache.latents, cache.indexKeys, cache.poolGates, cache.validity]
            #expect(cache.state.count == reference.arrays.count)
            guard !reference.arrays.isEmpty else {
                #expect(views.allSatisfy { $0 == nil })
                return
            }
            for (actual, expected) in zip(views, reference.arrays) {
                guard let actual else {
                    Issue.record("missing DSA history plane")
                    continue
                }
                #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
            for (actual, expected) in zip(cache.state, reference.arrays) {
                #expect(actual.shape == expected.shape)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
        }

        @Test(arguments: [GLM5NextDSACache.Growth.geometric, .exact])
        func appendTruncateCopyAndAdoptMatchConcatenatedHistory(
            growth: GLM5NextDSACache.Growth
        ) throws {
            let cache = GLM5NextDSACache(growth: growth)
            let reference = Reference()
            expectEqual(cache, reference)
            var position = 0
            for count in [5, 1, 1, 300, 7, 1] {
                let chunk = Chunk(start: position, count: count)
                try append(cache, chunk)
                reference.append(chunk)
                position += count
                expectEqual(cache, reference)
            }
            if growth == .geometric {
                // Spare columns exist and remain invisible to every reader.
                #expect(cache.capacity > cache.offset)
            } else {
                #expect(cache.capacity == cache.offset)
            }

            // A copy is an independent snapshot of the same logical history.
            let snapshot = cache.copy() as! GLM5NextDSACache
            let snapshotReference = Reference()
            snapshotReference.arrays = reference.arrays

            // Rewind, then overwrite the rewound columns with different data.
            try cache.truncate(to: 200)
            reference.truncate(to: 200)
            expectEqual(cache, reference)
            let rewritten = Chunk(start: 200, count: 20, marker: 0.5)
            try append(cache, rewritten)
            reference.append(rewritten)
            expectEqual(cache, reference)
            // The snapshot never observes the overwrite.
            expectEqual(snapshot, snapshotReference)

            #expect(cache.trim(15) == 15)
            reference.truncate(to: 205)
            expectEqual(cache, reference)

            // Adopt replaces the whole logical history; later appends to
            // either side remain independent.
            let other = GLM5NextDSACache(growth: growth)
            other.adopt(snapshot)
            expectEqual(other, snapshotReference)
            let tail = Chunk(start: snapshotReference.offset, count: 2, marker: 0.25)
            try append(other, tail)
            snapshotReference.append(tail)
            expectEqual(other, snapshotReference)
            snapshotReference.truncate(to: snapshotReference.offset - 2)
            expectEqual(snapshot, snapshotReference)

            try cache.truncate(to: 0)
            reference.truncate(to: 0)
            expectEqual(cache, reference)
            #expect(cache.capacity == 0)
        }

        @Test func serializedStateIsExactlyTheLogicalHistory() throws {
            let cache = GLM5NextDSACache()
            try append(cache, Chunk(start: 0, count: 3))
            try append(cache, Chunk(start: 3, count: 2))
            #expect(cache.capacity >= GLM5NextDSACache.minimumCapacity)
            let restored = GLM5NextDSACache()
            restored.state = cache.state
            restored.metaState = cache.metaState
            #expect(restored.offset == 5 && restored.capacity == 5)
            let reference = Reference()
            reference.append(Chunk(start: 0, count: 3))
            reference.append(Chunk(start: 3, count: 2))
            expectEqual(restored, reference)
        }

        @Test func appendRefusesChangedGeometry() throws {
            let cache = GLM5NextDSACache()
            try append(cache, Chunk(start: 0, count: 2))
            #expect(throws: GLM5NextLoadError.self) {
                try cache.append(
                    latents: MLXArray.zeros([1, 1, 4]), keys: MLXArray.zeros([1, 1, 2]),
                    gates: MLXArray.zeros([1, 1, 2]), validity: MLXArray.ones([1, 1], dtype: .bool))
            }
            #expect(cache.offset == 2)
        }

        /// The CBv2 payload rows of `GLM5NextDSAPayloadKernel` have the bits of
        /// the composed validity cast, concatenation and FP32 casts, for each
        /// activation dtype and for strided views of a wider projection.
        @Test(arguments: [DType.bfloat16, .float16, .float32])
        func payloadKernelMatchesComposedRowsBitwise(dtype: DType) throws {
            try Device.withDefaultDevice(.gpu) {
                for (batch, count) in [(1, 1), (1, 4), (2, 3), (1, 2048)] {
                    let fused = MLXArray(
                        (0 ..< batch * count * 900).map { Float(($0 * 7919) % 4093) / 97 - 21 },
                        [batch, count, 900]
                    ).asType(dtype)
                    let latents = fused[0..., 0..., ..<512]
                    let keys = fused[0..., 0..., 512 ..< 640]
                    let gates = fused[0..., 0..., 700 ..< 828]
                    let validity = MLXArray(
                        (0 ..< batch * count).map { $0 % 5 != 2 }, [batch, count])
                    let staged = try #require(
                        GLM5NextDSAPayloadKernel.stage(
                            latents: latents, keys: keys, gates: gates, validity: validity))
                    let values = concatenated(
                        [keys, gates, validity.asType(keys.dtype).expandedDimensions(axis: -1)],
                        axis: -1)
                    let expectedKeys = latents.asType(.float32).expandedDimensions(axis: 1)
                    let expectedValues = values.asType(.float32).expandedDimensions(axis: 1)
                    #expect(staged.keys.shape == expectedKeys.shape)
                    #expect(staged.values.shape == expectedValues.shape)
                    #expect(staged.keys.dtype == .float32 && staged.values.dtype == .float32)
                    #expect(
                        staged.keys.asArray(Float.self) == expectedKeys.asArray(Float.self),
                        "B=\(batch) T=\(count) \(dtype)")
                    #expect(
                        staged.values.asArray(Float.self) == expectedValues.asArray(Float.self),
                        "B=\(batch) T=\(count) \(dtype)")
                }
                #expect(
                    GLM5NextDSAPayloadKernel.stage(
                        latents: MLXArray.zeros([1, 2, 4]), keys: MLXArray.zeros([1, 2, 3]),
                        gates: MLXArray.zeros([1, 3, 3]), validity: MLXArray.ones([1, 2]).asType(.bool))
                        == nil)
            }
        }
    }
}
