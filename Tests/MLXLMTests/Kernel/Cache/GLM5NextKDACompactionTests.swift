// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextKDACompactionTests {
        private typealias State = GLM5NextLinearAttention.State

        private func oversized(_ seed: Float, channels: Int = 192) -> State {
            let length = 2052
            let history = MLXArray(
                (0 ..< length * channels).map { seed + Float($0 % 251) },
                [1, length, channels])
            return State(
                convolution: history[0..., (length - 4)..., 0...].transposed(0, 2, 1),
                recurrent: MLXArray.ones([1, 2, 8, 8]) * seed)
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { eval(arrays) }
        }

        private func assertCompact(_ value: MLXArray) throws {
            let observed = try value.evaluatedBufferInfo()
            let info = try #require(observed)
            let upper = try Memory.allocationFootprintUpperBound(
                byteCount: value.nbytes + 16_384)
            #expect(info.allocatedBytes <= upper)
            #expect(info.isRowContiguous)
        }

        @Test(arguments: [1, 192])
        func advanceCompactsPhysicalBackingAndPreservesValues(channels: Int) throws {
            let raw = oversized(3, channels: channels)
            try fence([raw.convolution, raw.recurrent])
            let observed = try raw.convolution.evaluatedBufferInfo()
            let original = try #require(observed)
            // Prove the fixture really retains more than the proposed bound.
            let upper = try Memory.allocationFootprintUpperBound(
                byteCount: raw.convolution.nbytes + 16_384)
            if channels > 1 { #expect(original.allocatedBytes > upper) }
            let expected = raw.convolution.asArray(Float.self)
            let cache = GLM5NextKDACache()
            cache.advance(raw, count: 2048)
            try fence(cache.innerState())
            try assertCompact(cache.state[0])
            #expect(cache.state[0].asArray(Float.self) == expected)
            #expect(cache.state[1].asArray(Float.self) == raw.recurrent.asArray(Float.self))
            #expect(cache.offset == 2048)
        }

        @Test func rollbackCopyAndAdoptKeepIndependentCompactStates() throws {
            let cache = GLM5NextKDACache()
            cache.advance(oversized(7), count: 2048)
            try fence(cache.innerState())
            let first = cache.state.map { $0.asArray(Float.self) }
            let snapshot = try #require(cache.copy() as? GLM5NextKDACache)
            cache.advance(oversized(19), count: 2048)
            try fence(cache.innerState())
            #expect(cache.offset == 4096 && snapshot.offset == 2048)
            for value in cache.innerState() { try assertCompact(value) }
            #expect(snapshot.state.map { $0.asArray(Float.self) } == first)
            let adopted = GLM5NextKDACache()
            adopted.adopt(cache)
            try adopted.truncate(to: 2048)
            try fence(adopted.innerState())
            #expect(adopted.state.map { $0.asArray(Float.self) } == first)
            #expect(cache.offset == 4096)
            for value in adopted.innerState() { try assertCompact(value) }
            try cache.truncate(to: 2048)
            #expect(cache.state.map { $0.asArray(Float.self) } == first)
        }

        @Test func importedPendingRollbackIsIncludedInFence() throws {
            let imported = oversized(31)
            let cache = GLM5NextKDACache()
            cache.state = [imported.convolution, imported.recurrent]
            cache.metaState = ["glm5_kda_v1", "2048"]
            // Do not evaluate the import before it becomes rollback state.
            cache.advance(oversized(47), count: 1)
            try fence(cache.innerState())
            #expect(cache.innerState().count == 4)
            for value in cache.innerState() { try assertCompact(value) }
            try cache.truncate(to: 2048)
            #expect(cache.state[0].asArray(Float.self) == imported.convolution.asArray(Float.self))
            #expect(cache.state[1].asArray(Float.self) == imported.recurrent.asArray(Float.self))
            for value in cache.innerState() { try assertCompact(value) }
        }
    }
}
