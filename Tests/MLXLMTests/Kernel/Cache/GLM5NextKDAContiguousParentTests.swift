// Copyright © 2026 Eigen Labs.
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextKDAContiguousParentTests {
        @Test func rowContiguousOversizedImportDoesNotKeepParentAllocation() throws {
            let parent = MLXArray((0 ..< 131072).map { Float($0 % 251) })
            let window = parent[65536 ..< 65552].reshaped(1, 1, 16)
            try MLX.withError { eval(parent, window) }
            let beforeInfo = try window.evaluatedBufferInfo()
            let before = try #require(beforeInfo)
            let upper = try Memory.allocationFootprintUpperBound(
                byteCount: window.nbytes + 16_384)
            #expect(before.isRowContiguous)
            #expect(before.dataOffset > 0)
            #expect(before.allocatedBytes > upper)
            let expected = window.asArray(Float.self)
            let cache = GLM5NextKDACache()
            cache.state = [window, MLXArray.zeros([1, 1, 4, 4])]
            try MLX.withError { eval(cache.innerState()) }
            let afterInfo = try cache.state[0].evaluatedBufferInfo()
            let after = try #require(afterInfo)
            #expect(after.isRowContiguous)
            #expect(after.allocatedBytes <= upper)
            #expect(cache.state[0].asArray(Float.self) == expected)
            // Hold both parent and original view through the assertion: no donation shortcut.
            #expect(parent.asArray(Float.self)[65536] == expected[0])
            #expect(window.asArray(Float.self) == expected)
        }
    }
}
