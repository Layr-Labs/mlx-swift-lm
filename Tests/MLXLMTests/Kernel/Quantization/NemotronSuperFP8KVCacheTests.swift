import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    struct NemotronSuperFP8KVCacheTests {
        private func rows(_ values: [Float]) -> MLXArray {
            MLXArray(values).reshaped(1, 1, values.count / 2, 2)
        }
        @Test func roundTripAppendTrimAndTrustedReplay() throws {
            let cache = try NemotronSuperFP8KVCache(
                heads: 1, headDimension: 2, capacityTokens: 4, dtype: .float32)
            let first = rows([1.0625, 1.1875, -500, 0.001])
            let (keys, _) = try cache.checkedUpdate(keys: first, values: first)
            #expect(keys.asArray(Float.self) == [1, 1.25, -448, 1.0 / 512])
            let snapshot = cache.state
            _ = try cache.checkedUpdate(keys: rows([3, 4]), values: rows([5, 6]))
            #expect(cache.offset == 3 && cache.trim(1) == 1)
            let (replayed, _) = try cache.checkedUpdate(keys: rows([7, 8]), values: rows([9, 10]))
            #expect(replayed.asArray(Float.self).suffix(2) == [7, 8])
            try cache.restore(snapshot)
            #expect(cache.offset == 2)
            #expect(cache.trim(-1) == 0 && cache.trim(99) == 2 && cache.state.isEmpty)
        }
        @Test func copyAndInvalidAppendPreserveOriginalState() throws {
            let cache = try NemotronSuperFP8KVCache(
                heads: 1, headDimension: 2, capacityTokens: 2, dtype: .float32)
            _ = try cache.checkedUpdate(keys: rows([1, 2]), values: rows([3, 4]))
            let copied = try #require(cache.copy() as? NemotronSuperFP8KVCache)
            copied.trim(1)
            _ = try copied.checkedUpdate(keys: rows([9, 10]), values: rows([11, 12]))
            #expect(cache.offset == 1)
            let before = cache.state
            #expect(throws: (any Error).self) {
                try cache.checkedUpdate(keys: rows([1, 2, 3, 4]), values: rows([1, 2, 3, 4]))
            }
            #expect(throws: (any Error).self) {
                try cache.checkedUpdate(
                    keys: rows([1, 2]), values: rows([1, 2]).asType(.bfloat16))
            }
            #expect(throws: (any Error).self) { try cache.restore([rows([1, 2]), rows([3, 4])]) }
            #expect(cache.offset == 1)
            #expect(arrayEqual(cache.state[0], before[0]).item(Bool.self))
            #expect(arrayEqual(cache.state[1], before[1]).item(Bool.self))
        }
        @Test func configurationAndAccountingAreBounded() throws {
            #expect(throws: (any Error).self) {
                try NemotronSuperFP8KVCache(heads: Int.max, headDimension: 2, capacityTokens: 2)
            }
            let cache = try NemotronSuperFP8KVCache(
                heads: 2, headDimension: 128, capacityTokens: 1024)
            #expect(cache.maximumResidentBytes == 524288)
            #expect(cache.operationAllowanceBytes == 18_874_368)
            #expect(cache.maxSize == 1024 && cache.isTrimmable)
            #expect(throws: (any Error).self) {
                try cache.checkedMask(n: 1025, windowSize: nil, returnArray: true)
            }
            let long = try NemotronSuperFP8KVCache(heads: 1, headDimension: 1, capacityTokens: 2048)
            #expect(throws: (any Error).self) {
                try long.checkedMask(n: 2048, windowSize: nil, returnArray: true)
            }
        }
    }
}
