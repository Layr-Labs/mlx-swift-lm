import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    @Suite("CBv2 elastic sliding-window storage", .serialized)
    struct CBv2ElasticWindowStorageTests {
        private func tensor(
            start: Int, count: Int, heads: Int = 1, width: Int = 4, dtype: DType = .float32
        ) -> MLXArray {
            // Dense input matters for allocator receipts: a broadcast fixture
            // can occupy only `count` elements while advertising the whole shape.
            let elements = (0 ..< heads * count * width).map { index -> Float in
                let head = index / (count * width)
                let token = (index / width) % count
                let dimension = index % width
                return Float(start + token) + Float(head) * 0.25 + Float(dimension) * 0.001
            }
            return MLXArray(elements).reshaped([1, heads, count, width]).asType(dtype)
        }

        private func update(
            _ row: CBv2WindowedSequenceKV, start: Int, count: Int,
            heads: Int = 1, width: Int = 4, valueWidth: Int = 4, dtype: DType = .float32
        ) -> (MLXArray, MLXArray) {
            row.update(
                keys: tensor(start: start, count: count, heads: heads, width: width, dtype: dtype),
                values: tensor(
                    start: start + 100, count: count, heads: heads, width: valueWidth, dtype: dtype))
        }

        private func assertSnapshot(
            _ row: CBv2WindowedSequenceKV, start: Int, count: Int,
            heads: Int = 1, width: Int = 4, valueWidth: Int = 4, dtype: DType = .float32
        ) {
            let snapshot = row.snapshot()
            #expect(snapshot.offset == row.absoluteOffset)
            #expect(row.retainedCount == count)
            #expect(arrayEqual(
                snapshot.keys,
                tensor(start: start, count: count, heads: heads, width: width, dtype: dtype)
            ).item(Bool.self))
            #expect(arrayEqual(
                snapshot.values,
                tensor(start: start + 100, count: count, heads: heads, width: valueWidth, dtype: dtype)
            ).item(Bool.self))
        }

        @Test func shortGemmaRowsOwnOnlyPopulatedCapacity() throws {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                let row = CBv2WindowedSequenceKV(
                    window: 1_024, kvHeads: 8, headDim: 256, elasticStorage: true)
                #expect(row.byteCount == 0)
                _ = update(row, start: 0, count: 32, heads: 8, width: 256, valueWidth: 256, dtype: dtype)
                eval(row.cbv2InnerState())
                #expect(row.byteCount == 2 * 8 * 32 * 256 * dtype.size)
                #expect(row.cbv2InnerState().allSatisfy { $0.dim(2) == 32 })
                for array in row.cbv2InnerState() {
                    let backing = try #require(try array.evaluatedBufferInfo())
                    #expect(backing.dataOffset == 0 && backing.dataElements == array.size)
                    #expect(backing.allocatedBytes >= array.nbytes)
                    #expect(backing.allocatedBytes < 8 * 1_024 * 256 * dtype.size)
                }
                assertSnapshot(row, start: 0, count: 32, heads: 8, width: 256, valueWidth: 256, dtype: dtype)
            }
        }

        @Test func growthRemapsAbsoluteSlotsWithoutDiscardingHistory() {
            // Nonzero replay origin and a non-power-of-two window force wrapped
            // physical ranges during growth as well as the final semantic ring.
            let row = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 4, valueHeadDim: 3, initialOffset: 997,
                elasticStorage: true)
            for position in 997 ..< 1_070 {
                _ = update(row, start: position, count: 1, valueWidth: 3)
                let retained = min(33, position + 1 - 997)
                assertSnapshot(row, start: position + 1 - retained, count: retained, valueWidth: 3)
                let capacity = row.cbv2InnerState()[0].dim(2)
                #expect(capacity >= retained && capacity <= 33)
                #expect(capacity < 2 * retained || capacity == 33)
            }
        }

        @Test func oversizedChunkReturnsPreEvictionHistoryAndKeepsRecentTail() {
            let row = CBv2WindowedSequenceKV(window: 17, kvHeads: 1, headDim: 4, elasticStorage: true)
            _ = update(row, start: 0, count: 5)
            let before = row.snapshot()
            let returned = update(row, start: 5, count: 40)
            #expect(arrayEqual(returned.0, tensor(start: 0, count: 45)).item(Bool.self))
            #expect(arrayEqual(returned.1, tensor(start: 100, count: 45)).item(Bool.self))
            #expect(arrayEqual(before.keys, tensor(start: 0, count: 5)).item(Bool.self))
            let borrowed = row.borrowableViews()
            #expect(arrayEqual(borrowed.keys, returned.0).item(Bool.self))
            assertSnapshot(row, start: 28, count: 17)
            row.rollback(3)
            assertSnapshot(row, start: 28, count: 14)
            _ = update(row, start: 42, count: 4)
            assertSnapshot(row, start: 29, count: 17)
        }

        @Test func speculativeGrowthChargesOnlyConfirmedStorage() {
            let row = CBv2WindowedSequenceKV(window: 1_024, kvHeads: 1, headDim: 4, elasticStorage: true)
            _ = update(row, start: 0, count: 7)
            let initialBytes = row.byteCount
            row.beginSpeculativeWrite()
            _ = update(row, start: 7, count: 100)
            #expect(row.byteCount > initialBytes) // Includes the staged tensors.
            row.rollback(98)
            row.commitSpeculativeWrite()
            #expect(row.cbv2InnerState()[0].dim(2) == 16)
            #expect(row.byteCount == 2 * 16 * 4 * DType.float32.size)
            assertSnapshot(row, start: 0, count: 9)

            let beforeCancelBytes = row.byteCount
            row.beginSpeculativeWrite()
            _ = update(row, start: 9, count: 1_100)
            row.rollback(1_100)
            row.commitSpeculativeWrite()
            #expect(row.byteCount == beforeCancelBytes)
            assertSnapshot(row, start: 0, count: 9)
        }

        @Test func rollbackBeforeWindowFillAndFreshReplayRemainExact() {
            let row = CBv2WindowedSequenceKV(window: 128, kvHeads: 1, headDim: 4, elasticStorage: true)
            row.fastForward(to: 1_001)
            _ = update(row, start: 1_001, count: 5)
            row.rollback(5)
            #expect(row.retainedCount == 0)
            _ = update(row, start: 1_001, count: 20)
            #expect(row.cbv2InnerState()[0].dim(2) == 32)
            assertSnapshot(row, start: 1_001, count: 20)
        }

        @Test func defaultPreservesFullRingForUnqualifiedModels() {
            let row = CBv2WindowedSequenceKV(window: 128, kvHeads: 1, headDim: 4)
            _ = update(row, start: 0, count: 3)
            #expect(row.cbv2InnerState()[0].dim(2) == 128)
            assertSnapshot(row, start: 0, count: 3)
        }
    }
}
